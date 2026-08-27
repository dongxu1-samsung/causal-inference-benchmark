"""
CausalODE: IPM-Regularized Neural ODE for Causal Inference ITE Estimation.

Architecture:
- Input projection: Linear(input_dim → latent_dim)
- Neural ODE with Euler integration (no torchdiffeq dependency)
- Adaptive IPM (MMD with RBF kernel) regularization
- Dual outcome heads for Y(0) and Y(1)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def mmd_rbf(X, Y, sigma=1.0):
    """Compute MMD with RBF kernel between two sets of samples."""
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return torch.tensor(0.0, device=X.device)

    n_x = X.shape[0]
    n_y = Y.shape[0]

    # Pairwise squared distances
    XX = torch.mm(X, X.t())
    YY = torch.mm(Y, Y.t())
    XY = torch.mm(X, Y.t())

    X_sqnorms = torch.diag(XX)
    Y_sqnorms = torch.diag(YY)

    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i.x_j
    dists_XX = X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0) - 2 * XX
    dists_YY = Y_sqnorms.unsqueeze(1) + Y_sqnorms.unsqueeze(0) - 2 * YY
    dists_XY = X_sqnorms.unsqueeze(1) + Y_sqnorms.unsqueeze(0) - 2 * XY

    # RBF kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))
    gamma = 1.0 / (2.0 * sigma * sigma)
    K_XX = torch.exp(-gamma * dists_XX.clamp(min=0))
    K_YY = torch.exp(-gamma * dists_YY.clamp(min=0))
    K_XY = torch.exp(-gamma * dists_XY.clamp(min=0))

    # Unbiased MMD^2 estimate
    # Remove diagonal for unbiased estimate
    if n_x > 1:
        mmd_xx = (K_XX.sum() - K_XX.diag().sum()) / (n_x * (n_x - 1))
    else:
        mmd_xx = torch.tensor(0.0, device=X.device)

    if n_y > 1:
        mmd_yy = (K_YY.sum() - K_YY.diag().sum()) / (n_y * (n_y - 1))
    else:
        mmd_yy = torch.tensor(0.0, device=X.device)

    mmd_xy = K_XY.sum() / (n_x * n_y)

    mmd_sq = mmd_xx + mmd_yy - 2 * mmd_xy
    return mmd_sq.clamp(min=0)


def euler_integrate(f, z0, n_steps=15):
    """Simple Euler integration from t=0 to t=1."""
    dt = 1.0 / n_steps
    z = z0
    for i in range(n_steps):
        z = z + dt * f(z)
    return z


class ODEFunc(nn.Module):
    """ODE dynamics function f_θ: dΦ/dt = f_θ(Φ)"""

    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),  # Tanh to keep dynamics bounded
        )

    def forward(self, z):
        return self.net(z)


class OutcomeHead(nn.Module):
    """MLP outcome head."""

    def __init__(self, latent_dim, head_dim, n_layers):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, head_dim))
            layers.append(nn.ELU())
            in_dim = head_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z).squeeze(-1)


class CausalODEModel(nn.Module):
    """Full CausalODE model."""

    def __init__(self, input_dim, latent_dim=64, ode_steps=15,
                 n_head_layers=2, head_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.ode_steps = ode_steps

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ELU(),
        )

        # ODE dynamics
        self.ode_func = ODEFunc(latent_dim)

        # Outcome heads
        self.head_0 = OutcomeHead(latent_dim, head_dim, n_head_layers)
        self.head_1 = OutcomeHead(latent_dim, head_dim, n_head_layers)

    def get_representation(self, X):
        """Project input and integrate ODE to get final representation."""
        z0 = self.input_proj(X)
        z_final = euler_integrate(self.ode_func, z0, self.ode_steps)
        return z_final

    def forward(self, X):
        """Forward pass returning predictions for both potential outcomes."""
        z_final = self.get_representation(X)
        y0 = self.head_0(z_final)
        y1 = self.head_1(z_final)
        return y0, y1, z_final


def estimate_propensity_overlap(t_batch):
    """
    Estimate propensity overlap from treatment assignment in batch.
    Overlap is high when treatment proportions are balanced (close to 0.5/0.5).
    Returns value in [0, 1] where 1 = perfectly balanced.
    """
    if len(t_batch) == 0:
        return 0.5
    prop = t_batch.float().mean().item()
    # Overlap = 1 - |prop - 0.5| * 2, ranges from 0 (all one group) to 1 (balanced)
    overlap = 1.0 - abs(prop - 0.5) * 2.0
    return max(0.0, min(1.0, overlap))


def compute_adaptive_ipm(z_final, t_batch, lambda_0=1.0, sigma=1.0):
    """
    Compute adaptive IPM loss.
    IPM_strength = lambda_0 * (1 - overlap)
    When groups are balanced (overlap high), reduce regularization.
    When imbalanced (overlap low), increase regularization.
    """
    treated_mask = (t_batch == 1)
    control_mask = (t_batch == 0)

    n_treated = treated_mask.sum().item()
    n_control = control_mask.sum().item()

    # Edge case: all treated or all control
    if n_treated < 2 or n_control < 2:
        return torch.tensor(0.0, device=z_final.device)

    z_treated = z_final[treated_mask]
    z_control = z_final[control_mask]

    # Compute overlap
    overlap = estimate_propensity_overlap(t_batch)

    # Adaptive strength
    ipm_strength = lambda_0 * (1.0 - overlap)

    # Compute MMD
    mmd_loss = mmd_rbf(z_treated, z_control, sigma=sigma)

    return ipm_strength * mmd_loss


def train_causal_ode(X_train, t_train, y_train, input_dim, config=None):
    """
    Train the CausalODE model.

    Parameters:
        X_train: numpy array of covariates (n, input_dim)
        t_train: numpy array of binary treatment (n,)
        y_train: numpy array of outcomes (n,)
        input_dim: int, number of input features
        config: dict with optional keys for hyperparameters and validation data

    Returns:
        model_dict: dict with 'model' and 'config'
    """
    if config is None:
        config = {}

    # Hyperparameters
    latent_dim = config.get('latent_dim', 64)
    ode_steps = config.get('ode_steps', 15)
    n_epochs = config.get('n_epochs', 100)
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 256)
    patience = config.get('patience', 10)
    lambda_0 = config.get('lambda_0', 1.0)
    n_head_layers = config.get('n_head_layers', 2)
    head_dim = config.get('head_dim', 32)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(device)
    t_tr = torch.LongTensor(t_train.astype(int)).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)

    # Validation data
    X_val = config.get('_X_val', None)
    t_val = config.get('_t_val', None)
    y_val = config.get('_y_val', None)
    has_val = X_val is not None and t_val is not None and y_val is not None

    if has_val:
        X_v = torch.FloatTensor(X_val).to(device)
        t_v = torch.LongTensor(t_val.astype(int)).to(device)
        y_v = torch.FloatTensor(y_val).to(device)

    # Create model
    model = CausalODEModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        ode_steps=ode_steps,
        n_head_layers=n_head_layers,
        head_dim=head_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # DataLoader
    dataset = TensorDataset(X_tr, t_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, t_batch, y_batch in loader:
            optimizer.zero_grad()

            y0_pred, y1_pred, z_final = model(X_batch)

            # Factual loss: MSE on observed outcome
            treated_mask = (t_batch == 1)
            control_mask = (t_batch == 0)

            loss_parts = []

            if treated_mask.sum() > 0:
                loss_treated = nn.functional.mse_loss(
                    y1_pred[treated_mask], y_batch[treated_mask]
                )
                loss_parts.append(loss_treated)

            if control_mask.sum() > 0:
                loss_control = nn.functional.mse_loss(
                    y0_pred[control_mask], y_batch[control_mask]
                )
                loss_parts.append(loss_control)

            if len(loss_parts) == 0:
                continue

            factual_loss = sum(loss_parts) / len(loss_parts)

            # Adaptive IPM loss
            ipm_loss = compute_adaptive_ipm(z_final, t_batch, lambda_0=lambda_0)

            # Total loss
            loss = factual_loss + ipm_loss

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation for early stopping
        if has_val:
            model.eval()
            with torch.no_grad():
                y0_val_pred, y1_val_pred, _ = model(X_v)
                treated_val = (t_v == 1)
                control_val = (t_v == 0)

                val_losses = []
                if treated_val.sum() > 0:
                    val_losses.append(
                        nn.functional.mse_loss(y1_val_pred[treated_val], y_v[treated_val]).item()
                    )
                if control_val.sum() > 0:
                    val_losses.append(
                        nn.functional.mse_loss(y0_val_pred[control_val], y_v[control_val]).item()
                    )

                val_loss = np.mean(val_losses) if val_losses else float('inf')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        else:
            # Without validation, just save the last model
            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best state
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()

    return {
        'model': model,
        'device': device,
        'input_dim': input_dim,
        'config': {
            'latent_dim': latent_dim,
            'ode_steps': ode_steps,
            'n_head_layers': n_head_layers,
            'head_dim': head_dim,
        }
    }


def predict_causal_ode(model_dict, X_test):
    """
    Predict ITE for test data.

    Parameters:
        model_dict: dict returned by train_causal_ode
        X_test: numpy array of covariates (n, input_dim)

    Returns:
        ite: numpy array of ITE predictions (n,)
    """
    model = model_dict['model']
    device = model_dict['device']

    model.eval()
    X_t = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        y0_pred, y1_pred, _ = model(X_t)
        ite = (y1_pred - y0_pred).cpu().numpy()

    return ite
