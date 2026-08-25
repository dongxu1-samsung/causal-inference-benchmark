"""
CFRNet - Counterfactual Regression Network (Shalit et al., 2017)
Architecture: Shared representation + two outcome heads + IPM regularization
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class CFRNet(nn.Module):
    def __init__(self, input_dim, repr_dim=200, hypo_dim=100, n_repr_layers=3,
                 n_hypo_layers=3, activation="elu"):
        super().__init__()
        self.repr_dim = repr_dim

        # Representation network (shared)
        repr_layers = []
        prev_dim = input_dim
        for _ in range(n_repr_layers):
            repr_layers.append(nn.Linear(prev_dim, repr_dim))
            repr_layers.append(nn.ELU() if activation == "elu" else nn.ReLU())
            prev_dim = repr_dim
        self.repr_net = nn.Sequential(*repr_layers)

        # Outcome head for control (t=0)
        head0_layers = []
        prev_dim = repr_dim
        for _ in range(n_hypo_layers):
            head0_layers.append(nn.Linear(prev_dim, hypo_dim))
            head0_layers.append(nn.ELU() if activation == "elu" else nn.ReLU())
            prev_dim = hypo_dim
        head0_layers.append(nn.Linear(prev_dim, 1))
        self.head0 = nn.Sequential(*head0_layers)

        # Outcome head for treated (t=1)
        head1_layers = []
        prev_dim = repr_dim
        for _ in range(n_hypo_layers):
            head1_layers.append(nn.Linear(prev_dim, hypo_dim))
            head1_layers.append(nn.ELU() if activation == "elu" else nn.ReLU())
            prev_dim = hypo_dim
        head1_layers.append(nn.Linear(prev_dim, 1))
        self.head1 = nn.Sequential(*head1_layers)

    def forward(self, x, t):
        """Forward pass returning predicted outcomes."""
        phi = self.repr_net(x)
        y0 = self.head0(phi).squeeze(-1)
        y1 = self.head1(phi).squeeze(-1)
        # Return factual outcome
        y_pred = t * y1 + (1 - t) * y0
        return y_pred, y0, y1, phi

    def predict_ite(self, x):
        """Predict individual treatment effect."""
        phi = self.repr_net(x)
        y0 = self.head0(phi).squeeze(-1)
        y1 = self.head1(phi).squeeze(-1)
        return y1 - y0


def wasserstein_distance(phi_t, phi_c):
    """Approximate Wasserstein distance between representations."""
    # Sinkhorn-like approximation using sorted distance
    if len(phi_t) == 0 or len(phi_c) == 0:
        return torch.tensor(0.0)
    mean_t = phi_t.mean(dim=0)
    mean_c = phi_c.mean(dim=0)
    return torch.norm(mean_t - mean_c, p=2)


def mmd_distance(phi_t, phi_c, sigma=1.0):
    """Maximum Mean Discrepancy between treatment/control representations."""
    if len(phi_t) == 0 or len(phi_c) == 0:
        return torch.tensor(0.0)

    def rbf_kernel(x, y, sigma):
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.exp(-torch.sum(diff ** 2, dim=-1) / (2 * sigma ** 2))

    k_tt = rbf_kernel(phi_t, phi_t, sigma).mean()
    k_cc = rbf_kernel(phi_c, phi_c, sigma).mean()
    k_tc = rbf_kernel(phi_t, phi_c, sigma).mean()
    return k_tt + k_cc - 2 * k_tc


def train_cfrnet(X_train, t_train, y_train, input_dim, config=None):
    """Train CFRNet model with early stopping on validation set."""
    if config is None:
        config = {}

    repr_dim = config.get("repr_dim", 200)
    hypo_dim = config.get("hypo_dim", 100)
    n_repr_layers = config.get("n_repr_layers", 3)
    n_hypo_layers = config.get("n_hypo_layers", 3)
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 100)
    n_epochs = config.get("n_epochs", 300)
    alpha = config.get("alpha", 1.0)  # IPM weight
    weight_decay = config.get("weight_decay", 1e-4)
    ipm_type = config.get("ipm_type", "wass")  # 'wass' or 'mmd'
    patience = config.get("patience", 20)

    # Validation data for early stopping
    X_val = config.get("_X_val")
    t_val = config.get("_t_val")
    y_val = config.get("_y_val")

    model = CFRNet(input_dim, repr_dim, hypo_dim, n_repr_layers, n_hypo_layers)

    # Convert to tensors
    X_t = torch.FloatTensor(X_train)
    t_t = torch.FloatTensor(t_train)
    y_t = torch.FloatTensor(y_train)

    dataset = TensorDataset(X_t, t_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Validation tensors
    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val)
        t_val_t = torch.FloatTensor(t_val)
        y_val_t = torch.FloatTensor(y_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        for batch_x, batch_t, batch_y in loader:
            optimizer.zero_grad()

            y_pred, y0, y1, phi = model(batch_x, batch_t)

            # Factual loss
            loss_factual = nn.MSELoss()(y_pred, batch_y)

            # IPM regularization
            idx_t = batch_t == 1
            idx_c = batch_t == 0
            if ipm_type == "mmd":
                loss_ipm = mmd_distance(phi[idx_t], phi[idx_c])
            else:
                loss_ipm = wasserstein_distance(phi[idx_t], phi[idx_c])

            loss = loss_factual + alpha * loss_ipm
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            scheduler.step()

        # Early stopping on validation
        if X_val is not None and (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                y_pred_val, _, _, _ = model(X_val_t, t_val_t)
                val_loss = nn.MSELoss()(y_pred_val, y_val_t).item()
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_cfrnet(model, X_test):
    """Predict ITE using trained CFRNet."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test)
        ite = model.predict_ite(X_t).numpy()
    return ite
