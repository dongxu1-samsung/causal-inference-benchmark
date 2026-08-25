"""
DRNet - Dose-Response Network (Schwab et al., 2020)
Architecture: Shared representation + treatment-specific heads with dose strata
For binary treatment, this simplifies to a TARNet-like architecture with IPM regularization.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DRNet(nn.Module):
    """Dose-Response Network for binary treatment."""
    def __init__(self, input_dim, repr_dim=64, head_dim=64, n_repr_layers=3,
                 n_head_layers=2, n_treatments=2, activation="elu"):
        super().__init__()

        # Shared representation
        repr_layers = []
        prev_dim = input_dim
        for _ in range(n_repr_layers):
            repr_layers.append(nn.Linear(prev_dim, repr_dim))
            repr_layers.append(nn.ELU() if activation == "elu" else nn.ReLU())
            repr_layers.append(nn.BatchNorm1d(repr_dim))
            prev_dim = repr_dim
        self.repr_net = nn.Sequential(*repr_layers)

        # Treatment-specific heads
        self.heads = nn.ModuleList()
        for _ in range(n_treatments):
            head_layers = []
            prev_dim = repr_dim
            for _ in range(n_head_layers):
                head_layers.append(nn.Linear(prev_dim, head_dim))
                head_layers.append(nn.ELU() if activation == "elu" else nn.ReLU())
                prev_dim = head_dim
            head_layers.append(nn.Linear(prev_dim, 1))
            self.heads.append(nn.Sequential(*head_layers))

        # Propensity network (for targeted regularization)
        self.propensity_net = nn.Sequential(
            nn.Linear(repr_dim, repr_dim),
            nn.ELU(),
            nn.Linear(repr_dim, 1),
        )

    def forward(self, x, t):
        phi = self.repr_net(x)
        y0 = self.heads[0](phi).squeeze(-1)
        y1 = self.heads[1](phi).squeeze(-1)
        prop = torch.sigmoid(self.propensity_net(phi).squeeze(-1))
        y_pred = t * y1 + (1 - t) * y0
        return y_pred, y0, y1, phi, prop

    def predict_ite(self, x):
        phi = self.repr_net(x)
        y0 = self.heads[0](phi).squeeze(-1)
        y1 = self.heads[1](phi).squeeze(-1)
        return y1 - y0


def train_drnet(X_train, t_train, y_train, input_dim, config=None):
    """Train DRNet model with external validation for early stopping."""
    if config is None:
        config = {}

    repr_dim = config.get("repr_dim", 64)
    head_dim = config.get("head_dim", 64)
    n_repr_layers = config.get("n_repr_layers", 3)
    n_head_layers = config.get("n_head_layers", 2)
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 64)
    n_epochs = config.get("n_epochs", 300)
    weight_decay = config.get("weight_decay", 1e-4)
    alpha_prop = config.get("alpha_prop", 0.5)  # propensity weight
    patience = config.get("patience", 30)

    # Validation data for early stopping
    X_val_ext = config.get("_X_val")
    t_val_ext = config.get("_t_val")
    y_val_ext = config.get("_y_val")

    model = DRNet(input_dim, repr_dim, head_dim, n_repr_layers, n_head_layers)

    X_t = torch.FloatTensor(X_train)
    t_t = torch.FloatTensor(t_train)
    y_t = torch.FloatTensor(y_train)

    dataset = TensorDataset(X_t, t_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Use external val data if provided
    if X_val_ext is not None:
        X_val = torch.FloatTensor(X_val_ext)
        t_val = torch.FloatTensor(t_val_ext)
        y_val = torch.FloatTensor(y_val_ext)
    else:
        # Fallback: internal split
        n = len(X_t)
        n_val = int(0.2 * n)
        perm = torch.randperm(n)
        val_idx = perm[:n_val]
        X_val, t_val, y_val = X_t[val_idx], t_t[val_idx], y_t[val_idx]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    model.train()
    for epoch in range(n_epochs):
        for batch_x, batch_t, batch_y in loader:
            optimizer.zero_grad()

            y_pred, y0, y1, phi, prop = model(batch_x, batch_t)

            # Outcome loss
            loss_outcome = nn.MSELoss()(y_pred, batch_y)

            # Propensity loss
            loss_prop = nn.BCELoss()(prop, batch_t)

            loss = loss_outcome + alpha_prop * loss_prop
            loss.backward()
            optimizer.step()

        # Validation for early stopping
        model.eval()
        with torch.no_grad():
            y_pred_val, _, _, _, _ = model(X_val, t_val)
            val_loss = nn.MSELoss()(y_pred_val, y_val).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_drnet(model, X_test):
    """Predict ITE using trained DRNet."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test)
        ite = model.predict_ite(X_t).numpy()
    return ite
