"""
CATENets - Conditional Average Treatment Effect Networks (Curth & van der Schaar, 2021)
Implements: TARNet, SNet (with disentangled representations), FlexTENet, DragonNet

All use shared representation + treatment-specific outcome heads with various regularization.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TARNet(nn.Module):
    """Treatment-Agnostic Representation Network (baseline)."""
    def __init__(self, input_dim, repr_dim=200, out_dim=100, n_repr_layers=3, n_out_layers=2):
        super().__init__()

        # Shared representation
        repr_layers = []
        prev = input_dim
        for _ in range(n_repr_layers):
            repr_layers.extend([nn.Linear(prev, repr_dim), nn.ELU()])
            prev = repr_dim
        self.repr_net = nn.Sequential(*repr_layers)

        # Head 0 (control)
        h0 = []
        prev = repr_dim
        for _ in range(n_out_layers):
            h0.extend([nn.Linear(prev, out_dim), nn.ELU()])
            prev = out_dim
        h0.append(nn.Linear(prev, 1))
        self.head0 = nn.Sequential(*h0)

        # Head 1 (treated)
        h1 = []
        prev = repr_dim
        for _ in range(n_out_layers):
            h1.extend([nn.Linear(prev, out_dim), nn.ELU()])
            prev = out_dim
        h1.append(nn.Linear(prev, 1))
        self.head1 = nn.Sequential(*h1)

    def forward(self, x, t):
        phi = self.repr_net(x)
        y0 = self.head0(phi).squeeze(-1)
        y1 = self.head1(phi).squeeze(-1)
        y_pred = t * y1 + (1 - t) * y0
        return y_pred, y0, y1, phi

    def predict_ite(self, x):
        phi = self.repr_net(x)
        y0 = self.head0(phi).squeeze(-1)
        y1 = self.head1(phi).squeeze(-1)
        return y1 - y0


class SNet(nn.Module):
    """SNet with disentangled representations (confounding + outcome-specific + instrumental)."""
    def __init__(self, input_dim, repr_dim_big=100, repr_dim_small=50,
                 out_dim=100, n_repr_layers=3, n_out_layers=2):
        super().__init__()

        # Confounding representation (affects both T and Y)
        conf_layers = []
        prev = input_dim
        for _ in range(n_repr_layers):
            conf_layers.extend([nn.Linear(prev, repr_dim_big), nn.ELU()])
            prev = repr_dim_big
        self.conf_net = nn.Sequential(*conf_layers)

        # Outcome-specific representation
        out_layers = []
        prev = input_dim
        for _ in range(n_repr_layers):
            out_layers.extend([nn.Linear(prev, repr_dim_small), nn.ELU()])
            prev = repr_dim_small
        self.out_net = nn.Sequential(*out_layers)

        # Instrumental representation (affects T only)
        inst_layers = []
        prev = input_dim
        for _ in range(n_repr_layers):
            inst_layers.extend([nn.Linear(prev, repr_dim_small), nn.ELU()])
            prev = repr_dim_small
        self.inst_net = nn.Sequential(*inst_layers)

        # Outcome heads (receive conf + outcome repr)
        total_repr = repr_dim_big + repr_dim_small
        h0 = []
        prev = total_repr
        for _ in range(n_out_layers):
            h0.extend([nn.Linear(prev, out_dim), nn.ELU()])
            prev = out_dim
        h0.append(nn.Linear(prev, 1))
        self.head0 = nn.Sequential(*h0)

        h1 = []
        prev = total_repr
        for _ in range(n_out_layers):
            h1.extend([nn.Linear(prev, out_dim), nn.ELU()])
            prev = out_dim
        h1.append(nn.Linear(prev, 1))
        self.head1 = nn.Sequential(*h1)

        # Propensity head (receives conf + instrumental)
        prop_repr = repr_dim_big + repr_dim_small
        self.prop_net = nn.Sequential(
            nn.Linear(prop_repr, out_dim),
            nn.ELU(),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x, t):
        phi_conf = self.conf_net(x)
        phi_out = self.out_net(x)
        phi_inst = self.inst_net(x)

        # Outcome prediction
        phi_y = torch.cat([phi_conf, phi_out], dim=-1)
        y0 = self.head0(phi_y).squeeze(-1)
        y1 = self.head1(phi_y).squeeze(-1)
        y_pred = t * y1 + (1 - t) * y0

        # Propensity
        phi_t = torch.cat([phi_conf, phi_inst], dim=-1)
        prop = torch.sigmoid(self.prop_net(phi_t).squeeze(-1))

        return y_pred, y0, y1, prop, phi_conf, phi_out, phi_inst

    def predict_ite(self, x):
        phi_conf = self.conf_net(x)
        phi_out = self.out_net(x)
        phi_y = torch.cat([phi_conf, phi_out], dim=-1)
        y0 = self.head0(phi_y).squeeze(-1)
        y1 = self.head1(phi_y).squeeze(-1)
        return y1 - y0


class DragonNet(nn.Module):
    """DragonNet: TARNet + propensity head (Shi et al., 2019)."""
    def __init__(self, input_dim, repr_dim=200, out_dim=100, n_repr_layers=3, n_out_layers=2):
        super().__init__()

        # Shared representation
        repr_layers = []
        prev = input_dim
        for _ in range(n_repr_layers):
            repr_layers.extend([nn.Linear(prev, repr_dim), nn.ELU()])
            prev = repr_dim
        self.repr_net = nn.Sequential(*repr_layers)

        # Outcome head 0
        h0 = []
        prev = repr_dim
        for _ in range(n_out_layers):
            h0.extend([nn.Linear(prev, out_dim), nn.ELU()])
            prev = out_dim
        h0.append(nn.Linear(prev, 1))
        self.head0 = nn.Sequential(*h0)

        # Outcome head 1
        h1 = []
        prev = repr_dim
        for _ in range(n_out_layers):
            h1.extend([nn.Linear(prev, out_dim), nn.ELU()])
            prev = out_dim
        h1.append(nn.Linear(prev, 1))
        self.head1 = nn.Sequential(*h1)

        # Propensity head
        self.prop_head = nn.Sequential(
            nn.Linear(repr_dim, out_dim),
            nn.ELU(),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x, t):
        phi = self.repr_net(x)
        y0 = self.head0(phi).squeeze(-1)
        y1 = self.head1(phi).squeeze(-1)
        prop = torch.sigmoid(self.prop_head(phi).squeeze(-1))
        y_pred = t * y1 + (1 - t) * y0
        return y_pred, y0, y1, prop, phi

    def predict_ite(self, x):
        phi = self.repr_net(x)
        y0 = self.head0(phi).squeeze(-1)
        y1 = self.head1(phi).squeeze(-1)
        return y1 - y0


class FlexTENet(nn.Module):
    """FlexTENet: Shared + Private subspaces at every layer."""
    def __init__(self, input_dim, shared_dim=100, private_dim=100, n_layers=3):
        super().__init__()
        self.n_layers = n_layers

        # Shared layers
        self.shared_layers = nn.ModuleList()
        # Private layers (one per treatment)
        self.private0_layers = nn.ModuleList()
        self.private1_layers = nn.ModuleList()

        prev_s = input_dim
        prev_p = input_dim
        for i in range(n_layers):
            self.shared_layers.append(nn.Sequential(
                nn.Linear(prev_s, shared_dim), nn.ELU()))
            self.private0_layers.append(nn.Sequential(
                nn.Linear(prev_p, private_dim), nn.ELU()))
            self.private1_layers.append(nn.Sequential(
                nn.Linear(prev_p, private_dim), nn.ELU()))
            prev_s = shared_dim
            prev_p = private_dim

        # Output heads
        self.out0 = nn.Linear(shared_dim + private_dim, 1)
        self.out1 = nn.Linear(shared_dim + private_dim, 1)

    def forward(self, x, t):
        hs = x
        hp0 = x
        hp1 = x
        for i in range(self.n_layers):
            hs = self.shared_layers[i](hs)
            hp0 = self.private0_layers[i](hp0)
            hp1 = self.private1_layers[i](hp1)

        y0 = self.out0(torch.cat([hs, hp0], dim=-1)).squeeze(-1)
        y1 = self.out1(torch.cat([hs, hp1], dim=-1)).squeeze(-1)
        y_pred = t * y1 + (1 - t) * y0
        return y_pred, y0, y1

    def predict_ite(self, x):
        hs = x
        hp0 = x
        hp1 = x
        for i in range(self.n_layers):
            hs = self.shared_layers[i](hs)
            hp0 = self.private0_layers[i](hp0)
            hp1 = self.private1_layers[i](hp1)
        y0 = self.out0(torch.cat([hs, hp0], dim=-1)).squeeze(-1)
        y1 = self.out1(torch.cat([hs, hp1], dim=-1)).squeeze(-1)
        return y1 - y0


def orthogonal_penalty(model):
    """Compute orthogonality penalty between shared and private representations."""
    penalty = 0.0
    if hasattr(model, 'shared_layers'):
        for s_layer, p0_layer, p1_layer in zip(
            model.shared_layers, model.private0_layers, model.private1_layers
        ):
            # Get weight matrices
            s_w = list(s_layer.parameters())[0]
            p0_w = list(p0_layer.parameters())[0]
            p1_w = list(p1_layer.parameters())[0]
            # Frobenius norm of cross-product
            min_d = min(s_w.shape[1], p0_w.shape[1])
            penalty += torch.norm(s_w[:, :min_d].T @ p0_w[:, :min_d]) ** 2
            penalty += torch.norm(s_w[:, :min_d].T @ p1_w[:, :min_d]) ** 2
    return penalty


def train_catenet(X_train, t_train, y_train, input_dim, model_type="tarnet", config=None,
                  X_val=None, t_val=None, y_val=None):
    """Train a CATENet model with external validation data."""
    if config is None:
        config = {}

    repr_dim = config.get("repr_dim", 200)
    out_dim = config.get("out_dim", 100)
    n_repr_layers = config.get("n_repr_layers", 3)
    n_out_layers = config.get("n_out_layers", 2)
    lr = config.get("lr", 1e-4)
    batch_size = config.get("batch_size", 100)
    n_epochs = config.get("n_epochs", 400)
    weight_decay = config.get("weight_decay", 1e-4)
    alpha_prop = config.get("alpha_prop", 1.0)
    alpha_ortho = config.get("alpha_ortho", 0.01)
    patience = config.get("patience", 30)

    # Create model
    if model_type == "tarnet":
        model = TARNet(input_dim, repr_dim, out_dim, n_repr_layers, n_out_layers)
    elif model_type == "snet":
        model = SNet(input_dim, repr_dim_big=100, repr_dim_small=50, out_dim=out_dim,
                     n_repr_layers=n_repr_layers, n_out_layers=n_out_layers)
    elif model_type == "dragonnet":
        model = DragonNet(input_dim, repr_dim, out_dim, n_repr_layers, n_out_layers)
    elif model_type == "flextenet":
        model = FlexTENet(input_dim, shared_dim=100, private_dim=100, n_layers=n_repr_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    X_t = torch.FloatTensor(X_train)
    t_t = torch.FloatTensor(t_train)
    y_t = torch.FloatTensor(y_train)

    # Use external val data if provided, else internal split
    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val)
        t_val_t = torch.FloatTensor(t_val)
        y_val_t = torch.FloatTensor(y_val)
    else:
        n = len(X_t)
        n_val = int(0.2 * n)
        perm = torch.randperm(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        X_val_t, t_val_t, y_val_t = X_t[val_idx], t_t[val_idx], y_t[val_idx]
        X_t, t_t, y_t = X_t[train_idx], t_t[train_idx], y_t[train_idx]

    dataset = TensorDataset(X_t, t_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    model.train()
    for epoch in range(n_epochs):
        for batch_x, batch_t, batch_y in loader:
            optimizer.zero_grad()

            if model_type in ["tarnet"]:
                y_pred, y0, y1, phi = model(batch_x, batch_t)
                loss = nn.MSELoss()(y_pred, batch_y)
            elif model_type == "dragonnet":
                y_pred, y0, y1, prop, phi = model(batch_x, batch_t)
                loss_y = nn.MSELoss()(y_pred, batch_y)
                loss_t = nn.BCELoss()(prop, batch_t)
                loss = loss_y + alpha_prop * loss_t
            elif model_type == "snet":
                y_pred, y0, y1, prop, phi_c, phi_o, phi_i = model(batch_x, batch_t)
                loss_y = nn.MSELoss()(y_pred, batch_y)
                loss_t = nn.BCELoss()(prop, batch_t)
                # Orthogonality between representations
                ortho = (torch.norm(phi_c.T @ phi_o) ** 2 +
                         torch.norm(phi_c.T @ phi_i) ** 2 +
                         torch.norm(phi_o.T @ phi_i) ** 2)
                loss = loss_y + alpha_prop * loss_t + alpha_ortho * ortho
            elif model_type == "flextenet":
                y_pred, y0, y1 = model(batch_x, batch_t)
                loss_y = nn.MSELoss()(y_pred, batch_y)
                ortho = orthogonal_penalty(model)
                loss = loss_y + alpha_ortho * ortho

            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            if model_type == "tarnet":
                y_pred_val, _, _, _ = model(X_val_t, t_val_t)
            elif model_type == "dragonnet":
                y_pred_val, _, _, _, _ = model(X_val_t, t_val_t)
            elif model_type == "snet":
                y_pred_val, _, _, _, _, _, _ = model(X_val_t, t_val_t)
            elif model_type == "flextenet":
                y_pred_val, _, _ = model(X_val_t, t_val_t)
            val_loss = nn.MSELoss()(y_pred_val, y_val_t).item()
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


def predict_catenet(model, X_test):
    """Predict ITE using trained CATENet."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test)
        ite = model.predict_ite(X_t).numpy()
    return ite
