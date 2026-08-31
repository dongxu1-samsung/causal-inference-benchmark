"""
DDRNet - Disentangled Dose-Response Network (2025)
Architecture: Mixture-of-Experts with disentangled representations.
  Splits latent space into Instrumental (I), Confounding (C), and
  Adjustment (A) factors using MoE routing + orthogonality constraints.

Key idea: Uses Mixture-of-Experts for flexible representation learning,
then disentangles into three subspaces. Instrumental variables affect
treatment only, confounders affect both, adjustment variables affect
outcome only. Orthogonality + MI minimization separate subspaces.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class ExpertBlock(nn.Module):
    """Single expert MLP."""
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_layers=2):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ELU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """MoE layer with soft routing."""
    def __init__(self, input_dim, output_dim, n_experts=4, hidden_dim=128):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            ExpertBlock(input_dim, output_dim, hidden_dim)
            for _ in range(n_experts)
        ])
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, n_experts),
        )

    def forward(self, x):
        # Get gating weights
        gate_logits = self.gate(x)  # [B, n_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Run all experts
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, n_experts, D]

        # Weighted combination
        out = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # [B, D]
        return out, gate_weights


class DDRNet(nn.Module):
    """Disentangled Dose-Response Network with MoE."""
    def __init__(self, input_dim, repr_dim=64, n_experts=4, hidden_dim=128,
                 n_head_layers=2, dropout=0.1):
        super().__init__()
        self.repr_dim = repr_dim

        # Shared encoder
        self.shared_enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # MoE for each disentangled subspace
        self.moe_instr = MixtureOfExperts(hidden_dim, repr_dim, n_experts, hidden_dim)   # Instrumental
        self.moe_conf = MixtureOfExperts(hidden_dim, repr_dim, n_experts, hidden_dim)    # Confounding
        self.moe_adj = MixtureOfExperts(hidden_dim, repr_dim, n_experts, hidden_dim)     # Adjustment

        # Treatment predictor (from Instrumental + Confounding)
        self.treat_pred = nn.Sequential(
            nn.Linear(repr_dim * 2, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Outcome heads (from Confounding + Adjustment)
        def make_outcome_head():
            layers = []
            p = repr_dim * 2
            for _ in range(n_head_layers):
                layers.append(nn.Linear(p, hidden_dim))
                layers.append(nn.ELU())
                layers.append(nn.Dropout(dropout))
                p = hidden_dim
            layers.append(nn.Linear(p, 1))
            return nn.Sequential(*layers)

        self.head0 = make_outcome_head()
        self.head1 = make_outcome_head()

    def encode(self, x):
        h = self.shared_enc(x)
        z_i, gate_i = self.moe_instr(h)
        z_c, gate_c = self.moe_conf(h)
        z_a, gate_a = self.moe_adj(h)
        return z_i, z_c, z_a, (gate_i, gate_c, gate_a)

    def forward(self, x, t):
        z_i, z_c, z_a, gates = self.encode(x)

        # Treatment prediction (instrumental + confounding)
        e = self.treat_pred(torch.cat([z_i, z_c], dim=-1)).squeeze(-1)

        # Outcome prediction (confounding + adjustment)
        z_outcome = torch.cat([z_c, z_a], dim=-1)
        y0 = self.head0(z_outcome).squeeze(-1)
        y1 = self.head1(z_outcome).squeeze(-1)
        y_pred = t * y1 + (1 - t) * y0

        return y_pred, y0, y1, e, z_i, z_c, z_a, gates

    def predict_ite(self, x):
        z_i, z_c, z_a, _ = self.encode(x)
        z_outcome = torch.cat([z_c, z_a], dim=-1)
        y0 = self.head0(z_outcome).squeeze(-1)
        y1 = self.head1(z_outcome).squeeze(-1)
        return y1 - y0


def orthogonality_loss(z_i, z_c, z_a):
    """Encourage orthogonal subspaces via cosine similarity penalty."""
    def cos_penalty(a, b):
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        return (a_norm * b_norm).sum(dim=-1).pow(2).mean()

    return cos_penalty(z_i, z_c) + cos_penalty(z_i, z_a) + cos_penalty(z_c, z_a)


def mi_penalty(z1, z2):
    """Approximate MI minimization via HSIC (Hilbert-Schmidt Independence Criterion)."""
    B = z1.size(0)
    if B < 4:
        return torch.tensor(0.0)
    # Center
    z1c = z1 - z1.mean(0)
    z2c = z2 - z2.mean(0)
    # Gram matrices (linear kernel)
    K1 = z1c @ z1c.T
    K2 = z2c @ z2c.T
    # Centered HSIC
    H = torch.eye(B, device=z1.device) - 1.0 / B
    hsic = (K1 @ H @ K2 @ H).trace() / ((B - 1) ** 2)
    return hsic.clamp(min=0)


def mmd_balance(phi_t, phi_c, sigma=1.0):
    """MMD for balanced representations."""
    if len(phi_t) == 0 or len(phi_c) == 0:
        return torch.tensor(0.0)
    def rbf(x, y, s):
        d = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.exp(-torch.sum(d ** 2, dim=-1) / (2 * s ** 2))
    return rbf(phi_t, phi_t, sigma).mean() + rbf(phi_c, phi_c, sigma).mean() \
           - 2 * rbf(phi_t, phi_c, sigma).mean()


def train_ddrnet(X_train, t_train, y_train, input_dim, config=None):
    if config is None:
        config = {}

    repr_dim = config.get("repr_dim", 64)
    n_experts = config.get("n_experts", 4)
    hidden_dim = config.get("hidden_dim", 128)
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 256)
    n_epochs = config.get("n_epochs", 300)
    weight_decay = config.get("weight_decay", 1e-4)
    alpha_prop = config.get("alpha_prop", 0.3)
    alpha_orth = config.get("alpha_orth", 1.0)     # orthogonality weight
    alpha_mi = config.get("alpha_mi", 0.5)         # MI penalty weight
    alpha_ipm = config.get("alpha_ipm", 0.5)       # IPM balance on confounders
    patience = config.get("patience", 20)
    dropout = config.get("dropout", 0.1)

    X_val = config.get("_X_val")
    t_val = config.get("_t_val")
    y_val = config.get("_y_val")

    model = DDRNet(input_dim, repr_dim, n_experts, hidden_dim, dropout=dropout)

    X_t = torch.FloatTensor(X_train)
    t_t = torch.FloatTensor(t_train)
    y_t = torch.FloatTensor(y_train)

    dataset = TensorDataset(X_t, t_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val)
        t_val_t = torch.FloatTensor(t_val)
        y_val_t = torch.FloatTensor(y_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    model.train()
    for epoch in range(n_epochs):
        for bx, bt, by in loader:
            optimizer.zero_grad()

            y_pred, y0, y1, e, z_i, z_c, z_a, gates = model(bx, bt)

            # 1. Factual loss
            loss_fact = nn.MSELoss()(y_pred, by)

            # 2. Propensity loss
            loss_prop = nn.BCELoss()(e.clamp(1e-6, 1 - 1e-6), bt)

            # 3. Orthogonality between subspaces
            loss_orth = orthogonality_loss(z_i, z_c, z_a)

            # 4. MI penalty (pairwise HSIC)
            loss_mi = mi_penalty(z_i, z_c) + mi_penalty(z_i, z_a) + mi_penalty(z_c, z_a)

            # 5. Balance confounding representations across treatment groups
            idx_t = bt == 1
            idx_c = bt == 0
            loss_ipm = mmd_balance(z_c[idx_t], z_c[idx_c])

            # 6. Gate load balancing (prevent expert collapse)
            for _, _, gate_w in [gates]:
                pass  # skip for simplicity — soft gating rarely collapses
            avg_gates = sum(g.mean(0) for g in gates) / 3
            loss_balance = ((avg_gates - 1.0 / avg_gates.size(0)) ** 2).sum()

            loss = loss_fact + alpha_prop * loss_prop + alpha_orth * loss_orth \
                   + alpha_mi * loss_mi + alpha_ipm * loss_ipm + 0.01 * loss_balance
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scheduler.step()

        # Early stopping
        if X_val is not None and (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                y_pred_v, _, _, _, _, _, _, _ = model(X_val_t, t_val_t)
                val_loss = nn.MSELoss()(y_pred_v, y_val_t).item()
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


def predict_ddrnet(model, X_test):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test)
        ite = model.predict_ite(X_t).numpy()
    return ite
