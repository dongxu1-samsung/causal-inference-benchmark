"""
MOCA - Modular One-way Cross-Attention (2026)
Architecture: Modular encoder with one-way cross-attention between
  treatment-specific representations and shared confounders.

Key idea: Instead of full self-attention, uses asymmetric cross-attention
where treatment queries attend to confounder keys/values (one-way), preventing
information leakage from treatment to confounders. Modular heads allow
independent capacity per treatment arm.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class CrossAttentionBlock(nn.Module):
    """One-way cross-attention: queries from treatment, keys/values from confounders."""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query, context):
        """query: treatment repr [B, D], context: confounder repr [B, D]"""
        B = query.size(0)

        # Reshape for multi-head: [B, 1, D] -> [B, H, 1, d_head]
        q = self.W_q(query).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(context).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(context).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)

        # Attention
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, self.d_model)
        out = self.W_o(out)

        # Residual + norm + feedforward
        x = self.norm1(query + out)
        x = self.norm2(x + self.ff(x))
        return x


class MOCA(nn.Module):
    """Modular One-way Cross-Attention network."""
    def __init__(self, input_dim, hidden_dim=128, n_shared_layers=3,
                 n_cross_blocks=2, n_heads=4, n_head_layers=2, dropout=0.1):
        super().__init__()
        act = nn.ELU()

        # Shared confounder encoder
        shared = []
        prev = input_dim
        for _ in range(n_shared_layers):
            shared.append(nn.Linear(prev, hidden_dim))
            shared.append(nn.ELU())
            shared.append(nn.Dropout(dropout))
            prev = hidden_dim
        self.shared_enc = nn.Sequential(*shared)

        # Treatment-specific encoders (modular)
        def make_branch_enc():
            layers = []
            p = input_dim
            for _ in range(2):
                layers.append(nn.Linear(p, hidden_dim))
                layers.append(nn.ELU())
                layers.append(nn.Dropout(dropout))
                p = hidden_dim
            return nn.Sequential(*layers)

        self.branch0_enc = make_branch_enc()  # control branch
        self.branch1_enc = make_branch_enc()  # treated branch

        # Cross-attention blocks (one-way: treatment queries → confounder context)
        self.cross_attn0 = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_cross_blocks)
        ])
        self.cross_attn1 = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_cross_blocks)
        ])

        # Outcome heads
        def make_head():
            layers = []
            p = hidden_dim
            for _ in range(n_head_layers):
                layers.append(nn.Linear(p, hidden_dim // 2))
                layers.append(nn.ELU())
                p = hidden_dim // 2
            layers.append(nn.Linear(p, 1))
            return nn.Sequential(*layers)

        self.head0 = make_head()
        self.head1 = make_head()

        # Propensity head (from shared)
        self.prop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, t):
        # Shared confounder representation
        phi_shared = self.shared_enc(x)

        # Branch-specific representations
        phi0 = self.branch0_enc(x)
        phi1 = self.branch1_enc(x)

        # One-way cross-attention: treatment queries attend to confounder keys
        for ca_block in self.cross_attn0:
            phi0 = ca_block(phi0, phi_shared)
        for ca_block in self.cross_attn1:
            phi1 = ca_block(phi1, phi_shared)

        # Outcome predictions
        y0 = self.head0(phi0).squeeze(-1)
        y1 = self.head1(phi1).squeeze(-1)
        y_pred = t * y1 + (1 - t) * y0

        # Propensity
        e = self.prop_head(phi_shared).squeeze(-1)

        return y_pred, y0, y1, e, phi_shared

    def predict_ite(self, x):
        phi_shared = self.shared_enc(x)
        phi0 = self.branch0_enc(x)
        phi1 = self.branch1_enc(x)
        for ca_block in self.cross_attn0:
            phi0 = ca_block(phi0, phi_shared)
        for ca_block in self.cross_attn1:
            phi1 = ca_block(phi1, phi_shared)
        y0 = self.head0(phi0).squeeze(-1)
        y1 = self.head1(phi1).squeeze(-1)
        return y1 - y0


def mmd_loss(phi_t, phi_c, sigma=1.0):
    if len(phi_t) == 0 or len(phi_c) == 0:
        return torch.tensor(0.0)
    def rbf(x, y, s):
        d = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.exp(-torch.sum(d ** 2, dim=-1) / (2 * s ** 2))
    return rbf(phi_t, phi_t, sigma).mean() + rbf(phi_c, phi_c, sigma).mean() \
           - 2 * rbf(phi_t, phi_c, sigma).mean()


def train_moca(X_train, t_train, y_train, input_dim, config=None):
    if config is None:
        config = {}

    hidden_dim = config.get("hidden_dim", 128)
    n_shared_layers = config.get("n_shared_layers", 3)
    n_cross_blocks = config.get("n_cross_blocks", 2)
    n_heads = config.get("n_heads", 4)
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 256)
    n_epochs = config.get("n_epochs", 300)
    weight_decay = config.get("weight_decay", 1e-4)
    alpha_prop = config.get("alpha_prop", 0.3)
    alpha_ipm = config.get("alpha_ipm", 0.5)
    patience = config.get("patience", 20)
    dropout = config.get("dropout", 0.1)

    X_val = config.get("_X_val")
    t_val = config.get("_t_val")
    y_val = config.get("_y_val")

    model = MOCA(input_dim, hidden_dim, n_shared_layers, n_cross_blocks,
                 n_heads, dropout=dropout)

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

            y_pred, y0, y1, e, phi = model(bx, bt)

            # Factual loss
            loss_fact = nn.MSELoss()(y_pred, by)

            # Propensity loss
            loss_prop = nn.BCELoss()(e.clamp(1e-6, 1 - 1e-6), bt)

            # IPM on shared representation
            idx_t = bt == 1
            idx_c = bt == 0
            loss_ipm = mmd_loss(phi[idx_t], phi[idx_c])

            loss = loss_fact + alpha_prop * loss_prop + alpha_ipm * loss_ipm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        scheduler.step()

        # Early stopping
        if X_val is not None and (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                y_pred_v, _, _, _, _ = model(X_val_t, t_val_t)
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


def predict_moca(model, X_test):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test)
        ite = model.predict_ite(X_t).numpy()
    return ite
