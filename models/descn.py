"""
DESCN - Deep Entire Space Cross Network (Alibaba, Zhong et al. 2022)
Architecture: 5-head network — propensity (e), μ₀, μ₁, pseudo-treatment (τ₀, τ₁)
  with cross-network constraints for ITE estimation.

Key idea: Uses the "entire space" approach where propensity, base outcomes, and
treatment effects are all jointly estimated with shared representations.
The cross constraints enforce consistency: τ = μ₁ - μ₀.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DESCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=200, n_layers=3, activation="elu"):
        super().__init__()
        act = nn.ELU if activation == "elu" else nn.ReLU

        # Shared representation
        shared = []
        prev = input_dim
        for _ in range(n_layers):
            shared.append(nn.Linear(prev, hidden_dim))
            shared.append(act())
            prev = hidden_dim
        self.shared = nn.Sequential(*shared)

        # Head builders
        def make_head(out_dim=1, n_head_layers=2):
            layers = []
            p = hidden_dim
            for _ in range(n_head_layers):
                layers.append(nn.Linear(p, hidden_dim // 2))
                layers.append(act())
                p = hidden_dim // 2
            layers.append(nn.Linear(p, out_dim))
            return nn.Sequential(*layers)

        # 5 heads
        self.head_e = nn.Sequential(*[*make_head(1).children()], nn.Sigmoid())  # propensity
        self.head_mu0 = make_head(1)   # E[Y|T=0,X]
        self.head_mu1 = make_head(1)   # E[Y|T=1,X]
        self.head_tau0 = make_head(1)  # pseudo treatment effect (control perspective)
        self.head_tau1 = make_head(1)  # pseudo treatment effect (treated perspective)

    def forward(self, x):
        phi = self.shared(x)
        e = self.head_e(phi).squeeze(-1)        # propensity
        mu0 = self.head_mu0(phi).squeeze(-1)     # base outcome control
        mu1 = self.head_mu1(phi).squeeze(-1)     # base outcome treated
        tau0 = self.head_tau0(phi).squeeze(-1)   # CATE from control
        tau1 = self.head_tau1(phi).squeeze(-1)   # CATE from treated
        return e, mu0, mu1, tau0, tau1

    def predict_ite(self, x):
        phi = self.shared(x)
        e = self.head_e(phi).squeeze(-1)
        mu0 = self.head_mu0(phi).squeeze(-1)
        mu1 = self.head_mu1(phi).squeeze(-1)
        tau0 = self.head_tau0(phi).squeeze(-1)
        tau1 = self.head_tau1(phi).squeeze(-1)
        # Weighted combination like X-learner
        tau = e * tau0 + (1 - e) * tau1
        # Cross constraint: also consider mu1 - mu0
        tau_mu = mu1 - mu0
        # Final: average of direct and cross estimates
        return 0.5 * (tau + tau_mu)


def train_descn(X_train, t_train, y_train, input_dim, config=None):
    if config is None:
        config = {}

    hidden_dim = config.get("hidden_dim", 200)
    n_layers = config.get("n_layers", 3)
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 256)
    n_epochs = config.get("n_epochs", 300)
    weight_decay = config.get("weight_decay", 1e-4)
    alpha_prop = config.get("alpha_prop", 0.5)     # propensity loss weight
    alpha_cross = config.get("alpha_cross", 0.5)   # cross constraint weight
    patience = config.get("patience", 20)

    X_val = config.get("_X_val")
    t_val = config.get("_t_val")
    y_val = config.get("_y_val")

    model = DESCN(input_dim, hidden_dim, n_layers)

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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    model.train()
    for epoch in range(n_epochs):
        for bx, bt, by in loader:
            optimizer.zero_grad()

            e, mu0, mu1, tau0, tau1 = model(bx)

            # 1. Factual outcome loss
            y_pred = bt * mu1 + (1 - bt) * mu0
            loss_outcome = nn.MSELoss()(y_pred, by)

            # 2. Propensity loss (BCE)
            loss_prop = nn.BCELoss()(e.clamp(1e-6, 1 - 1e-6), bt)

            # 3. Cross constraints: τ₁ ≈ y - μ₀ for treated, τ₀ ≈ μ₁ - y for control
            idx_t = bt == 1
            idx_c = bt == 0
            loss_cross = torch.tensor(0.0)
            if idx_t.sum() > 0:
                loss_cross = loss_cross + nn.MSELoss()(tau1[idx_t], (by[idx_t] - mu0[idx_t]).detach())
            if idx_c.sum() > 0:
                loss_cross = loss_cross + nn.MSELoss()(tau0[idx_c], (mu1[idx_c] - by[idx_c]).detach())

            # 4. Consistency: tau heads should agree with mu difference
            loss_consist = nn.MSELoss()(tau0, (mu1 - mu0).detach()) + \
                           nn.MSELoss()(tau1, (mu1 - mu0).detach())

            loss = loss_outcome + alpha_prop * loss_prop + \
                   alpha_cross * (loss_cross + 0.1 * loss_consist)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if (epoch + 1) % 50 == 0:
            scheduler.step()

        # Early stopping
        if X_val is not None and (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                e_v, mu0_v, mu1_v, _, _ = model(X_val_t)
                y_pred_v = t_val_t * mu1_v + (1 - t_val_t) * mu0_v
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


def predict_descn(model, X_test):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test)
        ite = model.predict_ite(X_t).numpy()
    return ite
