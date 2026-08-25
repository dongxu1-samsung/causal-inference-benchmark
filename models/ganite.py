"""
GANITE - Generative Adversarial Nets for Inference of Individualized Treatment Effects
(Yoon et al., 2018)
Architecture: Generator + Discriminator (counterfactual block) + Inference network (ITE block)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    """Generates counterfactual outcomes."""
    def __init__(self, input_dim, h_dim=100):
        super().__init__()
        # Input: X + T + Y (dim + 2)
        self.shared = nn.Sequential(
            nn.Linear(input_dim + 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        # Head for Y(0)
        self.head0 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )
        # Head for Y(1)
        self.head1 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, x, t, y):
        inp = torch.cat([x, t.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        h = self.shared(inp)
        y0_hat = self.head0(h).squeeze(-1)
        y1_hat = self.head1(h).squeeze(-1)
        return y0_hat, y1_hat


class Discriminator(nn.Module):
    """Discriminates real vs generated counterfactuals."""
    def __init__(self, input_dim, h_dim=100):
        super().__init__()
        # Input: X + Y0 + Y1 (dim + 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, x, y0, y1):
        inp = torch.cat([x, y0.unsqueeze(-1), y1.unsqueeze(-1)], dim=-1)
        return self.net(inp).squeeze(-1)


class InferenceNet(nn.Module):
    """Predicts ITE from features only."""
    def __init__(self, input_dim, h_dim=100):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.head0 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )
        self.head1 = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, x):
        h = self.shared(x)
        y0 = self.head0(h).squeeze(-1)
        y1 = self.head1(h).squeeze(-1)
        return y0, y1


def train_ganite(X_train, t_train, y_train, input_dim, config=None):
    """Train GANITE model (2 phases) with early stopping on validation."""
    if config is None:
        config = {}

    h_dim = config.get("h_dim", 100)
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 256)
    n_iter_gan = config.get("n_iter_gan", 5000)
    n_iter_inf = config.get("n_iter_inf", 5000)
    alpha = config.get("alpha", 1.0)
    patience = config.get("patience", 20)

    # Validation data for early stopping
    X_val = config.get("_X_val")
    t_val = config.get("_t_val")
    y_val = config.get("_y_val")

    generator = Generator(input_dim, h_dim)
    discriminator = Discriminator(input_dim, h_dim)
    inference = InferenceNet(input_dim, h_dim)

    X_t = torch.FloatTensor(X_train)
    t_t = torch.FloatTensor(t_train)
    y_t = torch.FloatTensor(y_train)

    dataset = TensorDataset(X_t, t_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Phase 1: Train Generator + Discriminator
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    generator.train()
    discriminator.train()

    for iteration in range(n_iter_gan):
        for batch_x, batch_t, batch_y in loader:
            # Generate counterfactual outcomes
            y0_gen, y1_gen = generator(batch_x, batch_t, batch_y)

            # Combine with factual
            y0_full = batch_t * y0_gen + (1 - batch_t) * batch_y
            y1_full = (1 - batch_t) * y1_gen + batch_t * batch_y

            # Train Discriminator
            opt_d.zero_grad()
            d_out = discriminator(batch_x, y0_full.detach(), y1_full.detach())
            d_loss = nn.BCEWithLogitsLoss()(d_out, batch_t)
            d_loss.backward()
            opt_d.step()

            # Train Generator
            opt_g.zero_grad()
            y0_gen, y1_gen = generator(batch_x, batch_t, batch_y)
            y0_full = batch_t * y0_gen + (1 - batch_t) * batch_y
            y1_full = (1 - batch_t) * y1_gen + batch_t * batch_y

            d_out = discriminator(batch_x, y0_full, y1_full)
            # Generator wants to fool discriminator
            g_loss_adv = -nn.BCEWithLogitsLoss()(d_out, batch_t)
            # Supervised loss on factual
            y_pred = batch_t * y1_gen + (1 - batch_t) * y0_gen
            g_loss_sup = nn.MSELoss()(y_pred, batch_y) * alpha
            g_loss = g_loss_adv + g_loss_sup
            g_loss.backward()
            opt_g.step()

    # Phase 2: Train Inference Network with early stopping
    opt_i = torch.optim.Adam(inference.parameters(), lr=lr)
    inference.train()
    generator.eval()

    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val)
        t_val_t = torch.FloatTensor(t_val)
        y_val_t = torch.FloatTensor(y_val)

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for iteration in range(n_iter_inf):
        for batch_x, batch_t, batch_y in loader:
            with torch.no_grad():
                y0_gen, y1_gen = generator(batch_x, batch_t, batch_y)
                # Target: factual for observed, generated for counterfactual
                y0_target = batch_t * y0_gen + (1 - batch_t) * batch_y
                y1_target = (1 - batch_t) * y1_gen + batch_t * batch_y

            opt_i.zero_grad()
            y0_pred, y1_pred = inference(batch_x)
            loss = nn.MSELoss()(y0_pred, y0_target) + nn.MSELoss()(y1_pred, y1_target)
            loss.backward()
            opt_i.step()

        # Early stopping on validation (check every 5 iterations)
        if X_val is not None and (iteration + 1) % 5 == 0:
            inference.eval()
            with torch.no_grad():
                y0_v, y1_v = inference(X_val_t)
                y_pred_v = t_val_t * y1_v + (1 - t_val_t) * y0_v
                val_loss = nn.MSELoss()(y_pred_v, y_val_t).item()
            inference.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in inference.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if best_state is not None:
        inference.load_state_dict(best_state)
    inference.eval()
    return inference


def predict_ganite(model, X_test):
    """Predict ITE using trained GANITE inference network."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test)
        y0, y1 = model(X_t)
        ite = (y1 - y0).numpy()
    return ite
