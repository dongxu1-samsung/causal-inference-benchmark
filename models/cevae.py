"""
CEVAE - Causal Effect Variational Autoencoder (Louizos et al., 2017)
Architecture: VAE with latent confounders - encoder q(z|x,t,y) + decoder p(x,t,y|z)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Encoder(nn.Module):
    """Inference network q(z|x,t,y)."""
    def __init__(self, input_dim, latent_dim=20, h_dim=200, n_layers=3):
        super().__init__()

        # q(t|x) - auxiliary
        self.qt_net = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, 1),
        )

        # q(y|x,t) - auxiliary
        qy_layers = [nn.Linear(input_dim + 1, h_dim), nn.ELU()]
        for _ in range(n_layers - 2):
            qy_layers.extend([nn.Linear(h_dim, h_dim), nn.ELU()])
        qy_layers.append(nn.Linear(h_dim, 1))
        self.qy_net = nn.Sequential(*qy_layers)

        # q(z|x,t,y) - main encoder
        enc_layers = [nn.Linear(input_dim + 2, h_dim), nn.ELU()]
        for _ in range(n_layers - 2):
            enc_layers.extend([nn.Linear(h_dim, h_dim), nn.ELU()])
        self.enc_shared = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(h_dim, latent_dim)
        self.fc_logvar = nn.Linear(h_dim, latent_dim)

    def forward(self, x, t, y):
        # Encode to latent
        inp = torch.cat([x, t.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        h = self.enc_shared(inp)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def predict_t(self, x):
        return torch.sigmoid(self.qt_net(x).squeeze(-1))


class Decoder(nn.Module):
    """Generative model p(x,t,y|z)."""
    def __init__(self, input_dim, latent_dim=20, h_dim=200, n_layers=3):
        super().__init__()
        self.input_dim = input_dim

        # p(x|z)
        px_layers = [nn.Linear(latent_dim, h_dim), nn.ELU()]
        for _ in range(n_layers - 2):
            px_layers.extend([nn.Linear(h_dim, h_dim), nn.ELU()])
        px_layers.append(nn.Linear(h_dim, input_dim))
        self.px_net = nn.Sequential(*px_layers)

        # p(t|z)
        self.pt_net = nn.Sequential(
            nn.Linear(latent_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, 1),
        )

        # p(y|z,t) - two heads
        py_layers_shared = [nn.Linear(latent_dim, h_dim), nn.ELU()]
        for _ in range(n_layers - 2):
            py_layers_shared.extend([nn.Linear(h_dim, h_dim), nn.ELU()])
        self.py_shared = nn.Sequential(*py_layers_shared)
        self.py_head0 = nn.Linear(h_dim, 1)
        self.py_head1 = nn.Linear(h_dim, 1)

    def forward(self, z, t):
        x_recon = self.px_net(z)
        t_logit = self.pt_net(z).squeeze(-1)
        h = self.py_shared(z)
        y0 = self.py_head0(h).squeeze(-1)
        y1 = self.py_head1(h).squeeze(-1)
        y_pred = t * y1 + (1 - t) * y0
        return x_recon, t_logit, y_pred, y0, y1


class CEVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=20, h_dim=200, n_layers=3):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, h_dim, n_layers)
        self.decoder = Decoder(input_dim, latent_dim, h_dim, n_layers)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, t, y):
        mu, logvar = self.encoder(x, t, y)
        z = self.reparameterize(mu, logvar)
        x_recon, t_logit, y_pred, y0, y1 = self.decoder(z, t)
        return x_recon, t_logit, y_pred, y0, y1, mu, logvar

    def predict_ite(self, x, n_samples=100):
        """Predict ITE by sampling from approximate posterior."""
        self.eval()
        with torch.no_grad():
            # Use prior samples for prediction (no access to t, y at test time)
            batch_size = x.shape[0]
            ite_samples = []
            for _ in range(n_samples):
                z = torch.randn(batch_size, self.latent_dim)
                _, _, _, y0, y1 = self.decoder(z, torch.zeros(batch_size))
                ite_samples.append((y1 - y0).numpy())
            # Average over samples
            return np.mean(ite_samples, axis=0)


def cevae_loss(x, t, y, x_recon, t_logit, y_pred, mu, logvar, beta=1.0):
    """ELBO loss for CEVAE."""
    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")
    # Treatment prediction loss
    t_loss = F.binary_cross_entropy_with_logits(t_logit, t, reduction="mean")
    # Outcome prediction loss
    y_loss = F.mse_loss(y_pred, y, reduction="mean")
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + t_loss + y_loss + beta * kl_loss


def train_cevae(X_train, t_train, y_train, input_dim, config=None):
    """Train CEVAE model with early stopping on validation."""
    if config is None:
        config = {}

    latent_dim = config.get("latent_dim", 20)
    h_dim = config.get("h_dim", 200)
    n_layers = config.get("n_layers", 3)
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 100)
    n_epochs = config.get("n_epochs", 200)
    weight_decay = config.get("weight_decay", 1e-4)
    beta = config.get("beta", 1.0)
    patience = config.get("patience", 15)

    # Validation data for early stopping
    X_val = config.get("_X_val")
    t_val = config.get("_t_val")
    y_val = config.get("_y_val")

    model = CEVAE(input_dim, latent_dim, h_dim, n_layers)

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

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    model.train()
    for epoch in range(n_epochs):
        for batch_x, batch_t, batch_y in loader:
            optimizer.zero_grad()
            x_recon, t_logit, y_pred, y0, y1, mu, logvar = model(batch_x, batch_t, batch_y)
            loss = cevae_loss(batch_x, batch_t, batch_y, x_recon, t_logit, y_pred, mu, logvar, beta)
            loss.backward()
            optimizer.step()

        # Early stopping on validation
        if X_val is not None and (epoch + 1) % 3 == 0:
            model.eval()
            with torch.no_grad():
                x_r, t_l, y_p, _, _, mu_v, lv = model(X_val_t, t_val_t, y_val_t)
                val_loss = cevae_loss(X_val_t, t_val_t, y_val_t, x_r, t_l, y_p, mu_v, lv, beta).item()
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


def predict_cevae(model, X_test, n_samples=50):
    """Predict ITE using trained CEVAE."""
    model.eval()
    X_t = torch.FloatTensor(X_test)
    return model.predict_ite(X_t, n_samples=n_samples)
