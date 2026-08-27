"""
TransDCA: Transformer Disentangled Causal Attention

Architecture:
- Input features split into token groups → Transformer Encoder
- Disentangle into 3 subspaces: Z_I (instrumental), Z_C (confounding), Z_A (adjustment)
- Propensity head on [Z_I, Z_C]
- Outcome heads on [Z_C, Z_A]
- Loss: factual + alpha*propensity + beta*orthogonality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransDCAModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, n_layers=2, dim_ff=128,
                 repr_dim=32, head_dim=32, n_head_layers=2, group_size=4, max_tokens=64):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        # Adaptively increase group_size to cap tokens at max_tokens for high-dim inputs
        effective_group_size = max(group_size, (input_dim + max_tokens - 1) // max_tokens)
        self.group_size = effective_group_size
        self.n_tokens = max(1, (input_dim + effective_group_size - 1) // effective_group_size)

        # Input: project each group of features into d_model
        self.input_proj = nn.Linear(effective_group_size, d_model)
        # Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Disentanglement: project pooled representation into 3 subspaces
        self.proj_I = nn.Linear(d_model, repr_dim)  # Instrumental
        self.proj_C = nn.Linear(d_model, repr_dim)  # Confounding
        self.proj_A = nn.Linear(d_model, repr_dim)  # Adjustment

        # Propensity head: [Z_I, Z_C] → P(T=1)
        prop_layers = [nn.Linear(repr_dim * 2, head_dim), nn.ELU()]
        for _ in range(n_head_layers - 1):
            prop_layers += [nn.Linear(head_dim, head_dim), nn.ELU()]
        prop_layers.append(nn.Linear(head_dim, 1))
        self.propensity_head = nn.Sequential(*prop_layers)

        # Outcome head 0: [Z_C, Z_A] → Y(0)
        out_layers_0 = [nn.Linear(repr_dim * 2, head_dim), nn.ELU()]
        for _ in range(n_head_layers - 1):
            out_layers_0 += [nn.Linear(head_dim, head_dim), nn.ELU()]
        out_layers_0.append(nn.Linear(head_dim, 1))
        self.outcome_head_0 = nn.Sequential(*out_layers_0)

        # Outcome head 1: [Z_C, Z_A] → Y(1)
        out_layers_1 = [nn.Linear(repr_dim * 2, head_dim), nn.ELU()]
        for _ in range(n_head_layers - 1):
            out_layers_1 += [nn.Linear(head_dim, head_dim), nn.ELU()]
        out_layers_1.append(nn.Linear(head_dim, 1))
        self.outcome_head_1 = nn.Sequential(*out_layers_1)

    def _tokenize(self, X):
        """Split input features into token groups, pad if needed."""
        B = X.shape[0]
        # Pad input to multiple of group_size
        pad_size = self.n_tokens * self.group_size - self.input_dim
        if pad_size > 0:
            X = F.pad(X, (0, pad_size))
        # Reshape: (B, n_tokens, group_size)
        return X.view(B, self.n_tokens, self.group_size)

    def encode(self, X):
        """Encode input through transformer and return 3 subspaces."""
        tokens = self._tokenize(X)  # (B, n_tokens, group_size)
        tokens = self.input_proj(tokens)  # (B, n_tokens, d_model)
        tokens = tokens + self.pos_enc[:, :self.n_tokens, :]

        # Transformer
        encoded = self.transformer(tokens)  # (B, n_tokens, d_model)

        # Mean pooling over tokens
        pooled = encoded.mean(dim=1)  # (B, d_model)

        # Project into 3 disentangled subspaces
        Z_I = self.proj_I(pooled)
        Z_C = self.proj_C(pooled)
        Z_A = self.proj_A(pooled)

        return Z_I, Z_C, Z_A

    def forward(self, X, t):
        Z_I, Z_C, Z_A = self.encode(X)

        # Propensity prediction
        prop_input = torch.cat([Z_I, Z_C], dim=-1)
        logit_prop = self.propensity_head(prop_input).squeeze(-1)

        # Outcome predictions
        out_input = torch.cat([Z_C, Z_A], dim=-1)
        y0 = self.outcome_head_0(out_input).squeeze(-1)
        y1 = self.outcome_head_1(out_input).squeeze(-1)

        # Factual outcome
        y_pred = t * y1 + (1 - t) * y0

        return y_pred, y0, y1, logit_prop, Z_I, Z_C, Z_A

    def predict_ite(self, X):
        Z_I, Z_C, Z_A = self.encode(X)
        out_input = torch.cat([Z_C, Z_A], dim=-1)
        y0 = self.outcome_head_0(out_input).squeeze(-1)
        y1 = self.outcome_head_1(out_input).squeeze(-1)
        return y1 - y0


def orthogonality_loss(Z_I, Z_C, Z_A):
    """Penalize correlation between subspaces using cosine similarity."""
    def cos_penalty(A, B):
        # Normalize columns, compute cross-correlation
        A_n = F.normalize(A, dim=0)
        B_n = F.normalize(B, dim=0)
        return (A_n.T @ B_n).pow(2).mean()

    loss = cos_penalty(Z_I, Z_C) + cos_penalty(Z_I, Z_A) + cos_penalty(Z_C, Z_A)
    return loss / 3.0


def train_transdca(X_train, t_train, y_train, input_dim, config=None):
    """Train TransDCA model with early stopping on validation set."""
    if config is None:
        config = {}

    d_model = config.get("d_model", 64)
    nhead = config.get("nhead", 4)
    n_layers = config.get("n_layers", 2)
    dim_ff = config.get("dim_ff", 128)
    repr_dim = config.get("repr_dim", 32)
    head_dim = config.get("head_dim", 32)
    n_head_layers = config.get("n_head_layers", 2)
    n_epochs = config.get("n_epochs", 100)
    lr = config.get("lr", 1e-3)
    batch_size = config.get("batch_size", 256)
    patience = config.get("patience", 10)
    alpha = config.get("alpha", 0.5)  # propensity weight
    beta = config.get("beta", 0.1)   # orthogonality weight
    group_size = config.get("group_size", 4)

    # Validation data
    X_val = config.get("_X_val")
    t_val = config.get("_t_val")
    y_val = config.get("_y_val")

    # Build model
    model = TransDCAModel(
        input_dim=input_dim, d_model=d_model, nhead=nhead, n_layers=n_layers,
        dim_ff=dim_ff, repr_dim=repr_dim, head_dim=head_dim,
        n_head_layers=n_head_layers, group_size=group_size
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_t = torch.FloatTensor(X_train)
    t_t = torch.FloatTensor(t_train)
    y_t = torch.FloatTensor(y_train)

    if X_val is not None:
        X_val_t = torch.FloatTensor(X_val)
        t_val_t = torch.FloatTensor(t_val)
        y_val_t = torch.FloatTensor(y_val)

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    n = len(X_t)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, tb, yb = X_t[idx], t_t[idx], y_t[idx]

            y_pred, y0, y1, logit_prop, Z_I, Z_C, Z_A = model(xb, tb)

            # Factual loss
            fact_loss = F.mse_loss(y_pred, yb)

            # Propensity loss
            prop_loss = F.binary_cross_entropy_with_logits(logit_prop, tb)

            # Orthogonality loss
            orth_loss = orthogonality_loss(Z_I, Z_C, Z_A)

            loss = fact_loss + alpha * prop_loss + beta * orth_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += fact_loss.item()
            n_batches += 1

        # Validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                y_pred_v, _, _, _, _, _, _ = model(X_val_t, t_val_t)
                val_loss = F.mse_loss(y_pred_v, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        else:
            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {'model': model, 'input_dim': input_dim}


def predict_transdca(model_dict, X_test):
    """Predict ITE using trained TransDCA model."""
    model = model_dict['model']
    model.eval()
    X_t = torch.FloatTensor(X_test)
    with torch.no_grad():
        ite = model.predict_ite(X_t).numpy()
    return ite
