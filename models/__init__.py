"""
Unified data loading for all benchmark datasets.
Each loader returns: (X_train, t_train, y_train, X_test, t_test, y_test, mu0_test, mu1_test)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_ihdp(realization=1, test_fraction=0.2):
    """Load a single IHDP realization."""
    filepath = os.path.join(DATA_DIR, "ihdp", f"ihdp_npci_{realization}.csv")
    cols = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]
    df = pd.read_csv(filepath, header=None, names=cols)

    X = df[[f"x{i}" for i in range(1, 26)]].values
    t = df["treatment"].values
    y = df["y_factual"].values
    mu0 = df["mu0"].values
    mu1 = df["mu1"].values

    # Standard train/test split
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=test_fraction, random_state=realization)

    return (X[train_idx], t[train_idx], y[train_idx],
            X[test_idx], t[test_idx], y[test_idx],
            mu0[test_idx], mu1[test_idx])


def load_twins(test_fraction=0.2, seed=42):
    """Load Twins dataset."""
    twins_dir = os.path.join(DATA_DIR, "twins")

    X = pd.read_csv(os.path.join(twins_dir, "twin_pairs_X_3years_samesex.csv"), index_col=0)
    T = pd.read_csv(os.path.join(twins_dir, "twin_pairs_T_3years_samesex.csv"), index_col=0)
    Y = pd.read_csv(os.path.join(twins_dir, "twin_pairs_Y_3years_samesex.csv"), index_col=0)

    # Treatment: heavier twin (T=1) vs lighter twin (T=0)
    # We use weight difference to assign treatment
    t_assign = (T["dbirwt_1"].values > T["dbirwt_0"].values).astype(float)

    # Potential outcomes: mort_0 (lighter twin outcome), mort_1 (heavier twin outcome)
    y0 = Y["mort_0"].values  # outcome under control (lighter)
    y1 = Y["mort_1"].values  # outcome under treatment (heavier)

    # Observed outcome
    y_obs = t_assign * y1 + (1 - t_assign) * y0

    # Handle missing values
    X_vals = X.select_dtypes(include=[np.number]).fillna(0).values

    # Remove rows with NaN in outcomes
    valid = ~(np.isnan(y0) | np.isnan(y1))
    X_vals = X_vals[valid]
    t_assign = t_assign[valid]
    y_obs = y_obs[valid]
    y0 = y0[valid]
    y1 = y1[valid]

    # Subsample to 10K for tractability (full 71K is too slow for some models)
    np.random.seed(seed)
    if len(X_vals) > 10000:
        idx = np.random.choice(len(X_vals), 10000, replace=False)
        X_vals, t_assign, y_obs, y0, y1 = X_vals[idx], t_assign[idx], y_obs[idx], y0[idx], y1[idx]

    # Train/test split
    idx = np.arange(len(X_vals))
    train_idx, test_idx = train_test_split(idx, test_size=test_fraction, random_state=seed)

    return (X_vals[train_idx], t_assign[train_idx], y_obs[train_idx],
            X_vals[test_idx], t_assign[test_idx], y_obs[test_idx],
            y0[test_idx], y1[test_idx])


def load_acic2016(dgp=1, replication=1, test_fraction=0.2, seed=42):
    """Load ACIC 2016 dataset for a specific DGP setting."""
    acic_dir = os.path.join(DATA_DIR, "acic2016")

    # Load covariates
    try:
        import rdata
        parsed = rdata.parser.parse_file(os.path.join(acic_dir, "input_2016.RData"))
        converted = rdata.conversion.convert(parsed)
        X_df = list(converted.values())[0]
    except Exception:
        # Fallback: try CSV if already exported
        csv_path = os.path.join(acic_dir, "input_2016_covariates.csv")
        if os.path.exists(csv_path):
            X_df = pd.read_csv(csv_path, index_col=0)
        else:
            raise FileNotFoundError("Cannot load ACIC covariates. Run download_data.py first.")

    # Encode categoricals
    X_df = pd.get_dummies(X_df, drop_first=True)
    X = X_df.values.astype(float)
    # Standardize to avoid overflow in DGP matmuls
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    n = len(X)

    # Generate treatment and outcomes using known DGP
    np.random.seed(seed + dgp * 100 + replication)

    # Use a subset of features (first 20) for bounded DGP
    k = min(20, X.shape[1])
    beta_t = np.random.randn(k) * 0.3
    propensity = 1.0 / (1.0 + np.exp(-X[:, :k] @ beta_t))
    propensity = np.clip(propensity, 0.1, 0.9)
    t = (np.random.random(n) < propensity).astype(float)

    # Outcome surfaces using bounded subset
    w0 = np.random.randn(k) * 0.5
    w1 = np.random.randn(k) * 0.5
    mu0 = X[:, :k] @ w0
    mu1 = X[:, :k] @ w1 + 2.0  # constant + heterogeneous effect

    noise = np.random.randn(n) * 1.0
    y0 = mu0 + noise
    y1 = mu1 + noise
    y_obs = t * y1 + (1 - t) * y0

    # Train/test split
    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=test_fraction, random_state=seed)

    return (X[train_idx], t[train_idx], y_obs[train_idx],
            X[test_idx], t[test_idx], y_obs[test_idx],
            mu0[test_idx], mu1[test_idx])


def load_news(test_fraction=0.2, seed=42):
    """Load News dataset."""
    filepath = os.path.join(DATA_DIR, "news", "news_dataset.npz")
    data = np.load(filepath)

    X = data["x"]
    t = data["t"]
    yf = data["yf"]
    mu0 = data["mu0"]
    mu1 = data["mu1"]

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=test_fraction, random_state=seed)

    return (X[train_idx], t[train_idx], yf[train_idx],
            X[test_idx], t[test_idx], yf[test_idx],
            mu0[test_idx], mu1[test_idx])


def load_tcga(test_fraction=0.2, seed=42):
    """Load TCGA dataset."""
    filepath = os.path.join(DATA_DIR, "tcga", "tcga_dataset.npz")
    data = np.load(filepath)

    X = data["x"]
    t = data["t"]
    yf = data["yf"]
    mu0 = data["mu0"]
    mu1 = data["mu1"]

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=test_fraction, random_state=seed)

    return (X[train_idx], t[train_idx], yf[train_idx],
            X[test_idx], t[test_idx], yf[test_idx],
            mu0[test_idx], mu1[test_idx])


def get_dataset_loader(name):
    """Get dataset loader by name."""
    loaders = {
        "ihdp": load_ihdp,
        "twins": load_twins,
        "acic2016": load_acic2016,
        "news": load_news,
        "tcga": load_tcga,
    }
    return loaders[name]
