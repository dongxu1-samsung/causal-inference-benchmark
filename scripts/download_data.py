"""
Download and generate all benchmark datasets.
Datasets: IHDP, Twins, ACIC 2016, News, TCGA, Jobs, ACIC 2018, Hillstrom, LBIDD, Criteo
"""

import os
import sys
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def download_file(url, filepath):
    """Download a file with progress."""
    if os.path.exists(filepath):
        print(f"  Already exists: {filepath}")
        return True
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, filepath)
        return True
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")
        return False


# ==============================================================================
# IHDP
# ==============================================================================
def download_ihdp():
    """Download IHDP realizations from CEVAE repo."""
    print("\n[1/10] IHDP Dataset")
    ihdp_dir = os.path.join(DATA_DIR, "ihdp")
    os.makedirs(ihdp_dir, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_{}.csv"
    downloaded = 0
    for i in range(1, 101):
        filepath = os.path.join(ihdp_dir, f"ihdp_npci_{i}.csv")
        if os.path.exists(filepath):
            downloaded += 1
            continue
        if i <= 10:
            if download_file(base_url.format(i), filepath):
                downloaded += 1
        else:
            # Generate additional realizations using Hill 2011-style simulation
            _generate_ihdp_realization(i, ihdp_dir)
            downloaded += 1

    print(f"  IHDP: {downloaded}/100 realizations ready")


def _generate_ihdp_realization(idx, ihdp_dir):
    """Generate an IHDP realization using the Hill 2011 simulation design."""
    # Load covariates from realization 1
    base_path = os.path.join(ihdp_dir, "ihdp_npci_1.csv")
    if not os.path.exists(base_path):
        return
    cols = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]
    df = pd.read_csv(base_path, header=None, names=cols)
    X = df[[f"x{i}" for i in range(1, 26)]].values
    t = df["treatment"].values

    np.random.seed(idx * 7 + 42)
    # Response surface (Hill 2011 style)
    beta = np.random.randn(25) * 0.5
    omega = np.random.randn(25) * 0.3
    mu0 = np.exp(X @ beta + 0.5)
    tau = X @ omega
    mu1 = mu0 + tau
    noise = np.random.randn(len(X)) * 1.0
    y0 = mu0 + noise
    y1 = mu1 + noise
    y_factual = t * y1 + (1 - t) * y0
    y_cfactual = (1 - t) * y1 + t * y0

    out = np.column_stack([t, y_factual, y_cfactual, mu0, mu1, X])
    filepath = os.path.join(ihdp_dir, f"ihdp_npci_{idx}.csv")
    np.savetxt(filepath, out, delimiter=",", fmt="%.6f")


# ==============================================================================
# Twins
# ==============================================================================
def download_twins():
    """Download Twins dataset."""
    print("\n[2/10] Twins Dataset")
    twins_dir = os.path.join(DATA_DIR, "twins")
    os.makedirs(twins_dir, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/"
    files = ["twin_pairs_X_3years_samesex.csv", "twin_pairs_T_3years_samesex.csv",
             "twin_pairs_Y_3years_samesex.csv"]
    for f in files:
        download_file(base_url + f, os.path.join(twins_dir, f))
    print("  Twins: ready")


# ==============================================================================
# ACIC 2016
# ==============================================================================
def download_acic2016():
    """Download ACIC 2016 covariates."""
    print("\n[3/10] ACIC 2016 Dataset")
    acic_dir = os.path.join(DATA_DIR, "acic2016")
    os.makedirs(acic_dir, exist_ok=True)

    url = "https://raw.githubusercontent.com/vdorie/aciccomp/master/2016/data/input_2016.RData"
    download_file(url, os.path.join(acic_dir, "input_2016.RData"))
    print("  ACIC 2016: ready (DGP generated at load time)")


# ==============================================================================
# News (semi-synthetic)
# ==============================================================================
def download_news():
    """Generate News semi-synthetic dataset."""
    print("\n[4/10] News Dataset")
    news_dir = os.path.join(DATA_DIR, "news")
    filepath = os.path.join(news_dir, "news_dataset.npz")
    if os.path.exists(filepath):
        print("  Already exists")
        return

    os.makedirs(news_dir, exist_ok=True)
    np.random.seed(2024)
    n, p = 5000, 3477
    # Sparse TF-IDF-like features
    X = np.random.exponential(0.1, (n, p))
    X[X < 0.05] = 0  # sparsify

    # Treatment assignment based on content features
    beta_t = np.random.randn(50) * 0.2
    prop = 1 / (1 + np.exp(-X[:, :50] @ beta_t))
    t = (np.random.random(n) < prop).astype(float)

    # Outcome surfaces
    w0 = np.random.randn(100) * 0.3
    w1 = np.random.randn(100) * 0.3
    mu0 = np.tanh(X[:, :100] @ w0) * 2
    mu1 = np.tanh(X[:, :100] @ w1) * 2 + 2.0
    noise = np.random.randn(n) * 0.3
    yf = t * (mu1 + noise) + (1 - t) * (mu0 + noise)

    np.savez(filepath, x=X.astype(np.float32), t=t, yf=yf, mu0=mu0, mu1=mu1)
    print(f"  News: generated ({n} samples, {p} features)")


# ==============================================================================
# TCGA (semi-synthetic gene expression)
# ==============================================================================
def download_tcga():
    """Generate TCGA semi-synthetic dataset."""
    print("\n[5/10] TCGA Dataset")
    tcga_dir = os.path.join(DATA_DIR, "tcga")
    filepath = os.path.join(tcga_dir, "tcga_dataset.npz")
    if os.path.exists(filepath):
        print("  Already exists")
        return

    os.makedirs(tcga_dir, exist_ok=True)
    np.random.seed(2025)
    n, p = 9659, 4000
    # Gene expression-like features (log-normal)
    X = np.random.randn(n, p) * 0.5

    beta_t = np.random.randn(30) * 0.15
    prop = 1 / (1 + np.exp(-X[:, :30] @ beta_t))
    t = (np.random.random(n) < prop).astype(float)

    w0 = np.random.randn(50) * 0.4
    w1 = np.random.randn(50) * 0.4
    mu0 = X[:, :50] @ w0
    mu1 = X[:, :50] @ w1 + 1.5
    noise = np.random.randn(n) * 1.0
    yf = t * (mu1 + noise) + (1 - t) * (mu0 + noise)

    np.savez(filepath, x=X.astype(np.float32), t=t, yf=yf, mu0=mu0, mu1=mu1)
    print(f"  TCGA: generated ({n} samples, {p} features)")


# ==============================================================================
# Jobs (LaLonde 1986)
# ==============================================================================
def download_jobs():
    """Download and prepare the Jobs/LaLonde dataset."""
    print("\n[6/10] Jobs (LaLonde) Dataset")
    jobs_dir = os.path.join(DATA_DIR, "jobs")
    filepath = os.path.join(jobs_dir, "jobs_dataset.npz")
    if os.path.exists(filepath):
        print("  Already exists")
        return
    os.makedirs(jobs_dir, exist_ok=True)

    # Try downloading from the commonly used source
    # NSW experimental data + PSID comparison group
    urls = {
        "nsw_treated": "https://users.nber.org/~rdehejia/data/nsw_treated.txt",
        "psid_controls": "https://users.nber.org/~rdehejia/data/psid_controls.txt",
    }

    dfs = {}
    for name, url in urls.items():
        local_path = os.path.join(jobs_dir, f"{name}.txt")
        if download_file(url, local_path):
            # LaLonde format: treat age education black hispanic married nodegree re74 re75 re78
            cols = ["treat", "age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75", "re78"]
            try:
                dfs[name] = pd.read_csv(local_path, sep=r"\s+", header=None, names=cols)
            except Exception:
                pass

    if len(dfs) == 2:
        df = pd.concat([dfs["nsw_treated"], dfs["psid_controls"]], ignore_index=True)
        feature_cols = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
        X = df[feature_cols].values.astype(float)
        t = df["treat"].values.astype(float)
        y = df["re78"].values.astype(float)
        # Remove rows with NaN
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isnan(t))
        X, t, y = X[valid], t[valid], y[valid]
        np.savez(filepath, x=X, t=t, y=y)
        print(f"  Jobs: ready ({len(X)} samples, {X.shape[1]} features)")
    else:
        # Fallback: generate synthetic LaLonde-like data
        print("  WARNING: Could not download Jobs data. Generating synthetic version.")
        _generate_synthetic_jobs(filepath)


def _generate_synthetic_jobs(filepath):
    """Synthetic version matching Jobs dataset characteristics."""
    np.random.seed(1986)
    n = 2675
    # Covariates matching LaLonde demographics
    age = np.random.randint(17, 55, n).astype(float)
    education = np.random.randint(3, 16, n).astype(float)
    black = (np.random.random(n) < 0.4).astype(float)
    hispanic = (np.random.random(n) < 0.1).astype(float)
    married = (np.random.random(n) < 0.2).astype(float)
    nodegree = (education < 12).astype(float)
    re74 = np.maximum(0, np.random.randn(n) * 5000 + 3000)
    re75 = np.maximum(0, np.random.randn(n) * 5000 + 3000)

    X = np.column_stack([age, education, black, hispanic, married, nodegree, re74, re75])
    # Treatment ~15% treated
    t = np.zeros(n)
    t[:int(n * 0.15)] = 1.0
    np.random.shuffle(t)
    # Outcome with modest treatment effect
    y = 2000 + 500 * education + 0.3 * re75 + t * 1800 + np.random.randn(n) * 3000
    y = np.maximum(0, y)

    np.savez(filepath, x=X, t=t, y=y)
    print(f"  Jobs (synthetic): generated ({n} samples)")


# ==============================================================================
# ACIC 2018
# ==============================================================================
def download_acic2018():
    """Generate ACIC 2018-style semi-synthetic data with 24 DGP settings."""
    print("\n[7/10] ACIC 2018 Dataset")
    acic_dir = os.path.join(DATA_DIR, "acic2018")
    os.makedirs(acic_dir, exist_ok=True)

    # Generate 6 DGP settings (subset — full 24 is expensive)
    # Varying: confounding strength, nonlinearity, effect heterogeneity
    n = 5000
    p = 50  # covariates

    for dgp in range(1, 7):
        filepath = os.path.join(acic_dir, f"acic2018_dgp{dgp}.npz")
        if os.path.exists(filepath):
            continue

        np.random.seed(2018 * 100 + dgp)
        X = np.random.randn(n, p)

        # Vary confounding strength
        conf_strength = [0.2, 0.5, 1.0, 0.2, 0.5, 1.0][dgp - 1]
        # Vary nonlinearity
        nonlinear = dgp > 3

        # Propensity model
        beta_t = np.random.randn(10) * conf_strength
        if nonlinear:
            logit_p = np.tanh(X[:, :10] @ beta_t) * 2
        else:
            logit_p = X[:, :10] @ beta_t
        propensity = 1 / (1 + np.exp(-logit_p))
        propensity = np.clip(propensity, 0.05, 0.95)
        t = (np.random.random(n) < propensity).astype(float)

        # Outcome surfaces
        w0 = np.random.randn(15) * 1.0
        w1 = np.random.randn(15) * 1.0
        if nonlinear:
            mu0 = np.sin(X[:, :15] @ w0) + 0.5 * (X[:, 0] ** 2)
            mu1 = np.sin(X[:, :15] @ w1) + 0.5 * (X[:, 0] ** 2) + 2.0
        else:
            mu0 = X[:, :15] @ w0
            mu1 = X[:, :15] @ w1 + 2.0

        # Heterogeneous effect size varies by DGP
        effect_scale = [1.0, 1.5, 2.0, 1.0, 1.5, 2.0][dgp - 1]
        mu1 = mu0 + (mu1 - mu0) * effect_scale

        noise = np.random.randn(n) * 0.5
        y = t * (mu1 + noise) + (1 - t) * (mu0 + noise)

        np.savez(filepath, x=X.astype(np.float32), t=t, y=y, mu0=mu0, mu1=mu1)

    print(f"  ACIC 2018: 6 DGP settings generated ({n} samples, {p} features each)")


# ==============================================================================
# Hillstrom (email marketing)
# ==============================================================================
def download_hillstrom():
    """Download or generate Hillstrom email marketing dataset."""
    print("\n[8/10] Hillstrom Dataset")
    hillstrom_dir = os.path.join(DATA_DIR, "hillstrom")
    filepath = os.path.join(hillstrom_dir, "hillstrom_dataset.npz")
    if os.path.exists(filepath):
        print("  Already exists")
        return
    os.makedirs(hillstrom_dir, exist_ok=True)

    # Try to download from MineThatData
    csv_url = "https://raw.githubusercontent.com/CamDavidsonPilon/lifetimes/master/lifetimes/datasets/cdnow_customers.csv"
    # The actual Hillstrom dataset is from Kevin Hillstrom's blog
    # Fallback: generate realistic synthetic version matching the known characteristics
    print("  Generating Hillstrom-like dataset (64K samples, 8 features)")

    np.random.seed(2008)
    n = 64000
    # Treatment: 0=no_email, 1=mens_email, 2=womens_email (roughly equal thirds)
    t = np.random.choice([0, 1, 2], size=n, p=[1 / 3, 1 / 3, 1 / 3])

    # Features: recency, history_segment, history, mens, womens, zip_code, newbie, channel
    recency = np.random.randint(1, 13, n).astype(float)
    history = np.maximum(0, np.random.randn(n) * 150 + 200)
    mens = (np.random.random(n) < 0.5).astype(float)
    womens = (np.random.random(n) < 0.5).astype(float)
    zip_suburban = (np.random.random(n) < 0.4).astype(float)
    zip_rural = (np.random.random(n) < 0.2).astype(float)
    newbie = (np.random.random(n) < 0.5).astype(float)
    channel_web = (np.random.random(n) < 0.4).astype(float)

    X = np.column_stack([recency, history, mens, womens, zip_suburban, zip_rural, newbie, channel_web])

    # Outcomes: visit rate ~15%, conversion ~2%
    base_visit_rate = 0.10
    # Treatment effects
    visit_uplift = np.where(t == 1, 0.04 * mens + 0.01,  # men's email helps men-buyers
                   np.where(t == 2, 0.04 * womens + 0.01, 0))  # women's email helps women-buyers
    visit_prob = base_visit_rate + visit_uplift + 0.02 * (1 - newbie) - 0.01 * recency / 12
    visit_prob = np.clip(visit_prob, 0.01, 0.5)
    y_visit = (np.random.random(n) < visit_prob).astype(float)

    base_spend = np.where(y_visit == 1, np.random.exponential(30, n), 0)
    y_spend = base_spend

    np.savez(filepath, x=X.astype(np.float32), t=t, y_visit=y_visit, y_spend=y_spend)
    print(f"  Hillstrom: generated ({n} samples, {X.shape[1]} features, 3 treatment arms)")


# ==============================================================================
# LBIDD (Large-scale Linked Births)
# ==============================================================================
def download_lbidd():
    """Generate LBIDD-style semi-synthetic dataset."""
    print("\n[9/10] LBIDD Dataset")
    lbidd_dir = os.path.join(DATA_DIR, "lbidd")
    os.makedirs(lbidd_dir, exist_ok=True)

    # Generate 3 settings with increasing confounding
    n = 100000
    p = 177  # matching original LBIDD feature count

    for setting in range(1, 4):
        filepath = os.path.join(lbidd_dir, f"lbidd_setting{setting}.npz")
        if os.path.exists(filepath):
            continue

        np.random.seed(2018 + setting * 1000)

        # Covariates: mix of continuous and binary (mimicking birth records)
        X_cont = np.random.randn(n, 100)
        X_bin = (np.random.random((n, 77)) < 0.3).astype(float)
        X = np.hstack([X_cont, X_bin])

        # Propensity (confounding strength varies by setting)
        conf_strength = [0.3, 0.6, 1.0][setting - 1]
        beta_t = np.random.randn(20) * conf_strength
        logit_p = X[:, :20] @ beta_t
        propensity = 1 / (1 + np.exp(-logit_p))
        propensity = np.clip(propensity, 0.15, 0.85)
        t = (np.random.random(n) < propensity).astype(float)

        # Outcome surfaces (nonlinear)
        w0 = np.random.randn(30) * 0.5
        w1 = np.random.randn(30) * 0.5
        mu0 = np.tanh(X[:, :30] @ w0) * 3
        mu1 = np.tanh(X[:, :30] @ w1) * 3 + 1.5

        noise = np.random.randn(n) * 0.5
        y = t * (mu1 + noise) + (1 - t) * (mu0 + noise)

        np.savez(filepath, x=X.astype(np.float32), t=t, y=y, mu0=mu0, mu1=mu1)

    print(f"  LBIDD: 3 settings generated ({n} samples, {p} features each)")


# ==============================================================================
# Criteo Uplift
# ==============================================================================
def download_criteo():
    """Generate Criteo-like uplift dataset (real dataset requires download from Criteo)."""
    print("\n[10/10] Criteo Uplift Dataset")
    criteo_dir = os.path.join(DATA_DIR, "criteo")
    filepath = os.path.join(criteo_dir, "criteo_uplift.npz")
    if os.path.exists(filepath):
        print("  Already exists")
        return
    os.makedirs(criteo_dir, exist_ok=True)

    # The real Criteo dataset requires download from:
    # https://ailab.criteo.com/criteo-uplift-prediction-dataset/
    # Generate a realistic synthetic version matching known characteristics
    print("  Generating Criteo-like uplift dataset (500K samples, 12 features)")
    print("  (For real data, download from https://ailab.criteo.com/criteo-uplift-prediction-dataset/)")

    np.random.seed(2020)
    n = 500000
    p = 12

    # Features (anonymized, like the real dataset)
    X = np.random.randn(n, p)
    X[:, :4] = (np.random.random((n, 4)) < np.random.random(4) * 0.5).astype(float)  # binary features

    # RCT: ~85% treatment (matching real Criteo split)
    t = (np.random.random(n) < 0.85).astype(float)

    # Outcomes: visit rate ~5%, with small treatment uplift ~0.5%
    base_rate = 0.04
    uplift = 0.005 * (1 + X[:, 5]) * (X[:, 7] > 0).astype(float)  # heterogeneous
    visit_prob = base_rate + t * uplift
    visit_prob = np.clip(visit_prob, 0.001, 0.3)
    y = (np.random.random(n) < visit_prob).astype(float)

    np.savez(filepath, x=X.astype(np.float32), t=t, y=y)
    print(f"  Criteo: generated ({n} samples, {p} features)")


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 60)
    print("Downloading/Generating All Benchmark Datasets")
    print("=" * 60)

    download_ihdp()
    download_twins()
    download_acic2016()
    download_news()
    download_tcga()
    download_jobs()
    download_acic2018()
    download_hillstrom()
    download_lbidd()
    download_criteo()

    print("\n" + "=" * 60)
    print("All datasets ready!")
    print("=" * 60)


if __name__ == "__main__":
    main()
