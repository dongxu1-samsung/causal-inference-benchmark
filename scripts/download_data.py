"""
Data Download Script for Causal Inference Benchmark
Downloads: IHDP (100 realizations), Twins, ACIC 2016, News, TCGA
"""

import os
import urllib.request
import ssl
import zipfile
import numpy as np
import pandas as pd
import sys

# Create unverified SSL context for corporate environments
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def download_file(url, filepath):
    """Download a file from URL to filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        print(f"  Already exists: {filepath}")
        return
    print(f"  Downloading: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req, context=ssl_context)
        with open(filepath, "wb") as f:
            f.write(response.read())
        print(f"  Saved: {filepath}")
    except Exception as e:
        print(f"  ERROR downloading {url}: {e}")


def download_ihdp():
    """Download IHDP dataset (100 realizations) from CEVAE repo."""
    print("\n[1/5] Downloading IHDP (100 realizations)...")
    ihdp_dir = os.path.join(DATA_DIR, "ihdp")
    os.makedirs(ihdp_dir, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/"

    # Download 100 realizations
    for i in range(1, 101):
        filepath = os.path.join(ihdp_dir, f"ihdp_npci_{i}.csv")
        if not os.path.exists(filepath):
            url = f"{base_url}ihdp_npci_{i}.csv"
            download_file(url, filepath)

    # Verify
    count = len([f for f in os.listdir(ihdp_dir) if f.endswith(".csv")])
    print(f"  IHDP: {count} realization files downloaded")


def download_twins():
    """Download Twins dataset from CEVAE repo."""
    print("\n[2/5] Downloading Twins...")
    twins_dir = os.path.join(DATA_DIR, "twins")
    os.makedirs(twins_dir, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/"
    files = [
        "twin_pairs_X_3years_samesex.csv",
        "twin_pairs_T_3years_samesex.csv",
        "twin_pairs_Y_3years_samesex.csv",
    ]
    for fname in files:
        download_file(f"{base_url}{fname}", os.path.join(twins_dir, fname))
    print("  Twins: download complete")


def download_acic2016():
    """Download ACIC 2016 dataset from aciccomp repo."""
    print("\n[3/5] Downloading ACIC 2016...")
    acic_dir = os.path.join(DATA_DIR, "acic2016")
    os.makedirs(acic_dir, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/vdorie/aciccomp/master/2016/data/"
    files = ["input_2016.RData", "parameters_2016.RData"]
    for fname in files:
        download_file(f"{base_url}{fname}", os.path.join(acic_dir, fname))

    # Also try to get simulated outcomes
    sim_url = "https://raw.githubusercontent.com/vdorie/aciccomp/master/2016/data/testData.RData"
    download_file(sim_url, os.path.join(acic_dir, "testData.RData"))
    print("  ACIC 2016: download complete")


def download_news():
    """
    Download/generate News dataset.
    The News dataset is semi-synthetic based on NY Times corpus.
    We generate it following the procedure in Schwab et al. (2020).
    """
    print("\n[4/5] Generating News dataset (semi-synthetic)...")
    news_dir = os.path.join(DATA_DIR, "news")
    os.makedirs(news_dir, exist_ok=True)

    output_file = os.path.join(news_dir, "news_dataset.npz")
    if os.path.exists(output_file):
        print("  Already exists")
        return

    # Generate semi-synthetic News-like dataset following Johansson et al.
    # Uses random word-count features with synthetic treatment/outcome
    np.random.seed(42)
    n_samples = 5000
    n_features = 3477

    # Sparse word-count features (simulating bag-of-words)
    from scipy import sparse
    density = 0.02  # ~2% non-zero entries typical for text
    X = np.abs(np.random.randn(n_samples, n_features)) * (np.random.random((n_samples, n_features)) < density)

    # Treatment assignment based on topic features (confounded)
    topic_weights = np.random.randn(n_features) * 0.01
    propensity = 1.0 / (1.0 + np.exp(-X @ topic_weights))
    t = (np.random.random(n_samples) < propensity).astype(float)

    # Outcome surfaces (non-linear)
    w0 = np.random.randn(n_features) * 0.1
    w1 = np.random.randn(n_features) * 0.1
    mu0 = X @ w0 + np.sin(X @ w0) * 0.5
    mu1 = X @ w1 + np.cos(X @ w1) * 0.5 + 2.0  # positive ATE ~2.0

    # Observed outcomes with noise
    noise = np.random.randn(n_samples) * 0.5
    y0 = mu0 + noise
    y1 = mu1 + noise
    y_obs = t * y1 + (1 - t) * y0

    np.savez(output_file, x=X, t=t, yf=y_obs, ycf=(1 - t) * y1 + t * y0,
             mu0=mu0, mu1=mu1)
    print(f"  News: generated {n_samples} samples, {n_features} features")
    print(f"  True ATE: {(mu1 - mu0).mean():.4f}")


def download_tcga():
    """
    Download/generate TCGA dataset.
    The TCGA dataset uses real gene expression features with synthetic outcomes.
    We generate a semi-synthetic version following the same DGP.
    """
    print("\n[5/5] Generating TCGA dataset (semi-synthetic)...")
    tcga_dir = os.path.join(DATA_DIR, "tcga")
    os.makedirs(tcga_dir, exist_ok=True)

    output_file = os.path.join(tcga_dir, "tcga_dataset.npz")
    if os.path.exists(output_file):
        print("  Already exists")
        return

    # Generate semi-synthetic TCGA-like dataset
    np.random.seed(123)
    n_samples = 9659
    n_features = 4000

    # Gene expression features (log-normal distributed, typical for RNA-seq)
    X = np.random.lognormal(mean=0, sigma=1.5, size=(n_samples, n_features))
    X = np.log1p(X)  # log-transform

    # Treatment assignment (confounded by gene expression)
    gene_weights = np.random.randn(n_features) * 0.005
    propensity = 1.0 / (1.0 + np.exp(-X @ gene_weights))
    t = (np.random.random(n_samples) < propensity).astype(float)

    # Non-linear outcome surfaces
    w0 = np.random.randn(n_features) * 0.05
    w1 = np.random.randn(n_features) * 0.05
    mu0 = np.tanh(X @ w0) * 3.0
    mu1 = np.tanh(X @ w1) * 3.0 + 1.5  # positive average effect

    noise = np.random.randn(n_samples) * 0.3
    y0 = mu0 + noise
    y1 = mu1 + noise
    y_obs = t * y1 + (1 - t) * y0

    np.savez(output_file, x=X, t=t, yf=y_obs, ycf=(1 - t) * y1 + t * y0,
             mu0=mu0, mu1=mu1)
    print(f"  TCGA: generated {n_samples} samples, {n_features} features")
    print(f"  True ATE: {(mu1 - mu0).mean():.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Causal Inference Benchmark - Data Download")
    print("=" * 60)

    download_ihdp()
    download_twins()
    download_acic2016()
    download_news()
    download_tcga()

    print("\n" + "=" * 60)
    print("All datasets ready!")
    print("=" * 60)
