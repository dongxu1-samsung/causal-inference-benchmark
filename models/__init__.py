"""
Unified data loading for all benchmark datasets.
Each loader returns: (X_train, t_train, y_train, X_val, t_val, y_val, X_test, t_test, y_test, mu0_test, mu1_test)

For datasets without ground-truth ITE (Jobs, Criteo, Hillstrom):
  mu0_test and mu1_test are None — use policy risk / ATT / uplift metrics instead.

Split ratios: 60% train / 20% validation / 20% test
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _split_3way(X, t, y, mu0, mu1, seed=42, val_fraction=0.2, test_fraction=0.2):
    """Helper: 3-way train/val/test split preserving ground truth on test."""
    n = len(X)
    idx = np.arange(n)
    # First split off test
    train_val_idx, test_idx = train_test_split(idx, test_size=test_fraction, random_state=seed)
    # Then split train_val into train and val
    val_frac_of_remaining = val_fraction / (1 - test_fraction)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_frac_of_remaining, random_state=seed + 1)

    result = {
        'X_train': X[train_idx], 't_train': t[train_idx], 'y_train': y[train_idx],
        'X_val': X[val_idx], 't_val': t[val_idx], 'y_val': y[val_idx],
        'X_test': X[test_idx], 't_test': t[test_idx], 'y_test': y[test_idx],
        'mu0_test': mu0[test_idx] if mu0 is not None else None,
        'mu1_test': mu1[test_idx] if mu1 is not None else None,
    }
    return result


def _split_3way_no_ground_truth(X, t, y, seed=42, val_fraction=0.2, test_fraction=0.2):
    """Helper for real-world datasets without ground-truth ITE."""
    return _split_3way(X, t, y, None, None, seed, val_fraction, test_fraction)


# ==============================================================================
# DATASET 1: IHDP (Hill 2011)
# ==============================================================================
def load_ihdp(realization=1, seed=None):
    """Load a single IHDP realization. Seed defaults to realization number."""
    if seed is None:
        seed = realization
    filepath = os.path.join(DATA_DIR, "ihdp", f"ihdp_npci_{realization}.csv")
    cols = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]
    df = pd.read_csv(filepath, header=None, names=cols)

    X = df[[f"x{i}" for i in range(1, 26)]].values
    t = df["treatment"].values
    y = df["y_factual"].values
    mu0 = df["mu0"].values
    mu1 = df["mu1"].values

    return _split_3way(X, t, y, mu0, mu1, seed=seed)


# ==============================================================================
# DATASET 2: Twins
# ==============================================================================
def load_twins(seed=42):
    """Load Twins dataset (binary outcomes)."""
    twins_dir = os.path.join(DATA_DIR, "twins")

    X = pd.read_csv(os.path.join(twins_dir, "twin_pairs_X_3years_samesex.csv"), index_col=0)
    T = pd.read_csv(os.path.join(twins_dir, "twin_pairs_T_3years_samesex.csv"), index_col=0)
    Y = pd.read_csv(os.path.join(twins_dir, "twin_pairs_Y_3years_samesex.csv"), index_col=0)

    t_assign = (T["dbirwt_1"].values > T["dbirwt_0"].values).astype(float)
    y0 = Y["mort_0"].values
    y1 = Y["mort_1"].values
    y_obs = t_assign * y1 + (1 - t_assign) * y0

    X_vals = X.select_dtypes(include=[np.number]).fillna(0).values
    valid = ~(np.isnan(y0) | np.isnan(y1))
    X_vals, t_assign, y_obs, y0, y1 = X_vals[valid], t_assign[valid], y_obs[valid], y0[valid], y1[valid]

    # Subsample to 10K
    np.random.seed(seed)
    if len(X_vals) > 10000:
        idx = np.random.choice(len(X_vals), 10000, replace=False)
        X_vals, t_assign, y_obs, y0, y1 = X_vals[idx], t_assign[idx], y_obs[idx], y0[idx], y1[idx]

    return _split_3way(X_vals, t_assign, y_obs, y0, y1, seed=seed)


# ==============================================================================
# DATASET 3: ACIC 2016
# ==============================================================================
def load_acic2016(dgp=1, replication=1, seed=42):
    """Load ACIC 2016 with semi-synthetic DGP."""
    acic_dir = os.path.join(DATA_DIR, "acic2016")

    try:
        import rdata
        parsed = rdata.parser.parse_file(os.path.join(acic_dir, "input_2016.RData"))
        converted = rdata.conversion.convert(parsed)
        X_df = list(converted.values())[0]
    except Exception:
        csv_path = os.path.join(acic_dir, "input_2016_covariates.csv")
        if os.path.exists(csv_path):
            X_df = pd.read_csv(csv_path, index_col=0)
        else:
            raise FileNotFoundError("Cannot load ACIC 2016 covariates.")

    X_df = pd.get_dummies(X_df, drop_first=True)
    X = X_df.values.astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    n = len(X)

    np.random.seed(seed + dgp * 100 + replication)
    k = min(20, X.shape[1])
    beta_t = np.random.randn(k) * 0.3
    propensity = 1.0 / (1.0 + np.exp(-X[:, :k] @ beta_t))
    propensity = np.clip(propensity, 0.1, 0.9)
    t = (np.random.random(n) < propensity).astype(float)

    w0 = np.random.randn(k) * 0.5
    w1 = np.random.randn(k) * 0.5
    mu0 = X[:, :k] @ w0
    mu1 = X[:, :k] @ w1 + 2.0

    noise = np.random.randn(n) * 1.0
    y_obs = t * (mu1 + noise) + (1 - t) * (mu0 + noise)

    return _split_3way(X, t, y_obs, mu0, mu1, seed=seed)


# ==============================================================================
# DATASET 4: News (semi-synthetic)
# ==============================================================================
def load_news(seed=42):
    """Load News dataset."""
    filepath = os.path.join(DATA_DIR, "news", "news_dataset.npz")
    data = np.load(filepath)
    X, t, yf, mu0, mu1 = data["x"], data["t"], data["yf"], data["mu0"], data["mu1"]
    return _split_3way(X, t, yf, mu0, mu1, seed=seed)


# ==============================================================================
# DATASET 5: TCGA (semi-synthetic gene expression)
# ==============================================================================
def load_tcga(seed=42):
    """Load TCGA dataset."""
    filepath = os.path.join(DATA_DIR, "tcga", "tcga_dataset.npz")
    data = np.load(filepath)
    X, t, yf, mu0, mu1 = data["x"], data["t"], data["yf"], data["mu0"], data["mu1"]
    return _split_3way(X, t, yf, mu0, mu1, seed=seed)


# ==============================================================================
# DATASET 6: Jobs (LaLonde 1986) — Real-world RCT + observational
# ==============================================================================
def load_jobs(seed=42):
    """
    Load Jobs/LaLonde dataset. No ground-truth ITE.
    Uses experimental data for treatment group + PSID/CPS for control.
    Outcome: post-intervention earnings (RE78).
    """
    jobs_dir = os.path.join(DATA_DIR, "jobs")
    filepath = os.path.join(jobs_dir, "jobs_dataset.npz")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Jobs dataset not found at {filepath}. Run download_data.py first.")

    data = np.load(filepath)
    X, t, y = data["x"], data["t"], data["y"]
    # Jobs has no ground-truth counterfactuals
    return _split_3way_no_ground_truth(X, t, y, seed=seed)


# ==============================================================================
# DATASET 7: ACIC 2018 (24 DGP settings, harder benchmark)
# ==============================================================================
def load_acic2018(dgp=1, seed=42):
    """
    Load ACIC 2018 dataset. Semi-synthetic with known ground truth.
    24 DGP settings varying confounding strength and nonlinearity.
    """
    acic_dir = os.path.join(DATA_DIR, "acic2018")
    filepath = os.path.join(acic_dir, f"acic2018_dgp{dgp}.npz")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ACIC 2018 DGP {dgp} not found. Run download_data.py first.")

    data = np.load(filepath)
    X, t, y, mu0, mu1 = data["x"], data["t"], data["y"], data["mu0"], data["mu1"]
    return _split_3way(X, t, y, mu0, mu1, seed=seed)


# ==============================================================================
# DATASET 8: Hillstrom (email marketing RCT, multi-treatment)
# ==============================================================================
def load_hillstrom(treatment="mens", seed=42):
    """
    Load Hillstrom email marketing dataset.
    Multi-treatment: 'mens' email, 'womens' email, or 'no_email' control.
    Binary outcomes: visit, conversion. Continuous: spend.
    We use visit as primary outcome, binary treatment (specified arm vs no_email).
    """
    hillstrom_dir = os.path.join(DATA_DIR, "hillstrom")
    filepath = os.path.join(hillstrom_dir, "hillstrom_dataset.npz")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Hillstrom dataset not found. Run download_data.py first.")

    data = np.load(filepath, allow_pickle=True)
    X, t_raw, y_visit, y_spend = data["x"], data["t"], data["y_visit"], data["y_spend"]

    # Filter to: treatment arm vs no_email control
    if treatment == "mens":
        mask = (t_raw == 1) | (t_raw == 0)  # 1=mens, 0=no_email
        t_binary = t_raw[mask]
    elif treatment == "womens":
        mask = (t_raw == 2) | (t_raw == 0)  # 2=womens, 0=no_email
        t_binary = (t_raw[mask] == 2).astype(float)
    else:
        raise ValueError("treatment must be 'mens' or 'womens'")

    X_filtered = X[mask]
    y_filtered = y_visit[mask].astype(float)

    # No ground-truth ITE (real RCT)
    return _split_3way_no_ground_truth(X_filtered, t_binary, y_filtered, seed=seed)


# ==============================================================================
# DATASET 9: LBIDD (Large-scale Linked Births)
# ==============================================================================
def load_lbidd(setting=1, seed=42):
    """
    Load LBIDD dataset (Shimoni et al. 2018). Semi-synthetic.
    Real covariates from linked birth/infant death records, synthetic outcomes.
    """
    lbidd_dir = os.path.join(DATA_DIR, "lbidd")
    filepath = os.path.join(lbidd_dir, f"lbidd_setting{setting}.npz")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"LBIDD setting {setting} not found. Run download_data.py first.")

    data = np.load(filepath)
    X, t, y, mu0, mu1 = data["x"], data["t"], data["y"], data["mu0"], data["mu1"]

    # Subsample to 50K for tractability (full is 100K+)
    np.random.seed(seed)
    if len(X) > 50000:
        idx = np.random.choice(len(X), 50000, replace=False)
        X, t, y, mu0, mu1 = X[idx], t[idx], y[idx], mu0[idx], mu1[idx]

    return _split_3way(X, t, y, mu0, mu1, seed=seed)


# ==============================================================================
# DATASET 10: Criteo Uplift (large-scale industrial RCT)
# ==============================================================================
def load_criteo(seed=42, max_samples=50000):
    """
    Load Criteo Uplift dataset. Real industrial RCT — no ground-truth ITE.
    Subsampled for tractability (full dataset is 14M).
    """
    criteo_dir = os.path.join(DATA_DIR, "criteo")
    filepath = os.path.join(criteo_dir, "criteo_uplift.npz")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Criteo dataset not found. Run download_data.py first.")

    data = np.load(filepath)
    X, t, y = data["x"], data["t"], data["y"]

    # Subsample
    np.random.seed(seed)
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X, t, y = X[idx], t[idx], y[idx]

    return _split_3way_no_ground_truth(X, t, y, seed=seed)


# ==============================================================================
# DATASET 11: NLSM (Non-Linear Synthetic Model)
# ==============================================================================
def load_nlsm(difficulty="medium", n=10000, p=20, seed=42):
    """Load NLSM dataset with configurable HTE difficulty.
    difficulty: 'easy' (linear HTE), 'medium' (nonlinear), 'hard' (complex interactions)
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)

    # Propensity: logistic on first few features
    logit_e = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]
    e = 1.0 / (1.0 + np.exp(-logit_e))
    t = rng.binomial(1, e).astype(float)

    # Baseline outcome
    mu0 = 2 * X[:, 0] + X[:, 1]**2 - 0.5 * X[:, 2] * X[:, 3]

    # Treatment effect by difficulty
    if difficulty == "easy":
        tau = 1.0 + 0.5 * X[:, 0]
    elif difficulty == "medium":
        tau = 1.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1]**2 - 0.2 * np.abs(X[:, 2])
    else:  # hard
        tau = (1.0 + 0.5 * np.sin(2 * X[:, 0]) + 0.3 * X[:, 1] * X[:, 2]
               + 0.2 * np.exp(-X[:, 3]**2) - 0.1 * X[:, 4]**3)

    mu1 = mu0 + tau
    noise = rng.randn(n) * 0.5
    y = t * mu1 + (1 - t) * mu0 + noise

    return _split_3way(X, t, y, mu0, mu1, seed=seed)


# ==============================================================================
# DATASET 12: IBM Causal Benchmark
# ==============================================================================
def load_ibm_causal(confounding_strength=0.5, n=10000, p=30, seed=42):
    """IBM-style synthetic SCM with tunable confounding strength.
    confounding_strength: 0.0 (no confounding) to 1.0 (strong confounding)
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)

    # Confounders: first 5 features affect both T and Y
    conf = X[:, :5]
    conf_effect = confounding_strength * (conf @ rng.randn(5))

    # Treatment: logistic model with confounding
    logit_t = conf_effect + 0.3 * rng.randn(n)
    e = 1.0 / (1.0 + np.exp(-logit_t))
    t = rng.binomial(1, e).astype(float)

    # Outcome: nonlinear function of X with confounding-influenced heterogeneity
    w0 = rng.randn(p) * 0.5
    w0[:5] += confounding_strength  # confounders influence outcome
    mu0 = np.tanh(X @ w0) * 2

    # Treatment effect: varies with confounders and some other features
    tau = (1.5 + 0.5 * X[:, 0] - 0.3 * X[:, 1]**2
           + confounding_strength * 0.5 * X[:, 2] * X[:, 3])
    mu1 = mu0 + tau

    noise = rng.randn(n) * 0.3
    y = t * mu1 + (1 - t) * mu0 + noise

    return _split_3way(X, t, y, mu0, mu1, seed=seed)


# ==============================================================================
# DATASET 13: Continuous Treatment DGP
# ==============================================================================
def load_continuous_treatment(n=10000, p=25, seed=42):
    """Continuous treatment t ~ Uniform(0,1). Dose-response benchmark.
    Returns mu0_test=Y(t=0), mu1_test=Y(t=1) for ITE compatibility.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)

    # Continuous treatment influenced by covariates
    # t = sigmoid(X features) clipped to [0.05, 0.95]
    logit_t = 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + 0.5 * rng.randn(n)
    t = 1.0 / (1.0 + np.exp(-logit_t))
    t = np.clip(t, 0.05, 0.95)

    # Baseline: f(X)
    f_x = 2 * X[:, 0] + X[:, 1]**2 - 0.5 * X[:, 2] * X[:, 3] + 0.3 * X[:, 4]

    # Dose-response: g(X) * t + h(X) * t^2 (quadratic in dose)
    g_x = 1.5 + 0.5 * X[:, 0] + 0.3 * X[:, 1]  # linear dose coefficient
    h_x = -0.5 * X[:, 2]  # quadratic dose coefficient

    # Y = f(X) + g(X)*t + h(X)*t^2 + noise
    y = f_x + g_x * t + h_x * t**2 + rng.randn(n) * 0.3

    # Ground truth: mu0 = Y(t=0) = f(X), mu1 = Y(t=1) = f(X) + g(X) + h(X)
    mu0 = f_x.copy()
    mu1 = f_x + g_x + h_x  # at t=1

    return _split_3way(X, t, y, mu0, mu1, seed=seed)


# ==============================================================================
# DATASET 14: ACIC 2022 (Longitudinal/Panel)
# ==============================================================================
def load_acic2022(n=5000, p=30, n_periods=5, treatment_start=3, seed=42):
    """ACIC 2022-style longitudinal semi-synthetic dataset.
    Treatment starts at period `treatment_start` for treated units.
    Returns cross-sectional view at final period for benchmark compatibility.
    """
    rng = np.random.RandomState(seed)

    # Static covariates
    X_static = rng.randn(n, p)

    # Treatment assignment (based on covariates)
    logit_t = 0.4 * X_static[:, 0] + 0.3 * X_static[:, 1] - 0.2 * X_static[:, 2]
    e = 1.0 / (1.0 + np.exp(-logit_t))
    treated = rng.binomial(1, e).astype(float)

    # Baseline trajectory (AR(1) process)
    Y_panels = np.zeros((n, n_periods))
    Y_panels[:, 0] = X_static[:, 0] + 0.5 * X_static[:, 1] + rng.randn(n) * 0.3

    for period in range(1, n_periods):
        # AR(1) + time trend
        Y_panels[:, period] = (0.7 * Y_panels[:, period - 1]
                               + 0.2 * X_static[:, 2]
                               + 0.1 * period
                               + rng.randn(n) * 0.3)
        # Treatment effect kicks in after treatment_start
        if period >= treatment_start:
            # Growing treatment effect over time
            periods_since = period - treatment_start + 1
            tau_t = treated * (1.0 + 0.3 * periods_since
                               + 0.2 * X_static[:, 0] * periods_since
                               - 0.1 * X_static[:, 3])
            Y_panels[:, period] += tau_t

    # Final period outcome for benchmark
    y_final = Y_panels[:, -1]

    # Ground truth ITE at final period
    periods_since_final = n_periods - treatment_start
    tau_final = (1.0 + 0.3 * periods_since_final
                 + 0.2 * X_static[:, 0] * periods_since_final
                 - 0.1 * X_static[:, 3])

    mu0 = y_final - treated * tau_final  # counterfactual no-treatment
    mu1 = mu0 + tau_final                # counterfactual with treatment

    # Augment covariates with pre-treatment outcomes (standard in panel methods)
    X_aug = np.hstack([X_static, Y_panels[:, :treatment_start]])

    return _split_3way(X_aug, treated, y_final, mu0, mu1, seed=seed)


# ==============================================================================
# DATASET 15: STAR (Tennessee Student/Teacher Achievement Ratio)
# ==============================================================================
def load_star(n=11000, p=30, seed=42):
    """Simulated STAR-like education RCT.
    Binary treatment: small class (13-17 students) vs regular (22-25).
    Known HTE: ~0.2 SD average effect, varies by demographics.
    """
    rng = np.random.RandomState(seed)

    # Features: school/student characteristics
    # Continuous features
    X_cont = rng.randn(n, 20)
    # Binary features (race, gender, free lunch, urban, etc.)
    X_bin = rng.binomial(1, 0.4, size=(n, 10)).astype(float)
    X = np.hstack([X_cont, X_bin])

    # Treatment: random assignment (RCT) with slight school-level clustering
    school_id = rng.randint(0, 80, size=n)
    school_effect = rng.randn(80) * 0.1
    base_prob = 0.33 + school_effect[school_id]  # ~1/3 assigned to small class
    base_prob = np.clip(base_prob, 0.2, 0.5)
    t = rng.binomial(1, base_prob).astype(float)

    # Outcome: test scores (standardized)
    # Baseline depends on demographics
    mu0 = (0.5 * X[:, 0]  # prior achievement
           + 0.3 * X[:, 1]  # parent education
           - 0.4 * X[:, 20]  # free lunch (poverty proxy)
           + 0.2 * X[:, 21]  # gender
           + 0.3 * X[:, 2]   # school quality
           + rng.randn(n) * 0.1)  # small noise in potential outcome

    # Treatment effect: HTE by demographics
    # Average ~0.2 SD, larger for disadvantaged students
    tau = (0.2  # base effect
           + 0.15 * X[:, 20]   # larger for free-lunch students
           + 0.1 * (1 - X[:, 21])  # slightly larger for girls
           - 0.05 * X[:, 0]    # smaller for high-achievers
           + 0.08 * X[:, 22])  # minority students benefit more

    mu1 = mu0 + tau
    noise = rng.randn(n) * 0.5
    y = t * mu1 + (1 - t) * mu0 + noise

    return _split_3way(X, t, y, mu0, mu1, seed=seed)


# ==============================================================================
# Registry
# ==============================================================================
def get_dataset_loader(name):
    """Get dataset loader by name."""
    loaders = {
        "ihdp": load_ihdp,
        "twins": load_twins,
        "acic2016": load_acic2016,
        "news": load_news,
        "tcga": load_tcga,
        "jobs": load_jobs,
        "acic2018": load_acic2018,
        "hillstrom": load_hillstrom,
        "lbidd": load_lbidd,
        "criteo": load_criteo,
        "nlsm": load_nlsm,
        "ibm_causal": load_ibm_causal,
        "continuous": load_continuous_treatment,
        "acic2022": load_acic2022,
        "star": load_star,
    }
    return loaders.get(name)


# Metadata for reporting
DATASET_INFO = {
    "ihdp": {"has_ground_truth": True, "type": "semi-synthetic", "source": "Hill 2011"},
    "twins": {"has_ground_truth": True, "type": "semi-synthetic", "source": "Linked Birth/Death Records"},
    "acic2016": {"has_ground_truth": True, "type": "semi-synthetic", "source": "ACIC Competition 2016"},
    "news": {"has_ground_truth": True, "type": "semi-synthetic", "source": "Generated (text features)"},
    "tcga": {"has_ground_truth": True, "type": "semi-synthetic", "source": "Generated (gene expression)"},
    "jobs": {"has_ground_truth": False, "type": "real-world RCT", "source": "LaLonde 1986"},
    "acic2018": {"has_ground_truth": True, "type": "semi-synthetic", "source": "ACIC Competition 2018"},
    "hillstrom": {"has_ground_truth": False, "type": "real-world RCT", "source": "Hillstrom 2008"},
    "lbidd": {"has_ground_truth": True, "type": "semi-synthetic", "source": "Shimoni et al. 2018"},
    "criteo": {"has_ground_truth": False, "type": "real-world RCT", "source": "Criteo AI Lab"},
    "nlsm": {"has_ground_truth": True, "type": "synthetic", "source": "GRF-style NLSM"},
    "ibm_causal": {"has_ground_truth": True, "type": "synthetic", "source": "IBM Causal Benchmark"},
    "continuous": {"has_ground_truth": True, "type": "synthetic", "source": "Continuous Treatment DGP"},
    "acic2022": {"has_ground_truth": True, "type": "semi-synthetic", "source": "ACIC 2022 Longitudinal"},
    "star": {"has_ground_truth": True, "type": "semi-synthetic", "source": "STAR Education RCT"},
}
