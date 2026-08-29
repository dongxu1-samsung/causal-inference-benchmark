#!/usr/bin/env python3
"""
Benchmark runner for 6 new models (CausalForestDML, DR-Learner, X-Learner,
R-Learner, Uplift Random Forest, BART) on 15 datasets.

Reports:
- PEHE + ITE Correlation on 12 GT datasets
- AUUC + Qini on 3 no-GT datasets (Jobs, Hillstrom, Criteo)
"""
import sys, os, time, json, warnings
import numpy as np
from collections import defaultdict

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_dataset_loader

# ==============================================================================
# METRICS
# ==============================================================================
def compute_metrics_gt(ite_pred, mu0_test, mu1_test):
    """Metrics for datasets with ground truth."""
    ite_true = mu1_test - mu0_test
    pehe = float(np.sqrt(np.mean((ite_pred - ite_true) ** 2)))
    if np.std(ite_true) > 1e-8 and np.std(ite_pred) > 1e-8:
        ite_corr = float(np.corrcoef(ite_pred, ite_true)[0, 1])
    else:
        ite_corr = 0.0
    return {'pehe': round(pehe, 2), 'ite_corr': round(ite_corr, 2)}


def compute_auuc(ite_pred, t_test, y_test, n_bins=20):
    """Area Under the Uplift Curve (normalized over random)."""
    n = len(ite_pred)
    order = np.argsort(-ite_pred)
    t_sorted = t_test[order]
    y_sorted = y_test[order]

    fractions = np.linspace(0.05, 1.0, n_bins)
    uplift_curve = []

    for frac in fractions:
        k = int(frac * n)
        t_topk = t_sorted[:k]
        y_topk = y_sorted[:k]
        treated_mask = t_topk == 1
        control_mask = t_topk == 0
        if treated_mask.sum() > 10 and control_mask.sum() > 10:
            uplift = y_topk[treated_mask].mean() - y_topk[control_mask].mean()
        else:
            uplift = 0.0
        uplift_curve.append(uplift * frac)

    auuc = float(np.trapz(uplift_curve, fractions))
    t1_mask = t_test == 1
    t0_mask = t_test == 0
    if t1_mask.sum() > 0 and t0_mask.sum() > 0:
        overall_ate = y_test[t1_mask].mean() - y_test[t0_mask].mean()
    else:
        overall_ate = 0.0
    random_auuc = float(np.trapz([overall_ate * f for f in fractions], fractions))
    return round(auuc - random_auuc, 4)


def compute_qini(ite_pred, t_test, y_test, n_bins=20):
    """Qini coefficient."""
    n = len(ite_pred)
    order = np.argsort(-ite_pred)
    t_sorted = t_test[order].astype(float)
    y_sorted = y_test[order].astype(float)

    fractions = np.linspace(0.05, 1.0, n_bins)
    qini_curve = []

    for frac in fractions:
        k = int(frac * n)
        t_k = t_sorted[:k]
        y_k = y_sorted[:k]
        n_t = t_k.sum()
        n_c = (1 - t_k).sum()
        if n_t > 10 and n_c > 10:
            uplift = (y_k[t_k == 1].sum() / n_t) - (y_k[t_k == 0].sum() / n_c)
        else:
            uplift = 0.0
        qini_curve.append(uplift * frac)

    qini_area = float(np.trapz(qini_curve, fractions))
    t1_mask = t_test == 1
    t0_mask = t_test == 0
    if t1_mask.sum() > 0 and t0_mask.sum() > 0:
        overall_uplift = y_test[t1_mask].mean() - y_test[t0_mask].mean()
    else:
        overall_uplift = 0.0
    random_area = float(np.trapz([overall_uplift * f for f in fractions], fractions))
    return round(qini_area - random_area, 4)


def compute_metrics_no_gt(ite_pred, t_test, y_test):
    """AUUC + Qini for no-GT datasets."""
    auuc = compute_auuc(ite_pred, t_test, y_test)
    qini = compute_qini(ite_pred, t_test, y_test)
    return {'auuc': auuc, 'qini': qini}


# ==============================================================================
# MODEL IMPLEMENTATIONS
# ==============================================================================

def estimate_propensity(X, t):
    """Estimate propensity scores."""
    from sklearn.linear_model import LogisticRegression
    n_features = X.shape[1]
    C_val = 1.0 if n_features < 100 else 0.1
    prop = LogisticRegression(C=C_val, max_iter=1000, solver='lbfgs')
    prop.fit(X, t)
    return prop


def train_causal_forest_dml(X_train, t_train, y_train, X_val, t_val, y_val,
                            X_test, t_test, config):
    """EconML CausalForestDML."""
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingRegressor

    X_all = np.vstack([X_train, X_val])
    t_all = np.concatenate([t_train, t_val])
    y_all = np.concatenate([y_train, y_val])

    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1),
        n_estimators=config.get('n_estimators', 300),
        max_depth=config.get('max_depth', None),
        min_samples_leaf=config.get('min_samples_leaf', 5),
        honest=True,
        cv=3,
        random_state=config.get('seed', 42)
    )
    try:
        cf.fit(y_all, t_all, X=X_all)
    except np.linalg.LinAlgError:
        # Fallback: disable sensitivity analysis by using fewer CV folds
        cf = CausalForestDML(
            model_y=GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1),
            model_t=GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1),
            n_estimators=config.get('n_estimators', 300),
            max_depth=config.get('max_depth', None),
            min_samples_leaf=config.get('min_samples_leaf', 5),
            honest=True,
            cv=2,
            random_state=config.get('seed', 42)
        )
        cf.fit(y_all, t_all, X=X_all)
    return cf.effect(X_test).flatten()


def train_dr_learner(X_train, t_train, y_train, X_val, t_val, y_val,
                     X_test, t_test, config):
    """CausalML DR-Learner with LightGBM."""
    from causalml.inference.meta import BaseDRLearner
    from lightgbm import LGBMRegressor

    X_all = np.vstack([X_train, X_val])
    t_all = np.concatenate([t_train, t_val])
    y_all = np.concatenate([y_train, y_val])

    # Propensity
    prop = estimate_propensity(X_all, t_all)
    p = np.clip(prop.predict_proba(X_all)[:, 1], 0.05, 0.95)

    dr = BaseDRLearner(
        learner=LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                              verbosity=-1, n_jobs=1),
        control_name=0
    )
    dr.fit(X=X_all, treatment=t_all, y=y_all, p=p)
    return dr.predict(X_test).flatten()


def train_x_learner(X_train, t_train, y_train, X_val, t_val, y_val,
                    X_test, t_test, config):
    """CausalML X-Learner with XGBoost."""
    from causalml.inference.meta import BaseXLearner
    from xgboost import XGBRegressor

    X_all = np.vstack([X_train, X_val])
    t_all = np.concatenate([t_train, t_val])
    y_all = np.concatenate([y_train, y_val])

    # Propensity
    prop = estimate_propensity(X_all, t_all)
    p_train = np.clip(prop.predict_proba(X_all)[:, 1], 0.05, 0.95)
    p_test = np.clip(prop.predict_proba(X_test)[:, 1], 0.05, 0.95)

    xl = BaseXLearner(
        learner=XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                             verbosity=0, n_jobs=1, tree_method='hist'),
        control_name=0
    )
    xl.fit(X=X_all, treatment=t_all, y=y_all, p=p_train)
    return xl.predict(X_test, treatment=None, y=None, p=p_test).flatten()


def train_r_learner(X_train, t_train, y_train, X_val, t_val, y_val,
                    X_test, t_test, config):
    """CausalML R-Learner with LightGBM."""
    from causalml.inference.meta import BaseRLearner
    from lightgbm import LGBMRegressor

    X_all = np.vstack([X_train, X_val])
    t_all = np.concatenate([t_train, t_val])
    y_all = np.concatenate([y_train, y_val])

    # Propensity
    prop = estimate_propensity(X_all, t_all)
    p = np.clip(prop.predict_proba(X_all)[:, 1], 0.05, 0.95)

    rl = BaseRLearner(
        learner=LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                              verbosity=-1, n_jobs=1),
        outcome_learner=LGBMRegressor(n_estimators=200, max_depth=6,
                                      learning_rate=0.1, verbosity=-1, n_jobs=1),
        effect_learner=LGBMRegressor(n_estimators=200, max_depth=6,
                                     learning_rate=0.1, verbosity=-1, n_jobs=1),
        control_name=0
    )
    rl.fit(X=X_all, treatment=t_all, y=y_all, p=p)
    return rl.predict(X_test).flatten()


def train_uplift_rf(X_train, t_train, y_train, X_val, t_val, y_val,
                    X_test, t_test, config):
    """CausalML Uplift Random Forest."""
    from causalml.inference.tree import UpliftRandomForestClassifier

    X_all = np.vstack([X_train, X_val])
    t_all = np.concatenate([t_train, t_val])
    y_all = np.concatenate([y_train, y_val])

    # UpliftRF needs string treatment labels and binary outcome
    t_str = np.where(t_all == 1, 'treatment', 'control')

    # Discretize outcome for UpliftRF (it's a classifier)
    y_median = np.median(y_all)
    y_binary = (y_all > y_median).astype(int)

    urf = UpliftRandomForestClassifier(
        n_estimators=config.get('n_estimators', 300),
        max_depth=config.get('max_depth', 10),
        min_samples_leaf=config.get('min_samples_leaf', 20),
        evaluationFunction='KL',
        control_name='control',
        random_state=config.get('seed', 42),
        n_jobs=1
    )
    urf.fit(X_all, treatment=t_str, y=y_binary)

    pred = urf.predict(X_test)
    ite_pred = pred.flatten()

    # Scale: UpliftRF predicts P(Y=1|T=1)-P(Y=1|T=0), rescale to outcome units
    y_std = np.std(y_all)
    if y_std > 1e-8:
        ite_pred = ite_pred * y_std * 4  # heuristic: prob_diff → outcome scale
    return ite_pred


def train_bart(X_train, t_train, y_train, X_val, t_val, y_val,
               X_test, t_test, config):
    """BART-like T-learner with GBM (many shallow trees + shrinkage)."""
    from sklearn.ensemble import GradientBoostingRegressor

    X_all = np.vstack([X_train, X_val])
    t_all = np.concatenate([t_train, t_val])
    y_all = np.concatenate([y_train, y_val])

    bart_params = {
        'n_estimators': config.get('n_estimators', 200),
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'min_samples_leaf': 5,
        'random_state': config.get('seed', 42)
    }

    treated_mask = t_all == 1
    control_mask = t_all == 0

    model_1 = GradientBoostingRegressor(**bart_params)
    model_0 = GradientBoostingRegressor(**bart_params)

    model_1.fit(X_all[treated_mask], y_all[treated_mask])
    model_0.fit(X_all[control_mask], y_all[control_mask])

    return model_1.predict(X_test) - model_0.predict(X_test)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

GT_DATASETS = [
    'ihdp', 'twins', 'acic2016', 'news', 'tcga', 'acic2018',
    'lbidd', 'nlsm', 'ibm_causal', 'continuous', 'acic2022', 'star'
]

NO_GT_DATASETS = ['jobs', 'hillstrom', 'criteo']

ALL_DATASETS = GT_DATASETS + NO_GT_DATASETS

MODELS = {
    'CausalForestDML': train_causal_forest_dml,
    'DR-Learner': train_dr_learner,
    'X-Learner': train_x_learner,
    'R-Learner': train_r_learner,
    'UpliftRF': train_uplift_rf,
    'BART': train_bart,
}

SEEDS = [42, 43, 44]


# ==============================================================================
# RUNNER
# ==============================================================================

def run_single(dataset_name, model_name, model_fn, seed, config):
    """Run a single model on a single dataset with a single seed."""
    config['seed'] = seed

    loader = get_dataset_loader(dataset_name)
    data = loader(seed=seed)

    X_train = np.nan_to_num(data['X_train'].astype(np.float64), nan=0.0)
    t_train = data['t_train'].astype(np.float64)
    y_train = np.nan_to_num(data['y_train'].astype(np.float64), nan=0.0)
    X_val = np.nan_to_num(data['X_val'].astype(np.float64), nan=0.0)
    t_val = data['t_val'].astype(np.float64)
    y_val = np.nan_to_num(data['y_val'].astype(np.float64), nan=0.0)
    X_test = np.nan_to_num(data['X_test'].astype(np.float64), nan=0.0)
    t_test = data['t_test'].astype(np.float64)
    y_test = np.nan_to_num(data['y_test'].astype(np.float64), nan=0.0)
    mu0_test = data.get('mu0_test', None)
    mu1_test = data.get('mu1_test', None)

    # For continuous treatment, binarize at median
    if dataset_name == 'continuous':
        t_median = np.median(np.concatenate([t_train, t_val, t_test]))
        t_train = (t_train > t_median).astype(np.float64)
        t_val = (t_val > t_median).astype(np.float64)
        t_test_bin = (t_test > t_median).astype(np.float64)
    else:
        t_test_bin = t_test

    start = time.time()
    try:
        ite_pred = model_fn(X_train, t_train, y_train, X_val, t_val, y_val,
                            X_test, t_test_bin, config)
    except Exception as e:
        print(f"\n    ERROR: {model_name} on {dataset_name} seed={seed}: {e}")
        return None
    elapsed = time.time() - start

    # Handle NaN predictions
    if np.any(np.isnan(ite_pred)):
        ite_pred = np.nan_to_num(ite_pred, nan=0.0)

    # Compute metrics
    has_gt = mu0_test is not None and mu1_test is not None
    if has_gt:
        metrics = compute_metrics_gt(ite_pred, mu0_test, mu1_test)
    else:
        metrics = compute_metrics_no_gt(ite_pred, t_test, y_test)

    metrics['time'] = round(elapsed, 1)
    return metrics


def run_dataset_model(dataset_name, model_name, model_fn, seeds):
    """Run model across seeds, return aggregated results."""
    # Determine dataset scale for config adjustments
    loader = get_dataset_loader(dataset_name)
    test_data = loader(seed=42)
    n_samples = len(test_data['X_train']) + len(test_data['X_val']) + len(test_data['X_test'])
    n_features = test_data['X_train'].shape[1]

    config = {}
    if n_samples > 30000 or n_features > 500:
        config['n_estimators'] = 100
        config['max_depth'] = 8
        config['min_samples_leaf'] = 30

    all_metrics = []
    for seed in seeds:
        m = run_single(dataset_name, model_name, model_fn, seed, config.copy())
        if m is not None:
            all_metrics.append(m)

    if not all_metrics:
        return None

    # Aggregate mean ± std
    agg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if m.get(key) is not None and not np.isnan(m.get(key, float('nan')))]
        if vals:
            agg[f'{key}_mean'] = round(np.mean(vals), 2)
            if len(vals) > 1:
                agg[f'{key}_std'] = round(np.std(vals), 2)
    return agg


def main():
    print("=" * 70)
    print("CAUSAL INFERENCE BENCHMARK — 6 NEW MODELS")
    print("=" * 70)
    print(f"Models: {list(MODELS.keys())}")
    print(f"GT datasets ({len(GT_DATASETS)}): {GT_DATASETS}")
    print(f"No-GT datasets ({len(NO_GT_DATASETS)}): {NO_GT_DATASETS}")
    print(f"Seeds: {SEEDS}")
    print()

    results = {}

    for ds_idx, dataset_name in enumerate(ALL_DATASETS, 1):
        print(f"\n{'='*70}")
        print(f"[{ds_idx}/{len(ALL_DATASETS)}] Dataset: {dataset_name}")
        print(f"{'='*70}")

        results[dataset_name] = {}

        for model_name, model_fn in MODELS.items():
            print(f"  {model_name}...", end=" ", flush=True)
            agg = run_dataset_model(dataset_name, model_name, model_fn, SEEDS)

            if agg is None:
                print("FAILED")
                results[dataset_name][model_name] = {'error': True}
            else:
                if 'pehe_mean' in agg:
                    print(f"PEHE={agg['pehe_mean']:.2f}  corr={agg.get('ite_corr_mean', 0):.2f}  ({agg.get('time_mean', '?')}s)")
                elif 'auuc_mean' in agg:
                    print(f"AUUC={agg['auuc_mean']:.4f}  Qini={agg.get('qini_mean', 0):.4f}  ({agg.get('time_mean', '?')}s)")
                results[dataset_name][model_name] = agg

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results', 'new_models_results.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")

    # Print summary
    print_summary(results)


def print_summary(results):
    """Print formatted results tables."""
    model_names = list(MODELS.keys())

    print("\n\n" + "=" * 120)
    print("√PEHE (lower is better)")
    print("=" * 120)
    header = f"{'Dataset':<14}" + "".join(f"{m:<17}" for m in model_names)
    print(header)
    print("-" * 120)
    for ds in GT_DATASETS:
        if ds not in results:
            continue
        row = f"{ds:<14}"
        for mn in model_names:
            if mn in results[ds] and 'pehe_mean' in results[ds][mn]:
                val = results[ds][mn]['pehe_mean']
                row += f"{val:<17.2f}"
            else:
                row += f"{'ERR':<17}"
        print(row)

    print("\n\n" + "=" * 120)
    print("ITE Correlation (higher is better)")
    print("=" * 120)
    print(header)
    print("-" * 120)
    for ds in GT_DATASETS:
        if ds not in results:
            continue
        row = f"{ds:<14}"
        for mn in model_names:
            if mn in results[ds] and 'ite_corr_mean' in results[ds][mn]:
                val = results[ds][mn]['ite_corr_mean']
                row += f"{val:<17.2f}"
            else:
                row += f"{'ERR':<17}"
        print(row)

    print("\n\n" + "=" * 120)
    print("No-GT Datasets: AUUC & Qini (higher is better)")
    print("=" * 120)
    print(header)
    print("-" * 120)
    for ds in NO_GT_DATASETS:
        if ds not in results:
            continue
        row = f"{ds+' AUUC':<14}"
        for mn in model_names:
            if mn in results[ds] and 'auuc_mean' in results[ds][mn]:
                row += f"{results[ds][mn]['auuc_mean']:<17.4f}"
            else:
                row += f"{'ERR':<17}"
        print(row)
        row = f"{ds+' Qini':<14}"
        for mn in model_names:
            if mn in results[ds] and 'qini_mean' in results[ds][mn]:
                row += f"{results[ds][mn]['qini_mean']:<17.4f}"
            else:
                row += f"{'ERR':<17}"
        print(row)

    print("\n\nDone!")


if __name__ == '__main__':
    main()
