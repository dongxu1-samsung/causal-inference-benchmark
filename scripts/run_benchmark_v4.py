"""
Benchmark runner v4: Run 3 new neural models (DESCN, MOCA, DDRNet) on all 15 datasets.
- 12 GT datasets: report PEHE and ITE Correlation
- 3 no-GT datasets: report AUUC and Qini
- Merge results with existing v3 results (8 neural + 6 ML models)
"""

import sys
import os
import time
import json
import numpy as np
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_dataset_loader, DATASET_INFO
from models.descn import train_descn, predict_descn
from models.moca import train_moca, predict_moca
from models.ddrnet import train_ddrnet, predict_ddrnet


# ==============================================================================
# Metrics (reuse from v3)
# ==============================================================================
def compute_gt_metrics(ite_pred, mu0_test, mu1_test):
    ite_true = mu1_test - mu0_test
    pehe = float(np.sqrt(np.mean((ite_pred - ite_true) ** 2)))
    if np.std(ite_true) > 1e-8 and np.std(ite_pred) > 1e-8:
        ite_corr = float(np.corrcoef(ite_pred, ite_true)[0, 1])
    else:
        ite_corr = 0.0
    return {'pehe': pehe, 'ite_corr': ite_corr}


def compute_auuc(ite_pred, t_test, y_test, n_bins=20):
    n = len(ite_pred)
    order = np.argsort(-ite_pred)
    t_sorted = t_test[order]
    y_sorted = y_test[order]
    fractions = np.linspace(0.05, 1.0, n_bins)
    uplift_curve = []
    for frac in fractions:
        k = max(1, int(frac * n))
        t_topk, y_topk = t_sorted[:k], y_sorted[:k]
        treated_mask, control_mask = t_topk == 1, t_topk == 0
        if treated_mask.sum() > 5 and control_mask.sum() > 5:
            uplift = y_topk[treated_mask].mean() - y_topk[control_mask].mean()
        else:
            uplift = 0.0
        uplift_curve.append(uplift * frac)
    auuc = float(np.trapz(uplift_curve, fractions))
    t1m, t0m = t_test == 1, t_test == 0
    overall_ate = (y_test[t1m].mean() - y_test[t0m].mean()) if (t1m.sum() > 0 and t0m.sum() > 0) else 0.0
    random_auuc = float(np.trapz([overall_ate * f for f in fractions], fractions))
    return {'auuc': round(auuc, 4), 'auuc_norm': round(auuc - random_auuc, 4)}


def compute_qini(ite_pred, t_test, y_test, n_bins=20):
    n = len(ite_pred)
    order = np.argsort(-ite_pred)
    t_sorted, y_sorted = t_test[order], y_test[order]
    cum_t1 = np.cumsum(t_sorted)
    cum_t0 = np.cumsum(1 - t_sorted)
    cum_y1 = np.cumsum(t_sorted * y_sorted)
    cum_y0 = np.cumsum((1 - t_sorted) * y_sorted)
    fractions = np.linspace(0.05, 1.0, n_bins)
    qini_values = []
    for frac in fractions:
        k = min(max(1, int(frac * n)) - 1, n - 1)
        n_t, n_c = cum_t1[k], cum_t0[k]
        qini_val = (cum_y1[k] / n_t - cum_y0[k] / n_c) if (n_t > 5 and n_c > 5) else 0.0
        qini_values.append(qini_val * frac)
    qini_area = float(np.trapz(qini_values, fractions))
    n_t_all, n_c_all = cum_t1[-1], cum_t0[-1]
    overall_uplift = (cum_y1[-1] / n_t_all - cum_y0[-1] / n_c_all) if (n_t_all > 0 and n_c_all > 0) else 0.0
    return {'qini': round(qini_area, 4), 'qini_norm': round(qini_area - overall_uplift * 0.5, 4)}


# ==============================================================================
# Model registry
# ==============================================================================
NEW_MODELS = {
    'descn': (train_descn, predict_descn),
    'moca': (train_moca, predict_moca),
    'ddrnet': (train_ddrnet, predict_ddrnet),
}

# ==============================================================================
# Dataset configurations
# ==============================================================================
DATASET_CONFIGS = {
    'ihdp': [{'seed': s} for s in [42, 43, 44]],
    'twins': [{'seed': s} for s in [42, 43, 44]],
    'acic2016': [{'dgp': d, 'seed': 42} for d in [1, 2, 3]],
    'news': [{'seed': s} for s in [42, 43, 44]],
    'tcga': [{'seed': s} for s in [42, 43, 44]],
    'jobs': [{'seed': s} for s in [42, 43, 44]],
    'acic2018': [{'dgp': d, 'seed': 42} for d in [1, 2, 3, 4, 5, 6]],
    'hillstrom': [{'seed': s} for s in [42, 43, 44]],
    'lbidd': [{'setting': s, 'seed': 42} for s in [1, 2, 3]],
    'criteo': [{'seed': s} for s in [42, 43, 44]],
    'nlsm': [{'difficulty': d, 'seed': 42} for d in [1, 2, 3]],
    'ibm_causal': [{'confounding_strength': c, 'seed': 42} for c in [1, 2, 3]],
    'continuous': [{'seed': s} for s in [42, 43, 44]],
    'acic2022': [{'seed': s} for s in [42, 43, 44]],
    'star': [{'seed': s} for s in [42, 43, 44]],
}

GT_DATASETS = ['ihdp', 'twins', 'acic2016', 'news', 'tcga', 'acic2018',
               'lbidd', 'nlsm', 'ibm_causal', 'continuous', 'acic2022', 'star']
NOGT_DATASETS = ['jobs', 'hillstrom', 'criteo']

# Reduced epochs for CPU training
EPOCH_CONFIG = {
    'default': {'n_epochs': 100, 'patience': 15},
    'large': {'n_epochs': 50, 'patience': 10, 'batch_size': 512},
}
LARGE_DATASETS = {'tcga', 'news', 'lbidd', 'criteo', 'acic2022', 'star'}


# ==============================================================================
# Main benchmark
# ==============================================================================
def run_benchmark():
    total_start = time.time()
    all_results = {}
    ds_order = GT_DATASETS + NOGT_DATASETS

    # Count total runs
    total_runs = sum(len(DATASET_CONFIGS[ds]) * len(NEW_MODELS) for ds in ds_order)
    run_count = 0

    for ds_name in ds_order:
        is_gt = ds_name in GT_DATASETS
        configs = DATASET_CONFIGS[ds_name]
        epoch_cfg = EPOCH_CONFIG['large'] if ds_name in LARGE_DATASETS else EPOCH_CONFIG['default']

        print(f"\nDataset: {ds_name} ({'GT' if is_gt else 'No-GT'}) — {len(configs)} configs")
        ds_results = {}

        for model_name in NEW_MODELS:
            train_fn, predict_fn = NEW_MODELS[model_name]
            config_metrics = []

            for ci, cfg in enumerate(configs):
                run_count += 1

                try:
                    loader = get_dataset_loader(ds_name)
                    data = loader(**cfg)
                    X_train = data['X_train']
                    t_train = data['t_train']
                    y_train = data['y_train']
                    X_val = data['X_val']
                    t_val = data['t_val']
                    y_val = data['y_val']
                    X_test = data['X_test']
                    t_test = data['t_test']
                    y_test = data['y_test']
                    mu0_test = data.get('mu0_test')
                    mu1_test = data.get('mu1_test')

                    # Check both classes exist
                    if len(np.unique(t_train)) < 2:
                        raise ValueError("Single treatment class in training data")

                    # Normalize
                    y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
                    y_train_n = (y_train - y_mean) / y_std
                    y_val_n = (y_val - y_mean) / y_std

                    x_mean, x_std = X_train.mean(0), X_train.std(0) + 1e-8
                    X_train_n = (X_train - x_mean) / x_std
                    X_val_n = (X_val - x_mean) / x_std
                    X_test_n = (X_test - x_mean) / x_std

                    input_dim = X_train_n.shape[1]

                    # Build config
                    model_cfg = {
                        '_X_val': X_val_n,
                        '_t_val': t_val,
                        '_y_val': y_val_n,
                    }
                    model_cfg.update(epoch_cfg)

                    # Special config for TransDCA-like large datasets
                    if ds_name in LARGE_DATASETS:
                        model_cfg['lr'] = 2e-3

                    t0 = time.time()
                    model = train_fn(X_train_n, t_train, y_train_n, input_dim, model_cfg)
                    ite_pred = predict_fn(model, X_test_n) * y_std
                    ite_pred = np.nan_to_num(ite_pred, nan=0, posinf=0, neginf=0)
                    elapsed = time.time() - t0

                    if is_gt and mu0_test is not None and mu1_test is not None:
                        metrics = compute_gt_metrics(ite_pred, mu0_test, mu1_test)
                        print(f"    {model_name:<14} PEHE={metrics['pehe']:.4f}  "
                              f"ITE_corr={metrics['ite_corr']:.4f}  ({elapsed:.1f}s)  [{run_count}/{total_runs}]")
                    else:
                        auuc_m = compute_auuc(ite_pred, t_test, y_test)
                        qini_m = compute_qini(ite_pred, t_test, y_test)
                        metrics = {**auuc_m, **qini_m}
                        print(f"    {model_name:<14} AUUC={auuc_m['auuc']:.4f}  "
                              f"AUUC_n={auuc_m['auuc_norm']:.4f}  Qini={qini_m['qini']:.4f}  "
                              f"Qini_n={qini_m['qini_norm']:.4f}  ({elapsed:.1f}s)  [{run_count}/{total_runs}]")

                    config_metrics.append(metrics)

                except Exception as e:
                    print(f"    {model_name:<14} ERROR: {str(e)[:80]}  [{run_count}/{total_runs}]")
                    run_count_skipped = True

            # Aggregate across configs
            if config_metrics:
                agg = {}
                for key in config_metrics[0]:
                    vals = [m[key] for m in config_metrics if key in m]
                    agg[f'{key}_mean'] = round(float(np.mean(vals)), 4)
                    agg[f'{key}_std'] = round(float(np.std(vals)), 4)
                ds_results[model_name] = agg

        all_results[ds_name] = ds_results
        print(f"  --- {ds_name} Summary ---")
        for mn, r in ds_results.items():
            if 'pehe_mean' in r:
                print(f"    {mn:<14} avg_PEHE={r['pehe_mean']:.2f}±{r['pehe_std']:.2f}  "
                      f"avg_corr={r['ite_corr_mean']:.2f}±{r['ite_corr_std']:.2f}")
            elif 'auuc_norm_mean' in r:
                print(f"    {mn:<14} avg_AUUC_n={r['auuc_norm_mean']:.4f}  "
                      f"avg_Qini_n={r['qini_norm_mean']:.4f}")

    # Save new model results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'v4_new_models_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved v4 new model results to results/v4_new_models_results.json")

    # Merge with existing v3 results
    v3_path = os.path.join(results_dir, 'all_results_v3.json')
    if os.path.exists(v3_path):
        with open(v3_path) as f:
            merged = json.load(f)
        for ds_name, models in all_results.items():
            if ds_name not in merged:
                merged[ds_name] = {}
            merged[ds_name].update(models)
        with open(os.path.join(results_dir, 'all_results_v4.json'), 'w') as f:
            json.dump(merged, f, indent=2)
        print(f"Saved merged results (17 models) to results/all_results_v4.json")

    elapsed_total = (time.time() - total_start) / 60
    print(f"\nBenchmark v4 complete in {elapsed_total:.1f} minutes")


if __name__ == '__main__':
    run_benchmark()
