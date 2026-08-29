"""
Benchmark runner v3: Run 6 new ML-based models on all 15 datasets.
- 12 GT datasets: report PEHE and ITE Correlation
- 3 no-GT datasets: report AUUC and Qini
- Merge results with existing v2 results (8 neural models)

Only runs the 6 new models; does NOT rerun the existing 8 neural models.
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
from models.ml_models import train_ml_model, predict_ml_model, MODEL_REGISTRY


# ==============================================================================
# Metrics for GT datasets
# ==============================================================================
def compute_gt_metrics(ite_pred, mu0_test, mu1_test):
    """Compute PEHE and ITE Correlation for datasets with ground truth."""
    ite_true = mu1_test - mu0_test
    pehe = float(np.sqrt(np.mean((ite_pred - ite_true) ** 2)))

    if np.std(ite_true) > 1e-8 and np.std(ite_pred) > 1e-8:
        ite_corr = float(np.corrcoef(ite_pred, ite_true)[0, 1])
    else:
        ite_corr = 0.0

    return {'pehe': pehe, 'ite_corr': ite_corr}


# ==============================================================================
# AUUC and Qini for no-GT datasets
# ==============================================================================
def compute_auuc(ite_pred, t_test, y_test, n_bins=20):
    """
    AUUC: Area Under the Uplift Curve.
    Sort by predicted ITE descending, compute cumulative uplift at each fraction.
    Returns normalized AUUC (excess over random baseline).
    """
    n = len(ite_pred)
    order = np.argsort(-ite_pred)
    t_sorted = t_test[order]
    y_sorted = y_test[order]

    fractions = np.linspace(0.05, 1.0, n_bins)
    uplift_curve = []

    for frac in fractions:
        k = max(1, int(frac * n))
        t_topk = t_sorted[:k]
        y_topk = y_sorted[:k]

        treated_mask = t_topk == 1
        control_mask = t_topk == 0

        if treated_mask.sum() > 5 and control_mask.sum() > 5:
            uplift = y_topk[treated_mask].mean() - y_topk[control_mask].mean()
        else:
            uplift = 0.0

        uplift_curve.append(uplift * frac)

    auuc = float(np.trapz(uplift_curve, fractions))

    # Random baseline
    t1_mask = t_test == 1
    t0_mask = t_test == 0
    if t1_mask.sum() > 0 and t0_mask.sum() > 0:
        overall_ate = y_test[t1_mask].mean() - y_test[t0_mask].mean()
    else:
        overall_ate = 0.0
    random_auuc = float(np.trapz([overall_ate * f for f in fractions], fractions))

    auuc_normalized = auuc - random_auuc
    return {'auuc': round(auuc, 4), 'auuc_norm': round(auuc_normalized, 4)}


def compute_qini(ite_pred, t_test, y_test, n_bins=20):
    """
    Qini coefficient: area between model's Qini curve and random diagonal.
    """
    n = len(ite_pred)
    order = np.argsort(-ite_pred)
    t_sorted = t_test[order]
    y_sorted = y_test[order]

    cum_t1 = np.cumsum(t_sorted)
    cum_t0 = np.cumsum(1 - t_sorted)
    cum_y1 = np.cumsum(t_sorted * y_sorted)
    cum_y0 = np.cumsum((1 - t_sorted) * y_sorted)

    fractions = np.linspace(0.05, 1.0, n_bins)
    qini_values = []

    for frac in fractions:
        k = max(1, int(frac * n)) - 1  # 0-indexed
        k = min(k, n - 1)
        n_t = cum_t1[k]
        n_c = cum_t0[k]
        if n_t > 5 and n_c > 5:
            qini_val = cum_y1[k] / n_t - cum_y0[k] / n_c
        else:
            qini_val = 0.0
        qini_values.append(qini_val * frac)

    qini_area = float(np.trapz(qini_values, fractions))

    # Random baseline: the value at k=N is overall ATE
    n_t_all = cum_t1[-1]
    n_c_all = cum_t0[-1]
    if n_t_all > 0 and n_c_all > 0:
        overall_uplift = cum_y1[-1] / n_t_all - cum_y0[-1] / n_c_all
    else:
        overall_uplift = 0.0
    random_area = overall_uplift * 0.5  # triangle

    qini_coeff = qini_area - random_area
    return {'qini': round(qini_area, 4), 'qini_norm': round(qini_coeff, 4)}


# ==============================================================================
# Dataset configurations (which settings/DGPs to iterate over)
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

NEW_MODELS = list(MODEL_REGISTRY.keys())

GT_DATASETS = [ds for ds, info in DATASET_INFO.items() if info['has_ground_truth']]
NO_GT_DATASETS = [ds for ds, info in DATASET_INFO.items() if not info['has_ground_truth']]


# ==============================================================================
# Main benchmark
# ==============================================================================
def run_benchmark():
    all_results = {}
    all_datasets = list(DATASET_CONFIGS.keys())
    total_combos = sum(len(DATASET_CONFIGS[ds]) * len(NEW_MODELS) for ds in all_datasets)
    done = 0

    print(f"=" * 70)
    print(f"Benchmark v3: {len(NEW_MODELS)} ML models × {len(all_datasets)} datasets")
    print(f"Models: {', '.join(NEW_MODELS)}")
    print(f"GT datasets ({len(GT_DATASETS)}): {', '.join(GT_DATASETS)}")
    print(f"No-GT datasets ({len(NO_GT_DATASETS)}): {', '.join(NO_GT_DATASETS)}")
    print(f"Total model-dataset-config runs: {total_combos}")
    print(f"=" * 70)

    for ds_name in all_datasets:
        configs = DATASET_CONFIGS[ds_name]
        has_gt = DATASET_INFO[ds_name]['has_ground_truth']
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({'GT' if has_gt else 'No-GT'}) — {len(configs)} configs")
        print(f"{'='*60}")

        # Collect per-model results across configs
        model_results = {mn: [] for mn in NEW_MODELS}

        for ci, cfg in enumerate(configs):
            cfg_str = ', '.join(f'{k}={v}' for k, v in cfg.items())
            print(f"\n  Config {ci+1}/{len(configs)}: {cfg_str}")

            # Load data
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
                mu0_test = data.get('mu0_test', None)
                mu1_test = data.get('mu1_test', None)
            except Exception as e:
                print(f"    ERROR loading data: {e}")
                for mn in NEW_MODELS:
                    model_results[mn].append(None)
                    done += 1
                continue

            n_train = len(X_train) + len(X_val)
            n_test = len(X_test)
            n_feat = X_train.shape[1]
            print(f"    N_train+val={n_train}, N_test={n_test}, D={n_feat}")

            for mn in NEW_MODELS:
                done += 1
                try:
                    t0 = time.time()

                    # For tree-based models, no need to normalize (they're scale-invariant)
                    # But some causalml internals may benefit from finite values
                    # Clip extreme values to prevent overflow
                    X_tr_c = np.clip(X_train, -1e6, 1e6)
                    X_va_c = np.clip(X_val, -1e6, 1e6)
                    X_te_c = np.clip(X_test, -1e6, 1e6)
                    y_tr_c = np.clip(y_train, -1e6, 1e6)
                    y_va_c = np.clip(y_val, -1e6, 1e6)

                    # Handle NaN/Inf
                    X_tr_c = np.nan_to_num(X_tr_c, nan=0.0, posinf=1e6, neginf=-1e6)
                    X_va_c = np.nan_to_num(X_va_c, nan=0.0, posinf=1e6, neginf=-1e6)
                    X_te_c = np.nan_to_num(X_te_c, nan=0.0, posinf=1e6, neginf=-1e6)

                    model_dict = train_ml_model(mn, X_tr_c, t_train, y_tr_c,
                                                X_va_c, t_val, y_va_c)
                    ite_pred = predict_ml_model(mn, model_dict, X_te_c)
                    ite_pred = np.nan_to_num(ite_pred, nan=0.0, posinf=0.0, neginf=0.0)

                    elapsed = time.time() - t0

                    # Compute metrics
                    if has_gt and mu0_test is not None and mu1_test is not None:
                        metrics = compute_gt_metrics(ite_pred, mu0_test, mu1_test)
                    else:
                        metrics = compute_auuc(ite_pred, t_test, y_test)
                        metrics.update(compute_qini(ite_pred, t_test, y_test))

                    metrics['time'] = elapsed
                    model_results[mn].append(metrics)

                    # Print result
                    if has_gt:
                        print(f"    {mn:<22} PEHE={metrics['pehe']:.4f}  "
                              f"ITE_corr={metrics['ite_corr']:.4f}  "
                              f"({elapsed:.1f}s)  [{done}/{total_combos}]")
                    else:
                        print(f"    {mn:<22} AUUC={metrics['auuc']:.4f}  "
                              f"AUUC_n={metrics['auuc_norm']:.4f}  "
                              f"Qini={metrics['qini']:.4f}  "
                              f"Qini_n={metrics['qini_norm']:.4f}  "
                              f"({elapsed:.1f}s)  [{done}/{total_combos}]")

                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"    {mn:<22} ERROR: {str(e)[:100]}  ({elapsed:.1f}s)")
                    model_results[mn].append(None)

        # Aggregate results per model
        print(f"\n  --- {ds_name} Summary ---")
        if ds_name not in all_results:
            all_results[ds_name] = {}

        for mn in NEW_MODELS:
            valid = [r for r in model_results[mn] if r is not None]
            if not valid:
                print(f"    {mn:<22} ALL FAILED")
                continue

            agg = {'n_valid': len(valid), 'n_total': len(configs)}

            if has_gt:
                pehe_vals = [r['pehe'] for r in valid]
                corr_vals = [r['ite_corr'] for r in valid]
                agg['pehe_mean'] = float(np.mean(pehe_vals))
                agg['pehe_std'] = float(np.std(pehe_vals))
                agg['ite_corr_mean'] = float(np.mean(corr_vals))
                agg['ite_corr_std'] = float(np.std(corr_vals))
                print(f"    {mn:<22} PEHE={agg['pehe_mean']:.2f}±{agg['pehe_std']:.2f}  "
                      f"ITE_corr={agg['ite_corr_mean']:.2f}±{agg['ite_corr_std']:.2f}")
            else:
                for metric in ['auuc', 'auuc_norm', 'qini', 'qini_norm']:
                    vals = [r[metric] for r in valid]
                    agg[f'{metric}_mean'] = float(np.mean(vals))
                    agg[f'{metric}_std'] = float(np.std(vals))
                print(f"    {mn:<22} AUUC_n={agg['auuc_norm_mean']:.4f}±{agg['auuc_norm_std']:.4f}  "
                      f"Qini_n={agg['qini_norm_mean']:.4f}±{agg['qini_norm_std']:.4f}")

            time_vals = [r['time'] for r in valid]
            agg['time_mean'] = float(np.mean(time_vals))
            agg['time_std'] = float(np.std(time_vals))
            all_results[ds_name][mn] = agg

    return all_results


def main():
    print("Starting benchmark v3 (6 new ML models)...")
    t_start = time.time()

    results = run_benchmark()

    # Save new model results
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'new_models_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Merge with existing results
    existing_path = os.path.join(os.path.dirname(out_path), 'all_results.json')
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        for ds_name, models in results.items():
            if ds_name not in existing:
                existing[ds_name] = {}
            existing[ds_name].update(models)
        merged_path = os.path.join(os.path.dirname(out_path), 'all_results_v3.json')
        with open(merged_path, 'w') as f:
            json.dump(existing, f, indent=2)
        print(f"\nMerged results saved to {merged_path}")
    else:
        print(f"\nNo existing results to merge. New results at {out_path}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Benchmark v3 complete in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
