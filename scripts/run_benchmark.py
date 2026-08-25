"""
Enhanced benchmark runner with:
- Proper train/val/test splits (early stopping on val for all models)
- Metrics: sqrt_PEHE, epsilon_ATE, epsilon_ATT, ITE_Correlation, Policy_Risk
- Support for 10 datasets
"""

import sys
import os
import time
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_dataset_loader, DATASET_INFO
from models.cfrnet import train_cfrnet, predict_cfrnet
from models.ganite import train_ganite, predict_ganite
from models.cevae import train_cevae, predict_cevae
from models.drnet import train_drnet, predict_drnet
from models.catenets import train_catenet, predict_catenet


# ==============================================================================
# Metrics
# ==============================================================================
def compute_metrics(ite_pred, t_test, y_test, mu0_test=None, mu1_test=None):
    """
    Compute all metrics. For datasets with ground truth (mu0, mu1),
    compute PEHE, ATE error, ATT error, ITE correlation.
    For datasets without ground truth, compute policy risk.
    """
    results = {}

    if mu0_test is not None and mu1_test is not None:
        ite_true = mu1_test - mu0_test
        ate_true = ite_true.mean()
        ate_pred = ite_pred.mean()

        # sqrt PEHE
        results['pehe'] = float(np.sqrt(np.mean((ite_pred - ite_true) ** 2)))

        # epsilon ATE
        results['ate_error'] = float(np.abs(ate_pred - ate_true))

        # epsilon ATT (Average Treatment Effect on the Treated)
        treated_mask = t_test == 1
        if treated_mask.sum() > 0:
            att_true = ite_true[treated_mask].mean()
            att_pred = ite_pred[treated_mask].mean()
            results['att_error'] = float(np.abs(att_pred - att_true))
        else:
            results['att_error'] = float('nan')

        # ITE Correlation
        if np.std(ite_true) > 1e-8 and np.std(ite_pred) > 1e-8:
            results['ite_corr'] = float(np.corrcoef(ite_pred, ite_true)[0, 1])
        else:
            results['ite_corr'] = 0.0

    # Policy Risk (works for all datasets, including those without ground truth)
    # R_pol = 1 - E[Y * 1(T = pi(X))] / E[Y] (simplified version using IPW)
    # For semi-synthetic: R_pol = E[Y(1-pi(x))] where pi is model's recommended treatment
    if mu0_test is not None and mu1_test is not None:
        # Optimal policy: treat if ITE > 0
        policy_pred = (ite_pred > 0).astype(float)
        # Expected outcome under predicted policy
        y_policy = policy_pred * mu1_test + (1 - policy_pred) * mu0_test
        # Expected outcome under oracle policy
        policy_oracle = (ite_true > 0).astype(float)
        y_oracle = policy_oracle * mu1_test + (1 - policy_oracle) * mu0_test
        # Policy risk = normalized regret
        if y_oracle.mean() != 0:
            results['policy_risk'] = float(1 - y_policy.mean() / y_oracle.mean())
        else:
            results['policy_risk'] = float(np.abs(y_policy.mean() - y_oracle.mean()))
    else:
        # For real-world datasets: use observed outcome under predicted policy
        # This is a proxy — true policy risk requires counterfactuals
        policy_pred = (ite_pred > 0).astype(float)
        # Agreement rate with actual treatment
        agreement = np.mean(policy_pred == t_test)
        results['policy_agreement'] = float(agreement)

    return results


# ==============================================================================
# Model training with validation-based early stopping
# ==============================================================================
def train_and_predict(model_name, X_train, t_train, y_train, X_val, t_val, y_val, X_test, config):
    """
    Train model with early stopping on validation set, predict ITE on test set.
    Returns: ite_pred (numpy array), training_time (float)
    """
    input_dim = X_train.shape[1]

    # Normalize features
    mean = X_train.mean(0)
    std = X_train.std(0) + 1e-8
    X_tr_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    X_te_n = (X_test - mean) / std

    # Normalize outcome
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_tr_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    t0 = time.time()

    try:
        if model_name == 'cfrnet':
            # CFRNet: pass val data for early stopping
            cfg = {**config, '_X_val': X_val_n, '_t_val': t_val, '_y_val': y_val_n}
            model = train_cfrnet(X_tr_n, t_train, y_tr_n, input_dim, cfg)
            ite = predict_cfrnet(model, X_te_n) * y_std

        elif model_name == 'ganite':
            cfg = {**config, '_X_val': X_val_n, '_t_val': t_val, '_y_val': y_val_n}
            model = train_ganite(X_tr_n, t_train, y_tr_n, input_dim, cfg)
            ite = predict_ganite(model, X_te_n) * y_std

        elif model_name == 'cevae':
            cfg = {**config, '_X_val': X_val_n, '_t_val': t_val, '_y_val': y_val_n}
            model = train_cevae(X_tr_n, t_train, y_tr_n, input_dim, cfg)
            ite = predict_cevae(model, X_te_n, n_samples=5) * y_std

        elif model_name == 'drnet':
            cfg = {**config, '_X_val': X_val_n, '_t_val': t_val, '_y_val': y_val_n}
            model = train_drnet(X_tr_n, t_train, y_tr_n, input_dim, cfg)
            ite = predict_drnet(model, X_te_n) * y_std

        elif model_name in ['tarnet', 'snet', 'dragonnet', 'flextenet']:
            # CATENets already have internal val split — we override with our val set
            # Concatenate train+val and let internal split use our val
            cfg = {**config}
            model = train_catenet(X_tr_n, t_train, y_tr_n, input_dim,
                                  model_type=model_name, config=cfg,
                                  X_val=X_val_n, t_val=t_val, y_val=y_val_n)
            ite = predict_catenet(model, X_te_n) * y_std
        else:
            raise ValueError(f"Unknown model: {model_name}")

        elapsed = time.time() - t0
        return ite, elapsed, None

    except Exception as e:
        elapsed = time.time() - t0
        return None, elapsed, str(e)[:200]


# ==============================================================================
# Configurations per dataset size
# ==============================================================================
def get_model_configs(dataset_size, input_dim):
    """Get model hyperparameters based on dataset characteristics."""
    if dataset_size < 2000:
        # Small (IHDP, Jobs)
        return {
            'cfrnet': {'repr_dim': 100, 'hypo_dim': 50, 'n_epochs': 100, 'lr': 1e-3,
                       'n_repr_layers': 2, 'n_hypo_layers': 2, 'patience': 20},
            'ganite': {'h_dim': 50, 'n_iter_gan': 600, 'n_iter_inf': 600, 'lr': 1e-3, 'patience': 20},
            'cevae': {'h_dim': 64, 'latent_dim': 15, 'n_epochs': 60, 'lr': 1e-3, 'n_layers': 2, 'patience': 15},
            'drnet': {'repr_dim': 50, 'head_dim': 50, 'n_epochs': 100, 'lr': 1e-3,
                      'n_repr_layers': 2, 'n_head_layers': 2, 'patience': 20},
            'tarnet': {'repr_dim': 100, 'out_dim': 50, 'n_epochs': 100, 'lr': 1e-4,
                       'n_repr_layers': 2, 'n_out_layers': 2, 'patience': 20},
            'snet': {'repr_dim': 100, 'out_dim': 50, 'n_epochs': 100, 'lr': 1e-4,
                     'n_repr_layers': 2, 'n_out_layers': 2, 'patience': 20},
            'dragonnet': {'repr_dim': 100, 'out_dim': 50, 'n_epochs': 100, 'lr': 1e-4,
                          'n_repr_layers': 2, 'n_out_layers': 2, 'patience': 20},
            'flextenet': {'repr_dim': 100, 'out_dim': 50, 'n_epochs': 100, 'lr': 1e-4,
                          'n_repr_layers': 2, 'n_out_layers': 2, 'patience': 20},
        }
    elif dataset_size < 10000:
        # Medium (Twins, ACIC, News, ACIC2018)
        return {
            'cfrnet': {'repr_dim': 64, 'hypo_dim': 32, 'n_epochs': 80, 'lr': 1e-3,
                       'n_repr_layers': 2, 'n_hypo_layers': 2, 'patience': 15},
            'ganite': {'h_dim': 32, 'n_iter_gan': 500, 'n_iter_inf': 500, 'lr': 1e-3, 'patience': 15},
            'cevae': {'h_dim': 50, 'latent_dim': 10, 'n_epochs': 40, 'lr': 1e-3, 'n_layers': 2, 'patience': 10},
            'drnet': {'repr_dim': 32, 'head_dim': 32, 'n_epochs': 80, 'lr': 1e-3,
                      'n_repr_layers': 2, 'n_head_layers': 2, 'patience': 15},
            'tarnet': {'repr_dim': 64, 'out_dim': 32, 'n_epochs': 80, 'lr': 1e-4,
                       'n_repr_layers': 2, 'n_out_layers': 2, 'patience': 15},
            'snet': {'repr_dim': 64, 'out_dim': 32, 'n_epochs': 80, 'lr': 1e-4,
                     'n_repr_layers': 2, 'n_out_layers': 2, 'patience': 15},
            'dragonnet': {'repr_dim': 64, 'out_dim': 32, 'n_epochs': 80, 'lr': 1e-4,
                          'n_repr_layers': 2, 'n_out_layers': 2, 'patience': 15},
            'flextenet': {'repr_dim': 64, 'out_dim': 32, 'n_epochs': 80, 'lr': 1e-4,
                          'n_repr_layers': 2, 'n_out_layers': 2, 'patience': 15},
        }
    else:
        # Large (TCGA, LBIDD, Criteo, Hillstrom)
        hdim = min(32, max(16, input_dim // 100))
        return {
            'cfrnet': {'repr_dim': hdim * 2, 'hypo_dim': hdim, 'n_epochs': 40, 'lr': 1e-3,
                       'n_repr_layers': 2, 'n_hypo_layers': 2, 'patience': 10},
            'ganite': {'h_dim': hdim, 'n_iter_gan': 300, 'n_iter_inf': 300, 'lr': 1e-3, 'patience': 10},
            'cevae': {'h_dim': hdim * 2, 'latent_dim': 8, 'n_epochs': 25, 'lr': 1e-3, 'n_layers': 2, 'patience': 8},
            'drnet': {'repr_dim': hdim, 'head_dim': hdim, 'n_epochs': 40, 'lr': 1e-3,
                      'n_repr_layers': 2, 'n_head_layers': 2, 'patience': 10},
            'tarnet': {'repr_dim': hdim * 2, 'out_dim': hdim, 'n_epochs': 40, 'lr': 5e-4,
                       'n_repr_layers': 2, 'n_out_layers': 2, 'patience': 10},
            'snet': {'repr_dim': hdim * 2, 'out_dim': hdim, 'n_epochs': 40, 'lr': 5e-4,
                     'n_repr_layers': 2, 'n_out_layers': 2, 'patience': 10},
            'dragonnet': {'repr_dim': hdim * 2, 'out_dim': hdim, 'n_epochs': 40, 'lr': 5e-4,
                          'n_repr_layers': 2, 'n_out_layers': 2, 'patience': 10},
            'flextenet': {'repr_dim': hdim * 2, 'out_dim': hdim, 'n_epochs': 40, 'lr': 5e-4,
                          'n_repr_layers': 2, 'n_out_layers': 2, 'patience': 10},
        }


# ==============================================================================
# Main benchmark runner
# ==============================================================================
ALL_MODELS = ['cfrnet', 'ganite', 'cevae', 'drnet', 'tarnet', 'snet', 'dragonnet', 'flextenet']
ALL_DATASETS = ['ihdp', 'twins', 'acic2016', 'news', 'tcga', 'jobs', 'acic2018', 'hillstrom', 'lbidd', 'criteo']


def run_single_experiment(dataset_name, model_name, data, config):
    """Run a single model on a single dataset split."""
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

    ite_pred, train_time, error = train_and_predict(
        model_name, X_train, t_train, y_train,
        X_val, t_val, y_val, X_test, config
    )

    if error is not None:
        return {'error': error, 'time': train_time}

    metrics = compute_metrics(ite_pred, t_test, y_test, mu0_test, mu1_test)
    metrics['time'] = train_time
    return metrics


def run_dataset_benchmark(dataset_name, models=None, seeds=None, ihdp_realizations=10, verbose=True):
    """Run benchmark for a single dataset across seeds/realizations."""
    if models is None:
        models = ALL_MODELS
    if seeds is None:
        seeds = [42, 43, 44]

    loader = get_dataset_loader(dataset_name)
    if loader is None:
        print(f"  ERROR: No loader for {dataset_name}")
        return {}

    info = DATASET_INFO.get(dataset_name, {})
    has_gt = info.get("has_ground_truth", True)

    results = {m: [] for m in models}

    # Determine iterations based on dataset type
    if dataset_name == 'ihdp':
        iterations = [(r, r) for r in range(1, ihdp_realizations + 1)]
        iter_label = "realization"
    elif dataset_name == 'acic2016':
        iterations = [(dgp, 42 + dgp) for dgp in range(1, 4)]
        iter_label = "DGP"
    elif dataset_name == 'acic2018':
        iterations = [(dgp, 42 + dgp) for dgp in range(1, 7)]
        iter_label = "DGP"
    elif dataset_name == 'lbidd':
        iterations = [(s, 42 + s) for s in range(1, 4)]
        iter_label = "setting"
    else:
        iterations = [(s, s) for s in seeds]
        iter_label = "seed"

    for iter_val, seed in iterations:
        # Load data
        try:
            if dataset_name == 'ihdp':
                data = loader(realization=iter_val, seed=seed)
            elif dataset_name == 'acic2016':
                data = loader(dgp=iter_val, seed=seed)
            elif dataset_name == 'acic2018':
                data = loader(dgp=iter_val, seed=seed)
            elif dataset_name == 'lbidd':
                data = loader(setting=iter_val, seed=seed)
            elif dataset_name == 'hillstrom':
                data = loader(treatment="mens", seed=seed)
            else:
                data = loader(seed=seed)
        except Exception as e:
            print(f"  ERROR loading {dataset_name} ({iter_label}={iter_val}): {e}")
            continue

        n_train = len(data['X_train'])
        n_val = len(data['X_val'])
        n_test = len(data['X_test'])
        input_dim = data['X_train'].shape[1]

        if verbose:
            print(f"  {iter_label}={iter_val}: train={n_train}, val={n_val}, test={n_test}, dim={input_dim}")

        configs = get_model_configs(n_train + n_val, input_dim)

        for mn in models:
            config = configs[mn]
            res = run_single_experiment(dataset_name, mn, data, config)

            if 'error' in res:
                if verbose:
                    print(f"    {mn:<12} ERROR: {res['error'][:60]}")
            else:
                pehe_str = f"PEHE={res.get('pehe', 'N/A'):.4f}" if 'pehe' in res else "no GT"
                if verbose:
                    print(f"    {mn:<12} {pehe_str}, t={res['time']:.1f}s")

            results[mn].append(res)

    return results


def aggregate_results(results):
    """Aggregate results across seeds/realizations."""
    aggregated = {}
    for mn, runs in results.items():
        valid = [r for r in runs if 'error' not in r]
        if not valid:
            aggregated[mn] = {'status': 'failed', 'n_valid': 0, 'n_total': len(runs)}
            continue

        agg = {'n_valid': len(valid), 'n_total': len(runs)}
        for metric in ['pehe', 'ate_error', 'att_error', 'ite_corr', 'policy_risk', 'policy_agreement', 'time']:
            values = [r[metric] for r in valid if metric in r and not np.isnan(r.get(metric, float('nan')))]
            if values:
                agg[f'{metric}_mean'] = float(np.mean(values))
                agg[f'{metric}_std'] = float(np.std(values))

        aggregated[mn] = agg
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Enhanced Causal Inference Benchmark")
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS, choices=ALL_DATASETS)
    parser.add_argument('--models', nargs='+', default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--ihdp-realizations', type=int, default=10)
    parser.add_argument('--output-dir', default='results')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"  {dataset.upper()} Benchmark")
        print(f"{'='*60}")

        results = run_dataset_benchmark(
            dataset, models=args.models, seeds=args.seeds,
            ihdp_realizations=args.ihdp_realizations,
            verbose=not args.quiet
        )

        aggregated = aggregate_results(results)
        all_results[dataset] = aggregated

        # Save per-dataset results
        with open(os.path.join(args.output_dir, f"{dataset}_results.json"), 'w') as f:
            json.dump(aggregated, f, indent=2)

        # Print summary
        print(f"\n  {dataset.upper()} Summary:")
        has_gt = DATASET_INFO.get(dataset, {}).get("has_ground_truth", True)
        if has_gt:
            print(f"  {'Model':<12} {'√PEHE':<16} {'εATE':<16} {'εATT':<16} {'Corr':<12} {'Time'}")
            for mn in args.models:
                agg = aggregated.get(mn, {})
                if agg.get('status') == 'failed':
                    print(f"  {mn:<12} FAILED")
                elif 'pehe_mean' in agg:
                    print(f"  {mn:<12} {agg['pehe_mean']:.4f}±{agg['pehe_std']:.4f}  "
                          f"{agg.get('ate_error_mean', 0):.4f}±{agg.get('ate_error_std', 0):.4f}  "
                          f"{agg.get('att_error_mean', 0):.4f}±{agg.get('att_error_std', 0):.4f}  "
                          f"{agg.get('ite_corr_mean', 0):.3f}±{agg.get('ite_corr_std', 0):.3f}  "
                          f"{agg.get('time_mean', 0):.1f}s")
        else:
            print(f"  {'Model':<12} {'Policy Agree':<16} {'Time'}")
            for mn in args.models:
                agg = aggregated.get(mn, {})
                if agg.get('status') == 'failed':
                    print(f"  {mn:<12} FAILED")
                elif 'policy_agreement_mean' in agg:
                    print(f"  {mn:<12} {agg['policy_agreement_mean']:.4f}±{agg.get('policy_agreement_std', 0):.4f}  "
                          f"{agg.get('time_mean', 0):.1f}s")

    # Save combined results
    with open(os.path.join(args.output_dir, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  All results saved to {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
