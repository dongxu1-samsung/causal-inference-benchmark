"""
Main Benchmark Runner
Trains all models on all datasets and computes evaluation metrics.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import load_ihdp, load_twins, load_acic2016, load_news, load_tcga
from models.cfrnet import train_cfrnet, predict_cfrnet
from models.ganite import train_ganite, predict_ganite
from models.cevae import train_cevae, predict_cevae
from models.drnet import train_drnet, predict_drnet
from models.catenets import train_catenet, predict_catenet


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_metrics(ite_pred, mu0_true, mu1_true):
    """Compute all causal inference metrics."""
    ite_true = mu1_true - mu0_true
    ate_true = ite_true.mean()
    ate_pred = ite_pred.mean()

    metrics = {
        # Precision in Estimation of Heterogeneous Effects (lower is better)
        "pehe": np.sqrt(np.mean((ite_pred - ite_true) ** 2)),
        # Absolute ATE error (lower is better)
        "ate_error": np.abs(ate_pred - ate_true),
        # ATE bias (signed)
        "ate_bias": ate_pred - ate_true,
        # True ATE
        "ate_true": ate_true,
        # Predicted ATE
        "ate_pred": ate_pred,
        # Correlation between predicted and true ITE
        "ite_corr": np.corrcoef(ite_pred, ite_true)[0, 1] if np.std(ite_true) > 1e-8 else 0.0,
    }
    return metrics


# ============================================================================
# Model Training Wrappers
# ============================================================================

MODEL_CONFIGS = {
    "ihdp": {
        "cfrnet": {"repr_dim": 200, "hypo_dim": 100, "n_epochs": 300, "lr": 1e-3},
        "ganite": {"h_dim": 100, "n_iter_gan": 3000, "n_iter_inf": 3000, "lr": 1e-3},
        "cevae": {"h_dim": 200, "latent_dim": 20, "n_epochs": 150, "lr": 1e-3},
        "drnet": {"repr_dim": 64, "n_epochs": 300, "lr": 1e-3},
        "tarnet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 300, "lr": 1e-4},
        "snet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 300, "lr": 1e-4},
        "dragonnet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 300, "lr": 1e-4},
        "flextenet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 300, "lr": 1e-4},
    },
    "twins": {
        "cfrnet": {"repr_dim": 200, "hypo_dim": 100, "n_epochs": 200, "lr": 1e-3},
        "ganite": {"h_dim": 100, "n_iter_gan": 3000, "n_iter_inf": 3000, "lr": 1e-3},
        "cevae": {"h_dim": 200, "latent_dim": 20, "n_epochs": 100, "lr": 1e-3},
        "drnet": {"repr_dim": 64, "n_epochs": 200, "lr": 1e-3},
        "tarnet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 200, "lr": 1e-4},
        "snet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 200, "lr": 1e-4},
        "dragonnet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 200, "lr": 1e-4},
        "flextenet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 200, "lr": 1e-4},
    },
    "acic2016": {
        "cfrnet": {"repr_dim": 200, "hypo_dim": 100, "n_epochs": 300, "lr": 1e-3},
        "ganite": {"h_dim": 100, "n_iter_gan": 3000, "n_iter_inf": 3000, "lr": 1e-3},
        "cevae": {"h_dim": 200, "latent_dim": 20, "n_epochs": 150, "lr": 1e-3},
        "drnet": {"repr_dim": 64, "n_epochs": 300, "lr": 1e-3},
        "tarnet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 300, "lr": 1e-4},
        "snet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 300, "lr": 1e-4},
        "dragonnet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 300, "lr": 1e-4},
        "flextenet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 300, "lr": 1e-4},
    },
    "news": {
        "cfrnet": {"repr_dim": 200, "hypo_dim": 100, "n_epochs": 200, "lr": 1e-3, "batch_size": 200},
        "ganite": {"h_dim": 100, "n_iter_gan": 2000, "n_iter_inf": 2000, "lr": 1e-3, "batch_size": 256},
        "cevae": {"h_dim": 200, "latent_dim": 20, "n_epochs": 100, "lr": 1e-3},
        "drnet": {"repr_dim": 64, "n_epochs": 200, "lr": 1e-3, "batch_size": 128},
        "tarnet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 200, "lr": 1e-4},
        "snet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 200, "lr": 1e-4},
        "dragonnet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 200, "lr": 1e-4},
        "flextenet": {"repr_dim": 100, "out_dim": 100, "n_epochs": 200, "lr": 1e-4},
    },
    "tcga": {
        "cfrnet": {"repr_dim": 200, "hypo_dim": 100, "n_epochs": 200, "lr": 1e-3, "batch_size": 200},
        "ganite": {"h_dim": 100, "n_iter_gan": 2000, "n_iter_inf": 2000, "lr": 1e-3, "batch_size": 256},
        "cevae": {"h_dim": 200, "latent_dim": 20, "n_epochs": 100, "lr": 1e-3},
        "drnet": {"repr_dim": 64, "n_epochs": 200, "lr": 1e-3, "batch_size": 128},
        "tarnet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 200, "lr": 1e-4},
        "snet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 200, "lr": 1e-4},
        "dragonnet": {"repr_dim": 200, "out_dim": 100, "n_epochs": 200, "lr": 1e-4},
        "flextenet": {"repr_dim": 100, "out_dim": 100, "n_epochs": 200, "lr": 1e-4},
    },
}


def run_single_experiment(model_name, dataset_name, X_train, t_train, y_train,
                          X_test, t_test, y_test, mu0_test, mu1_test, config):
    """Run a single model on a single dataset split."""
    input_dim = X_train.shape[1]

    # Standardize features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # Standardize outcome
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    y_train_norm = (y_train - y_mean) / y_std

    start_time = time.time()

    try:
        if model_name == "cfrnet":
            model = train_cfrnet(X_train_norm, t_train, y_train_norm, input_dim, config)
            ite_pred = predict_cfrnet(model, X_test_norm) * y_std
        elif model_name == "ganite":
            model = train_ganite(X_train_norm, t_train, y_train_norm, input_dim, config)
            ite_pred = predict_ganite(model, X_test_norm) * y_std
        elif model_name == "cevae":
            model = train_cevae(X_train_norm, t_train, y_train_norm, input_dim, config)
            ite_pred = predict_cevae(model, X_test_norm) * y_std
        elif model_name == "drnet":
            model = train_drnet(X_train_norm, t_train, y_train_norm, input_dim, config)
            ite_pred = predict_drnet(model, X_test_norm) * y_std
        elif model_name in ["tarnet", "snet", "dragonnet", "flextenet"]:
            model = train_catenet(X_train_norm, t_train, y_train_norm, input_dim,
                                  model_type=model_name, config=config)
            ite_pred = predict_catenet(model, X_test_norm) * y_std
        else:
            raise ValueError(f"Unknown model: {model_name}")

        elapsed = time.time() - start_time
        metrics = compute_metrics(ite_pred, mu0_test, mu1_test)
        metrics["train_time"] = elapsed
        metrics["status"] = "success"

    except Exception as e:
        elapsed = time.time() - start_time
        metrics = {
            "pehe": float("nan"),
            "ate_error": float("nan"),
            "ate_bias": float("nan"),
            "ate_true": float("nan"),
            "ate_pred": float("nan"),
            "ite_corr": float("nan"),
            "train_time": elapsed,
            "status": f"error: {str(e)[:100]}",
        }

    return metrics


# ============================================================================
# Benchmark Orchestration
# ============================================================================

def run_ihdp_benchmark(models, n_realizations=100):
    """Run benchmark on IHDP with multiple realizations."""
    print(f"\n{'='*60}")
    print(f"IHDP Benchmark ({n_realizations} realizations)")
    print(f"{'='*60}")

    results = {m: [] for m in models}

    for real in range(1, n_realizations + 1):
        if real % 10 == 0:
            print(f"  Realization {real}/{n_realizations}...")

        try:
            X_train, t_train, y_train, X_test, t_test, y_test, mu0_test, mu1_test = load_ihdp(real)
        except FileNotFoundError:
            print(f"  Skipping realization {real} (file not found)")
            continue

        for model_name in models:
            config = MODEL_CONFIGS["ihdp"].get(model_name, {})
            metrics = run_single_experiment(
                model_name, "ihdp", X_train, t_train, y_train,
                X_test, t_test, y_test, mu0_test, mu1_test, config
            )
            results[model_name].append(metrics)

    return results


def run_dataset_benchmark(dataset_name, models, n_runs=5):
    """Run benchmark on a single dataset with multiple seeds."""
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} Benchmark ({n_runs} runs)")
    print(f"{'='*60}")

    results = {m: [] for m in models}
    loader = {"twins": load_twins, "acic2016": load_acic2016,
              "news": load_news, "tcga": load_tcga}[dataset_name]

    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}...")
        seed = 42 + run

        try:
            if dataset_name == "acic2016":
                data = loader(dgp=run + 1, replication=1, seed=seed)
            else:
                data = loader(test_fraction=0.2, seed=seed)
            X_train, t_train, y_train, X_test, t_test, y_test, mu0_test, mu1_test = data
        except Exception as e:
            print(f"  Error loading data: {e}")
            continue

        for model_name in models:
            config = MODEL_CONFIGS[dataset_name].get(model_name, {})
            metrics = run_single_experiment(
                model_name, dataset_name, X_train, t_train, y_train,
                X_test, t_test, y_test, mu0_test, mu1_test, config
            )
            results[model_name].append(metrics)
            if metrics["status"] == "success":
                print(f"    {model_name}: PEHE={metrics['pehe']:.4f}, "
                      f"|ATE|={metrics['ate_error']:.4f}, "
                      f"time={metrics['train_time']:.1f}s")

    return results


def aggregate_results(results):
    """Aggregate results across runs."""
    summary = {}
    for model_name, runs in results.items():
        if not runs:
            continue
        valid_runs = [r for r in runs if r["status"] == "success"]
        if not valid_runs:
            summary[model_name] = {"n_valid": 0, "n_total": len(runs)}
            continue

        pehes = [r["pehe"] for r in valid_runs]
        ates = [r["ate_error"] for r in valid_runs]
        corrs = [r["ite_corr"] for r in valid_runs]
        times = [r["train_time"] for r in valid_runs]

        summary[model_name] = {
            "pehe_mean": np.mean(pehes),
            "pehe_std": np.std(pehes),
            "ate_error_mean": np.mean(ates),
            "ate_error_std": np.std(ates),
            "ite_corr_mean": np.mean(corrs),
            "ite_corr_std": np.std(corrs),
            "train_time_mean": np.mean(times),
            "n_valid": len(valid_runs),
            "n_total": len(runs),
        }
    return summary


def print_summary_table(dataset_name, summary):
    """Print a formatted summary table."""
    print(f"\n{'─'*80}")
    print(f"  {dataset_name.upper()} Results Summary")
    print(f"{'─'*80}")
    print(f"{'Model':<12} {'√PEHE (↓)':<18} {'|ATE Error| (↓)':<18} {'ITE Corr (↑)':<16} {'Time (s)':<10} {'N'}")
    print(f"{'─'*80}")

    # Sort by PEHE
    sorted_models = sorted(summary.items(),
                          key=lambda x: x[1].get("pehe_mean", float("inf")))

    for model_name, s in sorted_models:
        if s.get("n_valid", 0) == 0:
            print(f"{model_name:<12} {'FAILED':<18} {'—':<18} {'—':<16} {'—':<10} {s.get('n_total', 0)}")
            continue
        print(f"{model_name:<12} "
              f"{s['pehe_mean']:.4f}±{s['pehe_std']:.4f}  "
              f"{s['ate_error_mean']:.4f}±{s['ate_error_std']:.4f}  "
              f"{s['ite_corr_mean']:.3f}±{s['ite_corr_std']:.3f}  "
              f"{s['train_time_mean']:.1f}     "
              f"{s['n_valid']}/{s['n_total']}")
    print(f"{'─'*80}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Causal Inference Benchmark")
    parser.add_argument("--datasets", nargs="+",
                       default=["ihdp", "twins", "acic2016", "news", "tcga"],
                       help="Datasets to benchmark")
    parser.add_argument("--models", nargs="+",
                       default=["cfrnet", "ganite", "cevae", "drnet",
                                "tarnet", "snet", "dragonnet", "flextenet"],
                       help="Models to benchmark")
    parser.add_argument("--ihdp-realizations", type=int, default=100,
                       help="Number of IHDP realizations")
    parser.add_argument("--n-runs", type=int, default=5,
                       help="Number of runs per dataset (non-IHDP)")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for results")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CAUSAL INFERENCE BENCHMARK")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = {}

    for dataset in args.datasets:
        if dataset == "ihdp":
            results = run_ihdp_benchmark(args.models, args.ihdp_realizations)
        else:
            results = run_dataset_benchmark(dataset, args.models, args.n_runs)

        summary = aggregate_results(results)
        print_summary_table(dataset, summary)
        all_results[dataset] = summary

        # Save per-dataset results
        with open(os.path.join(args.output_dir, f"{dataset}_results.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

    # Save full results
    with open(os.path.join(args.output_dir, "benchmark_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
