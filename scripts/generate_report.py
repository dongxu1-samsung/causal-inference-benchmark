"""
Generate Results Report
Creates a comprehensive markdown report of benchmark results.
"""

import os
import json
import sys
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def generate_report():
    """Generate markdown report from benchmark results."""
    results_file = os.path.join(RESULTS_DIR, "benchmark_results.json")
    if not os.path.exists(results_file):
        print("No results found. Run the benchmark first.")
        return

    with open(results_file) as f:
        all_results = json.load(f)

    report = []
    report.append("# Causal Inference Benchmark Results\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("## Overview\n")
    report.append("| Metric | Description | Direction |")
    report.append("|--------|-------------|-----------|")
    report.append("| √PEHE | Root Precision in Estimation of Heterogeneous Effects | ↓ Lower is better |")
    report.append("| \\|ATE Error\\| | Absolute error in Average Treatment Effect estimation | ↓ Lower is better |")
    report.append("| ITE Corr | Pearson correlation between predicted and true ITE | ↑ Higher is better |")
    report.append("")

    for dataset, summary in all_results.items():
        report.append(f"\n## {dataset.upper()}\n")
        report.append(f"| Model | √PEHE (↓) | \\|ATE Error\\| (↓) | ITE Corr (↑) | Train Time (s) | Runs |")
        report.append(f"|-------|-----------|-----------------|-------------|----------------|------|")

        sorted_models = sorted(summary.items(),
                              key=lambda x: x[1].get("pehe_mean", float("inf")))

        for model_name, s in sorted_models:
            if s.get("n_valid", 0) == 0:
                report.append(f"| {model_name} | FAILED | — | — | — | 0/{s.get('n_total', 0)} |")
                continue
            report.append(
                f"| **{model_name}** | "
                f"{s['pehe_mean']:.4f} ± {s['pehe_std']:.4f} | "
                f"{s['ate_error_mean']:.4f} ± {s['ate_error_std']:.4f} | "
                f"{s['ite_corr_mean']:.3f} ± {s['ite_corr_std']:.3f} | "
                f"{s['train_time_mean']:.1f} | "
                f"{s['n_valid']}/{s['n_total']} |"
            )
        report.append("")

    # Summary across datasets
    report.append("\n## Cross-Dataset Summary\n")
    report.append("Average rank across all datasets (lower = better):\n")

    model_ranks = {}
    for dataset, summary in all_results.items():
        sorted_models = sorted(
            [(m, s.get("pehe_mean", float("inf"))) for m, s in summary.items() if s.get("n_valid", 0) > 0],
            key=lambda x: x[1]
        )
        for rank, (model, _) in enumerate(sorted_models, 1):
            if model not in model_ranks:
                model_ranks[model] = []
            model_ranks[model].append(rank)

    report.append("| Model | Avg PEHE Rank | Datasets Completed |")
    report.append("|-------|---------------|-------------------|")
    for model, ranks in sorted(model_ranks.items(), key=lambda x: sum(x[1]) / len(x[1])):
        avg_rank = sum(ranks) / len(ranks)
        report.append(f"| **{model}** | {avg_rank:.1f} | {len(ranks)}/{len(all_results)} |")

    report.append("\n\n## Methodology\n")
    report.append("- **IHDP**: 10 realizations, 747 samples, 25 features, 80/20 train/test split")
    report.append("- **Twins**: 10K subsampled from 71K pairs, 50+ features, 3 random seeds")
    report.append("- **ACIC 2016**: 4,802 samples, 58+ features, 3 DGP settings")
    report.append("- **News**: 5,000 samples, 3,477 features (bag-of-words), 3 random seeds")
    report.append("- **TCGA**: 9,659 samples, 4,000 features (gene expression), 3 random seeds")
    report.append("- All models trained on CPU (Apple Silicon M-series Mac)")
    report.append("- Outcome standardized before training, predictions rescaled")

    output_path = os.path.join(RESULTS_DIR, "RESULTS.md")
    with open(output_path, "w") as f:
        f.write("\n".join(report))
    print(f"Report saved to: {output_path}")
    return "\n".join(report)


if __name__ == "__main__":
    report = generate_report()
    if report:
        print("\n" + report)
