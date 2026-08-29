# Causal Inference Benchmark

A comprehensive benchmark for Individual Treatment Effect (ITE) estimation,
spanning **15 datasets** and **14 model architectures** (8 neural + 6 ML/tree-based).

## Models (14)

### Neural Models (8)

| Model | Paper/Method | Key Innovation |
|-------|-------------|----------------|
| **CFRNet** | Shalit et al. 2017 | Counterfactual regression with IPM regularization |
| **DRNet** | Schwab et al. 2020 | Dosage-response network with targeted regularization |
| **TARNet** | Shalit et al. 2017 | Treatment-Agnostic Representation Network |
| **SNet** | Curth & van der Schaar 2021 | Shared/specific representation learning |
| **DragonNet** | Shi et al. 2019 | Targeted regularization via propensity head |
| **FlexTENet** | Curth & van der Schaar 2021 | Flexible TE with orthogonality constraints |
| **TransDCA** ★ | Novel | Transformer Disentangled Causal Attention |
| **CausalODE** ★ | Novel | IPM-regularized Neural ODE dynamics |

★ = Novel architectures implemented in this benchmark

### ML / Tree-Based Models (6)

| Model | Package | Key Innovation |
|-------|---------|----------------|
| **CausalForestDML** | EconML | Honest causal forest + double ML orthogonalization |
| **DR-Learner** | CausalML + LGBM | Doubly-robust meta-learner with gradient boosting |
| **X-Learner** | CausalML + XGBoost | Cross-imputation for unbalanced treatment groups |
| **R-Learner** | CausalML + LGBM | Residual-on-residual orthogonal meta-learner |
| **UpliftRF** | CausalML | Uplift random forest with KL-divergence splitting |
| **BART** | sklearn HistGBR | Bayesian-style T-learner with HistGradientBoosting |

## Datasets (15)

### With Ground-Truth Counterfactuals (12)

| Dataset | N | Features | Type | Domain |
|---------|---|----------|------|--------|
| IHDP | 747 | 25 | Semi-synthetic | Healthcare |
| Twins | 10,000 | 53 | Semi-synthetic | Healthcare |
| ACIC 2016 | 4,802 | 79 | Semi-synthetic (3 DGPs) | Competition |
| News | 5,000 | 3,477 | Semi-synthetic | Text/NLP |
| TCGA | 9,659 | 4,000 | Semi-synthetic | Genomics |
| ACIC 2018 | 5,000 | 50 | Semi-synthetic (6 DGPs) | Competition |
| LBIDD | 50,000 | 177 | Semi-synthetic (3 settings) | Large-scale |
| NLSM | 10,000 | 20 | Synthetic (3 difficulties) | Causal Forest std |
| IBM Causal | 10,000 | 30 | Synthetic (4 confounding levels) | Tunable |
| Continuous DGP | 10,000 | 25 | Synthetic (continuous treatment) | Dose-response |
| ACIC 2022 | 10,000 | 33 | Semi-synthetic (3 time steps) | Longitudinal |
| STAR | 11,000 | 30 | Real RCT | Education |

### Real-World Without Ground Truth (3)

| Dataset | N | Features | Type | Domain |
|---------|---|----------|------|--------|
| Jobs (LaLonde) | 2,787 | 8 | Real RCT | Employment |
| Hillstrom | 42,693 | 8 | Real RCT | E-commerce |
| Criteo | 50,000 | 12 | Real observational | Advertising |

## Quick Start

```bash
# Generate/download all datasets
python3 scripts/download_data.py

# Run neural models benchmark (v2)
python3 scripts/run_benchmark.py --datasets all --seeds 42 43 44

# Run ML/tree-based models benchmark (v3)
python3 scripts/run_benchmark_v3.py
```

## Results Summary

See [RESULTS.md](RESULTS.md) for full tables.

### Overall Model Rankings (√PEHE across 11 GT datasets, lower rank = better)

| Rank | Model | Type | Avg Rank | #1 Wins |
|------|-------|------|----------|---------|
| 1 | **DRNet** | Neural | 5.64 | 0 |
| 2 | **FlexTENet** | Neural | 5.64 | 3 |
| 3 | **CausalODE** | Neural | 5.82 | 0 |
| 4 | **CFRNet** | Neural | 5.91 | 0 |
| 5 | **TARNet** | Neural | 5.91 | 0 |
| 6 | **TransDCA** | Neural | 6.27 | 3 |
| 7 | **X-Learner** | ML | 6.60 | 1 |
| 8 | **DragonNet** | Neural | 6.73 | 0 |
| 9 | **BART** | ML | 7.70 | 1 |
| 10 | **CausalForestDML** | ML | 7.90 | 1 |

### Key Findings

- **Neural models outperform ML/tree-based on average** (avg rank 6.31 vs 8.82)
- However, **ML models dominate specific niches**: CausalForestDML wins IHDP, X-Learner wins News, BART wins TCGA
- **No single model dominates** — dataset characteristics strongly determine the winner
- **BART** is the best overall ML model — simple T-learner with HistGBR beats complex neural architectures on high-dim data
- **R-Learner** underperforms — residual-on-residual can be unstable

## Metrics

### Ground-Truth Datasets (12)
- **√PEHE**: Root Precision in Estimation of Heterogeneous Effects (primary, lower = better)
- **ITE Correlation**: Pearson correlation between predicted and true ITE (higher = better)

### Real-World Datasets (3, no ground truth)
- **AUUC**: Area Under the Uplift Curve (normalized, higher = better)
- **Qini**: Qini coefficient (normalized, higher = better)

## Architecture Details

### TransDCA (Novel)
- Splits input features into token groups → Transformer encoder
- Disentangles into 3 subspaces: Instrumental (Z_I), Confounding (Z_C), Adjustment (Z_A)
- Cross-attention between treatment and latent representations
- Orthogonality + MI minimization losses for subspace separation

### CausalODE (Novel)
- Encoder maps (X, T) → latent state z_0
- Neural ODE evolves z_0 → z_T via learned dynamics f(z, t)
- IPM regularization (MMD) for balanced representations
- Separate outcome heads per treatment arm

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- EconML, CausalML (for ML models)
- LightGBM, XGBoost (for meta-learners)

## Citation

If you use this benchmark, please cite:
```
@misc{causal-inference-benchmark-2024,
  title={Comprehensive Causal Inference Benchmark: 15 Datasets, 14 Models},
  author={Xu, Darren},
  year={2024},
  url={https://github.com/dongxu1-samsung/causal-inference-benchmark}
}
```
