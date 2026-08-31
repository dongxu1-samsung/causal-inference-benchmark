# Causal Inference Benchmark

A comprehensive benchmark for Individual Treatment Effect (ITE) estimation,
spanning **15 datasets** and **17 model architectures** (11 neural + 6 ML/tree-based).

## Models (17)

### Neural Models (11)

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
| **DESCN** | Zhong et al. 2022 (Alibaba) | Entire Space Cross Network — 5-head joint estimation |
| **MOCA** | 2026 | Modular One-way Cross-Attention between treatment branches and confounders |
| **DDRNet** | 2025 | Mixture-of-Experts with disentangled I/C/A subspaces |

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

# Run neural models benchmark (v2 — 8 models)
python3 scripts/run_benchmark.py --datasets all --seeds 42 43 44

# Run ML/tree-based models benchmark (v3 — 6 models)
python3 scripts/run_benchmark_v3.py

# Run DESCN/MOCA/DDRNet benchmark (v4 — 3 models)
python3 scripts/run_benchmark_v4.py

# Regenerate RESULTS.md from all results
python3 scripts/generate_results.py
```

## Results Summary

See [RESULTS.md](RESULTS.md) for full tables.

### Overall Model Rankings (√PEHE across 12 GT datasets, lower rank = better)

| Rank | Model | Type | Avg Rank | #1 Wins |
|------|-------|------|----------|---------|
| 1 | **DDRNet** | Neural | 4.73 | 3 |
| 2 | **CFRNet** | Neural | 6.75 | 0 |
| 3 | **DRNet** | Neural | 6.83 | 0 |
| 4 | **FlexTENet** | Neural | 6.83 | 2 |
| 5 | **CausalODE** | Neural | 6.92 | 0 |
| 6 | **TARNet** | Neural | 7.08 | 0 |
| 7 | **TransDCA** | Neural | 7.25 | 3 |
| 8 | **DESCN** | Neural | 8.09 | 0 |
| 9 | **DragonNet** | Neural | 8.17 | 0 |
| 10 | **X-Learner** | ML | 8.80 | 1 |

### Key Findings

- **DDRNet is #1 overall** — MoE disentanglement (I/C/A subspaces) + orthogonality constraints provide the most consistent performance across diverse datasets
- **Neural models outperform ML/tree-based on average** (avg rank 7.52 vs 10.94)
- However, **ML models dominate specific niches**: X-Learner wins News, BART wins TCGA
- **No single model dominates** — dataset characteristics strongly determine the winner
- **TransDCA** wins 3 datasets (NLSM, IBM Causal, Continuous) despite mid-table avg rank — bimodal performer

## Metrics

### Ground-Truth Datasets (12)
- **√PEHE**: Root Precision in Estimation of Heterogeneous Effects (primary, lower = better)
- **ITE Correlation**: Pearson correlation between predicted and true ITE (higher = better)

### Real-World Datasets (3, no ground truth)
- **AUUC**: Area Under the Uplift Curve (normalized, higher = better)
- **Qini**: Qini coefficient (normalized, higher = better)

## Architecture Details

### DDRNet (MoE Disentanglement, 2025)
- Shared encoder → 3 Mixture-of-Experts modules (Instrumental, Confounding, Adjustment)
- Orthogonality loss + HSIC MI penalty separate subspaces
- Treatment prediction from I+C, outcome prediction from C+A
- Gate load balancing prevents expert collapse

### MOCA (Modular Cross-Attention, 2026)
- Shared confounder encoder + per-treatment branch encoders
- One-way cross-attention: treatment queries attend to confounder keys/values
- Prevents information leakage from treatment to confounders
- Propensity head from shared representation + MMD balance

### DESCN (Entire Space Cross Network, Alibaba 2022)
- 5 heads: propensity (e), base outcomes (μ₀, μ₁), pseudo treatment effects (τ₀, τ₁)
- Cross-constraints: τ₁ ≈ y − μ₀ for treated, τ₀ ≈ μ₁ − y for control
- Weighted combination like X-learner: τ = e·τ₀ + (1−e)·τ₁

### TransDCA (Novel)
- Splits input features into token groups → Transformer encoder
- Disentangles into 3 subspaces: Instrumental (Z_I), Confounding (Z_C), Adjustment (Z_A)
- Cross-attention between treatment and latent representations

### CausalODE (Novel)
- Encoder maps (X, T) → latent state z_0
- Neural ODE evolves z_0 → z_T via learned dynamics f(z, t)
- IPM regularization (MMD) for balanced representations

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
  title={Comprehensive Causal Inference Benchmark: 17 Models, 15 Datasets},
  author={Xu, Darren},
  year={2024},
  url={https://github.com/dongxu1-samsung/causal-inference-benchmark}
}
```
