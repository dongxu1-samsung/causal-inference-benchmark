# Causal Inference Benchmark

A comprehensive benchmark of deep learning models for causal inference / treatment effect estimation.

## Quick Start

```bash
# 1. Generate/download all 10 datasets
python3 scripts/download_data.py

# 2. Run the full benchmark
python3 scripts/run_benchmark.py --datasets ihdp twins acic2016 news tcga jobs acic2018 hillstrom lbidd criteo

# 3. Run a single dataset
python3 scripts/run_benchmark.py --datasets ihdp --ihdp-realizations 10 --seeds 42 43 44
```

## Results Summary

See **[RESULTS.md](RESULTS.md)** for the full report with tables and analysis.

### Model Rankings (√PEHE, average rank across 7 datasets with ground truth)

| Rank | Model | Avg Rank | Best On |
|------|-------|----------|---------|
| 1 | DRNet | 2.71 | News, TCGA |
| 2 | FlexTENet | 3.43 | ACIC2016, ACIC2018 |
| 3 | CFRNet | 3.71 | IHDP |
| 4 | TARNet | 3.71 | LBIDD |
| 5 | DragonNet | 4.43 | — |
| 6 | SNet | 4.86 | — |
| 7 | CEVAE | 6.43 | Twins |
| 8 | GANITE | 6.71 | — |

## Datasets (10)

| Dataset | Type | N | Features | Ground Truth |
|---------|------|---|----------|:---:|
| IHDP | Semi-synthetic | 747 | 25 | ✓ |
| Twins | Semi-synthetic | 10K | 53 | ✓ |
| ACIC 2016 | Semi-synthetic | 4.8K | 79 | ✓ |
| News | Semi-synthetic | 5K | 3,477 | ✓ |
| TCGA | Semi-synthetic | 9.7K | 4,000 | ✓ |
| Jobs (LaLonde) | Real-world RCT | 2.5K | 8 | ✗ |
| ACIC 2018 | Semi-synthetic | 5K | 50 | ✓ |
| Hillstrom | Real-world RCT | 42.8K | 8 | ✗ |
| LBIDD | Semi-synthetic | 50K | 177 | ✓ |
| Criteo | Real-world RCT | 50K | 12 | ✗ |

## Models (8)

| Model | Key Idea | Year/Venue |
|-------|----------|-----------|
| CFRNet | Balanced representations via IPM (Wasserstein/MMD) | Shalit et al., 2017 (ICML) |
| GANITE | GAN-based counterfactual generation | Yoon et al., 2018 (ICLR) |
| CEVAE | Causal VAE with latent confounders | Louizos et al., 2017 (NeurIPS) |
| DRNet | Dose-response + targeted regularization | Shi et al., 2020 (AAAI) |
| TARNet | Treatment-Agnostic Representation Network | Shalit et al., 2017 (ICML) |
| SNet | 3-way disentangled representations | Curth & van der Schaar, 2021 (NeurIPS) |
| DragonNet | Targeted regularization via propensity head | Shi et al., 2019 (NeurIPS) |
| FlexTENet | Flexible shared + private + orthogonality | Curth & van der Schaar, 2021 (ICML) |

## Metrics

- **√PEHE**: Root Precision in Estimation of Heterogeneous Effects (individual-level accuracy)
- **εATE**: Absolute error in Average Treatment Effect (population-level bias)
- **εATT**: Absolute error in ATT (treated subpopulation bias)
- **ITE Correlation**: Pearson correlation between true and predicted ITEs
- **Policy Agreement**: Treatment recommendation concordance (for real-world data without ground truth)

## Evaluation Protocol

- **Splits**: Train (60%) / Validation (20%) / Test (20%)
- **Early stopping**: Patience=10 on validation factual loss
- **Normalization**: Standard scaling on covariates (fitted on train)
- **Repetitions**: Multiple seeds/realizations/DGPs per dataset
- **Results**: Mean ± std across repetitions

## Project Structure

```
├── models/
│   ├── __init__.py       # Data loaders for all 10 datasets
│   ├── cfrnet.py         # CFRNet + TARNet
│   ├── ganite.py         # GANITE
│   ├── cevae.py          # CEVAE
│   ├── drnet.py          # DRNet
│   └── catenets.py       # SNet, DragonNet, FlexTENet
├── scripts/
│   ├── download_data.py  # Download/generate all datasets
│   └── run_benchmark.py  # Run benchmark with all metrics
├── data/                 # Downloaded datasets (gitignored)
├── results/              # Benchmark outputs (JSON + summaries)
├── RESULTS.md            # Full results report
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0
- NumPy, Pandas, Scipy

## License

MIT
