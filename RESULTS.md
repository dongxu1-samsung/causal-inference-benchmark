# Causal Inference Benchmark Results

## Overview

This benchmark evaluates **8 neural network models** for individual treatment effect (ITE) estimation across **5 standard datasets**. All models are implemented in PyTorch and trained on CPU.

### Models Evaluated

| Model | Paper | Year | Key Idea |
|-------|-------|------|----------|
| **CFRNet** | Shalit et al. (ICML 2017) | 2017 | Counterfactual regression with IPM (MMD) regularization |
| **GANITE** | Yoon et al. (ICLR 2018) | 2018 | GAN-based counterfactual + ITE inference network |
| **CEVAE** | Louizos et al. (NeurIPS 2017) | 2017 | Variational autoencoder with latent confounders |
| **DRNet** | Schwab et al. (AAAI 2020) | 2020 | Dose-response network (binary treatment variant) |
| **TARNet** | Shalit et al. (ICML 2017) | 2017 | Treatment-agnostic representation with separate heads |
| **SNet** | Curth & van der Schaar (NeurIPS 2021) | 2021 | Shared + treatment-specific representation components |
| **DragonNet** | Shi et al. (NeurIPS 2019) | 2019 | Targeted regularization via propensity head |
| **FlexTENet** | Curth & van der Schaar (NeurIPS 2021) | 2021 | Flexible architecture with shared/specific subspaces |

### Datasets

| Dataset | N | Features | Source | Treatment |
|---------|---|----------|--------|-----------|
| **IHDP** | 747 | 25 | Hill (2011), 10 realizations | Binary (neonatal health) |
| **Twins** | 10,000 | 53 | Linked birth/death records | Binary (birth weight) |
| **ACIC 2016** | 4,802 | 79 | Atlantic Causal Inference Competition | Semi-synthetic (3 DGPs) |
| **News** | 5,000 | 3,477 | Semi-synthetic text features | Binary (content exposure) |
| **TCGA** | 9,659 | 4,000 | Semi-synthetic gene expression | Binary (drug treatment) |

### Metrics

- **√PEHE** (Precision in Estimation of Heterogeneous Effects): RMSE of predicted vs. true ITE. **Lower is better.**
- **|εATE|** (Absolute ATE Error): |predicted ATE − true ATE|. **Lower is better.**
- **ITE Correlation**: Pearson correlation between predicted and true ITE. **Higher is better.**

---

## Results by Dataset

### 1. IHDP (10 realizations, 3 seeds each)

| Model | √PEHE ↓ | |εATE| ↓ | ITE Corr ↑ | Avg Time |
|-------|---------|---------|-----------|----------|
| **DragonNet** | **1.109 ± 0.791** | 0.299 ± 0.301 | **0.832 ± 0.153** | 1.4s |
| TARNet | 1.184 ± 0.506 | 0.295 ± 0.312 | 0.779 ± 0.239 | 1.6s |
| FlexTENet | 1.245 ± 0.862 | **0.277 ± 0.131** | 0.785 ± 0.218 | 2.5s |
| CFRNet | 1.644 ± 0.954 | 0.356 ± 0.530 | 0.671 ± 0.249 | 27.3s |
| SNet | 1.675 ± 2.002 | 0.263 ± 0.165 | 0.799 ± 0.192 | 3.0s |
| DRNet | 1.771 ± 0.931 | 0.590 ± 0.213 | 0.653 ± 0.286 | 1.0s |
| GANITE | 2.004 ± 1.590 | 0.655 ± 0.202 | 0.624 ± 0.260 | 30.7s |
| CEVAE | 5.710 ± 6.696 | 2.280 ± 0.761 | -0.020 ± 0.099 | 2.8s |

**Winner: DragonNet** — Best PEHE and correlation. TARNet and FlexTENet close behind.

---

### 2. Twins (3 seeds)

| Model | √PEHE ↓ | |εATE| ↓ | ITE Corr ↑ | Avg Time |
|-------|---------|---------|-----------|----------|
| **CEVAE** | **0.173 ± 0.014** | **0.007 ± 0.001** | -0.005 ± 0.021 | 2.6s |
| SNet | 0.181 ± 0.014 | 0.009 ± 0.003 | 0.003 ± 0.025 | 7.0s |
| FlexTENet | 0.185 ± 0.015 | 0.008 ± 0.004 | 0.013 ± 0.037 | 6.2s |
| DRNet | 0.195 ± 0.014 | 0.008 ± 0.003 | 0.016 ± 0.035 | 5.7s |
| TARNet | 0.197 ± 0.013 | 0.008 ± 0.004 | 0.018 ± 0.033 | 3.4s |
| DragonNet | 0.202 ± 0.010 | 0.007 ± 0.002 | 0.012 ± 0.029 | 3.9s |
| CFRNet | 0.220 ± 0.021 | 0.013 ± 0.004 | **0.037 ± 0.065** | 4.5s |
| GANITE | 0.250 ± 0.036 | 0.072 ± 0.043 | 0.005 ± 0.020 | 41.9s |

**Winner: CEVAE** — Best PEHE and ATE on this binary-outcome dataset. Latent variable approach well-suited here.

---

### 3. ACIC 2016 (3 DGP settings)

| Model | √PEHE ↓ | |εATE| ↓ | ITE Corr ↑ | Avg Time |
|-------|---------|---------|-----------|----------|
| **FlexTENet** | **0.436 ± 0.028** | 0.020 ± 0.006 | **0.988 ± 0.002** | 3.0s |
| DragonNet | 0.628 ± 0.067 | 0.015 ± 0.007 | 0.975 ± 0.004 | 1.9s |
| TARNet | 0.688 ± 0.117 | **0.007 ± 0.006** | 0.970 ± 0.008 | 1.6s |
| SNet | 0.859 ± 0.118 | 0.052 ± 0.009 | 0.952 ± 0.011 | 3.4s |
| CFRNet | 0.873 ± 0.049 | 0.028 ± 0.011 | 0.952 ± 0.009 | 2.4s |
| GANITE | 2.036 ± 0.228 | 0.144 ± 0.037 | 0.810 ± 0.023 | 21.5s |
| CEVAE | 3.348 ± 0.357 | 1.731 ± 0.657 | -0.036 ± 0.014 | 1.3s |
| DRNet | — | — | — | — (BatchNorm error) |

**Winner: FlexTENet** — Dominant on PEHE and correlation. DragonNet second.

---

### 4. News (3 seeds, 3477-dim features)

| Model | √PEHE ↓ | |εATE| ↓ | ITE Corr ↑ | Avg Time |
|-------|---------|---------|-----------|----------|
| **DRNet** | **0.552 ± 0.022** | 0.263 ± 0.052 | 0.135 ± 0.038 | 3.1s |
| CEVAE | 0.605 ± 0.049 | 0.255 ± 0.043 | 0.022 ± 0.027 | 2.5s |
| SNet | 0.654 ± 0.038 | 0.164 ± 0.054 | 0.050 ± 0.019 | 4.6s |
| CFRNet | 0.717 ± 0.027 | **0.111 ± 0.038** | **0.268 ± 0.008** | 2.6s |
| GANITE | 0.780 ± 0.019 | 0.461 ± 0.003 | 0.238 ± 0.021 | 34.0s |
| DragonNet | 0.796 ± 0.078 | 0.270 ± 0.031 | 0.204 ± 0.018 | 2.0s |
| TARNet | 0.867 ± 0.058 | 0.403 ± 0.066 | 0.236 ± 0.012 | 2.1s |
| FlexTENet | 0.870 ± 0.018 | 0.438 ± 0.014 | 0.229 ± 0.037 | 38.1s |

**Winner: DRNet** — Best PEHE in high-dimensional setting. CFRNet best on ATE and correlation.

---

### 5. TCGA (1 seed, 4000-dim gene expression)

| Model | √PEHE ↓ | |εATE| ↓ | ITE Corr ↑ | Avg Time |
|-------|---------|---------|-----------|----------|
| **CFRNet** | **2.871** | **0.008** | **0.626** | 2.6s |
| DRNet | 2.908 | 0.175 | 0.599 | 3.9s |
| FlexTENet | 2.913 | 0.301 | 0.604 | 175.0s |
| TARNet | 2.950 | 0.094 | 0.594 | 1.9s |
| DragonNet | 2.986 | 0.125 | 0.562 | 2.1s |
| GANITE | 3.126 | 0.230 | 0.474 | 45.4s |
| SNet | 3.287 | 0.123 | 0.538 | 5.3s |
| CEVAE | 3.966 | 0.529 | -0.027 | 3.3s |

**Winner: CFRNet** — IPM regularization helps in high-dim genome-scale data.

---

## Cross-Dataset Summary

### Model Rankings (by √PEHE, 1=best)

| Model | IHDP | Twins | ACIC | News | TCGA | **Avg Rank** |
|-------|------|-------|------|------|------|-------------|
| **DragonNet** | 1 | 6 | 2 | 6 | 5 | **4.0** |
| **FlexTENet** | 3 | 3 | 1 | 8 | 3 | **3.6** |
| **TARNet** | 2 | 5 | 3 | 7 | 4 | **4.2** |
| **CFRNet** | 4 | 7 | 5 | 4 | 1 | **4.2** |
| **DRNet** | 6 | 4 | — | 1 | 2 | **3.3** |
| **SNet** | 5 | 2 | 4 | 3 | 7 | **4.2** |
| **CEVAE** | 8 | 1 | 7 | 2 | 8 | **5.2** |
| **GANITE** | 7 | 8 | 6 | 5 | 6 | **6.4** |

### Key Takeaways

1. **No single model dominates** across all datasets — dataset characteristics drive model selection.

2. **DragonNet and FlexTENet** are the most robust choices for low-to-medium dimensional problems (IHDP, ACIC). Their propensity-aware and flexible architectures handle confounding well.

3. **CFRNet and DRNet** excel in **high-dimensional** settings (News, TCGA) where the IPM regularization and dose-response architecture provide useful inductive bias.

4. **CEVAE** shows extreme bimodal behavior: best on Twins (binary outcomes, strong latent structure) but worst on IHDP/TCGA (struggles with continuous outcomes and high dimensions without extensive tuning).

5. **GANITE** consistently ranks below average — the adversarial training is unstable and computationally expensive (~10-40× slower than alternatives).

6. **TARNet** (the simplest 2-head architecture) is surprisingly competitive — a strong baseline that more complex models only marginally improve upon.

---

## Experimental Setup

- **Framework**: PyTorch 2.8.0 (CPU only)
- **Hardware**: MacBook Pro (Apple Silicon)
- **Training**: Adam optimizer, batch size 128 (full-batch for smaller datasets)
- **Epochs**: 60-100 (small datasets), 20-40 (large datasets)
- **Validation**: 20% holdout, metrics computed on test set with known ground truth
- **IHDP**: 10 realizations × 3 random seeds
- **Twins/News**: 3 random seeds
- **ACIC 2016**: 3 DGP settings
- **TCGA**: 1 seed (due to computational cost at 4000 dimensions)

## Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Download/generate datasets
python scripts/download_data.py

# Run full benchmark
python scripts/run_benchmark.py --datasets ihdp twins acic2016 news tcga

# Generate this report
python scripts/generate_report.py
```
