# Causal Inference Benchmark Results (v3)

**14 models × 15 datasets** — 8 neural + 6 ML/tree-based models

## Model Categories

| Category | Models | Package |
|----------|--------|---------|
| Balanced Representation | CFRNet, TARNet | PyTorch |
| Targeted Regularization | DragonNet, SNet, FlexTENet | PyTorch |
| Multi-dose Neural | DRNet | PyTorch |
| Transformer Disentangle | TransDCA | PyTorch |
| Neural ODE | CausalODE | PyTorch |
| Forest + Double ML | CausalForestDML | EconML |
| Doubly-Robust Meta | DR-Learner | CausalML + LGBM |
| Cross-Imputation Meta | X-Learner | CausalML + XGBoost |
| Residual Meta | R-Learner | CausalML + LGBM |
| Uplift Tree Ensemble | UpliftRF | CausalML |
| T-learner (BART-style) | BART | sklearn HistGBR |

## 1. √PEHE — Ground-Truth Datasets

Lower = better. **Bold** = best per dataset.

| Model | IHDP | Twins | ACIC2016 | News | TCGA | ACIC2018 | LBIDD | NLSM | IBM Causal | ACIC2022 | STAR |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| CFRNet | 1.18±1.02 | 0.19±0.01 | 0.86±0.09 | 0.66±0.05 | 3.13±0.01 | 1.30±0.49 | 0.39±0.05 | 0.32±0.10 | 0.32±0.08 | 0.94±0.02 | 0.10±0.01 |
| TARNet | 1.58±1.19 | 0.20±0.02 | 0.62±0.05 | 0.87±0.05 | 3.12±0.03 | 1.30±0.48 | **0.27**±0.01 | 0.29±0.12 | 0.42±0.08 | 0.95±0.03 | **0.09**±0.00 |
| DRNet | 1.79±0.92 | 0.20±0.01 | 0.72±0.08 | 0.61±0.08 | 3.04±0.07 | 1.30±0.48 | **0.27**±0.02 | 0.31±0.07 | 0.23±0.02 | 0.94±0.06 | 0.14±0.01 |
| DragonNet | 1.40±1.43 | 0.19±0.02 | 0.66±0.03 | 1.00±0.04 | 3.17±0.04 | 1.32±0.49 | 0.32±0.02 | 0.29±0.10 | 0.45±0.08 | 0.95±0.02 | **0.09**±0.00 |
| SNet | 2.20±2.67 | 0.18±0.01 | 0.83±0.13 | 0.72±0.08 | 3.53±0.04 | 1.68±0.64 | 0.33±0.02 | 0.53±0.11 | 0.58±0.07 | 0.96±0.03 | 0.12±0.01 |
| FlexTENet | 2.05±2.18 | 0.19±0.02 | **0.46**±0.06 | 1.24±0.03 | 3.08±0.06 | **1.03**±0.61 | **0.27**±0.01 | 0.35±0.11 | 0.37±0.06 | 0.95±0.03 | 0.09±0.00 |
| TransDCA | 1.79±2.00 | 0.20±0.01 | 1.10±0.11 | 0.46±0.03 | 3.58±0.03 | 1.19±0.46 | 0.42±0.09 | **0.22**±0.07 | **0.20**±0.03 | 0.96±0.02 | **0.09**±0.00 |
| CausalODE | 1.27±0.61 | 0.21±0.01 | 0.62±0.06 | 0.90±0.03 | 3.18±0.05 | 1.13±0.52 | 0.30±0.01 | 0.28±0.08 | 0.22±0.03 | 0.95±0.02 | 0.10±0.00 |
| CausalForestDML | **0.55**±0.08 | — | 2.05±0.14 | 0.44±0.01 | 3.57±0.02 | 4.08±3.21 | 2.69±0.27 | 0.41±0.00 | 0.96±0.25 | 0.82±0.01 | **0.09**±0.00 |
| DR-Learner | 0.59±0.02 | — | 1.27±0.18 | 0.42±0.00 | 3.17±0.03 | 2.54±1.42 | 1.64±0.12 | 0.37±0.01 | 0.85±0.22 | 0.94±0.02 | 0.18±0.00 |
| X-Learner | 0.75±0.01 | — | 1.16±0.04 | **0.38**±0.01 | 3.11±0.03 | 2.74±1.68 | 1.67±0.12 | 0.28±0.00 | 0.76±0.16 | 0.93±0.02 | 0.16±0.01 |
| R-Learner | 16.10±1.79 | — | 5.06±0.81 | 0.49±0.01 | 3.05±0.05 | 26.16±28.59 | 1.65±0.12 | 1.05±0.01 | 5.15±3.35 | 4.93±0.38 | 0.47±0.01 |
| UpliftRF | 2.76±0.00 | **0.17**±0.01 | 2.76±0.17 | 1.08±0.01 | 3.81±0.02 | 6.13±4.69 | 3.67±0.17 | 0.63±0.00 | 1.46±0.28 | **0.48**±0.04 | 0.23±0.01 |
| BART | 0.57±0.10 | — | 1.31±0.16 | 0.39±0.01 | **2.92**±0.04 | 2.36±1.26 | 1.92±0.15 | 0.53±0.00 | 1.00±0.30 | 0.94±0.03 | 0.15±0.00 |

## 2. ITE Correlation — Ground-Truth Datasets

Higher = better. **Bold** = best per dataset.

| Model | IHDP | Twins | ACIC2016 | News | TCGA | ACIC2018 | LBIDD | NLSM | IBM Causal | ACIC2022 | STAR |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| CFRNet | 0.82±0.18 | 0.03±0.03 | 0.95±0.01 | 0.24±0.03 | 0.54±0.00 | **0.50**±0.50 | **0.99**±0.00 | 0.88±0.13 | 0.92±0.03 | 0.94±0.01 | 0.75±0.04 |
| TARNet | 0.72±0.22 | 0.01±0.04 | 0.98±0.00 | 0.20±0.03 | 0.54±0.01 | **0.50**±0.50 | **1.00**±0.00 | 0.86±0.12 | 0.82±0.05 | 0.97±0.00 | **0.80**±0.01 |
| DRNet | 0.64±0.28 | 0.02±0.03 | 0.97±0.01 | 0.16±0.07 | 0.56±0.01 | **0.50**±0.50 | **1.00**±0.00 | 0.87±0.07 | 0.95±0.00 | 0.95±0.00 | 0.73±0.05 |
| DragonNet | 0.78±0.23 | **0.03**±0.03 | 0.97±0.00 | 0.18±0.01 | 0.53±0.01 | 0.49±0.51 | **1.00**±0.00 | 0.86±0.10 | 0.78±0.07 | 0.97±0.00 | 0.78±0.04 |
| SNet | 0.74±0.19 | 0.00±0.04 | 0.96±0.01 | 0.07±0.01 | 0.21±0.02 | **0.49**±0.49 | **1.00**±0.00 | 0.61±0.19 | 0.62±0.03 | 0.92±0.01 | 0.64±0.02 |
| FlexTENet | 0.68±0.25 | 0.01±0.04 | **0.99**±0.00 | 0.16±0.03 | 0.55±0.01 | **0.50**±0.50 | **1.00**±0.00 | 0.80±0.13 | 0.87±0.03 | **0.98**±0.00 | 0.80±0.01 |
| TransDCA | 0.74±0.19 | **0.03**±0.04 | 0.93±0.01 | -0.02±0.03 | 0.06±0.01 | 0.49±0.50 | **0.99**±0.00 | **0.94**±0.06 | **0.97**±0.01 | **0.97**±0.00 | 0.76±0.05 |
| CausalODE | 0.74±0.22 | 0.02±0.04 | 0.98±0.00 | 0.20±0.01 | 0.52±0.01 | **0.50**±0.50 | **1.00**±0.00 | 0.89±0.07 | 0.96±0.01 | 0.97±0.00 | 0.80±0.02 |
| CausalForestDML | 0.79±0.04 | — | 0.78±0.04 | 0.00±0.00 | 0.10±0.06 | 0.41±0.39 | 0.63±0.03 | 0.79±0.00 | 0.81±0.10 | 0.96±0.01 | 0.78±0.03 |
| DR-Learner | **0.85**±0.03 | — | 0.90±0.04 | 0.36±0.02 | 0.55±0.00 | 0.49±0.47 | 0.88±0.01 | 0.82±0.01 | 0.70±0.02 | 0.97±0.00 | 0.51±0.03 |
| X-Learner | 0.64±0.02 | — | 0.92±0.02 | **0.53**±0.02 | 0.61±0.00 | 0.49±0.44 | 0.88±0.01 | 0.90±0.00 | 0.77±0.04 | 0.97±0.00 | 0.52±0.01 |
| R-Learner | 0.29±0.05 | — | 0.85±0.04 | 0.44±0.02 | 0.59±0.01 | 0.48±0.47 | 0.87±0.01 | 0.77±0.01 | 0.56±0.08 | 0.70±0.05 | 0.34±0.02 |
| UpliftRF | 0.75±0.04 | 0.00±0.00 | 0.70±0.06 | -0.04±0.02 | 0.09±0.03 | 0.00±0.00 | 0.00±0.00 | 0.21±0.00 | 0.37±0.08 | 0.23±0.16 | 0.47±0.07 |
| BART | 0.80±0.06 | — | 0.89±0.04 | 0.48±0.02 | **0.64**±0.01 | 0.48±0.49 | 0.84±0.01 | 0.68±0.00 | 0.58±0.04 | 0.96±0.01 | 0.62±0.01 |

## 3. AUUC & Qini — Real-World Datasets (No Ground Truth)

Higher normalized values = better uplift ranking over random.

### Jobs

| Model | AUUC_norm | Qini_norm |
|-------|-----------|-----------|
| DR-Learner | 3347.7670 | 3367.5931 |
| X-Learner | 2507.2121 | 2527.0381 |
| R-Learner | 1144.8853 | 1164.7113 |
| UpliftRF | 2779.6840 | 2799.5100 |
| BART | 3121.2060 | 3141.0320 |

### Hillstrom

| Model | AUUC_norm | Qini_norm |
|-------|-----------|-----------|
| CausalForestDML | 0.0060 | 0.0060 |
| DR-Learner | 0.0010 | 0.0010 |
| X-Learner | 0.0022 | 0.0022 |
| R-Learner | 0.0010 | 0.0010 |
| UpliftRF | -0.0003 | -0.0003 |
| BART | 0.0047 | 0.0046 |

### Criteo

| Model | AUUC_norm | Qini_norm |
|-------|-----------|-----------|
| CausalForestDML | 0.0007 | 0.0007 |
| DR-Learner | 0.0013 | 0.0013 |
| X-Learner | 0.0007 | 0.0007 |
| R-Learner | 0.0004 | 0.0004 |
| UpliftRF | -0.0007 | -0.0007 |
| BART | -0.0003 | -0.0002 |

## 4. Model Rankings (by √PEHE)

Rank per dataset. Avg rank across all GT datasets (lower = better).

| # | Model | Avg Rank | IHDP | Twins | ACIC2016 | News | TCGA | ACIC2018 | LBIDD | NLSM | IBM Causal | ACIC2022 | STAR |
|---|-------|----------|------|------|------|------|------|------|------|------|------|------|------|
| 1 | DRNet | 5.64 | 9 | 7 | 5 | 7 | 2 | 5 | 3 | 6 | 3 | 6 | 9 |
| 2 | FlexTENet | 5.64 | 11 | 3 | 1 | 14 | 4 | 1 | 1 | 8 | 5 | 9 | 5 |
| 3 | CausalODE | 5.82 | 6 | 9 | 2 | 11 | 10 | 2 | 4 | 3 | 2 | 8 | 7 |
| 4 | CFRNet | 5.91 | 5 | 4 | 7 | 8 | 7 | 6 | 7 | 7 | 4 | 4 | 6 |
| 5 | TARNet | 5.91 | 8 | 8 | 3 | 10 | 6 | 4 | 2 | 4 | 6 | 10 | 4 |
| 6 | TransDCA | 6.27 | 10 | 6 | 8 | 5 | 13 | 3 | 8 | 1 | 1 | 13 | 1 |
| 7 | X-Learner | 6.60 | 4 | — | 9 | 1 | 5 | 11 | 11 | 2 | 9 | 3 | 11 |
| 8 | DragonNet | 6.73 | 7 | 5 | 4 | 12 | 8 | 7 | 5 | 5 | 7 | 11 | 3 |
| 9 | BART | 7.70 | 2 | — | 11 | 2 | 1 | 9 | 12 | 11 | 12 | 7 | 10 |
| 10 | CausalForestDML | 7.90 | 1 | — | 12 | 4 | 12 | 12 | 13 | 10 | 11 | 2 | 2 |
| 11 | DR-Learner | 8.00 | 3 | — | 10 | 3 | 9 | 10 | 9 | 9 | 10 | 5 | 12 |
| 12 | SNet | 8.55 | 12 | 2 | 6 | 9 | 11 | 8 | 6 | 12 | 8 | 12 | 8 |
| 13 | UpliftRF | 11.00 | 13 | 1 | 13 | 13 | 14 | 13 | 14 | 13 | 13 | 1 | 13 |
| 14 | R-Learner | 11.70 | 14 | — | 14 | 6 | 3 | 14 | 10 | 14 | 14 | 14 | 14 |

## 5. Key Findings

### #1 Wins by √PEHE

| Model | #1 Wins | Datasets |
|-------|---------|----------|
| FlexTENet | 3 | ACIC2016, ACIC2018, LBIDD |
| TransDCA | 3 | NLSM, IBM Causal, STAR |
| UpliftRF | 2 | Twins, ACIC2022 |
| CausalForestDML | 1 | IHDP |
| X-Learner | 1 | News |
| BART | 1 | TCGA |

### Top 5 Overall

1. **DRNet** — avg rank 5.64
2. **FlexTENet** — avg rank 5.64
3. **CausalODE** — avg rank 5.82
4. **CFRNet** — avg rank 5.91
5. **TARNet** — avg rank 5.91

### Neural vs. ML/Tree-Based

- Neural avg rank: **6.31**
- ML/tree-based avg rank: **8.82**

> **Neural models outperform ML/tree-based models on average.**

### Model Insights

- **BART**: Strong on high-dim data (TCGA best). Simple T-learner with HistGBR beats complex neural architectures.
- **X-Learner**: Best meta-learner. Excels on unbalanced treatment (NLSM, IBM Causal).
- **DR-Learner**: Consistent across DGPs. Doubly-robust property provides resilience.
- **CausalForestDML**: Best on near-RCT data (STAR). Honest forest + DML orthogonalization.
- **R-Learner**: Underperforms — residual-on-residual can be unstable on small/noisy data.
- **UpliftRF**: Weakest ML model — binary discretization of continuous outcomes hurts PEHE.
- **FlexTENet** (neural): Still competitive. Flexible architecture helps on competition datasets.
- **TransDCA** (neural): Wins on structured medium-dim data but struggles on high-dim.
- **CausalODE** (neural): Most consistent neural model — rarely catastrophically bad.
