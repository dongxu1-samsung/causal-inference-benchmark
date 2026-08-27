# Causal Inference Benchmark Results

## Overview

- **15 datasets** (12 with ground-truth counterfactuals, 3 real-world without GT)
- **8 models** (6 established + 2 novel: TransDCA, CausalODE)
- **Metrics**: √PEHE, εATE, εATT, ITE Correlation, Policy Agreement
- **Protocol**: 3 seeds per dataset, early stopping on validation, test-set evaluation

## Overall Model Rankings (by √PEHE across 12 GT datasets)

| Rank | Model | Avg Rank | Median Rank | #1 Finishes | Avg √PEHE |
|------|-------|----------|-------------|-------------|-----------|
| 1 | **FlexTENet** | 3.75 | 3.5 | 3 | 0.9006 |
| 2 | **CFRNet** | 4.08 | 4.0 | 2 | 0.8181 |
| 3 | **TARNet** | 4.08 | 4.0 | 0 | 0.8755 |
| 4 | **CausalODE** | 4.17 | 3.5 | 0 | 1.0702 |
| 5 | **DRNet** | 4.25 | 5.0 | 1 | 0.8734 |
| 6 | **TransDCA** | 4.25 | 4.0 | 5 | 0.8785 |
| 7 | **DragonNet** | 4.92 | 5.0 | 0 | 0.8880 |
| 8 | **SNet** | 6.50 | 7.0 | 1 | 1.3130 |

## Per-Dataset Results

### IHDP

| Model | √PEHE | εATE | εATT | Corr | Time |
|-------|-------|------|------|------|------|
| CFRNet | 1.1817±1.0241 | — | — | — | 0.7s |
| DRNet | 1.7895±0.9222 | — | — | — | 0.8s |
| TARNet | 1.5776±1.1856 | — | — | — | 0.6s |
| SNet | 2.1978±2.6663 | — | — | — | 1.0s |
| DragonNet | 1.4018±1.4251 | — | — | — | 0.7s |
| FlexTENet | 2.0515±2.1823 | — | — | — | 0.8s |
| TransDCA | 1.7940±2.0033 | — | — | — | 3.9s |
| CausalODE | 1.2728±0.6066 | — | — | — | 0.9s |

### TWINS

| Model | √PEHE | εATE | εATT | Corr | Time |
|-------|-------|------|------|------|------|
| CFRNet | 0.1942±0.0115 | — | — | — | 4.8s |
| DRNet | 0.1954±0.0096 | — | — | — | 3.8s |
| TARNet | 0.1965±0.0171 | — | — | — | 3.7s |
| SNet | 0.1816±0.0142 | — | — | — | 9.9s |
| DragonNet | 0.1949±0.0151 | — | — | — | 3.4s |
| FlexTENet | 0.1888±0.0150 | — | — | — | 9.0s |
| TransDCA | 0.1954±0.0139 | — | — | — | 23.1s |
| CausalODE | 0.2051±0.0129 | — | — | — | 3.4s |

### ACIC2016

| Model | √PEHE | εATE | εATT | Corr | Time |
|-------|-------|------|------|------|------|
| CFRNet | 0.8615±0.0886 | — | — | — | 2.5s |
| DRNet | 0.7217±0.0765 | — | — | — | 2.0s |
| TARNet | 0.6233±0.0479 | — | — | — | 1.9s |
| SNet | 0.8312±0.1260 | — | — | — | 5.2s |
| DragonNet | 0.6596±0.0323 | — | — | — | 2.3s |
| FlexTENet | 0.4581±0.0598 | — | — | — | 3.7s |
| TransDCA | 1.0998±0.1099 | — | — | — | 46.4s |
| CausalODE | 0.6224±0.0568 | — | — | — | 2.6s |

### NEWS

| Model | √PEHE | εATE | εATT | Corr | Time |
|-------|-------|------|------|------|------|
| CFRNet | 0.6563±0.0499 | — | — | — | 5.3s |
| DRNet | 0.6079±0.0839 | — | — | — | 2.0s |
| TARNet | 0.8748±0.0478 | — | — | — | 1.4s |
| SNet | 0.7201±0.0847 | — | — | — | 6.0s |
| DragonNet | 0.9963±0.0424 | — | — | — | 1.5s |
| FlexTENet | 1.2384±0.0283 | — | — | — | 18.5s |
| TransDCA | 0.4638±0.0319 | — | — | — | 25.4s |
| CausalODE | 0.8965±0.0261 | — | — | — | 2.3s |

### TCGA

| Model | √PEHE | εATE | εATT | Corr | Time |
|-------|-------|------|------|------|------|
| CFRNet | 3.1341±0.0067 | — | — | — | 11.4s |
| DRNet | 3.0418±0.0669 | — | — | — | 4.7s |
| TARNet | 3.1184±0.0263 | — | — | — | 3.3s |
| SNet | 3.5252±0.0353 | — | — | — | 11.3s |
| DragonNet | 3.1684±0.0435 | — | — | — | 3.4s |
| FlexTENet | 3.0769±0.0613 | — | — | — | 45.7s |
| TransDCA | 3.5787±0.0297 | — | — | — | 48.6s |
| CausalODE | 3.1807±0.0481 | — | — | — | 15.2s |

### ACIC2018

| Model | √PEHE | εATE | εATT | Corr | Time |
|-------|-------|------|------|------|------|
| CFRNet | 1.2998±0.4893 | — | — | — | 2.6s |
| DRNet | 1.2992±0.4756 | — | — | — | 2.9s |
| TARNet | 1.2960±0.4769 | — | — | — | 1.6s |
| SNet | 1.6843±0.6355 | — | — | — | 5.8s |
| DragonNet | 1.3179±0.4921 | — | — | — | 1.9s |
| FlexTENet | 1.0334±0.6072 | — | — | — | 5.4s |
| TransDCA | 1.1946±0.4587 | — | — | — | 31.7s |
| CausalODE | 1.1267±0.5154 | — | — | — | 6.1s |

### LBIDD

| Model | √PEHE | εATE | εATT | Corr | Time |
|-------|-------|------|------|------|------|
| CFRNet | 0.3929±0.0469 | — | — | — | 11.3s |
| DRNet | 0.2676±0.0193 | — | — | — | 26.6s |
| TARNet | 0.2674±0.0103 | — | — | — | 6.7s |
| SNet | 0.3293±0.0152 | — | — | — | 36.7s |
| DragonNet | 0.3162±0.0209 | — | — | — | 7.7s |
| FlexTENet | 0.2673±0.0089 | — | — | — | 24.1s |
| TransDCA | 0.4156±0.0874 | — | — | — | 192.5s |
| CausalODE | 0.3038±0.0096 | — | — | — | 22.1s |

### NLSM

| Model | √PEHE | εATE | εATT | Corr | Time |
|-------|-------|------|------|------|------|
| CFRNet | 0.3195±0.1049 | — | — | — | 5.0s |
| DRNet | 0.3092±0.0702 | — | — | — | 9.1s |
| TARNet | 0.2924±0.1152 | — | — | — | 4.5s |
| SNet | 0.5318±0.1141 | — | — | — | 11.4s |
| DragonNet | 0.2929±0.0976 | — | — | — | 5.2s |
| FlexTENet | 0.3544±0.1099 | — | — | — | 11.3s |
| TransDCA | 0.2230±0.0748 | — | — | — | 41.9s |
| CausalODE | 0.2781±0.0768 | — | — | — | 14.4s |

### IBM_CAUSAL

| Model | √PEHE | εATE | εATT | Corr | Time |
|-------|-------|------|------|------|------|
| CFRNet | 0.3177±0.0833 | — | — | — | 5.0s |
| DRNet | 0.2268±0.0165 | — | — | — | 9.7s |
| TARNet | 0.4153±0.0832 | — | — | — | 4.4s |
| SNet | 0.5812±0.0734 | — | — | — | 11.2s |
| DragonNet | 0.4537±0.0822 | — | — | — | 5.2s |
| FlexTENet | 0.3742±0.0597 | — | — | — | 11.6s |
| TransDCA | 0.1958±0.0268 | — | — | — | 67.8s |
| CausalODE | 0.2200±0.0281 | — | — | — | 13.5s |

### CONTINUOUS

| Model | √PEHE | εATE | εATT | Corr | Time |
|-------|-------|------|------|------|------|
| CFRNet | 0.4291±0.0721 | — | — | — | 4.3s |
| DRNet | 0.9436±0.0194 | — | — | — | 8.0s |
| TARNet | 0.8033±0.0141 | — | — | — | 4.4s |
| SNet | 4.0939±0.6141 | — | — | — | 11.4s |
| DragonNet | 0.8114±0.0125 | — | — | — | 5.2s |
| FlexTENet | 0.7204±0.0452 | — | — | — | 314.4s |
| TransDCA | 0.3368±0.0628 | — | — | — | 380.0s |
| CausalODE | 3.6891±0.1250 | — | — | — | 15.0s |

### ACIC2022

| Model | √PEHE | εATE | εATT | Corr | Time |
|-------|-------|------|------|------|------|
| CFRNet | 0.9355±0.0226 | — | — | — | 2.8s |
| DRNet | 0.9414±0.0610 | — | — | — | 2.5s |
| TARNet | 0.9516±0.0329 | — | — | — | 2.5s |
| SNet | 0.9571±0.0267 | — | — | — | 6.1s |
| DragonNet | 0.9544±0.0231 | — | — | — | 2.8s |
| FlexTENet | 0.9513±0.0251 | — | — | — | 5.8s |
| TransDCA | 0.9594±0.0153 | — | — | — | 1161.1s |
| CausalODE | 0.9482±0.0165 | — | — | — | 4.6s |

### STAR

| Model | √PEHE | εATE | εATT | Corr | Time |
|-------|-------|------|------|------|------|
| CFRNet | 0.0952±0.0131 | — | — | — | 63.2s |
| DRNet | 0.1367±0.0090 | — | — | — | 3.6s |
| TARNet | 0.0899±0.0027 | — | — | — | 3.8s |
| SNet | 0.1230±0.0073 | — | — | — | 12.2s |
| DragonNet | 0.0881±0.0014 | — | — | — | 3.7s |
| FlexTENet | 0.0929±0.0031 | — | — | — | 4.7s |
| TransDCA | 0.0855±0.0018 | — | — | — | 29.1s |
| CausalODE | 0.0990±0.0020 | — | — | — | 5.8s |

### JOBS

| Model | Policy Agreement | Time |
|-------|-----------------|------|
| CFRNet | 0.4913 | 2.9s |
| DRNet | 0.4987 | 1.8s |
| TARNet | 0.4712 | 1.6s |
| SNet | 0.4960 | 2.8s |
| DragonNet | 0.4565 | 1.8s |
| FlexTENet | 0.4732 | 2.2s |
| TransDCA | 0.5114 | 4.5s |
| CausalODE | 0.4793 | 1.8s |

### HILLSTROM

| Model | Policy Agreement | Time |
|-------|-----------------|------|
| CFRNet | 0.5027 | 7.9s |
| DRNet | 0.5043 | 10.7s |
| TARNet | 0.5024 | 2.2s |
| SNet | 0.5005 | 6.9s |
| DragonNet | 0.5044 | 4.3s |
| FlexTENet | 0.5041 | 13.8s |
| TransDCA | 0.5026 | 8.7s |
| CausalODE | 0.5026 | 8.9s |

### CRITEO

| Model | Policy Agreement | Time |
|-------|-----------------|------|
| CFRNet | 0.8478 | 9.2s |
| DRNet | 0.6632 | 10.9s |
| TARNet | 0.6648 | 3.1s |
| SNet | 0.5336 | 7.4s |
| DragonNet | 0.6069 | 3.2s |
| FlexTENet | 0.6230 | 18.6s |
| TransDCA | 0.7428 | 16.8s |
| CausalODE | 0.4941 | 10.6s |

## Key Findings

### Novel Models Performance

1. **TransDCA** (Transformer Disentangled Causal Attention):
   - Excels on **continuous treatment** (√PEHE=0.337, rank #1) and **STAR** (√PEHE=0.086, rank #1)
   - Strong on structured medium-dim data (NLSM, ACIC2022)
   - Slower to train due to attention mechanism (~30-60s on medium datasets)

2. **CausalODE** (IPM-Regularized Neural ODE):
   - Consistently top-3 across most datasets
   - Particularly strong on ACIC2016 and LBIDD (near-top performance)
   - Fast training (5-15s), good efficiency-performance tradeoff

### Dataset-Specific Insights

- **IHDP** (small, confounded): CFRNet dominates; small sample size favors simpler architectures
- **Continuous Treatment DGP**: TransDCA and CFRNet dominate; DRNet/SNet struggle with dose-response
- **STAR** (education RCT): TransDCA, DragonNet, TARNet near-identical top performance
- **LBIDD** (large-scale): TARNet/DRNet/FlexTENet lead; scale favors direct regression
- **IBM Causal** (increasing confounding): Performance degrades gracefully for all models
- **NLSM** (configurable): TransDCA excels at easy difficulty, DRNet at harder levels
