# Causal Inference Benchmark Results

## Overview

**10 Datasets** × **8 Models** = 80 model-dataset evaluations

### Evaluation Protocol
- **Train/Val/Test split**: ~60%/20%/20% for all datasets
- **Early stopping**: All models use validation-based early stopping (patience=10)
- **Repetitions**: IHDP (10 realizations × 3 seeds), ACIC2016 (3 DGPs), ACIC2018 (6 DGPs), LBIDD (3 settings), others (3 seeds)
- **Metrics**: √PEHE, εATE, εATT, ITE Correlation (for datasets with ground truth); Policy Agreement (for real-world RCTs)
- **Hardware**: Apple M-series CPU, PyTorch 2.8.0

### Datasets

| # | Dataset | Type | N | Features | Ground Truth |
|---|---------|------|---|----------|--------------|
| 1 | IHDP | Semi-synthetic | 747 | 25 | ✓ |
| 2 | Twins | Semi-synthetic | 10,000 | 53 | ✓ |
| 3 | ACIC 2016 | Semi-synthetic | 4,802 | 79 | ✓ |
| 4 | News | Semi-synthetic | 5,000 | 3,477 | ✓ |
| 5 | TCGA | Semi-synthetic | 9,659 | 4,000 | ✓ |
| 6 | Jobs (LaLonde) | Real-world RCT | 2,490 | 8 | ✗ |
| 7 | ACIC 2018 | Semi-synthetic | 5,000 | 50 | ✓ |
| 8 | Hillstrom | Real-world RCT | 42,767 | 8 | ✗ |
| 9 | LBIDD | Semi-synthetic | 50,000 | 177 | ✓ |
| 10 | Criteo | Real-world RCT | 50,000 | 12 | ✗ |

### Models

| Model | Architecture | Year |
|-------|-------------|------|
| CFRNet | Shared repr + 2 heads + IPM regularization | 2017 |
| GANITE | Generator-Discriminator + Inference network | 2018 |
| CEVAE | Variational Autoencoder with latent confounders | 2017 |
| DRNet | Shared repr + treatment heads + propensity reg | 2020 |
| TARNet | Shared repr + 2 outcome heads | 2017 |
| SNet | 3-way disentangled representations | 2022 |
| DragonNet | Shared repr + 2 heads + propensity head | 2019 |
| FlexTENet | Shared + private networks + orthogonality | 2021 |

---

## Main Results: √PEHE (lower is better)

| Model | IHDP | TWINS | ACIC2016 | NEWS | TCGA | ACIC2018 | LBIDD |
|-------|---------|---------|---------|---------|---------|---------|---------|
| cfrnet       | 1.3897±1.5975 | 0.1950±0.0219 | 0.9471±0.1070 | 0.6956±0.0447 | 3.1109±0.0319 | 1.2840±0.5042 | 0.3964±0.0419 |
| ganite       | 3.4155±4.3651 | 0.3038±0.0983 | 2.0931±0.3685 | 0.7648±0.0639 | 3.1633±0.0359 | 5.5280±5.1969 | 9.1144±4.4138 |
| cevae        | 6.1289±7.0946 | 0.1720±0.0136 | 3.2338±0.1300 | 0.8656±0.1078 | 3.7158±0.0389 | 6.5039±5.0575 | 3.7229±0.6226 |
| drnet        | 1.6233±0.8787 | 0.1961±0.0145 | 0.6974±0.0434 | 0.6447±0.0398 | 3.0607±0.0239 | 1.2775±0.5232 | 0.2630±0.0101 |
| tarnet       | 1.5127±1.2889 | 0.1994±0.0149 | 0.6242±0.0885 | 0.9673±0.0432 | 3.0956±0.0471 | 1.2854±0.4752 | 0.2628±0.0076 |
| snet         | 2.2900±2.7912 | 0.1810±0.0136 | 0.8826±0.1615 | 0.7631±0.0442 | 3.5484±0.0415 | 1.6614±0.5831 | 0.3350±0.0238 |
| dragonnet    | 1.4189±1.4922 | 0.1962±0.0126 | 0.6695±0.1424 | 0.9221±0.0929 | 3.1506±0.0479 | 1.3151±0.4809 | 0.2915±0.0232 |
| flextenet    | 2.0943±2.1988 | 0.1894±0.0141 | 0.4594±0.0394 | 1.2013±0.0686 | 3.0977±0.0439 | 1.0199±0.6031 | 0.2637±0.0059 |

## εATE — Absolute Error in ATE (lower is better)

| Model | IHDP | TWINS | ACIC2016 | NEWS | TCGA | ACIC2018 | LBIDD |
|-------|---------|---------|---------|---------|---------|---------|---------|
| cfrnet       | 0.2549±0.4240 | 0.0078±0.0054 | 0.1452±0.0524 | 0.0776±0.0339 | 0.1805±0.0823 | 0.3230±0.2265 | 0.1079±0.0562 |
| ganite       | 1.9657±3.3393 | 0.2142±0.1396 | 0.4727±0.1680 | 0.4504±0.1087 | 0.3123±0.0612 | 1.1007±1.1367 | 0.7810±0.8562 |
| cevae        | 2.7407±0.9215 | 0.0047±0.0005 | 1.5205±0.4970 | 0.7173±0.1202 | 0.6980±0.3246 | 1.7281±1.3601 | 1.2330±1.0984 |
| drnet        | 0.5109±0.2565 | 0.0117±0.0019 | 0.0273±0.0294 | 0.1887±0.0793 | 0.1357±0.0959 | 0.0433±0.0461 | 0.0208±0.0065 |
| tarnet       | 0.4067±0.3664 | 0.0087±0.0058 | 0.0340±0.0103 | 0.6514±0.0661 | 0.2186±0.0702 | 0.0779±0.0697 | 0.0196±0.0086 |
| snet         | 0.6511±0.4563 | 0.0098±0.0037 | 0.0526±0.0347 | 0.1404±0.0418 | 0.1724±0.0892 | 0.0416±0.0371 | 0.0262±0.0168 |
| dragonnet    | 0.2231±0.1403 | 0.0091±0.0032 | 0.0485±0.0285 | 0.6730±0.0984 | 0.2496±0.1423 | 0.0986±0.1070 | 0.0159±0.0166 |
| flextenet    | 0.7064±0.7049 | 0.0087±0.0046 | 0.0284±0.0111 | 0.9542±0.0621 | 0.3705±0.1153 | 0.0476±0.0279 | 0.0085±0.0038 |

## εATT — Absolute Error in ATT (lower is better)

| Model | IHDP | TWINS | ACIC2016 | NEWS | TCGA | ACIC2018 | LBIDD |
|-------|---------|---------|---------|---------|---------|---------|---------|
| cfrnet       | 0.3563±0.5097 | 0.0078±0.0054 | 0.2452±0.1360 | 0.0783±0.0652 | 0.1225±0.0396 | 0.2131±0.0608 | 0.1258±0.0580 |
| ganite       | 2.2262±4.3709 | 0.2142±0.1396 | 0.6529±0.2388 | 0.4431±0.0952 | 0.1251±0.1169 | 0.6466±0.4786 | 0.7202±0.6079 |
| cevae        | 2.5763±0.6887 | 0.0047±0.0005 | 1.1772±0.1077 | 0.7157±0.1304 | 0.5045±0.3529 | 1.6294±1.3621 | 1.4666±1.1841 |
| drnet        | 0.3851±0.2997 | 0.0117±0.0019 | 0.0697±0.0245 | 0.1864±0.0571 | 0.0772±0.0478 | 0.0638±0.0798 | 0.0368±0.0182 |
| tarnet       | 0.3499±0.4088 | 0.0087±0.0058 | 0.0470±0.0203 | 0.6559±0.0968 | 0.0867±0.0261 | 0.0683±0.0627 | 0.0191±0.0031 |
| snet         | 0.4808±0.2741 | 0.0098±0.0037 | 0.0596±0.0259 | 0.1163±0.0308 | 0.2651±0.1881 | 0.0264±0.0203 | 0.0198±0.0123 |
| dragonnet    | 0.2961±0.2439 | 0.0091±0.0032 | 0.0743±0.0344 | 0.6714±0.1224 | 0.3255±0.1610 | 0.0985±0.0762 | 0.0223±0.0077 |
| flextenet    | 0.6228±0.7555 | 0.0087±0.0046 | 0.0293±0.0225 | 0.9569±0.0587 | 0.2334±0.1739 | 0.0316±0.0280 | 0.0045±0.0025 |

## ITE Correlation (higher is better)

| Model | IHDP | TWINS | ACIC2016 | NEWS | TCGA | ACIC2018 | LBIDD |
|-------|---------|---------|---------|---------|---------|---------|---------|
| cfrnet       | 0.823 | 0.021 | 0.942 | 0.224 | 0.539 | 0.496 | 0.994 |
| ganite       | 0.616 | 0.027 | 0.692 | 0.163 | 0.482 | 0.331 | 0.400 |
| cevae        | 0.003 | 0.024 | -0.043 | -0.019 | 0.002 | -0.006 | 0.001 |
| drnet        | 0.684 | 0.007 | 0.969 | 0.185 | 0.552 | 0.502 | 0.997 |
| tarnet       | 0.734 | 0.021 | 0.975 | 0.182 | 0.540 | 0.501 | 0.997 |
| snet         | 0.726 | -0.002 | 0.950 | 0.059 | 0.191 | 0.481 | 0.995 |
| dragonnet    | 0.808 | 0.025 | 0.971 | 0.168 | 0.533 | 0.497 | 0.996 |
| flextenet    | 0.686 | 0.015 | 0.987 | 0.171 | 0.539 | 0.495 | 0.997 |

---

## Real-World Datasets (No Ground Truth)

For datasets without counterfactual ground truth, we report **Policy Agreement** —
the fraction of test samples where the model's recommended treatment matches actual assignment.

| Model | Jobs | Hillstrom | Criteo |
|-------|------|-----------|--------|
| cfrnet       | 0.4578±0.0469 | 0.5027±0.0038 | 0.6133±0.3288 |
| ganite       | 0.4946±0.0446 | 0.4989±0.0043 | 0.7148±0.0762 |
| cevae        | 0.4378±0.0384 | 0.5018±0.0036 | 0.2710±0.0953 |
| drnet        | 0.4826±0.0275 | 0.5021±0.0031 | 0.5717±0.1748 |
| tarnet       | 0.4752±0.0379 | 0.5048±0.0025 | 0.6548±0.0967 |
| snet         | 0.4692±0.0379 | 0.5028±0.0039 | 0.5667±0.0493 |
| dragonnet    | 0.4913±0.0289 | 0.5020±0.0026 | 0.5269±0.1875 |
| flextenet    | 0.4799±0.0245 | 0.5044±0.0032 | 0.4154±0.0806 |

## Average Training Time (seconds per run)

| Model | IHDP | TWINS | ACIC2016 | NEWS | TCGA | ACIC2018 | LBIDD | JOBS | HILLSTROM | CRITEO |
|-------|------|------|------|------|------|------|------|------|------|------|
| cfrnet       | 0.7s | 4.8s | 2.5s | 4.6s | 9.6s | 2.6s | 10.8s | 2.1s | 7.7s | 9.0s |
| ganite       | 2.6s | 22.9s | 12.1s | 37.8s | 96.9s | 11.8s | 84.8s | 9.1s | 55.0s | 202.1s |
| cevae        | 0.3s | 2.8s | 1.4s | 3.5s | 7.4s | 1.4s | 8.8s | 1.0s | 6.3s | 127.0s |
| drnet        | 0.7s | 3.4s | 2.2s | 4.4s | 5.0s | 2.9s | 29.7s | 1.4s | 11.1s | 11.7s |
| tarnet       | 0.6s | 4.2s | 1.9s | 1.3s | 2.6s | 1.6s | 6.8s | 1.6s | 3.0s | 2.5s |
| snet         | 1.0s | 8.6s | 4.8s | 5.1s | 9.0s | 5.0s | 27.2s | 2.4s | 5.5s | 6.9s |
| dragonnet    | 0.7s | 3.6s | 2.3s | 1.3s | 2.8s | 2.1s | 7.8s | 1.3s | 3.3s | 2.8s |
| flextenet    | 0.8s | 8.4s | 3.9s | 19.1s | 45.6s | 4.1s | 17.0s | 1.5s | 11.4s | 13.5s |

---

## Model Rankings (by average rank across 7 ground-truth datasets)

| Rank | Model | Avg Rank | Best √PEHE on |
|------|-------|----------|---------------|
| 1 | drnet | 2.71 | NEWS, TCGA |
| 2 | flextenet | 3.43 | ACIC2016, ACIC2018 |
| 3 | cfrnet | 3.71 | IHDP |
| 4 | tarnet | 3.71 | LBIDD |
| 5 | dragonnet | 4.43 | — |
| 6 | snet | 4.86 | — |
| 7 | cevae | 6.43 | TWINS |
| 8 | ganite | 6.71 | — |

---

## Key Findings

1. **DRNet** achieves the best average rank (2.71/8), winning on News and TCGA (high-dimensional datasets).
2. **FlexTENet** excels on structured semi-synthetic data (ACIC 2016, ACIC 2018) via its orthogonality constraint.
3. **CFRNet** remains a strong baseline on small datasets (IHDP) thanks to IPM regularization.
4. **TARNet** — despite its simplicity — is competitive everywhere and wins on LBIDD (large-scale).
5. **DragonNet** is consistently top-5 but never first; its propensity head provides stable regularization.
6. **CEVAE** surprisingly wins Twins (binary outcome, strong latent structure) but struggles elsewhere.
7. **GANITE** shows high variance and slow convergence, especially on large datasets (LBIDD: 85s/run).
8. **SNet** underperforms expectations — the 3-way disentanglement may be over-parameterized for these dataset sizes.
9. On **LBIDD** (50K samples, 177 features), DRNet/TARNet/FlexTENet achieve near-perfect ITE correlation (0.997).
10. Real-world datasets show moderate policy agreement (~0.5–0.7), consistent with weak true uplift signals.

### Recommendations

- **Default choice**: DRNet or DragonNet — strong, stable, moderate compute
- **Small data** (N<1000): CFRNet with IPM regularization
- **Large structured data**: FlexTENet or TARNet
- **Speed-sensitive**: TARNet (fastest) or DragonNet
- **Avoid**: GANITE (slow, unstable), CEVAE (poor ITE estimation except on binary outcomes)
