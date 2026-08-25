# Causal Inference Benchmark

A comprehensive benchmark of neural network models for Individual Treatment Effect (ITE) estimation using PyTorch.

## 🎯 Models

| Model | Reference | Year |
|-------|-----------|------|
| CFRNet | Shalit et al., ICML 2017 | 2017 |
| GANITE | Yoon et al., ICLR 2018 | 2018 |
| CEVAE | Louizos et al., NeurIPS 2017 | 2017 |
| DRNet | Schwab et al., AAAI 2020 | 2020 |
| TARNet | Shalit et al., ICML 2017 | 2017 |
| SNet | Curth & van der Schaar, NeurIPS 2021 | 2021 |
| DragonNet | Shi et al., NeurIPS 2019 | 2019 |
| FlexTENet | Curth & van der Schaar, NeurIPS 2021 | 2021 |

## 📊 Datasets

| Dataset | Samples | Features | Description |
|---------|---------|----------|-------------|
| IHDP | 747 | 25 | Infant Health & Development Program (Hill 2011) |
| Twins | 10,000 | 53 | US Linked Birth/Infant Death Records |
| ACIC 2016 | 4,802 | 79 | Atlantic Causal Inference Competition |
| News | 5,000 | 3,477 | Semi-synthetic text-based features |
| TCGA | 9,659 | 4,000 | Semi-synthetic gene expression |

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/dongxu1-samsung/causal-inference-benchmark.git
cd causal-inference-benchmark

# Install dependencies
pip install -r requirements.txt

# Download/generate datasets
python scripts/download_data.py

# Run benchmark on specific dataset
python scripts/run_benchmark.py --datasets ihdp --ihdp-realizations 10

# Run full benchmark (all 5 datasets)
python scripts/run_benchmark.py --datasets ihdp twins acic2016 news tcga

# Generate markdown report
python scripts/generate_report.py
```

## 📁 Project Structure

```
causal-inference-benchmark/
├── models/
│   ├── __init__.py          # Data loaders (IHDP, Twins, ACIC, News, TCGA)
│   ├── cfrnet.py            # Counterfactual Regression Network
│   ├── ganite.py            # GAN for ITE
│   ├── cevae.py             # Causal Effect VAE
│   ├── drnet.py             # Dose-Response Network
│   └── catenets.py          # TARNet, SNet, DragonNet, FlexTENet
├── scripts/
│   ├── download_data.py     # Dataset download & generation
│   ├── run_benchmark.py     # Unified benchmark runner
│   └── generate_report.py   # Results → Markdown report
├── data/                    # Downloaded datasets (gitignored)
├── results/                 # JSON result files (gitignored)
├── configs/                 # Hyperparameter configs
├── RESULTS.md               # Full benchmark results
├── requirements.txt
└── README.md
```

## 📈 Key Results

| Model | IHDP (√PEHE↓) | Twins | ACIC 2016 | News | TCGA |
|-------|----------------|-------|-----------|------|------|
| CFRNet | 1.644 | 0.220 | 0.873 | 0.717 | **2.871** |
| GANITE | 2.004 | 0.250 | 2.036 | 0.780 | 3.126 |
| CEVAE | 5.710 | **0.173** | 3.348 | 0.605 | 3.966 |
| DRNet | 1.771 | 0.195 | — | **0.552** | 2.908 |
| TARNet | 1.184 | 0.197 | 0.688 | 0.867 | 2.950 |
| SNet | 1.675 | 0.181 | 0.859 | 0.654 | 3.287 |
| DragonNet | **1.109** | 0.202 | 0.628 | 0.796 | 2.986 |
| FlexTENet | 1.245 | 0.185 | **0.436** | 0.870 | 2.913 |

**Bold** = best per dataset. See [RESULTS.md](RESULTS.md) for full analysis.

## 🔑 Key Findings

1. **No single model dominates** — dataset properties determine the best model
2. **DragonNet/FlexTENet**: best for low-to-mid dimensional, structured confounding
3. **CFRNet/DRNet**: best for high-dimensional data (News, TCGA)
4. **CEVAE**: excels on binary outcomes (Twins) but struggles elsewhere
5. **TARNet**: surprisingly strong baseline — simple 2-head architecture

## ⚙️ Configuration

Edit `configs/` or pass arguments to `run_benchmark.py`:

```bash
# Custom epochs and architecture
python scripts/run_benchmark.py \
  --datasets ihdp \
  --ihdp-realizations 100 \
  --n-runs 5 \
  --models cfrnet tarnet dragonnet
```

## 📚 References

- Shalit et al. "Estimating individual treatment effect: generalization bounds and algorithms" (ICML 2017)
- Yoon et al. "GANITE: Estimation of Individualized Treatment Effects using Generative Adversarial Nets" (ICLR 2018)
- Louizos et al. "Causal Effect Inference with Deep Latent-Variable Models" (NeurIPS 2017)
- Schwab et al. "Learning Counterfactual Representations for Estimating Individual Dose-Response Curves" (AAAI 2020)
- Shi et al. "Adapting Neural Networks for the Estimation of Treatment Effects" (NeurIPS 2019)
- Curth & van der Schaar "Nonparametric Estimation of Heterogeneous Treatment Effects" (NeurIPS 2021)

## License

MIT
