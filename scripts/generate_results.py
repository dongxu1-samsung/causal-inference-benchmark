#!/usr/bin/env python3
"""Generate RESULTS.md from merged v4 results."""
import json
import numpy as np

with open('results/all_results_v4.json') as f:
    merged = json.load(f)

neural_models = ['cfrnet', 'tarnet', 'drnet', 'dragonnet', 'snet', 'flextenet', 'transdca', 'causal_ode', 'descn', 'moca', 'ddrnet']
ml_models = ['causal_forest_dml', 'dr_learner', 'x_learner', 'r_learner', 'uplift_rf', 'bart']
all_models = neural_models + ml_models

dn = {
    'cfrnet':'CFRNet','tarnet':'TARNet','drnet':'DRNet','dragonnet':'DragonNet',
    'snet':'SNet','flextenet':'FlexTENet','transdca':'TransDCA','causal_ode':'CausalODE',
    'descn':'DESCN','moca':'MOCA','ddrnet':'DDRNet',
    'causal_forest_dml':'CausalForestDML','dr_learner':'DR-Learner',
    'x_learner':'X-Learner','r_learner':'R-Learner','uplift_rf':'UpliftRF','bart':'BART',
}

gt_ds = ['ihdp','twins','acic2016','news','tcga','acic2018','lbidd','nlsm','ibm_causal','continuous','acic2022','star']
nogt_ds = ['jobs','hillstrom','criteo']
dd = {'ihdp':'IHDP','twins':'Twins','acic2016':'ACIC2016','news':'News','tcga':'TCGA',
      'acic2018':'ACIC2018','lbidd':'LBIDD','nlsm':'NLSM','ibm_causal':'IBM Causal',
      'continuous':'Continuous','acic2022':'ACIC2022','star':'STAR',
      'jobs':'Jobs','hillstrom':'Hillstrom','criteo':'Criteo'}

def get_val(ds, mn, metric):
    if ds in merged and mn in merged[ds]:
        m, s = f'{metric}_mean', f'{metric}_std'
        if m in merged[ds][mn]:
            return merged[ds][mn][m], merged[ds][mn][s]
    return None, None

def best_val(ds, metric, lower=True):
    best = float('inf') if lower else -float('inf')
    for mn in all_models:
        v, _ = get_val(ds, mn, metric)
        if v is not None:
            best = min(best, v) if lower else max(best, v)
    return best

# Rankings
rank_matrix = {}
for ds in gt_ds:
    vals = [(v, mn) for mn in all_models for v, _ in [get_val(ds, mn, 'pehe')] if v is not None]
    vals.sort()
    for rank, (_, mn) in enumerate(vals, 1):
        rank_matrix.setdefault(mn, {})[ds] = rank

avg_ranks = {mn: np.mean(list(ranks.values())) for mn, ranks in rank_matrix.items()}
sorted_m = sorted(avg_ranks.keys(), key=lambda m: avg_ranks[m])

wins = {}
win_details = {}
for ds in gt_ds:
    vals = [(merged[ds][mn]['pehe_mean'], mn) for mn in all_models
            if ds in merged and mn in merged[ds] and 'pehe_mean' in merged[ds][mn]]
    if vals:
        vals.sort()
        wins[vals[0][1]] = wins.get(vals[0][1], 0) + 1
        win_details.setdefault(vals[0][1], []).append(dd[ds])

# Build markdown
L = []
L.append("# Causal Inference Benchmark Results (v4)\n")
L.append("**17 models x 15 datasets** - 11 neural + 6 ML/tree-based models\n")

L.append("## Model Categories\n")
L.append("| Category | Models | Package |")
L.append("|----------|--------|---------|")
cats = [
    ("Balanced Representation", "CFRNet, TARNet", "PyTorch"),
    ("Targeted Regularization", "DragonNet, SNet, FlexTENet", "PyTorch"),
    ("Multi-dose Neural", "DRNet", "PyTorch"),
    ("Transformer Disentangle", "TransDCA *", "PyTorch"),
    ("Neural ODE", "CausalODE *", "PyTorch"),
    ("Entire Space Cross-Net", "DESCN", "PyTorch"),
    ("Modular Cross-Attention", "MOCA", "PyTorch"),
    ("MoE Disentanglement", "DDRNet", "PyTorch"),
    ("Forest + Double ML", "CausalForestDML", "EconML"),
    ("Doubly-Robust Meta", "DR-Learner", "CausalML + LGBM"),
    ("Cross-Imputation Meta", "X-Learner", "CausalML + XGBoost"),
    ("Residual Meta", "R-Learner", "CausalML + LGBM"),
    ("Uplift Tree Ensemble", "UpliftRF", "CausalML"),
    ("T-learner (BART-style)", "BART", "sklearn HistGBR"),
]
for cat, mods, pkg in cats:
    L.append(f"| {cat} | {mods} | {pkg} |")
L.append("")
L.append("\\* = Novel architectures implemented in this benchmark\n")

# PEHE table
L.append("## 1. sqrt(PEHE) - Ground-Truth Datasets\n")
L.append("Lower = better. **Bold** = best per dataset.\n")
h = "| Model |" + "".join(f" {dd[d]} |" for d in gt_ds)
s = "|-------|" + "".join("--------|" for _ in gt_ds)
L.append(h); L.append(s)
for mn in all_models:
    row = f"| {dn[mn]} |"
    for ds in gt_ds:
        v, sd = get_val(ds, mn, 'pehe')
        if v is not None:
            b = best_val(ds, 'pehe', lower=True)
            if abs(v - b) < 0.005:
                row += f" **{v:.2f}**+/-{sd:.2f} |"
            else:
                row += f" {v:.2f}+/-{sd:.2f} |"
        else:
            row += " - |"
    L.append(row)
L.append("")

# ITE Correlation table
L.append("## 2. ITE Correlation - Ground-Truth Datasets\n")
L.append("Higher = better. **Bold** = best per dataset.\n")
h = "| Model |" + "".join(f" {dd[d]} |" for d in gt_ds)
s = "|-------|" + "".join("--------|" for _ in gt_ds)
L.append(h); L.append(s)
for mn in all_models:
    row = f"| {dn[mn]} |"
    for ds in gt_ds:
        v, sd = get_val(ds, mn, 'ite_corr')
        if v is not None:
            b = best_val(ds, 'ite_corr', lower=False)
            if abs(v - b) < 0.005:
                row += f" **{v:.2f}**+/-{sd:.2f} |"
            else:
                row += f" {v:.2f}+/-{sd:.2f} |"
        else:
            row += " - |"
    L.append(row)
L.append("")

# AUUC/Qini
L.append("## 3. AUUC and Qini - Real-World Datasets (No Ground Truth)\n")
L.append("Higher normalized values = better uplift ranking over random.\n")
for ds in nogt_ds:
    L.append(f"### {dd[ds]}\n")
    L.append("| Model | AUUC_norm | Qini_norm |")
    L.append("|-------|-----------|-----------|")
    for mn in all_models:
        if ds in merged and mn in merged[ds] and 'auuc_norm_mean' in merged[ds][mn]:
            r = merged[ds][mn]
            L.append(f"| {dn[mn]} | {r['auuc_norm_mean']:.4f} | {r['qini_norm_mean']:.4f} |")
    L.append("")

# Rankings table
L.append("## 4. Model Rankings (by sqrt(PEHE))\n")
L.append("Rank per dataset, averaged across GT datasets where model ran.\n")
L.append("| # | Model | Avg Rank | #1 Wins | Datasets Ranked |")
L.append("|---|-------|----------|---------|-----------------|")
for i, mn in enumerate(sorted_m, 1):
    L.append(f"| {i} | {dn[mn]} | {avg_ranks[mn]:.2f} | {wins.get(mn,0)} | {len(rank_matrix[mn])}/12 |")
L.append("")

# Key findings
L.append("## 5. Key Findings\n")
L.append("### #1 Wins by sqrt(PEHE)\n")
L.append("| Model | #1 Wins | Datasets |")
L.append("|-------|---------|----------|")
for mn, datasets in sorted(win_details.items(), key=lambda x: -len(x[1])):
    L.append(f"| {dn[mn]} | {len(datasets)} | {', '.join(datasets)} |")
L.append("")

L.append("### Top 5 Overall\n")
for i, mn in enumerate(sorted_m[:5], 1):
    L.append(f"{i}. **{dn[mn]}** - avg rank {avg_ranks[mn]:.2f} (across {len(rank_matrix[mn])} datasets)")
L.append("")

na = np.mean([avg_ranks[mn] for mn in neural_models if mn in avg_ranks])
ma = np.mean([avg_ranks[mn] for mn in ml_models if mn in avg_ranks])
L.append("### Neural vs. ML/Tree-Based\n")
L.append(f"- Neural (11 models) avg rank: **{na:.2f}**")
L.append(f"- ML/tree-based (6 models) avg rank: **{ma:.2f}**")
verdict = "ML/tree-based" if ma < na else "Neural"
L.append(f"\n> **{verdict} models outperform on average.**")
L.append("")

L.append("### New Models (v4): DESCN, MOCA, DDRNet\n")
for mn in ['descn', 'moca', 'ddrnet']:
    L.append(f"- **{dn[mn]}**: avg rank {avg_ranks.get(mn,0):.2f}, {wins.get(mn,0)} #1 wins")
L.append("")

L.append("### Model Insights\n")
for ins in [
    "**DDRNet**: Strong disentanglement via MoE. Competitive on high-dim (TCGA) and structured data. Best new model.",
    "**MOCA**: Cross-attention helps on structured data but slower than simpler architectures.",
    "**DESCN**: Lightweight 5-head design. Fast but mediocre - cross-constraints don't always help.",
    "**FlexTENet**: Dominates competition datasets (ACIC2016, ACIC2018, LBIDD).",
    "**TransDCA**: Wins on structured medium-dim data (NLSM, IBM Causal, STAR).",
    "**CausalODE**: Most consistent neural model - rarely catastrophically bad.",
    "**BART**: Best ML model overall - simple T-learner with HistGBR is surprisingly effective.",
    "**CausalForestDML**: Wins IHDP (small sample). Industry standard for a reason.",
    "**R-Learner**: Consistently worst - residual-on-residual unstable on most benchmarks.",
]:
    L.append(f"- {ins}")
L.append("")

with open('RESULTS.md', 'w') as f:
    f.write("\n".join(L))

# Print summary
print(f"Written RESULTS.md: {len(L)} lines\n")
print("=== RANKINGS ===")
for i, mn in enumerate(sorted_m, 1):
    print(f"  {i:2d}. {dn[mn]:20s} avg_rank={avg_ranks[mn]:.2f}  wins={wins.get(mn,0)}  ds={len(rank_matrix[mn])}/12")
print(f"\nNeural avg rank: {na:.2f}")
print(f"ML avg rank: {ma:.2f}")
