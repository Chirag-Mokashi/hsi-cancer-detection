# coding: utf-8
# 6_ablation_analysis.py
# Step 6: Ablation study analysis and plots
#
# Reads hybridSN_v1_ablation.csv and vit_v1_ablation.csv.
# Generates patch-size sweep plots for HybridSN and ViT.
# Also reads main results CSVs for band-count sweep analysis.
#
# Run after 4c (HybridSN) and 4d (ViT) are complete.
# Safe to re-run: outputs always overwritten.

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG
# ============================================================
RESULTS_DIR = Path('results')

HYBRIDN_ABL  = RESULTS_DIR / 'HybridSN' / 'hybridSN_v1_ablation.csv'
VIT_ABL      = RESULTS_DIR / 'ViT'      / 'vit_v1_ablation.csv'
HYBRIDN_MAIN = RESULTS_DIR / 'HybridSN' / 'hybridSN_v1_results.csv'
VIT_MAIN     = RESULTS_DIR / 'ViT'      / 'vit_v1_results.csv'

PLOTS_DIR    = RESULTS_DIR / 'summary'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS     = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
PATCH_SIZES = [1, 6, 11]
BAND_COUNTS = [4, 10, 20, 50, 100]
METHODS     = ['PCA', 'MI', 'LASSO', 'FullSpectrum']


# ============================================================
# HELPERS
# ============================================================
def load_csv(path):
    if not path.exists():
        print('  [missing] {}'.format(path))
        return []
    rows = []
    with open(path, 'r', newline='') as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def group_mean(rows, group_keys, metric):
    """Return dict: group_key_tuple -> mean metric value."""
    buckets = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k, '') for k in group_keys)
        v = row.get(metric, '')
        if v != '':
            try:
                buckets[key].append(float(v))
            except ValueError:
                pass
    return {k: float(np.mean(v)) for k, v in buckets.items() if v}


# ============================================================
# LOAD DATA
# ============================================================
hyb_abl  = load_csv(HYBRIDN_ABL)
vit_abl  = load_csv(VIT_ABL)
hyb_main = load_csv(HYBRIDN_MAIN)
vit_main = load_csv(VIT_MAIN)

print('HybridSN ablation rows: {}'.format(len(hyb_abl)))
print('ViT ablation rows     : {}'.format(len(vit_abl)))
print('HybridSN main rows    : {}'.format(len(hyb_main)))
print('ViT main rows         : {}'.format(len(vit_main)))

if not hyb_abl and not vit_abl:
    sys.exit('No ablation data found. Run 4c_hybridSN.ipynb and 4d_vit.ipynb first.')


# ============================================================
# PLOT 1: patch_size_ablation.png
# Patch size vs mean AUC / F1 for HybridSN and ViT
# Aggregated over all band methods and counts.
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
metric_pairs = [('auc', 'Mean AUC'), ('f1', 'Mean F1')]
datasets = [('HybridSN', hyb_abl), ('ViT', vit_abl)]
colors_method = {'PCA': 'tab:blue', 'MI': 'tab:orange',
                 'LASSO': 'tab:green', 'FullSpectrum': 'tab:red'}

for ax, (metric, metric_label) in zip(axes, metric_pairs):
    for model_name, rows in datasets:
        if not rows:
            continue
        means_by_patch = group_mean(rows, ['patch_size'], metric)
        xs = sorted([int(k[0]) for k in means_by_patch if k[0] != ''],
                    key=lambda x: x)
        ys = [means_by_patch.get((str(p),), float('nan')) for p in xs]
        ax.plot(xs, ys, marker='o', linewidth=2, label=model_name)

    ax.set_xlabel('Patch size')
    ax.set_ylabel(metric_label)
    ax.set_title('Patch Size vs {}'.format(metric_label))
    ax.set_xticks(PATCH_SIZES)
    ax.set_xticklabels(['{}x{}'.format(p, p) for p in PATCH_SIZES])
    ax.grid(alpha=0.3)
    ax.legend()

fig.suptitle('Patch Size Ablation (mean over all band methods & counts)')
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'patch_size_ablation.png', dpi=150)
plt.close(fig)
print('Saved patch_size_ablation.png')


# ============================================================
# PLOT 2: patch_size_per_method.png
# One subplot per band method, x=patch size, y=AUC,
# separate lines per model (HybridSN vs ViT).
# ============================================================
methods_to_plot = [m for m in METHODS if m != 'FullSpectrum']
fig, axes = plt.subplots(1, len(methods_to_plot),
                         figsize=(4.5 * len(methods_to_plot), 5), sharey=True)

for ax, method in zip(axes, methods_to_plot):
    for model_name, rows in datasets:
        if not rows:
            continue
        method_rows = [r for r in rows if r.get('method') == method]
        means = group_mean(method_rows, ['patch_size'], 'auc')
        xs = sorted([int(k[0]) for k in means if k[0] != ''])
        ys = [means.get((str(p),), float('nan')) for p in xs]
        if xs:
            ax.plot(xs, ys, marker='o', linewidth=2, label=model_name)
    ax.set_title(method)
    ax.set_xlabel('Patch size')
    ax.set_xticks(PATCH_SIZES)
    ax.set_xticklabels(['{}x{}'.format(p, p) for p in PATCH_SIZES])
    ax.grid(alpha=0.3)
    if ax == axes[0]:
        ax.set_ylabel('Mean AUC')
        ax.legend()

fig.suptitle('Patch Size vs AUC by Band Method (HybridSN vs ViT)')
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'patch_size_per_method.png', dpi=150)
plt.close(fig)
print('Saved patch_size_per_method.png')


# ============================================================
# PLOT 3: band_count_deep.png
# Band count sweep for HybridSN and ViT (main results, patch=11)
# Similar to band_method_comparison.png but deep models only.
# ============================================================
if hyb_main or vit_main:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, method in zip(axes, ['PCA', 'MI', 'LASSO']):
        for model_name, rows in [('HybridSN', hyb_main), ('ViT', vit_main)]:
            if not rows:
                continue
            method_rows = [r for r in rows if r.get('method') == method]
            means = group_mean(method_rows, ['n_bands'], 'auc')
            xs = sorted([int(k[0]) for k in means if k[0] != ''])
            ys = [means.get((str(nb),), float('nan')) for nb in xs]
            if xs:
                ax.plot(xs, ys, marker='o', linewidth=2, label=model_name)
        ax.set_title(method)
        ax.set_xlabel('Band count')
        ax.set_xticks(BAND_COUNTS)
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel('Mean AUC')
            ax.legend()
    fig.suptitle('Band Count vs AUC — Deep Models (patch=11)')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'band_count_deep.png', dpi=150)
    plt.close(fig)
    print('Saved band_count_deep.png')


# ============================================================
# PLOT 4: learning_curves/ — referenced from Colab Drive plots
# (actual curves are saved by 4c/4d notebooks to Drive)
# Print reminder only.
# ============================================================
lc_dir = RESULTS_DIR / 'HybridSN' / 'plots'
vit_lc_dir = RESULTS_DIR / 'ViT' / 'plots'
print()
print('Learning curve PNGs (saved by Colab notebooks):')
print('  HybridSN: {}'.format(lc_dir))
print('  ViT     : {}'.format(vit_lc_dir))
print('  Download from Google Drive to include in results.')


# ============================================================
# SUMMARY TABLE: best patch size per model
# ============================================================
print()
print('Patch size ablation summary (mean AUC over all combos):')
print('{:10s} {:6s}  {}'.format('Model', 'Patch', 'AUC'))
for model_name, rows in datasets:
    if not rows:
        print('{:10s} (no data)'.format(model_name))
        continue
    means = group_mean(rows, ['patch_size'], 'auc')
    if not means:
        continue
    best_key = max(means, key=lambda k: means[k])
    for ps in PATCH_SIZES:
        val = means.get((str(ps),), float('nan'))
        marker = ' <-- best' if str(ps) == best_key[0] else ''
        print('{:10s} {:3d}x{:3d}  {:.4f}{}'.format(
            model_name, ps, ps, val, marker))


# ============================================================
# DONE
# ============================================================
print()
print('Ablation plots saved to: {}'.format(PLOTS_DIR))
print('  patch_size_ablation.png')
print('  patch_size_per_method.png')
print('  band_count_deep.png')
