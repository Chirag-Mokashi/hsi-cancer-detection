# coding: utf-8
# 5_compile_results.py
# Step 5: Cross-model results aggregation and comparison plots
#
# Reads all model vN_results.csv files, merges into combined_results.csv,
# and generates cross-model comparison visualisations in results/summary/.
#
# Run after all models (RF, SVM, HybridSN, ViT) are complete.
# Safe to re-run: outputs are always overwritten.

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
SUMMARY_DIR = RESULTS_DIR / 'summary'
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# Model result files (update version suffix if re-run with v2 etc.)
MODEL_CSVS = {
    'RF':       RESULTS_DIR / 'RF'       / 'rf_v1_results.csv',
    'SVM':      RESULTS_DIR / 'SVM'      / 'svm_v1_results.csv',
    'HybridSN': RESULTS_DIR / 'HybridSN' / 'hybridSN_v1_results.csv',
    'ViT':      RESULTS_DIR / 'ViT'      / 'vit_v1_results.csv',
}

METRICS      = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']
TIMING_COLS  = ['train_time_sec', 'inference_time_per_image_ms']
COMBINED_CSV = SUMMARY_DIR / 'combined_results.csv'


# ============================================================
# LOAD ALL RESULTS
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


all_rows = []
for model_name, csv_path in MODEL_CSVS.items():
    rows = load_csv(csv_path)
    print('{:10s}: {} rows from {}'.format(model_name, len(rows), csv_path))
    all_rows.extend(rows)

if not all_rows:
    sys.exit('No results found. Run all models first.')

print('\nTotal rows: {}'.format(len(all_rows)))

# ============================================================
# WRITE COMBINED CSV
# ============================================================
all_cols = list(all_rows[0].keys())
# Ensure optional cols exist; fill blank for rows that don't have them
for row in all_rows:
    row.setdefault('token_size', '')
    row.setdefault('patch_size', '')
    row.setdefault('loss_fn', '')
    row.setdefault('git_sha', '')
    row.setdefault('seed', '')
    row.setdefault('code_version', '')

# Canonical column order
COMBINED_COLS = ['model', 'method', 'n_bands', 'patch_size', 'token_size',
                 'fold', 'loss_fn'] + METRICS + TIMING_COLS + ['git_sha', 'seed', 'code_version']

with open(COMBINED_CSV, 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=COMBINED_COLS, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(all_rows)

print('Combined CSV -> {}'.format(COMBINED_CSV))


# ============================================================
# AGGREGATE: mean per (model, method, n_bands)
# ============================================================
agg = defaultdict(lambda: {m: [] for m in METRICS + TIMING_COLS})

for row in all_rows:
    key = (row.get('model', ''), row.get('method', ''), row.get('n_bands', ''))
    for m in METRICS + TIMING_COLS:
        val = row.get(m, '')
        if val != '':
            try:
                agg[key][m].append(float(val))
            except ValueError:
                pass

# Build summary dict: key -> metric -> (mean, std)
summary = {}
for key, metrics in agg.items():
    summary[key] = {}
    for m, vals in metrics.items():
        if vals:
            arr = np.array(vals)
            summary[key][m] = (float(arr.mean()), float(arr.std()))
        else:
            summary[key][m] = (float('nan'), float('nan'))


def mean_metric(model, method, n_bands, metric):
    key = (model, str(method), str(n_bands))
    entry = summary.get(key, {})
    return entry.get(metric, (float('nan'), float('nan')))[0]


# ============================================================
# PLOT 1: model_comparison.png
# Bar chart: mean AUC / F1 / sensitivity / specificity per model
# (aggregated over all band methods and counts)
# ============================================================
models = list(MODEL_CSVS.keys())
model_rows = {m: [r for r in all_rows if r.get('model') == m] for m in models}


def model_mean(model, metric):
    vals = []
    for row in model_rows[model]:
        v = row.get(metric, '')
        if v != '':
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return np.mean(vals) if vals else float('nan')


plot_metrics = ['auc', 'f1', 'sensitivity', 'specificity']
metric_labels = ['AUC', 'F1', 'Sensitivity', 'Specificity']

x = np.arange(len(plot_metrics))
width = 0.2
fig, ax = plt.subplots(figsize=(10, 5))

for i, model in enumerate(models):
    means = [model_mean(model, m) for m in plot_metrics]
    bars = ax.bar(x + i * width, means, width, label=model)
    for bar, val in zip(bars, means):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    '{:.3f}'.format(val), ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Model Comparison (mean over all combos & folds)')
ax.set_xticks(x + width * (len(models) - 1) / 2)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(SUMMARY_DIR / 'model_comparison.png', dpi=150)
plt.close(fig)
print('Saved model_comparison.png')


# ============================================================
# PLOT 2: auc_heatmap.png
# Heatmap: method (row) x model (col), cell = mean AUC
# ============================================================
methods_ordered = ['PCA', 'MI', 'LASSO', 'FullSpectrum']
band_counts = [4, 10, 20, 50, 100]

# Build label list: one row per (method, n_bands) combo
row_labels = []
for method in methods_ordered:
    if method == 'FullSpectrum':
        row_labels.append('FullSpectrum')
    else:
        for nb in band_counts:
            row_labels.append('{}/{}'.format(method, nb))

heatmap_data = np.full((len(row_labels), len(models)), float('nan'))

for ri, label in enumerate(row_labels):
    if '/' in label:
        method, nb = label.split('/')
        nb = int(nb)
    else:
        method, nb = label, 699

    for ci, model in enumerate(models):
        val = mean_metric(model, method, nb, 'auc')
        heatmap_data[ri, ci] = val

fig, ax = plt.subplots(figsize=(len(models) * 1.8 + 1, len(row_labels) * 0.42 + 1.5))
valid = heatmap_data[~np.isnan(heatmap_data)]
vmin = float(np.nanmin(heatmap_data)) if len(valid) else 0.5
vmax = float(np.nanmax(heatmap_data)) if len(valid) else 1.0
im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=vmin, vmax=vmax)

ax.set_xticks(range(len(models)))
ax.set_xticklabels(models)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=8)
ax.set_title('Mean AUC: Band Method x Model')

for ri in range(len(row_labels)):
    for ci in range(len(models)):
        val = heatmap_data[ri, ci]
        if not np.isnan(val):
            ax.text(ci, ri, '{:.3f}'.format(val), ha='center', va='center',
                    fontsize=7, color='black')

plt.colorbar(im, ax=ax, fraction=0.04)
fig.tight_layout()
fig.savefig(SUMMARY_DIR / 'auc_heatmap.png', dpi=150)
plt.close(fig)
print('Saved auc_heatmap.png')


# ============================================================
# PLOT 3: band_method_comparison.png
# Band count vs mean AUC per band selection method (lines per model)
# ============================================================
fig, axes = plt.subplots(1, len(methods_ordered[:3]), figsize=(14, 4), sharey=True)
colors = {'RF': 'tab:blue', 'SVM': 'tab:orange',
          'HybridSN': 'tab:green', 'ViT': 'tab:red'}

for ax, method in zip(axes, methods_ordered[:3]):
    for model in models:
        aucs = []
        for nb in band_counts:
            aucs.append(mean_metric(model, method, nb, 'auc'))
        valid_mask = [not np.isnan(v) for v in aucs]
        xs = [nb for nb, v in zip(band_counts, valid_mask) if v]
        ys = [v for v in aucs if not np.isnan(v)]
        if xs:
            ax.plot(xs, ys, marker='o', label=model, color=colors.get(model))
    ax.set_title(method)
    ax.set_xlabel('Band count')
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(band_counts)
    ax.grid(alpha=0.3)
    if ax == axes[0]:
        ax.set_ylabel('Mean AUC')
        ax.legend(fontsize=8)

fig.suptitle('Band Count vs AUC by Method')
fig.tight_layout()
fig.savefig(SUMMARY_DIR / 'band_method_comparison.png', dpi=150)
plt.close(fig)
print('Saved band_method_comparison.png')


# ============================================================
# PLOT 4: sens_spec_scatter.png
# Sensitivity vs Specificity per model (all combos)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
markers = {'RF': 'o', 'SVM': 's', 'HybridSN': '^', 'ViT': 'D'}

for model in models:
    sens, spec = [], []
    for row in model_rows[model]:
        s = row.get('sensitivity', '')
        p = row.get('specificity', '')
        if s != '' and p != '':
            try:
                sens.append(float(s))
                spec.append(float(p))
            except ValueError:
                pass
    if sens:
        ax.scatter(spec, sens, label=model, alpha=0.6,
                   marker=markers.get(model, 'o'), s=30,
                   color=colors.get(model))

ax.axhline(0.85, color='grey', linestyle='--', linewidth=0.8, label='Target 85%')
ax.axvline(0.85, color='grey', linestyle='--', linewidth=0.8)
ax.set_xlabel('Specificity')
ax.set_ylabel('Sensitivity')
ax.set_title('Sensitivity vs Specificity (all combos & folds)')
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(SUMMARY_DIR / 'sens_spec_scatter.png', dpi=150)
plt.close(fig)
print('Saved sens_spec_scatter.png')


# ============================================================
# NOTE: Wilcoxon signed-rank test was removed (Q4 decision, April 7 2026).
# Reason: LOPOCV produces only n=3 folds, which is insufficient for a
# meaningful signed-rank significance interpretation. Formal statistical
# testing (McNemar's test) was also skipped (Decision 1, April 14 2026):
# n=3 LOPOCV folds are insufficient for robust hypothesis testing.
# Results are framed as hypothesis-generating/exploratory.
# ============================================================


# ============================================================
# HELPERS for extra plots (April 14 2026)
# ============================================================

# Best combo per model (locked decisions April 14 2026)
BEST_COMBO = {
    'RF':       ('LASSO', '100'),
    'SVM':      ('LASSO', '100'),
    'HybridSN': ('LASSO', '100'),
    'ViT':      ('MI',    '100'),
}

MODEL_COLORS = {
    'RF':       'tab:blue',
    'SVM':      'tab:orange',
    'HybridSN': 'tab:green',
    'ViT':      'tab:red',
}

PATIENT_LABELS = {1: 'P1', 2: 'P2', 3: 'P3'}


def get_best_combo_folds(model):
    """Return list of 3 rows (fold 1/2/3) for a model's best combo, sorted by fold."""
    method, n_bands = BEST_COMBO[model]
    rows_out = []
    for row in model_rows[model]:
        if row.get('method') == method and row.get('n_bands') == str(n_bands):
            try:
                fold = int(row.get('fold', 0))
                if fold in (1, 2, 3):
                    rows_out.append((fold, row))
            except ValueError:
                pass
    rows_out.sort(key=lambda x: x[0])
    return rows_out


# ============================================================
# PLOT 5: per_patient_sens_spec.png
# Per-patient (per-fold) sensitivity and specificity at best combo
# 4 subplots, one per model; highlights P2 collapse
# ============================================================

fig5, axes5 = plt.subplots(1, len(models), figsize=(14, 4), sharey=True)
bar_width = 0.35
patient_x = np.array([0, 1, 2])
sens_color = '#4C72B0'
spec_color = '#DD8452'
p2_edge_color = '#C00000'

for ax, model in zip(axes5, models):
    fold_rows = get_best_combo_folds(model)
    if not fold_rows:
        ax.set_title('{}\n(no data)'.format(model))
        continue

    sens_vals = []
    spec_vals = []
    for fold, row in fold_rows:
        sens_vals.append(float(row.get('sensitivity', float('nan'))))
        spec_vals.append(float(row.get('specificity', float('nan'))))

    bars_sens = ax.bar(patient_x - bar_width / 2, sens_vals, bar_width,
                       label='Sensitivity', color=sens_color, alpha=0.85)
    bars_spec = ax.bar(patient_x + bar_width / 2, spec_vals, bar_width,
                       label='Specificity', color=spec_color, alpha=0.85)

    # Highlight P2 (index 1) with red edge
    for bars in (bars_sens, bars_spec):
        bars[1].set_edgecolor(p2_edge_color)
        bars[1].set_linewidth(2.0)

    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xticks(patient_x)
    ax.set_xticklabels(['P1\n(fold 1)', 'P2\n(fold 2)', 'P3\n(fold 3)'])
    ax.set_ylim(0, 1.08)
    method, nb = BEST_COMBO[model]
    ax.set_title('{}\n({}/{})'.format(model, method, nb))
    ax.grid(axis='y', alpha=0.3)
    if ax == axes5[0]:
        ax.set_ylabel('Score')
        ax.legend(fontsize=8)

    # Annotate P2 column with "collapse" label for models with P2 sens < 0.5
    if len(sens_vals) >= 2 and sens_vals[1] < 0.5:
        ax.annotate('collapse', xy=(0, sens_vals[1]),
                    xytext=(0, min(sens_vals[1] + 0.12, 0.95)),
                    ha='center', fontsize=7, color=p2_edge_color,
                    arrowprops=dict(arrowstyle='->', color=p2_edge_color, lw=1.2))

fig5.suptitle('Per-Patient Sensitivity & Specificity at Best Combo\n'
              '(red border = P2, severe collapse in RF/SVM/HybridSN)')
fig5.tight_layout()
fig5.savefig(SUMMARY_DIR / 'per_patient_sens_spec.png', dpi=150)
plt.close(fig5)
print('Saved per_patient_sens_spec.png')


# ============================================================
# PLOT 6: per_patient_auc_bar.png
# Grouped bar chart: AUC per patient (fold) for all 4 models at best combo
# Directly visualises P2 collapse across all models
# ============================================================

fig6, ax6 = plt.subplots(figsize=(9, 5))
n_models = len(models)
group_width = 0.8
bar_w = group_width / n_models
patient_positions = np.array([0, 1, 2])  # P1, P2, P3

for i, model in enumerate(models):
    fold_rows = get_best_combo_folds(model)
    auc_vals = [float('nan'), float('nan'), float('nan')]
    for fold, row in fold_rows:
        idx = fold - 1
        try:
            auc_vals[idx] = float(row.get('auc', float('nan')))
        except ValueError:
            pass

    offset = (i - (n_models - 1) / 2.0) * bar_w
    xpos = patient_positions + offset
    ax6.bar(xpos, auc_vals, bar_w, label=model,
            color=MODEL_COLORS[model], alpha=0.85, edgecolor='white', linewidth=0.5)

    # Annotate each bar with AUC value
    for x, val in zip(xpos, auc_vals):
        if not np.isnan(val):
            ax6.text(x, val + 0.01, '{:.3f}'.format(val),
                     ha='center', va='bottom', fontsize=6.5, rotation=90)

ax6.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.7, label='Chance (0.5)')
ax6.set_xticks(patient_positions)
ax6.set_xticklabels(['P1 (fold 1)', 'P2 (fold 2)', 'P3 (fold 3)'], fontsize=10)
ax6.set_ylabel('AUC')
ax6.set_ylim(0, 1.15)
ax6.set_title('Per-Patient AUC at Best Combo per Model\n'
              '(fold 2 = P2 held out; collapse visible in RF/SVM/HybridSN)')

# Shade P2 column to draw attention
ax6.axvspan(0.6, 1.4, alpha=0.07, color='red', label='P2 patient')

ax6.legend(fontsize=8, loc='lower right')
ax6.grid(axis='y', alpha=0.3)
fig6.tight_layout()
fig6.savefig(SUMMARY_DIR / 'per_patient_auc_bar.png', dpi=150)
plt.close(fig6)
print('Saved per_patient_auc_bar.png')


# ============================================================
# PLOT 7: auc_vs_bands_mean.png
# AUC vs band count (mean across all 3 methods) per model
# Shows "more bands = better, diminishing returns after 50"
# ============================================================

fig7, ax7 = plt.subplots(figsize=(7, 5))

for model in models:
    mean_aucs = []
    for nb in band_counts:
        vals = []
        for method in ['PCA', 'MI', 'LASSO']:
            v = mean_metric(model, method, nb, 'auc')
            if not np.isnan(v):
                vals.append(v)
        mean_aucs.append(np.mean(vals) if vals else float('nan'))

    ax7.plot(band_counts, mean_aucs, marker='o', label=model,
             color=MODEL_COLORS[model], linewidth=2)

    # Annotate each point
    for nb, val in zip(band_counts, mean_aucs):
        if not np.isnan(val):
            ax7.annotate('{:.3f}'.format(val), (nb, val),
                         textcoords='offset points', xytext=(0, 6),
                         ha='center', fontsize=7)

ax7.axvline(50, color='grey', linestyle=':', linewidth=1.0, alpha=0.8,
            label='50 bands (diminishing returns)')
ax7.set_xlabel('Band count')
ax7.set_ylabel('Mean AUC (averaged across PCA/MI/LASSO)')
ax7.set_title('AUC vs Band Count -- All Models\n'
              '(mean across PCA/MI/LASSO band selection methods)')
ax7.set_xticks(band_counts)
ax7.set_ylim(0.5, 0.95)
ax7.legend(fontsize=9)
ax7.grid(alpha=0.3)
fig7.tight_layout()
fig7.savefig(SUMMARY_DIR / 'auc_vs_bands_mean.png', dpi=150)
plt.close(fig7)
print('Saved auc_vs_bands_mean.png')


# ============================================================
# DONE
# ============================================================
print()
print('All outputs in: {}'.format(SUMMARY_DIR))
print('  combined_results.csv')
print('  model_comparison.png')
print('  auc_heatmap.png')
print('  band_method_comparison.png')
print('  sens_spec_scatter.png')
print('  per_patient_sens_spec.png')
print('  per_patient_auc_bar.png')
print('  auc_vs_bands_mean.png')
