# coding: utf-8
# plot_individual.py
# Per-model detailed plots — run any time after a single model completes.
# Safe to re-run: all outputs overwritten.
#
# Generates for each available model:
#   1. auc_by_method_bands.png  — AUC vs band count, one line per method
#   2. metrics_by_fold.png      — per-fold bar chart (acc/sens/spec/f1/auc)
#   3. best_combos.png          — top 5 combos by mean AUC with error bars
#   4. p2_collapse.png          — fold-by-fold sensitivity to highlight P2 issue

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

MODEL_CSVS = {
    'RF':       RESULTS_DIR / 'RF'       / 'rf_v1_results.csv',
    'SVM':      RESULTS_DIR / 'SVM'      / 'svm_v1_results.csv',
    'HybridSN': RESULTS_DIR / 'HybridSN' / 'hybridSN_v1_results.csv',
    'ViT':      RESULTS_DIR / 'ViT'      / 'vit_v1_results.csv',
}

BAND_COUNTS   = [4, 10, 20, 50, 100]
METHODS       = ['PCA', 'MI', 'LASSO']
METHOD_COLORS = {'PCA': 'tab:blue', 'MI': 'tab:orange', 'LASSO': 'tab:green'}
FOLD_COLORS   = ['#4C72B0', '#DD8452', '#55A868']
METRICS       = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']


# ============================================================
# HELPERS
# ============================================================
def load_csv(path):
    if not path.exists():
        return None
    rows = []
    with open(path, 'r', newline='') as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def flt(row, col):
    try:
        return float(row.get(col, ''))
    except (ValueError, TypeError):
        return float('nan')


def rows_for(rows, method, n_bands, fold=None):
    out = [r for r in rows
           if r.get('method') == str(method) and r.get('n_bands') == str(n_bands)]
    if fold is not None:
        out = [r for r in out if r.get('fold') == str(fold)]
    return out


# ============================================================
# GENERATE PLOTS FOR ONE MODEL
# ============================================================
def plot_model(model_name, rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'\n{model_name}: {len(rows)} rows -> {out_dir}')

    # ── Plot 1: AUC vs band count per method ─────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        means, stds = [], []
        for nb in BAND_COUNTS:
            vals = [flt(r, 'auc') for r in rows_for(rows, method, nb)]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else np.nan)
        ax.errorbar(BAND_COUNTS, means, yerr=stds, marker='o',
                    label=method, color=METHOD_COLORS[method],
                    capsize=4, linewidth=1.8)

    ax.set_xlabel('Band count')
    ax.set_ylabel('Mean AUC (±std, 3 folds)')
    ax.set_title(f'{model_name} — AUC vs Band Count by Method')
    ax.set_xticks(BAND_COUNTS)
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'auc_by_method_bands.png', dpi=150)
    plt.close(fig)
    print(f'  Saved auc_by_method_bands.png')

    # ── Plot 2: Per-fold bar chart (best method only) ─────────
    # Find best method = highest mean AUC across all combos
    best_method, best_nb, best_mean = None, None, -1
    for method in METHODS:
        for nb in BAND_COUNTS:
            vals = [flt(r, 'auc') for r in rows_for(rows, method, nb)]
            vals = [v for v in vals if not np.isnan(v)]
            if vals and np.mean(vals) > best_mean:
                best_mean = np.mean(vals)
                best_method, best_nb = method, nb

    if best_method:
        folds = [1, 2, 3]
        fold_labels = ['Fold 1\n(test P1)', 'Fold 2\n(test P2)', 'Fold 3\n(test P3)']
        n_metrics = len(METRICS)
        x = np.arange(len(folds))
        width = 0.15
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, metric in enumerate(METRICS):
            vals = []
            for fold in folds:
                fold_rows = rows_for(rows, best_method, best_nb, fold)
                v = flt(fold_rows[0], metric) if fold_rows else np.nan
                vals.append(v)
            ax.bar(x + i * width, vals, width, label=metric.capitalize(),
                   color=plt.cm.Set2(i / n_metrics))
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_title(f'{model_name} — Per-Fold Metrics\nBest combo: {best_method}/{best_nb} bands (mean AUC={best_mean:.3f})')
        ax.set_xticks(x + width * (n_metrics - 1) / 2)
        ax.set_xticklabels(fold_labels)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / 'metrics_by_fold.png', dpi=150)
        plt.close(fig)
        print(f'  Saved metrics_by_fold.png  (best combo: {best_method}/{best_nb})')

    # ── Plot 3: Top 5 combos by mean AUC with error bars ─────
    combo_means = []
    for method in METHODS:
        for nb in BAND_COUNTS:
            vals = [flt(r, 'auc') for r in rows_for(rows, method, nb)]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                combo_means.append((f'{method}/{nb}b', np.mean(vals), np.std(vals)))
    combo_means.sort(key=lambda x: x[1], reverse=True)
    top5 = combo_means[:5]

    if top5:
        labels = [t[0] for t in top5]
        means  = [t[1] for t in top5]
        stds   = [t[2] for t in top5]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(labels[::-1], means[::-1], xerr=stds[::-1],
                       color='steelblue', capsize=4, alpha=0.8)
        for bar, val in zip(bars, means[::-1]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9)
        ax.set_xlabel('Mean AUC (±std, 3 folds)')
        ax.set_title(f'{model_name} — Top 5 Combos by Mean AUC')
        ax.set_xlim(0.4, 1.05)
        ax.grid(axis='x', alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / 'best_combos.png', dpi=150)
        plt.close(fig)
        print(f'  Saved best_combos.png')

    # ── Plot 4: P2 sensitivity collapse ───────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, method in zip(axes, METHODS):
        for nb in BAND_COUNTS:
            sens_vals = []
            for fold in [1, 2, 3]:
                fold_rows = rows_for(rows, method, nb, fold)
                v = flt(fold_rows[0], 'sensitivity') if fold_rows else np.nan
                sens_vals.append(v)
            ax.plot([1, 2, 3], sens_vals, marker='o', alpha=0.6,
                    label=f'{nb}b', linewidth=1.2)
        ax.set_title(f'{method}')
        ax.set_xlabel('Fold (1=P1, 2=P2, 3=P3)')
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['P1', 'P2', 'P3'])
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(0.85, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.grid(alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel('Sensitivity')
            ax.legend(fontsize=7, title='bands')
    fig.suptitle(f'{model_name} — Sensitivity by Patient (P2 collapse visible)', y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / 'p2_collapse.png', dpi=150)
    plt.close(fig)
    print(f'  Saved p2_collapse.png')


# ============================================================
# RUN FOR ALL AVAILABLE MODELS
# ============================================================
generated = 0
for model_name, csv_path in MODEL_CSVS.items():
    rows = load_csv(csv_path)
    if rows is None:
        print(f'{model_name}: [missing] skipping')
        continue
    out_dir = RESULTS_DIR / model_name / 'plots'
    plot_model(model_name, rows, out_dir)
    generated += 1

print(f'\nDone. Generated plots for {generated} model(s).')
print('Outputs in results/<model>/plots/')
