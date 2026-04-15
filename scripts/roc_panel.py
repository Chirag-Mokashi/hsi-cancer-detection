# coding: utf-8
# scripts/roc_panel.py
#
# Assembles existing per-fold ROC curve PNG files (generated during Colab training)
# into a combined panel figure for HybridSN and ViT.
#
# HybridSN best combo: LASSO/100 -- folds 1, 2, 3
# ViT best combo:      MI/100    -- folds 1, 2, 3
#
# Input files (must exist):
#   results/HybridSN/plots/curve_LASSO_100b_p11_f{1,2,3}.png
#   results/ViT/plots/curve_MI_100b_p11_t4_f{1,2,3}.png
#
# Outputs:
#   results/summary/roc_dl_panel.png  -- 2-row x 3-col panel (model x fold)
#
# Must be run from the project root: python scripts/roc_panel.py

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

RESULTS_DIR = Path('results')
SUMMARY_DIR = RESULTS_DIR / 'summary'
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# ROC PNG files for each DL model at its best combo
DL_ROC_FILES = {
    'HybridSN (LASSO/100)': [
        RESULTS_DIR / 'HybridSN' / 'plots' / 'curve_LASSO_100b_p11_f1.png',
        RESULTS_DIR / 'HybridSN' / 'plots' / 'curve_LASSO_100b_p11_f2.png',
        RESULTS_DIR / 'HybridSN' / 'plots' / 'curve_LASSO_100b_p11_f3.png',
    ],
    'ViT (MI/100)': [
        RESULTS_DIR / 'ViT' / 'plots' / 'curve_MI_100b_p11_t4_f1.png',
        RESULTS_DIR / 'ViT' / 'plots' / 'curve_MI_100b_p11_t4_f2.png',
        RESULTS_DIR / 'ViT' / 'plots' / 'curve_MI_100b_p11_t4_f3.png',
    ],
}

FOLD_LABELS = ['Fold 1 (P1 held out)', 'Fold 2 (P2 held out)', 'Fold 3 (P3 held out)']

n_rows = len(DL_ROC_FILES)
n_cols = 3  # 3 folds

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 4.2))

for row_idx, (model_label, fold_paths) in enumerate(DL_ROC_FILES.items()):
    for col_idx, (img_path, fold_label) in enumerate(zip(fold_paths, FOLD_LABELS)):
        ax = axes[row_idx][col_idx]

        if img_path.exists():
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'Missing:\n{}'.format(img_path.name),
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=8, color='red')
            ax.set_facecolor('#f8f8f8')
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)

        if row_idx == 0:
            ax.set_title(fold_label, fontsize=10, pad=4)

        if col_idx == 0:
            ax.set_ylabel(model_label, fontsize=10, labelpad=6)

fig.suptitle('ROC Curves at Best Combo -- HybridSN and ViT (per fold / held-out patient)',
             fontsize=11, y=1.01)
fig.tight_layout()

out_path = SUMMARY_DIR / 'roc_dl_panel.png'
fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: {}'.format(out_path))
