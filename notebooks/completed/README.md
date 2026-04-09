# Completed Training Notebooks

Executed notebooks from Google Colab (A100 GPU) with full cell outputs preserved.
View directly on GitHub to see training progress, loss curves, and fold-by-fold results.

| Notebook | Model | Folds | Best AUC |
|----------|-------|-------|----------|
| [4c_hybridSN_completed.ipynb](4c_hybridSN_completed.ipynb) | HybridSN 3D+2D CNN | 45/45 | 0.918 (LASSO/100) |
| [4d_vit_completed.ipynb](4d_vit_completed.ipynb) | Vision Transformer | 45/45 | MI/100 |

## Notes
- Source notebooks (without outputs) are at the project root: `4c_hybridSN.ipynb`, `4d_vit.ipynb`
- Results CSVs: `results/HybridSN/hybridSN_v1_results.csv`, `results/ViT/vit_v1_results.csv`
- All training used Focal Loss, LOPOCV (3 patients), patch_size=11
