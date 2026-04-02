# coding: utf-8
# hybridSN_loader.py
# Data loader for HybridSN 3D+2D CNN (Step 4c - Colab T4)
# Returns 5D patches: (N, patch_size, patch_size, n_bands, 1)
#
# Imports shared LOPOCV core from data_loader.py.
# DO NOT redefine get_lopocv_folds() here - always import from data_loader.

from utils.data_loader import PREPROCESSED_DIR, PIXELS_PER_ROI


def get_fold_patches(fold_dict, band_indices, patch_size=11, patches_per_roi=300, seed=42):
    """
    Loads spatial patches from full h5 cubes for HybridSN input.

    Parameters
    ----------
    fold_dict      : dict from get_lopocv_folds() with train_files / test_files
    band_indices   : list of int band indices to select
    patch_size     : spatial size of each patch (default 11 -> 11x11)
    patches_per_roi: number of random patches to sample per ROI
    seed           : random seed

    Returns
    -------
    X : (N, patch_size, patch_size, n_bands, 1) float32  - 5D for HybridSN
    y : (N,) int8  0=NT  1=T
    """
    raise NotImplementedError(
        "HybridSN loader not yet implemented. "
        "Implement in step4c_hybridSN.ipynb on Colab."
    )
