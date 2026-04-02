# coding: utf-8
# vit_loader.py
# Data loader for Vision Transformer (Step 4d - Colab A100)
# Returns tokenized patches: (N, n_tokens, token_dim)
#
# Imports shared LOPOCV core from data_loader.py.
# DO NOT redefine get_lopocv_folds() here - always import from data_loader.

from utils.data_loader import PREPROCESSED_DIR, PIXELS_PER_ROI


def get_fold_tokens(fold_dict, band_indices, patch_size=11, token_size=4, seed=42):
    """
    Loads spatial patches and divides them into tokens for ViT input.

    Parameters
    ----------
    fold_dict  : dict from get_lopocv_folds() with train_files / test_files
    band_indices: list of int band indices to select
    patch_size : spatial size of patch (default 11)
    token_size : spatial size of each token sub-patch (default 4)
                 number of tokens = ceil(patch_size / token_size)^2
    seed       : random seed

    Returns
    -------
    X : (N, n_tokens, token_dim) float32  - sequence of token embeddings
    y : (N,) int8  0=NT  1=T
    """
    raise NotImplementedError(
        "ViT loader not yet implemented. "
        "Implement in step4d_vit.ipynb on Colab."
    )
