# coding: utf-8
# vit_loader.py
# Data loader for Vision Transformer (Step 4d - Colab A100)
# Returns tokenized patches: (N, n_tokens, token_dim)
#
# Imports shared LOPOCV core from data_loader.py.
# DO NOT redefine get_lopocv_folds() here - always import from data_loader.

import math

import h5py
import numpy as np

from utils.data_loader import PREPROCESSED_DIR, PIXELS_PER_ROI


def get_fold_tokens(fold_dict, band_indices, patch_size=11, token_size=4,
                    patches_per_roi=300, seed=42):
    """
    Extract spatial patches and divide into non-overlapping token sub-patches.

    A patch of shape (patch_size, patch_size, n_bands) is divided into a grid
    of (token_size, token_size, n_bands) sub-patches. Each sub-patch is flattened
    into a 1D token of dimension token_dim = token_size * token_size * n_bands.

    n_tokens = ceil(patch_size / token_size)^2
    token_dim = token_size * token_size * n_bands

    For patch_size=11, token_size=4: grid is 3x3 = 9 tokens (last row/col padded).
    For patch_size=1, token_size=1: 1 token of dim n_bands (pixel-level ViT).

    Parameters
    ----------
    fold_dict      : dict from get_lopocv_folds() with 'train_files'/'test_files'
    band_indices   : list of int band indices to select
    patch_size     : spatial size of each patch (1, 6, or 11)
    token_size     : spatial size of each token sub-patch (default 4)
    patches_per_roi: number of random patches per ROI file
    seed           : random seed

    Returns
    -------
    X : (N, n_tokens, token_dim) float32  - sequence of token embeddings
    y : (N,) int8   0=NT  1=T
    """
    rng = np.random.default_rng(seed)
    half   = patch_size // 2
    n_bands = len(band_indices)

    # Compute token grid dimensions (pad if needed)
    grid_size = math.ceil(patch_size / token_size)
    padded    = grid_size * token_size          # padded spatial size
    n_tokens  = grid_size * grid_size
    token_dim = token_size * token_size * n_bands

    all_X = []
    all_y = []

    train_files = fold_dict.get('train_files', fold_dict.get('files', []))

    for fpath in train_files:
        with h5py.File(fpath, 'r') as f:
            n_rows, n_cols, _ = f['cube'].shape
            label_str = str(f.attrs['label'])
            label_int = np.int8(1 if label_str == 'T' else 0)

            row_min, row_max = half, n_rows - half - 1
            col_min, col_max = half, n_cols - half - 1

            n_valid  = (row_max - row_min + 1) * (col_max - col_min + 1)
            n_sample = min(patches_per_roi, n_valid)

            rows = rng.integers(row_min, row_max + 1, size=n_sample)
            cols = rng.integers(col_min, col_max + 1, size=n_sample)

            tokens = np.zeros((n_sample, n_tokens, token_dim), dtype=np.float32)

            for i, (r, c) in enumerate(zip(rows, cols)):
                patch = f['cube'][r - half:r + half + 1,
                                   c - half:c + half + 1, :]
                patch = patch[:, :, band_indices]   # (patch, patch, n_bands)

                # Pad to padded x padded if necessary
                if padded != patch_size:
                    pad_h = padded - patch.shape[0]
                    pad_w = padded - patch.shape[1]
                    patch = np.pad(patch,
                                   ((0, pad_h), (0, pad_w), (0, 0)),
                                   mode='reflect')

                # Divide into token grid
                tok_idx = 0
                for tr in range(grid_size):
                    for tc in range(grid_size):
                        sub = patch[tr * token_size:(tr + 1) * token_size,
                                    tc * token_size:(tc + 1) * token_size, :]
                        tokens[i, tok_idx] = sub.reshape(-1)
                        tok_idx += 1

        all_X.append(tokens)
        all_y.append(np.full(n_sample, label_int, dtype=np.int8))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y)
    return X, y


def _get_test_tokens_deterministic(test_files, band_indices, patch_size=11,
                                   token_size=4, patches_per_roi=300):
    """Extract evenly spaced token patches from test files — no RNG."""
    half      = patch_size // 2
    n_bands   = len(band_indices)
    grid_size = math.ceil(patch_size / token_size)
    padded    = grid_size * token_size
    n_tokens  = grid_size * grid_size
    token_dim = token_size * token_size * n_bands

    all_X = []
    all_y = []

    for fpath in test_files:
        with h5py.File(fpath, 'r') as f:
            n_rows, n_cols, _ = f['cube'].shape
            label_str = str(f.attrs['label'])
            label_int = np.int8(1 if label_str == 'T' else 0)

            row_min, row_max = half, n_rows - half - 1
            col_min, col_max = half, n_cols - half - 1
            n_valid  = (row_max - row_min + 1) * (col_max - col_min + 1)
            n_sample = min(patches_per_roi, n_valid)

            flat_idx     = np.round(np.linspace(0, n_valid - 1, n_sample)).astype(np.intp)
            n_valid_cols = col_max - col_min + 1
            rows = flat_idx // n_valid_cols + row_min
            cols = flat_idx % n_valid_cols + col_min

            tokens = np.zeros((n_sample, n_tokens, token_dim), dtype=np.float32)
            for i, (r, c) in enumerate(zip(rows, cols)):
                patch = f['cube'][r - half:r + half + 1, c - half:c + half + 1, :]
                patch = patch[:, :, band_indices]
                if padded != patch_size:
                    pad_h = padded - patch.shape[0]
                    pad_w = padded - patch.shape[1]
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                tok_idx = 0
                for tr in range(grid_size):
                    for tc in range(grid_size):
                        sub = patch[tr * token_size:(tr + 1) * token_size,
                                    tc * token_size:(tc + 1) * token_size, :]
                        tokens[i, tok_idx] = sub.reshape(-1)
                        tok_idx += 1

        all_X.append(tokens)
        all_y.append(np.full(n_sample, label_int, dtype=np.int8))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y)
    return X, y


def get_fold_tokens_split(fold_dict, band_indices, patch_size=11, token_size=4,
                          patches_per_roi=300, seed=42):
    """
    Load train and test tokens for one LOPOCV fold.

    Train uses random sampling (seed). Test uses deterministic evenly-spaced
    sampling (no RNG) for reproducible LOPOCV metrics.

    Returns
    -------
    X_train, y_train, X_test, y_test
        X shape: (N, n_tokens, token_dim) float32
    """
    train_dict = {'train_files': fold_dict['train_files']}

    X_train, y_train = get_fold_tokens(
        train_dict, band_indices, patch_size=patch_size, token_size=token_size,
        patches_per_roi=patches_per_roi, seed=seed
    )
    X_test, y_test = _get_test_tokens_deterministic(
        fold_dict['test_files'], band_indices, patch_size=patch_size,
        token_size=token_size, patches_per_roi=patches_per_roi
    )
    return X_train, y_train, X_test, y_test
