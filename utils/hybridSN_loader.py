# coding: utf-8
# hybridSN_loader.py
# Data loader for HybridSN 3D+2D CNN (Step 4c - Colab T4)
# Returns 5D patches: (N, patch_size, patch_size, n_bands, 1)
#
# Imports shared LOPOCV core from data_loader.py.
# DO NOT redefine get_lopocv_folds() here - always import from data_loader.

import h5py
import numpy as np

from utils.data_loader import PREPROCESSED_DIR, PIXELS_PER_ROI


def get_fold_patches(fold_dict, band_indices, patch_size=11,
                     patches_per_roi=300, seed=42):
    """
    Extract random spatial patches from full HDF5 cubes for HybridSN input.

    Each patch is a (patch_size x patch_size) spatial window centred on a
    randomly sampled pixel. Patches that would extend outside the cube boundary
    are rejected (centre must be at least patch_size//2 from each edge).

    Parameters
    ----------
    fold_dict      : dict from get_lopocv_folds() with 'train_files'/'test_files'
    band_indices   : list of int band indices to select
    patch_size     : spatial size of each patch (1, 6, or 11)
    patches_per_roi: number of random patches to sample per ROI file
    seed           : random seed

    Returns
    -------
    X : (N, patch_size, patch_size, n_bands, 1) float32
    y : (N,) int8   0=NT  1=T
    """
    rng = np.random.default_rng(seed)
    half = patch_size // 2
    n_bands = len(band_indices)

    all_X = []
    all_y = []

    train_files = fold_dict.get('train_files', fold_dict.get('files', []))

    for fpath in train_files:
        with h5py.File(fpath, 'r') as f:
            n_rows, n_cols, _ = f['cube'].shape
            label_str = str(f.attrs['label'])
            label_int = np.int8(1 if label_str == 'T' else 0)

            # Valid centre range (avoid boundary)
            row_min, row_max = half, n_rows - half - 1
            col_min, col_max = half, n_cols - half - 1

            n_valid = (row_max - row_min + 1) * (col_max - col_min + 1)
            n_sample = min(patches_per_roi, n_valid)

            # Sample random centre pixels
            rows = rng.integers(row_min, row_max + 1, size=n_sample)
            cols = rng.integers(col_min, col_max + 1, size=n_sample)

            patches = np.empty((n_sample, patch_size, patch_size, n_bands),
                               dtype=np.float32)

            for i, (r, c) in enumerate(zip(rows, cols)):
                # One read per patch — smallest slice possible
                block = f['cube'][r - half:r + half + 1,
                                   c - half:c + half + 1, :]
                patches[i] = block[:, :, band_indices]

        # Add channel dim: (N, patch, patch, n_bands, 1)
        patches = patches[..., np.newaxis]
        all_X.append(patches)
        all_y.append(np.full(n_sample, label_int, dtype=np.int8))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y)
    return X, y


def _get_test_patches_deterministic(test_files, band_indices, patch_size=11,
                                    patches_per_roi=300):
    """Extract evenly spaced patches from test files — no RNG."""
    half    = patch_size // 2
    n_bands = len(band_indices)

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

            patches = np.empty((n_sample, patch_size, patch_size, n_bands), dtype=np.float32)
            for i, (r, c) in enumerate(zip(rows, cols)):
                block = f['cube'][r - half:r + half + 1, c - half:c + half + 1, :]
                patches[i] = block[:, :, band_indices]

        patches = patches[..., np.newaxis]
        all_X.append(patches)
        all_y.append(np.full(n_sample, label_int, dtype=np.int8))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y)
    return X, y


def get_fold_patches_split(fold_dict, band_indices, patch_size=11,
                           patches_per_roi=300, seed=42):
    """
    Load train and test patches for one LOPOCV fold.

    Train uses random sampling (seed). Test uses deterministic evenly-spaced
    sampling (no RNG) for reproducible LOPOCV metrics.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    train_dict = {'train_files': fold_dict['train_files']}

    X_train, y_train = get_fold_patches(
        train_dict, band_indices, patch_size=patch_size,
        patches_per_roi=patches_per_roi, seed=seed
    )
    X_test, y_test = _get_test_patches_deterministic(
        fold_dict['test_files'], band_indices, patch_size=patch_size,
        patches_per_roi=patches_per_roi
    )
    return X_train, y_train, X_test, y_test
