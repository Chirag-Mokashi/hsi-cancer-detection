# coding: utf-8
# data_loader.py
# Shared LOPOCV core for all HSI models (Step 4+)
# DO NOT modify per-model. This is the single source of truth for folds.

import csv
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.utils.class_weight import compute_class_weight as _skl_cw

# ---- Paths and constants ----
PREPROCESSED_DIR  = Path(r'C:\Users\mokas\OneDrive\Desktop\HSI\preprocessed')
BAND_SEL_DIR      = Path(r'C:\Users\mokas\OneDrive\Desktop\HSI\band_selection')
BLOCK_ROWS        = 10
PIXELS_PER_ROI    = 500
BAND_COUNTS       = [4, 10, 20, 50, 100]
# top-level ROIs (mislabeled during download) are permanently assigned to this patient group.
# All other patients are discovered dynamically from filename prefixes — adding P4, P5, P6
# requires no code changes: just preprocess + audit + sample_pixels.
TOPLEVEL_PATIENT  = 'P1'


# ---------------------------------------------------------------------------
# LOPOCV folds
# ---------------------------------------------------------------------------

def get_lopocv_folds():
    """
    Build Leave-One-Patient-Out folds from preprocessed/*.h5 (excludes samples.h5).

    Patient is parsed from filename stem — no h5py open required:
      prefix = stem.split('_')[0]
      prefix matches 'P' + digits  ->  that patient group
      anything else (e.g. 'top-level')  ->  TOPLEVEL_PATIENT group

    Patient groups are discovered dynamically: adding P4, P5, P6 files to
    preprocessed/ automatically generates a 4th, 5th, 6th fold with no code changes.

    Returns
    -------
    list of N dicts (one per discovered patient), sorted by patient name:
        [{'fold': 1, 'train_files': [...], 'test_files': [...]}, ...]
    """
    all_files = sorted(PREPROCESSED_DIR.glob('*.h5'))
    all_files = [f for f in all_files if f.name != 'samples.h5']

    groups = {}
    for f in all_files:
        prefix = f.stem.split('_')[0]
        # Assign non-standard prefixes (e.g. 'top-level') to TOPLEVEL_PATIENT
        if not (prefix.startswith('P') and prefix[1:].isdigit()):
            print("WARNING: file {} has unrecognised prefix '{}', treating as {} "
                  "(provenance unverified from TCIA)".format(f.name, prefix, TOPLEVEL_PATIENT))
            prefix = TOPLEVEL_PATIENT
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(f)

    patients = sorted(groups.keys())   # e.g. ['P1', 'P2', 'P3', 'P4', ...]

    folds = []
    for i, test_patient in enumerate(patients):
        train_files = []
        for p in patients:
            if p != test_patient:
                train_files.extend(groups[p])
        folds.append({
            'fold': i + 1,
            'train_files': train_files,
            'test_files': groups[test_patient],
        })
    return folds


# ---------------------------------------------------------------------------
# Band index loading
# ---------------------------------------------------------------------------

def load_band_indices(method, n_bands):
    """
    Return a list of integer band indices for (method, n_bands).

    - method='FullSpectrum' -> list(range(699))
    - Otherwise reads band_selection/bands_{method}.json
    - Returns None if: file missing, key missing, or indices list is empty
    """
    if method == 'FullSpectrum':
        return list(range(699))

    json_path = BAND_SEL_DIR / 'bands_{}.json'.format(method)
    if not json_path.exists():
        return None

    with open(json_path, 'r') as fh:
        data = json.load(fh)

    key = str(n_bands)
    if key not in data:
        return None

    indices = data[key].get('indices', [])
    if not indices:
        return None

    return [int(i) for i in indices]


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

def get_experiment_grid():
    """
    Return list of (method, n_bands) tuples for the active experiment grid.

    Iterates ['PCA', 'MI', 'LASSO', 'ACO'] x BAND_COUNTS.
    A combo is included only if load_band_indices() returns non-None.
    FullSpectrum (699 bands) is always appended.
    """
    grid = []
    for method in ('PCA', 'MI', 'LASSO', 'ACO'):
        for n in BAND_COUNTS:
            if load_band_indices(method, n) is not None:
                grid.append((method, n))
    grid.append(('FullSpectrum', 699))
    return grid


# ---------------------------------------------------------------------------
# Data loading (block sampling)
# ---------------------------------------------------------------------------

def load_patient_data(h5_paths, band_indices, pixels_per_roi=PIXELS_PER_ROI, seed=42):
    """
    Sample pixels from a list of HDF5 ROI files using block sampling.

    Strategy (mirrors sample_pixels.py):
      - Pick one random contiguous block of BLOCK_ROWS rows per file (1 h5py read)
      - Flatten block, randomly choose pixels_per_roi pixels
      - Apply band_indices AFTER the block read

    Parameters
    ----------
    h5_paths     : list of Path objects
    band_indices : list of int
    pixels_per_roi: int (default 500)
    seed         : int random seed

    Returns
    -------
    X : (N, n_bands) float32
    y : (N,) int8   0=NT  1=T
    """
    rng = np.random.default_rng(seed)
    all_X = []
    all_y = []

    for fpath in h5_paths:
        with h5py.File(fpath, 'r') as f:
            n_rows, n_cols, n_bands_full = f['cube'].shape
            label_str = str(f.attrs['label'])
            label_int = np.int8(1 if label_str == 'T' else 0)

            max_start = n_rows - BLOCK_ROWS
            start_row = int(rng.integers(0, max_start + 1))
            end_row   = start_row + BLOCK_ROWS

            # Single read: (BLOCK_ROWS, n_cols, n_bands_full)
            block = f['cube'][start_row:end_row, :, :]

        # Apply band selection after closing file
        block = block[:, :, band_indices]          # (BLOCK_ROWS, n_cols, n_bands)
        flat  = block.reshape(-1, len(band_indices))  # (BLOCK_ROWS*n_cols, n_bands)

        n_avail = flat.shape[0]
        chosen  = rng.choice(n_avail, size=min(pixels_per_roi, n_avail), replace=False)
        sampled = flat[chosen].astype(np.float32)

        all_X.append(sampled)
        all_y.append(np.full(len(chosen), label_int, dtype=np.int8))

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    return X, y


# ---------------------------------------------------------------------------
# Deterministic data loading (test folds only)
# ---------------------------------------------------------------------------

def load_patient_data_deterministic(h5_paths, band_indices, pixels_per_roi=PIXELS_PER_ROI):
    """
    Sample a FIXED pixel grid from each HDF5 file — no RNG.

    Reads just enough rows from the top of the cube to supply pixels_per_roi pixels,
    then picks evenly spaced indices via linspace. Result is byte-for-byte identical
    across runs and independent of any seed.

    Parameters
    ----------
    h5_paths     : list of Path objects
    band_indices : list of int
    pixels_per_roi: int (default 500)

    Returns
    -------
    X : (N, n_bands) float32
    y : (N,) int8   0=NT  1=T
    """
    all_X = []
    all_y = []

    for fpath in h5_paths:
        with h5py.File(fpath, 'r') as f:
            n_rows, n_cols, n_bands_full = f['cube'].shape
            label_str = str(f.attrs['label'])
            label_int = np.int8(1 if label_str == 'T' else 0)

            n_rows_needed = max(1, -(-pixels_per_roi // n_cols))  # ceiling div, no RNG
            block = f['cube'][0:n_rows_needed, :, :]
            block = block[:, :, band_indices]
            flat  = block.reshape(-1, len(band_indices))

        n_avail  = flat.shape[0]
        n_sample = min(pixels_per_roi, n_avail)
        chosen   = np.round(np.linspace(0, n_avail - 1, n_sample)).astype(np.intp)
        sampled  = flat[chosen].astype(np.float32)

        all_X.append(sampled)
        all_y.append(np.full(n_sample, label_int, dtype=np.int8))

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    return X, y


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(y):
    """
    Compute balanced class weights using sklearn.

    Returns
    -------
    dict {0: float, 1: float}
    """
    if len(np.unique(y)) < 2:
        print("WARNING: single-class fold detected, using unit weights")
        return {0: 1.0, 1: 1.0}
    classes = np.array([0, 1])
    weights = _skl_cw('balanced', classes=classes, y=y)
    return {0: float(weights[0]), 1: float(weights[1])}


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

def is_done(csv_path, model, method, n_bands, fold):
    """
    Return True if a row with matching (model, method, n_bands, fold) exists in csv_path.
    Returns False if the CSV does not exist yet.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return False

    with open(csv_path, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if (row.get('model')   == str(model)   and
                    row.get('method')  == str(method)  and
                    row.get('n_bands') == str(n_bands) and
                    row.get('fold')    == str(fold)):
                return True
    return False
