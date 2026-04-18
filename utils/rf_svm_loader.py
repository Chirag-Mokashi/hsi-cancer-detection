# coding: utf-8
# rf_svm_loader.py
# Data loader for RF and SVM (Steps 4a/4b)
# Returns flat pixel arrays: (N, n_bands)
#
# Imports shared LOPOCV core from data_loader.py.
# DO NOT redefine get_lopocv_folds() here - always import from data_loader.

from utils.data_loader import (load_patient_data, load_patient_data_deterministic,
                               PIXELS_PER_ROI)


def get_fold_data(fold_dict, band_indices, pixels_per_roi=PIXELS_PER_ROI, seed=42):
    """
    Load flat pixel data for one LOPOCV fold.

    Train split uses random block sampling (reproducible via seed).
    Test split uses deterministic evenly-spaced sampling (no RNG, seed ignored).

    Parameters
    ----------
    fold_dict    : dict from get_lopocv_folds() with 'train_files' / 'test_files'
    band_indices : list of int band indices to select
    pixels_per_roi: int (default 500)
    seed         : int random seed for train split only

    Returns
    -------
    X_train : (N_train, n_bands) float32
    y_train : (N_train,) int8
    X_test  : (N_test,  n_bands) float32
    y_test  : (N_test,)  int8
    """
    X_train, y_train = load_patient_data(
        fold_dict['train_files'], band_indices,
        pixels_per_roi=pixels_per_roi, seed=seed
    )
    X_test, y_test = load_patient_data_deterministic(
        fold_dict['test_files'], band_indices,
        pixels_per_roi=pixels_per_roi
    )
    return X_train, y_train, X_test, y_test
