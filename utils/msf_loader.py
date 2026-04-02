# coding: utf-8
# msf_loader.py
# Data loader for MSF + SVM (Step 4e - optional, not yet implemented)
# Returns full spatial cubes for graph construction via Prim's algorithm.
#
# Imports shared LOPOCV core from data_loader.py.
# DO NOT redefine get_lopocv_folds() here - always import from data_loader.

from utils.data_loader import load_patient_data, PREPROCESSED_DIR


def get_fold_cubes(fold_dict, band_indices):
    """
    Yields full spatial cubes for each ROI in the fold.
    Required by MSF which builds a pixel graph over the full spatial image.

    Parameters
    ----------
    fold_dict   : dict from get_lopocv_folds() with train_files / test_files
    band_indices: list of int band indices to select

    Yields
    ------
    (cube, label, patient) where cube is (800, 1004, n_bands) float32
    """
    raise NotImplementedError(
        "MSF loader not yet implemented. "
        "Implement after Step 4a-4d are complete."
    )
