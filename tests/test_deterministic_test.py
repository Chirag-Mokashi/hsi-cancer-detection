# coding: utf-8
# tests/test_deterministic_test.py
# Verify that test-set sampling is byte-for-byte identical across runs regardless of seed.

import numpy as np
import pytest

from utils.data_loader import get_lopocv_folds, load_band_indices
from utils.rf_svm_loader import get_fold_data


def test_rf_test_set_is_deterministic():
    folds = get_lopocv_folds()
    band_indices = load_band_indices('LASSO', 10)
    if band_indices is None:
        band_indices = load_band_indices('PCA', 10)
    if band_indices is None or not folds:
        pytest.skip("Required band indices or folds not available")

    fold = folds[0]
    _, _, X_test1, y_test1 = get_fold_data(fold, band_indices, seed=42)
    _, _, X_test2, y_test2 = get_fold_data(fold, band_indices, seed=99)

    assert np.array_equal(X_test1, X_test2), \
        "X_test differs across seeds — test sampling is not deterministic"
    assert np.array_equal(y_test1, y_test2), \
        "y_test differs across seeds — test sampling is not deterministic"


if __name__ == '__main__':
    test_rf_test_set_is_deterministic()
    print("PASS: test set is deterministic regardless of seed")
