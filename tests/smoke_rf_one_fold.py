# coding: utf-8
# tests/smoke_rf_one_fold.py
# Run one RF fold on the fixed code to catch import/loader regressions.
# Expected runtime: under 60 seconds (n_estimators=50).
# Exits 0 on success.

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from utils.data_loader import (compute_class_weights, get_experiment_grid,
                                get_lopocv_folds, load_band_indices)
from utils.rf_svm_loader import get_fold_data

t0 = time.time()

folds = get_lopocv_folds()
grid  = get_experiment_grid()
fold  = folds[0]
method, n_bands = grid[0]

print('Smoke test: fold={} method={} n_bands={}'.format(fold['fold'], method, n_bands))

band_indices = load_band_indices(method, n_bands)
X_tr, y_tr, X_te, y_te = get_fold_data(fold, band_indices, seed=42)
weights = compute_class_weights(y_tr)

print('  X_tr={} X_te={} class_weights={}'.format(X_tr.shape, X_te.shape, weights))

clf = RandomForestClassifier(n_estimators=50, class_weight=weights,
                             n_jobs=-1, random_state=42)
clf.fit(X_tr, y_tr)
proba = clf.predict_proba(X_te)[:, 1]
pred  = clf.predict(X_te)

try:
    auc = round(roc_auc_score(y_te, proba), 4)
except ValueError:
    auc = float('nan')

print('  acc={:.4f}  sens={:.4f}  spec={:.4f}  f1={:.4f}  auc={}'.format(
    accuracy_score(y_te, pred),
    recall_score(y_te, pred, pos_label=1, zero_division=0),
    recall_score(y_te, pred, pos_label=0, zero_division=0),
    f1_score(y_te, pred, average='macro', zero_division=0),
    auc
))
print('  elapsed: {:.1f}s'.format(time.time() - t0))
print('SMOKE TEST PASSED')
