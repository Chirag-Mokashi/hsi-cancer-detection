# coding: utf-8
# 4a_random_forest.py
# Step 4a: Random Forest classifier on HSI band-selected data
# LOPOCV (Leave-One-Patient-Out), 3 folds x 16 combos = 48 runs
# Results auto-saved after every fold; supports checkpoint/resume.

import csv
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             recall_score, roc_auc_score)
from tqdm import tqdm

from utils.data_loader import (compute_class_weights, get_experiment_grid,
                                get_lopocv_folds, is_done, load_band_indices)
from utils.rf_svm_loader import get_fold_data

# ---- Config ----
MODEL          = 'RF'
VERSION        = 'v2'
N_ESTIMATORS   = 500
RANDOM_SEED    = 42
RESULTS_DIR    = Path('results/RF_v2')
RESULTS_CSV    = RESULTS_DIR / 'rf_{}_results.csv'.format(VERSION)
SUMMARY_CSV    = RESULTS_DIR / 'rf_{}_summary.csv'.format(VERSION)
CSV_COLS       = ['model', 'method', 'n_bands', 'fold',
                  'accuracy', 'sensitivity', 'specificity', 'f1', 'auc',
                  'train_time_sec', 'inference_time_per_image_ms',
                  'git_sha', 'seed', 'code_version']
METRIC_COLS    = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc',
                  'train_time_sec', 'inference_time_per_image_ms']
# ----------------


def append_row(csv_path, row):
    """Append one result row to CSV, writing header if file is new."""
    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def generate_summary(results_csv, summary_csv):
    """
    Read results CSV and write summary with mean+-std per (method, n_bands).
    """
    from collections import defaultdict
    data = defaultdict(lambda: {m: [] for m in METRIC_COLS})

    with open(results_csv, 'r', newline='') as fh:
        for row in csv.DictReader(fh):
            key = (row['method'], row['n_bands'])
            for m in METRIC_COLS:
                if m in row and row[m] != '':
                    data[key][m].append(float(row[m]))

    sum_cols = ['model', 'method', 'n_bands', 'fold'] + METRIC_COLS
    with open(summary_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=sum_cols)
        writer.writeheader()
        for (method, n_bands), metrics in sorted(data.items()):
            row = {'model': MODEL, 'method': method,
                   'n_bands': n_bands, 'fold': 'summary'}
            for m, vals in metrics.items():
                if vals:
                    arr = np.array(vals)
                    row[m] = '{:.4f}+-{:.4f}'.format(arr.mean(), arr.std())
                else:
                    row[m] = ''
            writer.writerow(row)

    print('Summary saved to:', summary_csv)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    folds = get_lopocv_folds()
    grid  = get_experiment_grid()

    print('RF v1  |  {} combos x {} folds = {} runs'.format(
        len(grid), len(folds), len(grid) * len(folds)))
    print('Results -> {}'.format(RESULTS_CSV))
    print()

    combo_bar = tqdm(grid, desc='Combos', unit='combo')
    for method, n_bands in combo_bar:
        combo_bar.set_postfix({'method': method, 'bands': n_bands})
        band_indices = load_band_indices(method, n_bands)

        fold_bar = tqdm(folds, desc='  Folds', unit='fold', leave=False)
        for fold in fold_bar:
            fold_num = fold['fold']
            fold_bar.set_postfix({'fold': fold_num})

            if is_done(RESULTS_CSV, MODEL, method, n_bands, fold_num):
                fold_bar.write('  [skip] fold={} {}/{} already done'.format(
                    fold_num, method, n_bands))
                continue

            # ---- Load data ----
            t0 = time.time()
            X_tr, y_tr, X_te, y_te = get_fold_data(fold, band_indices, seed=RANDOM_SEED)
            weights = compute_class_weights(y_tr)
            load_sec = time.time() - t0

            # ---- Train ----
            t1 = time.time()
            clf = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                class_weight=weights,
                n_jobs=-1,
                random_state=RANDOM_SEED
            )
            clf.fit(X_tr, y_tr)
            train_time_sec = time.time() - t1

            # ---- Evaluate ----
            t2 = time.time()
            proba = clf.predict_proba(X_te)[:, 1]
            pred  = clf.predict(X_te)
            inf_ms_per_image = (time.time() - t2) / len(y_te) * 1000

            try:
                auc_val = round(roc_auc_score(y_te, proba), 6)
            except ValueError:
                auc_val = float('nan')

            try:
                git_sha = subprocess.check_output(
                    ['git', 'rev-parse', '--short', 'HEAD'],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except Exception:
                git_sha = 'unknown'

            row = {
                'model':                      MODEL,
                'method':                     method,
                'n_bands':                    n_bands,
                'fold':                       fold_num,
                'accuracy':                   round(accuracy_score(y_te, pred), 6),
                'sensitivity':                round(recall_score(y_te, pred, pos_label=1, zero_division=0), 6),
                'specificity':                round(recall_score(y_te, pred, pos_label=0, zero_division=0), 6),
                'f1':                         round(f1_score(y_te, pred, average='macro', zero_division=0), 6),
                'auc':                        auc_val,
                'train_time_sec':             round(train_time_sec, 3),
                'inference_time_per_image_ms': round(inf_ms_per_image, 6),
                'git_sha':                    git_sha,
                'seed':                       RANDOM_SEED,
                'code_version':               '{}-{}-{}'.format(MODEL, VERSION, git_sha),
            }
            append_row(RESULTS_CSV, row)

            fold_bar.write(
                '  fold={} {}/{:3d}  acc={:.3f} sens={:.3f} spec={:.3f} '
                'f1={:.3f} auc={:.3f}  train={:.1f}s inf={:.4f}ms/img'.format(
                    fold_num, method, n_bands,
                    row['accuracy'], row['sensitivity'], row['specificity'],
                    row['f1'], row['auc'],
                    train_time_sec, inf_ms_per_image
                )
            )

    print()
    print('All folds complete.')
    generate_summary(RESULTS_CSV, SUMMARY_CSV)
    print('Done.')


if __name__ == '__main__':
    main()
