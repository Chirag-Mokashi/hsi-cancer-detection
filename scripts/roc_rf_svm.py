# coding: utf-8
# scripts/roc_rf_svm.py
#
# Re-trains RF and SVM at their best combo (LASSO/100, 3 LOPOCV folds)
# to capture per-fold prediction probabilities, then generates overlaid
# ROC curve plots (3 folds on one axes per model).
#
# Must be run from the project root: python scripts/roc_rf_svm.py
#
# Outputs:
#   results/RF/roc_LASSO_100_fold{1,2,3}_ytrue.npy
#   results/RF/roc_LASSO_100_fold{1,2,3}_yproba.npy
#   results/RF/plots/roc_LASSO_100_overlay.png
#   results/SVM/roc_LASSO_100_fold{1,2,3}_ytrue.npy
#   results/SVM/roc_LASSO_100_fold{1,2,3}_yproba.npy
#   results/SVM/plots/roc_LASSO_100_overlay.png
#
# NOTE: Does NOT modify existing results CSVs. Read-only w.r.t. prior runs.
# NOTE: SVM LASSO/100 takes approx 12 min/fold x 3 folds = ~36 min total.
#       RF LASSO/100 takes approx 2-3 min/fold x 3 folds = ~8 min total.
#
# Decisions locked April 14 2026: RF and SVM best combo = LASSO/100.

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Add project root to path so utils can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import (compute_class_weights, get_lopocv_folds,
                                load_band_indices)
from utils.rf_svm_loader import get_fold_data

# ============================================================
# CONFIG
# ============================================================
RANDOM_SEED  = 42
ROC_METHOD   = 'LASSO'
ROC_N_BANDS  = 100

# RF hyperparams (must match 4a_random_forest.py)
RF_N_ESTIMATORS = 500

# SVM hyperparams (must match 4b_svm.py)
SVM_KERNEL = 'rbf'

RF_DIR  = Path('results/RF')
SVM_DIR = Path('results/SVM')

FOLD_COLORS = {1: '#4C72B0', 2: '#DD8452', 3: '#55A868'}
PATIENT_MAP = {1: 'P1', 2: 'P2', 3: 'P3'}


# ============================================================
# HELPERS
# ============================================================

def npy_prefix(results_dir, fold):
    return results_dir / 'roc_{}_{}_fold{}'.format(
        ROC_METHOD, ROC_N_BANDS, fold)


def save_fold_predictions(results_dir, fold, y_true, y_proba):
    prefix = npy_prefix(results_dir, fold)
    np.save(str(prefix) + '_ytrue.npy', y_true)
    np.save(str(prefix) + '_yproba.npy', y_proba)
    print('  Saved: {}'.format(prefix))


def load_fold_predictions(results_dir, fold):
    prefix = npy_prefix(results_dir, fold)
    ytrue_path  = str(prefix) + '_ytrue.npy'
    yproba_path = str(prefix) + '_yproba.npy'
    if not Path(ytrue_path).exists():
        return None, None
    y_true  = np.load(ytrue_path)
    y_proba = np.load(yproba_path)
    return y_true, y_proba


def plot_roc_overlay(results_dir, model_name, combo_label):
    """Load saved fold predictions and plot overlaid ROC curves."""
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    any_plotted = False

    for fold in [1, 2, 3]:
        y_true, y_proba = load_fold_predictions(results_dir, fold)
        if y_true is None:
            print('  [warn] No predictions for fold {}'.format(fold))
            continue

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        label = 'Fold {} ({} held out), AUC={:.3f}'.format(
            fold, PATIENT_MAP[fold], auc)
        ax.plot(fpr, tpr, color=FOLD_COLORS[fold], linewidth=2, label=label)
        any_plotted = True

    if not any_plotted:
        print('  [error] No fold predictions found for {}'.format(model_name))
        plt.close(fig)
        return

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Chance')

    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('{} -- Best Combo ({}) ROC Curves'.format(model_name, combo_label))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)

    out_path = plots_dir / 'roc_{}_overlay.png'.format(
        '{}_{}'.format(ROC_METHOD, ROC_N_BANDS))
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print('Saved: {}'.format(out_path))


# ============================================================
# RF ROC
# ============================================================

def run_rf_roc():
    print('=' * 60)
    print('RF ROC -- LASSO/{}'.format(ROC_N_BANDS))
    print('Re-training RF at best combo to capture predictions.')
    print('=' * 60)

    RF_DIR.mkdir(parents=True, exist_ok=True)
    band_indices = load_band_indices(ROC_METHOD, ROC_N_BANDS)
    if band_indices is None:
        print('[error] Could not load band indices for {}/{}'.format(
            ROC_METHOD, ROC_N_BANDS))
        return

    folds = get_lopocv_folds()
    for fold in folds:
        fold_num = fold['fold']
        # Skip if already saved
        prefix = npy_prefix(RF_DIR, fold_num)
        if Path(str(prefix) + '_yproba.npy').exists():
            print('  [skip] fold={} predictions already saved'.format(fold_num))
            continue

        print('  fold={} training...'.format(fold_num))
        t0 = time.time()
        X_tr, y_tr, X_te, y_te = get_fold_data(fold, band_indices, seed=RANDOM_SEED)
        weights = compute_class_weights(y_tr)
        load_sec = time.time() - t0

        t1 = time.time()
        clf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            class_weight=weights,
            n_jobs=-1,
            random_state=RANDOM_SEED
        )
        clf.fit(X_tr, y_tr)
        train_sec = time.time() - t1

        proba = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba)
        print('  fold={} done  train={:.1f}s  AUC={:.4f}'.format(
            fold_num, train_sec, auc))
        save_fold_predictions(RF_DIR, fold_num, y_te, proba)

    print('Generating RF ROC overlay plot...')
    plot_roc_overlay(RF_DIR, 'RF', '{}/{}'.format(ROC_METHOD, ROC_N_BANDS))


# ============================================================
# SVM ROC
# ============================================================

def run_svm_roc():
    print('=' * 60)
    print('SVM ROC -- LASSO/{}'.format(ROC_N_BANDS))
    print('Re-training SVM at best combo to capture predictions.')
    print('Estimated time: ~12 min/fold x 3 folds = ~36 min total.')
    print('=' * 60)

    SVM_DIR.mkdir(parents=True, exist_ok=True)
    band_indices = load_band_indices(ROC_METHOD, ROC_N_BANDS)
    if band_indices is None:
        print('[error] Could not load band indices for {}/{}'.format(
            ROC_METHOD, ROC_N_BANDS))
        return

    folds = get_lopocv_folds()
    for fold in folds:
        fold_num = fold['fold']
        # Skip if already saved
        prefix = npy_prefix(SVM_DIR, fold_num)
        if Path(str(prefix) + '_yproba.npy').exists():
            print('  [skip] fold={} predictions already saved'.format(fold_num))
            continue

        print('  fold={} loading data...'.format(fold_num))
        t0 = time.time()
        X_tr, y_tr, X_te, y_te = get_fold_data(fold, band_indices, seed=RANDOM_SEED)
        weights = compute_class_weights(y_tr)
        load_sec = time.time() - t0
        print('  fold={} data loaded ({:.1f}s)  training SVM...'.format(
            fold_num, load_sec))

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)
        X_te_sc  = scaler.transform(X_te)

        t1 = time.time()
        clf = SVC(
            kernel=SVM_KERNEL,
            probability=True,
            class_weight=weights,
            random_state=RANDOM_SEED
        )
        clf.fit(X_tr_sc, y_tr)
        train_sec = time.time() - t1

        proba = clf.predict_proba(X_te_sc)[:, 1]
        auc = roc_auc_score(y_te, proba)
        print('  fold={} done  train={:.1f}s  AUC={:.4f}'.format(
            fold_num, train_sec, auc))
        save_fold_predictions(SVM_DIR, fold_num, y_te, proba)

    print('Generating SVM ROC overlay plot...')
    plot_roc_overlay(SVM_DIR, 'SVM', '{}/{}'.format(ROC_METHOD, ROC_N_BANDS))


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate ROC curves for RF and SVM at best combo (LASSO/100).')
    parser.add_argument('--model', choices=['RF', 'SVM', 'both'], default='both',
                        help='Which model to run (default: both)')
    args = parser.parse_args()

    if args.model in ('RF', 'both'):
        run_rf_roc()
        print()

    if args.model in ('SVM', 'both'):
        run_svm_roc()
        print()

    print('Done. ROC overlays saved to results/RF/plots/ and results/SVM/plots/')
