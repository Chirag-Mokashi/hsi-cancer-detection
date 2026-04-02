# HSI Cancer Detection - Project Log
## Hyperspectral Imaging for Glioblastoma Detection

---

## How to Use This Log

This is a running project log for a hyperspectral imaging ML research project.
When starting a new conversation, read this file to understand:
- What has been done so far
- Current status and blockers
- Technical decisions and why they were made
- Exact file paths, Python version, and environment details
- What the next step is

---

## Project Overview

**Title:** Hyperspectral Imaging for Intraoperative Cancer Detection and Tumor Margin
Assessment: A Comparative Study of Machine Learning and Deep Learning Approaches

**Goal:** Compare multiple ML/DL approaches for detecting glioblastoma tumor tissue in
hyperspectral histology images. Find the best method balancing accuracy, speed, and
clinical feasibility.

**Dataset:** HistologyHSI-GB (The Cancer Imaging Archive / TCIA)

**GitHub:** https://github.com/Chirag-Mokashi/hsi-cancer-detection

---

## Environment

| Tool | Detail |
|------|--------|
| OS | Windows 11 |
| IDE | Antigravity (Google VS Code fork with Gemini) |
| Python | 3.14.3 |
| Shell | PowerShell (use PowerShell syntax, not cmd) |
| Key libraries | numpy 2.4.3, spectral 0.24, h5py, matplotlib |
| Training (Steps 4+) | Google Colab + Google Drive |
| Data path | C:\Users\mokas\OneDrive\Desktop\HSI |
| Preprocessed path | C:\Users\mokas\OneDrive\Desktop\HSI\preprocessed |

### Important Python 3.14 Notes
- numpy.savez_compressed produces zip files that Python 3.14 rejects as possible zip bomb
- Solution: use HDF5 (.h5) format via h5py instead of .npz
- Unicode/special characters in scripts cause SyntaxErrors - all scripts must be ASCII only
- Use python -m pip install not pip install directly

### PowerShell Syntax (not cmd)
- Delete folder: Remove-Item -Recurse -Force "path"
- NOT: rmdir /s /q "path"

---

## Dataset Details

### Source
- Name: HistologyHSI-GB
- From: The Cancer Imaging Archive (TCIA)
- Downloaded via: IBM Aspera
- Full dataset: 13 patients (P1 through P13)
- We are using: P1, P2, P3 + top-level ROIs only (134 ROI folders total)

### Raw Data Specifications
| Property | Value |
|----------|-------|
| Image dimensions | 800 x 1004 pixels per ROI |
| Spectral bands | 826 bands |
| Spectral range | 400.5 nm to 1000.7 nm |
| Spectral step | ~0.73 nm per band |
| Data type | uint16 (raw sensor counts 65-3106) |
| Storage format | ENVI BIL binary + .hdr header |
| Calibration files | darkReference + whiteReference (per ROI) |
| File size per ROI | ~1.33 GB raw |

### Folder Structure
```
HSI/
    P1/
        ROI_01_C01_T/
            raw                 Binary hyperspectral cube
            raw.hdr             ENVI header
            darkReference       Dark calibration frame
            darkReference.hdr
            whiteReference      White calibration frame
            whiteReference.hdr
            rgb.png             Quick-look preview
        ROI_02_C01_NT/
        ...
    P2/
    P3/
    ROI_01_C10_T/               Top-level ROIs (no patient subfolder)
    ROI_02_C10_NT/
    HistologyHSI-GB.sums        Checksum file
    preprocessed/               Output folder from Step 2
    dataset_summary/            Visuals and summary files
    PROJECT_LOG.md              This file
```

### ROI Naming Convention
```
ROI_01_C03_T
 |    |   |
 |    |   +-- T = Tumor  /  NT = Normal Tissue
 |    +------ C03 = Cut number 3
 +----------- 01 = ROI type (region on the slide)
```

### Dataset Statistics
| Metric | Value |
|--------|-------|
| Total ROI folders | 134 |
| Tumor ROIs (T) | 39 |
| Normal ROIs (NT) | 95 |
| T/NT ratio | 0.41 (imbalanced - handle with class weights) |
| Total raw size | ~177.8 GB |

### Per-Patient Breakdown
| Patient | Tumor (T) | Normal (NT) | Raw Size |
|---------|-----------|-------------|----------|
| P1 | 12 | 42 | ~71.7 GB |
| P2 | 12 | 21 | ~43.8 GB |
| P3 | 12 | 29 | ~54.4 GB |
| top-level | 3 | 3 | ~8.0 GB |

---

## Step 1 - Dataset Inspection (COMPLETE)

**Script:** inspect_dataset.py

**What it does:**
- Walks folder tree, finds all ROI folders
- Parses ENVI header for dimensions, dtype, wavelengths
- Checks calibration file presence
- Reads 20x20 patch to confirm data is readable
- Prints full summary

**Key findings:**
- 134 ROI folders confirmed
- Image shape: 800 x 1004 x 826
- Format: uint16, BIL interleave
- Spectral range: 400.5-1000.7 nm
- Calibration files present for all ROIs
- Values: range [65.0, 3106.0], mean 1186.91

**Git commit:** Step 1: dataset inspection complete - 134 ROIs, 3 patients, 826 bands, 400-1001nm

---

## Step 2 - Preprocessing (COMPLETE)

**Script:** preprocess.py (v4 - strict verification + auto-deletion)

**What it does per ROI:**
1. Load raw ENVI BIL cube -> float32 (rows, cols, bands)
2. Load darkReference and whiteReference calibration frames
3. Apply: Reflection = (Raw - Dark) / (White - Dark)
4. Clip values to [0.0, 1.0]
5. Drop bands above 909 nm: 826 -> 699 bands
6. Per-pixel normalization: divide each pixel spectrum by its own max
7. Save as HDF5 (.h5) with gzip compression
8. Strict verification before marking as done
9. Auto-delete patient raw folder after all their ROIs pass verification

**Output per file:** preprocessed/<patient>_<roi_name>.h5
- cube: float32 (800, 1004, 699)
- label: "T" or "NT"
- patient: "P1", "P2", "P3", "top-level"
- wavelengths: float32 array, 699 values, 400.5-909.0 nm

**Verification checks (strict - all must pass before raw deletion):**
1. File size >= 10 MB
2. Has cube, wavelengths, label, patient
3. Shape exactly (800, 1004, 699)
4. Label is T or NT
5. Wavelength count is 699
6. Values in [0, 1] checked on centre pixel patch
7. No NaN or Inf values

**Final verified state:**
| Patient | Tumor (T) | Normal (NT) | Total |
|---------|-----------|-------------|-------|
| P1 | 12 | 42 | 54 |
| P2 | 12 | 21 | 33 |
| P3 | 12 | 29 | 41 |
| top-level | 3 | 3 | 6 |
| TOTAL | 39 | 95 | 134 |

All 134 files passed full integrity check: shape (800, 1004, 699), values [0,1],
no NaN/Inf, correct patient and label attributes verified via audit.py.

**Disk space issues encountered and resolved:**
- h5 files are ~1.5 GB each compressed -> 134 files x 1.5 GB = ~200 GB total
- Raw (177 GB) + preprocessed (200 GB) approaches 510 GB drive limit
- Solution: manually delete verified raw folders before processing next batch
- Always pause OneDrive before running preprocess.py or any large deletion
- OneDrive syncing causes "Access Denied" errors on deletion

**P1 re-download complications:**
- P1 raw was deleted early (disk space), had to re-download
- First Aspera re-download failed at 53.7 GB (disk full, error code 25)
- Second attempt accidentally downloaded full PKG (all 13 patients) instead of P1 only
  - PKG download stopped mid-P8, wasted ~23.7 GB, had to delete
- Third attempt: correctly selected only the 9 missing P1 ROI folders in Aspera
  - Files downloaded to HSI root instead of HSI/P1/ subfolder
  - preprocess.py tagged them as patient="top-level" instead of "P1"
  - Fixed with fix_patient.py: renamed files and patched patient attribute in-place
  - Also deleted corrupt 800-byte stub file created during a failed earlier attempt

**Utility scripts created:**
- audit.py: integrity check on all 134 h5 files (shape, patient, label, wavelengths, values)
- fix_patient.py: renamed top-level_ROI_*.h5 -> P1_ROI_*.h5 and patched patient attribute
- regenerate_summary.py: regenerates all 4 stale dataset_summary files with full 134 ROIs

**How to load a preprocessed file:**
```python
import h5py
import numpy as np

with h5py.File('preprocessed/P1_ROI_01_C01_T.h5', 'r') as f:
    cube        = f['cube'][:]           # (800, 1004, 699) float32
    wavelengths = f['wavelengths'][:]    # (699,) float32
    label       = str(f.attrs['label'])  # "T" or "NT"
    patient     = str(f.attrs['patient'])
```

**Git commit:** Step 2: preprocessing complete - 134 ROIs verified, patient labels fixed, dataset summary regenerated

---

## Dataset Summary Files

**Location:** HSI/dataset_summary/
**Script to regenerate:** regenerate_summary.py

| File | Description | Status |
|------|-------------|--------|
| dataset_summary.txt | Full dataset specs and preprocessing details | Current (134 ROIs) |
| spectral_signatures.json | Mean spectra for tumor and normal tissue (699 bands, 39T/95NT) | Current (134 ROIs) |
| class_distribution.png | Bar chart: tumor vs normal ROIs per patient | Current (134 ROIs) |
| spectral_signatures.png | Spectral curves: tumor vs normal (400-909 nm) | Current (134 ROIs) |
| rgb_preview.png | Pseudo-RGB tissue images (R=650nm G=550nm B=450nm) | Unchanged - correct |
| preprocessing_summary.png | Before/after preprocessing pipeline | Unchanged - correct |

**Key spectral findings (from full 134 ROI dataset):**
- Tumor and normal tissue are spectrally distinct at 520-560 nm (hemoglobin absorption trough)
- Tumor drops to ~0.41 reflectance at trough vs normal ~0.46 - deeper absorption due to higher vascular activity
- Both spectra converge near 1.0 in NIR (>700 nm) due to per-pixel normalization
- Most discriminative region: 520-560 nm green band absorption trough
- This separation justifies hyperspectral imaging over standard RGB cameras

---

## Step 3 - Band Selection (COMPLETE)

**Goal:** Reduce 699 bands to informative subset. Compare 3 methods at 5 band counts.
ACO deferred — complex to implement, will add later if needed.

**Scripts:**
- sample_pixels.py: samples 500 pixels per ROI -> preprocessed/samples.h5 (67000 x 699)
- analyse_samples.py: spectral analysis, Cohen's d per band -> dataset_summary/sample_analysis/
- band_selection/band_selection.ipynb: PCA, MI, LASSO on Colab (CPU runtime, ~5 min)

**Methods run:**
| Method | Type | Library | Notes |
|--------|------|---------|-------|
| PCA | Unsupervised | scikit-learn | Top loading band per component |
| Mutual Information | Supervised | scikit-learn | Ranked by MI score, n_jobs=-1 |
| LASSO | Supervised | scikit-learn | L1 path, 50 alphas, 20000 pixel subsample |

**Band counts:** 4, 10, 20, 50, 100

**Output files (band_selection/):**
- bands_PCA.json, bands_MI.json, bands_LASSO.json
- Each: {"4": {"indices": [...], "wavelengths_nm": [...]}, "10": ..., ...}
- band_selection_PCA.png, band_selection_MI.png, band_selection_LASSO.png

**Key results:**

MI - selects exclusively from Red 570-644 nm (highest Cohen's d region, mean=0.88).
All band counts tightly clustered in this region. Scientifically strongest result.

LASSO - selects from 3 distinct spectral regions:
  n=4 : 596-597 nm + 788-805 nm
  n=10: 569-600 nm + 788-821 nm
  n=20: Red + Red-edge (685-700 nm) + NIR (788-832 nm)
  n=50+: expands all 3 regions
NIR selection (788-832 nm) aligns with Pike et al. (800-890 nm most informative).

PCA - works well at n=4 (544-558 nm). Degrades at n>=20: selects consecutive
correlated bands in 400-465 nm region. Band 698 (908.3 nm, worst discriminability
by Cohen's d=0.296) appears at n=50. This is a known limitation of unsupervised
band selection under high inter-band correlation (mean |r|=0.557 in this dataset).
Root causes: high HSI inter-band correlation, per-pixel normalization flattening
NIR variance, PCA being unsupervised (ignores class labels). Not a data size issue.
Will be reported as a finding in the paper.

**Samples.h5 location:** preprocessed/samples.h5 (121 MB, excluded from git)
Also uploaded to Google Drive: G:\My Drive\HSI\samples.h5 (Colab Pro account)

**Git commit:** Step 3 complete - band selection done (PCA, MI, LASSO at 5 band counts)

---

## Step 4 - Models (IN PROGRESS)

### Step 4a - Random Forest (COMPLETE)

**Script:** 4a_random_forest.py
**Config:** 500 trees, class_weight=balanced, n_jobs=-1, seed=42
**Grid:** 16 combos (PCA/MI/LASSO x 5 band counts + FullSpectrum) x 3 LOPOCV folds = 48 runs
**Results:** results/RF/rf_v1_results.csv (48 rows), rf_v1_summary.csv (16 rows)

**Key findings:**
- LASSO/100 best AUC (0.806 mean), LASSO consistently strong across band counts
- MI most consistent (lowest std ~0.06 across folds)
- PCA worst sensitivity — unsupervised band selection limitation confirmed in model results
- FullSpectrum AUC 0.767 — not much gain over LASSO despite 699 bands
- Fold 2 (P2 held out) hardest across ALL methods — sensitivity collapses to near-zero for
  LASSO/FullSpectrum on P2 test set. Patient-level generalization finding worth reporting.

**Top results (mean AUC across 3 folds):**
| Method | Bands | AUC | F1 |
|--------|-------|-----|----|
| LASSO | 100 | 0.806 | 0.614 |
| LASSO | 50 | 0.797 | 0.613 |
| LASSO | 20 | 0.784 | 0.608 |
| FullSpectrum | 699 | 0.767 | 0.594 |
| MI | 100 | 0.758 | 0.673 |

**Git commit:** Step 4a: RF complete - 48 runs, LASSO/100 best AUC 0.806

### Step 4b - SVM (RUNNING)

| Model | Type | Where | Status |
|-------|------|-------|--------|
| Random Forest | Classical ML | Local | COMPLETE |
| SVM | Classical ML | Local | RUNNING |
| HybridSN 3D+2D CNN | Deep Learning | Colab T4 | PENDING |
| Vision Transformer | Deep Learning | Colab A100 | PENDING |
| MSF + SVM | Spectral-spatial | Local | OPTIONAL/LAST |

**Class imbalance:** T/NT = 0.41 - use class weights (~2.4x for tumor class)

**MSF method steps (from scratch):**
1. Train SVM for pixel-wise probability map
2. Select high-confidence pixels as markers
3. Use mutual information to select optimal bands for edge weights
4. Build graph: nodes = pixels, edges = spectral dissimilarity (SAM)
5. Grow Minimum Spanning Forest from markers (Prim's algorithm)
6. Classify by tree root label
7. Majority vote with SVM

---

## Step 5 - Validation (PENDING)

| Method | Description | Purpose |
|--------|-------------|---------|
| Intra-patient split | Train/test same patient (80/20) | Upper bound |
| LOPOCV | Leave-one-patient-out | Clinical generalization |
| Simple split | Random 80/20 | Fast prototyping only |

**Target:** >85% F1 score on LOPOCV (inter-patient)

---

## Step 6 - Evaluation Metrics (PENDING)

For every model x band method x band count combination:
- Accuracy, Sensitivity, Specificity, F1 (macro), AUC
- Training time, inference time per image

---

## Step 7 - Ablation Studies (PENDING)

1. Band count sweep: 4 -> 10 -> 20 -> 50 -> 100 bands
2. Patch size sweep (CNN/ViT): 1x1, 6x6, 11x11, 22x22
3. Class weighting: balanced vs unbalanced

---

## Key Technical Decisions and Why

### Why HDF5 instead of NPZ
Python 3.14 introduced stricter zip validation that rejects numpy's overlapped zip entries.
All .npz files were unreadable with "possible zip bomb" error. Switched to h5py HDF5
format which has no such issue and is the standard for large scientific datasets.

### Why drop bands above 909 nm
Signal-to-noise ratio drops significantly above 909 nm. Consistent with Verbers et al.
(glioblastoma paper). 826 -> 699 bands kept.

### Why per-pixel normalization
After calibration, overall brightness still varies between pixels (blood vessels absorb
more light). Per-pixel normalization removes brightness variation and forces the classifier
to learn from spectral shape differences only - which is the meaningful signal.

### Why LOPOCV
Most clinically meaningful validation. Tests whether the model generalizes to a completely
unseen patient. Inter-patient accuracy is always lower than intra-patient but is the
realistic performance estimate for clinical deployment.

### Why class weights for imbalance
39 tumor vs 95 normal ROIs (ratio 0.41). Without correction, models are biased toward
predicting normal and achieve misleadingly high accuracy. Class weights of ~2.4x for
tumor class corrects this during training.

### Why in-place calibration operations
Each raw cube is ~2.5 GB as float32. Creating a second copy during calibration would
require 5 GB RAM. In-place operations (raw -= dark, raw /= denom) avoid this allocation.

---

## Literature Reference (6 Core Papers)

1. Simhadri et al. - Brain tumor ML: RF achieves 96.78% on HELICoiD dataset
2. Verbers et al. - Glioblastoma: ACO selects 20 wavelengths, MLP 86.65% F1
3. Cruz-Guerrero et al. - ViT: 99% intra-patient, 86% inter-patient on brain HSI
4. Pike et al. - MSF+SVM: 93.3% accuracy, 800-890 nm most informative
5. Manni et al. - HybridSN CNN: AUC 0.82 on colon cancer
6. Gopi et al. - MRF + active contour + optimization: 95.2% accuracy

---

## Commands Reference

### Install dependencies
```
python -m pip install spectral numpy h5py matplotlib scikit-learn
```

### Run scripts
```
python inspect_dataset.py
python preprocess.py
python audit.py
python regenerate_summary.py
```

### Git workflow
```
git add <files>
git commit -m "Step X: description"
git push
```

### Check disk space
```
wmic logicaldisk get caption,freespace,size
```

### Count preprocessed files
```
python -c "from pathlib import Path; print(len(list(Path(r'C:\Users\mokas\OneDrive\Desktop\HSI\preprocessed').glob('*.h5'))))"
```

### Delete raw folder (PowerShell)
```
Remove-Item -Recurse -Force "C:\Users\mokas\OneDrive\Desktop\HSI\P1"
```

### Pause OneDrive before long runs
Right click OneDrive tray icon -> Pause syncing -> 24 hours

### Run integrity audit on all h5 files
```
python audit.py
```

---

## Planned Git Commit History

```
Step 1: dataset inspection complete - 134 ROIs, 3 patients, 826 bands, 400-1001nm
Add gitignore - exclude raw data, npz files, and OS artifacts
Step 2: dataset summary and visuals - spectral signatures, RGB previews (partial data)
Step 2: preprocessing complete - 134 ROIs verified, patient labels fixed, dataset summary regenerated
Step 3: band selection - PCA, MI, LASSO comparison (ACO deferred)
Step 4a: Random Forest and SVM training
Step 4b: MSF + SVM training
Step 4c: HybridSN CNN training (Colab)
Step 4d: Vision Transformer training (Colab)
Step 5/6: LOPOCV evaluation and metrics
Step 7: ablation studies - band count, patch size, class weighting
```

---

## Next Actions

1. Commit Step 3 complete
2. Begin Step 4a: Random Forest + SVM (local, scikit-learn)
   - Load preprocessed h5 files, slice bands using band_selection JSON files
   - Train per band method (MI, LASSO, PCA) x band count (4,10,20,50,100)
   - LOPOCV: train on 2 patients, test on 1 (top-level ROIs assigned to P1)
   - Class weights: ~2.4x for tumor
   - Metrics: Accuracy, Sensitivity, Specificity, F1 (macro), AUC
3. Step 4b: MSF + SVM (local, custom Prim's algorithm)
4. Step 4c: HybridSN CNN (Colab, T4 GPU, PyTorch)
5. Step 4d: Vision Transformer (Colab, A100 GPU, PyTorch)
