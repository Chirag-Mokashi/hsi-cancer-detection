# Hyperspectral Imaging for Glioblastoma Detection

A comparative study of machine learning and deep learning approaches for intraoperative
tumor margin assessment using hyperspectral histology images.

---

## Overview

This project investigates whether hyperspectral imaging (HSI) can reliably distinguish
glioblastoma tumor tissue from normal brain tissue in histology slides. Unlike standard
RGB cameras which capture only 3 broad color bands, hyperspectral cameras capture
699 narrow spectral bands (~0.73 nm each) across 400-909 nm, producing a full
reflectance spectrum per pixel. This spectral resolution reveals biochemical differences
between tumor and normal tissue that are invisible to the human eye and standard cameras.

The key discriminative signal is the hemoglobin absorption trough at 520-560 nm: tumor
tissue shows deeper absorption (~0.41 normalized reflectance) than normal tissue (~0.46)
due to higher vascular activity, a difference undetectable by RGB imaging.

**Clinical motivation:** Accurate intraoperative tumor margin assessment could help
surgeons achieve cleaner resections and improve patient outcomes in glioblastoma surgery,
where leaving tumor tissue behind is a major cause of recurrence.

---

## Dataset

**HistologyHSI-GB** from The Cancer Imaging Archive (TCIA)

| Property | Value |
|----------|-------|
| Patients used | P1, P2, P3 + top-level ROIs |
| Total ROI folders | 134 |
| Tumor ROIs (T) | 39 |
| Normal ROIs (NT) | 95 |
| Image dimensions | 800 x 1004 pixels per ROI |
| Spectral bands (raw) | 826 (400.5 - 1000.7 nm) |
| Spectral bands (preprocessed) | 699 (400.5 - 909.0 nm) |
| Raw data size | ~177.8 GB |

> Note: Dataset is not included in this repository. Download from
> [TCIA](https://www.cancerimagingarchive.net) via IBM Aspera (search HistologyHSI-GB).

---

## Pipeline

```
Raw ENVI BIL cubes (uint16, 826 bands)
         |
         v
1. Reflectance calibration
   (Raw - Dark) / (White - Dark)
         |
         v
2. Band truncation
   826 -> 699 bands (drop > 909 nm, low SNR)
         |
         v
3. Per-pixel normalization
   Each pixel spectrum / its own max
   (removes brightness variation, preserves spectral shape)
         |
         v
4. Save as HDF5 (.h5)
   float32, gzip compressed
   shape: (800, 1004, 699) per ROI
         |
         v
5. Band selection  [COMPLETE]
   PCA / Mutual Information / LASSO
   Band counts: 4, 10, 20, 50, 100
   Results in band_selection/
         |
         v
6. Model training  [COMPLETE]
   Random Forest + SVM (local CPU)
   HybridSN 3D+2D CNN (Colab A100)
   Vision Transformer (Colab A100)
         |
         v
7. Evaluation  [COMPLETE]
   Leave-One-Patient-Out Cross Validation (LOPOCV)
   186 total runs — 48 RF + 48 SVM + 45 HybridSN + 45 ViT
   Metrics: Accuracy, Sensitivity, Specificity, F1, AUC
```

---

## Repository Structure

```
hsi-cancer-detection/
    1_inspect_dataset.py        Step 1: dataset inspection
    2_preprocess.py             Step 2: preprocessing pipeline
    3a_sample_pixels.py         Step 3: pixel sampling
    3b_analyse_samples.py       Step 3: spectral analysis
    4a_random_forest.py         Step 4a: RF training (local)
    4b_svm.py                   Step 4b: SVM training (local)
    4c_hybridSN.ipynb           Step 4c: HybridSN source notebook (Colab)
    4d_vit.ipynb                Step 4d: ViT source notebook (Colab)
    5_compile_results.py        Step 5: cross-model comparison plots
    plot_individual.py          Per-model detailed plots
    band_selection/
        3c_band_selection.ipynb Colab notebook (PCA, MI, LASSO)
        bands_PCA.json          Selected band indices + wavelengths
        bands_MI.json
        bands_LASSO.json
    results/
        RF/                     rf_v1_results.csv, summary, plots/
        SVM/                    svm_v1_results.csv, summary, plots/
        HybridSN/               hybridSN_v1_results.csv, summary, plots/
        ViT/                    vit_v1_results.csv, summary, plots/
        summary/                combined_results.csv, cross-model plots
    notebooks/
        completed/              Executed notebooks with full cell outputs
    docs/
        DECISIONS_01_session_planning.md
        DECISIONS_02_locked_decisions.md
        APRIL09_CHECKPOINT.md   Results analysis + paper decisions
    scripts/
        audit.py / audit.ipynb  Data integrity checks
        powershell/             Drive verification and upload scripts
    dataset_summary/            EDA plots and spectral signatures
    PROJECT_LOG.md              Full running project log
    .gitignore                  Excludes raw data, h5 files, model weights
```

---

## Current Results

**All 4 models complete — 186 LOPOCV runs (April 2026)**

### Model Comparison (best combo per model)

| Model | Best Combo | AUC | Accuracy | Sensitivity | Specificity | F1 |
|-------|-----------|-----|----------|------------|------------|-----|
| **HybridSN** | LASSO/100 | **0.918 ± 0.039** | 0.789 ± 0.106 | 0.574 ± 0.447 | 0.910 ± 0.075 | 0.699 ± 0.191 |
| **SVM** | LASSO/100 | 0.873 ± 0.072 | **0.851 ± 0.047** | 0.626 ± 0.263 | **0.852 ± 0.178** | **0.777 ± 0.073** |
| **RF** | LASSO/100 | 0.806 ± 0.076 | 0.797 ± 0.040 | 0.476 ± 0.411 | 0.856 ± 0.184 | 0.700 ± 0.071 |
| **ViT** | MI/100 | 0.799 ± 0.099 | 0.619 ± 0.149 | **0.836 ± 0.091** | 0.613 ± 0.215 | 0.637 ± 0.105 |

*Mean ± std across 3 LOPOCV folds*

### Key Findings

- **HybridSN achieves the highest peak AUC (0.918)** via LASSO band selection
- **SVM is the most practical classifier** — best accuracy (0.851) and F1 (0.777)
- **LASSO dominates for RF/SVM/HybridSN; ViT uniquely prefers MI** — suggesting
  attention-based models benefit from distributed spectral information
- **P2 sensitivity collapse** in RF/SVM/HybridSN: trained on P1+P3, these models
  fail to detect tumours in P2 (sens < 0.26 at best combo). ViT avoids this collapse
  (P2 sens = 0.837) through its attention mechanism, though at lower overall specificity
- **Classical ML competitive with DL at n=3**: mean AUC across all combos —
  SVM (0.764) ≈ HybridSN (0.762) ≈ ViT (0.757) > RF (0.739)

> Full analysis: [docs/APRIL09_CHECKPOINT.md](docs/APRIL09_CHECKPOINT.md)
> All results: [results/](results/) | Cross-model plots: [results/summary/](results/summary/)

---

## Setup

### Requirements

```bash
python -m pip install spectral numpy h5py matplotlib scikit-learn
```

Python 3.14+ required. Note: numpy.savez_compressed is broken in Python 3.14
(zip bomb false positive) — this project uses HDF5 (.h5) via h5py instead.

### Running the pipeline

```bash
# Step 1: Inspect dataset
python inspect_dataset.py

# Step 2: Preprocess all ROIs
python preprocess.py

# Verify all preprocessed files
python audit.py

# Regenerate dataset summary visuals
python regenerate_summary.py
```

### Loading a preprocessed file

```python
import h5py
import numpy as np

with h5py.File('preprocessed/P1_ROI_01_C01_T.h5', 'r') as f:
    cube        = f['cube'][:]           # (800, 1004, 699) float32
    wavelengths = f['wavelengths'][:]    # (699,) float32
    label       = str(f.attrs['label'])  # "T" or "NT"
    patient     = str(f.attrs['patient'])
```

---

## Validation Strategy

**Leave-One-Patient-Out Cross Validation (LOPOCV):** Train on 2 patients, test on the
third. Repeated for all 3 combinations. This is the most clinically meaningful
validation — it tests whether the model generalizes to a completely unseen patient,
which is the realistic deployment scenario.

**Class imbalance:** T/NT ratio = 0.41. All models use class weights (~2.4x for tumor)
to prevent bias toward the majority normal class.

> **Provenance caveat:** A subset of ROI files in the TCIA download lacked a patient
> sub-folder (`P1/`, `P2/`, `P3/`), appearing at the top level of the archive. These
> "top-level" ROIs are assumed to belong to patient P1 based on TCIA archive structure,
> but their exact patient provenance is **unverified**. All 134 preprocessed files have
> been patched with a `patient` attribute (via `scripts/fix_patient.py`) and pass full
> audit checks, but readers should note this assumption as a potential threat to validity
> in the LOPOCV fold assignments.

---

## References

1. Simhadri et al. - Brain tumor classification with ML on HELICoiD dataset (RF: 96.78%)
2. Verbers et al. - Glioblastoma wavelength selection with ACO, MLP achieves 86.65% F1
3. Cruz-Guerrero et al. - ViT for brain HSI: 99% intra-patient, 86% inter-patient
4. Pike et al. - MSF+SVM: 93.3% accuracy, 800-890 nm most informative range
5. Manni et al. - HybridSN on colon cancer HSI: AUC 0.82
6. Gopi et al. - MRF + active contour segmentation: 95.2% accuracy

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Notes

- Training for CNN and ViT models runs on Google Colab with Google Drive
- Raw data and preprocessed .h5 files are excluded from this repository via .gitignore
- See PROJECT_LOG.md for full session-by-session development notes
