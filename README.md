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
5. Band selection  [PENDING]
   PCA / Mutual Information / LASSO / Ant Colony Optimization
   Band counts: 4, 10, 20, 50, 100
         |
         v
6. Model training  [PENDING]
   Random Forest + SVM
   MSF + SVM (spectral-spatial graph)
   HybridSN (3D+2D CNN)
   Vision Transformer (ViT)
         |
         v
7. Evaluation  [PENDING]
   Leave-One-Patient-Out Cross Validation (LOPOCV)
   Metrics: Accuracy, Sensitivity, Specificity, F1, AUC
```

---

## Repository Structure

```
hsi-cancer-detection/
    inspect_dataset.py          Step 1: dataset inspection
    preprocess.py               Step 2: preprocessing pipeline (v4)
    audit.py                    Integrity check on all preprocessed h5 files
    fix_patient.py              Utility: fix patient attribute in h5 files
    regenerate_summary.py       Regenerate dataset summary visuals
    dataset_summary/
        class_distribution.png  ROI counts per patient and class
        spectral_signatures.png Mean spectra: tumor vs normal (400-909 nm)
        spectral_signatures.json Mean spectra data (JSON)
        dataset_summary.txt     Full dataset statistics
        rgb_preview.png         Pseudo-RGB tissue preview
        preprocessing_summary.png Before/after preprocessing
    PROJECT_LOG.md              Full running project log
    .gitignore                  Excludes raw data, h5 files, model weights
```

---

## Current Results

**Step 2 complete — Preprocessing verified:**

All 134 ROIs preprocessed and validated:

| Patient | Tumor (T) | Normal (NT) | Total |
|---------|-----------|-------------|-------|
| P1 | 12 | 42 | 54 |
| P2 | 12 | 21 | 33 |
| P3 | 12 | 29 | 41 |
| top-level | 3 | 3 | 6 |
| **TOTAL** | **39** | **95** | **134** |

**Key spectral finding:** The 520-560 nm absorption trough is the most discriminative
spectral region between tumor and normal tissue, consistent with findings in
Verbers et al. and Pike et al. Both classes converge to ~0.98-0.99 in NIR (>700 nm)
after per-pixel normalization.

---

## Planned Models

| Model | Type | Expected F1 (LOPOCV) | Reference |
|-------|------|----------------------|-----------|
| Random Forest | Classical ML | ~0.80 | Simhadri et al. |
| SVM | Classical ML | ~0.82 | - |
| MSF + SVM | Spectral-spatial | ~0.87 | Pike et al. |
| HybridSN | 3D+2D CNN | ~0.84 | Manni et al. |
| Vision Transformer | ViT | ~0.86 | Cruz-Guerrero et al. |

> Target: >85% F1 on Leave-One-Patient-Out cross validation

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
