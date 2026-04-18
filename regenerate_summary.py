import h5py
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from utils.config import DATA_ROOT, PREPROCESSED_DIR

PREPROCESSED = PREPROCESSED_DIR
SUMMARY_DIR  = DATA_ROOT / "dataset_summary"
SUMMARY_DIR.mkdir(exist_ok=True)

files = sorted(PREPROCESSED.glob("*.h5"))
total = len(files)
print(f"Found {total} h5 files\n")

# ----------------------------------------------------------------
# 1. Collect per-patient counts + spectral signatures
# ----------------------------------------------------------------
patients = ["P1", "P2", "P3", "top-level"]
counts = {p: {"T": 0, "NT": 0} for p in patients}

tumor_spectra  = []
normal_spectra = []
wavelengths    = None

for i, f in enumerate(files, 1):
    print(f"[{i}/{total}] Reading {f.name} ...", flush=True)
    with h5py.File(f, "r") as hf:
        label   = str(hf.attrs["label"])
        patient = str(hf.attrs["patient"])
        wl      = hf["wavelengths"][:]
        if wavelengths is None:
            wavelengths = wl

        # Centre 50x50 patch mean spectrum
        r0, r1 = 375, 425
        c0, c1 = 477, 527
        patch = hf["cube"][r0:r1, c0:c1, :]   # (50, 50, 699)
        mean_spectrum = patch.reshape(-1, patch.shape[2]).mean(axis=0)

        counts[patient][label] += 1
        if label == "T":
            tumor_spectra.append(mean_spectrum)
        else:
            normal_spectra.append(mean_spectrum)

tumor_mean  = np.mean(tumor_spectra,  axis=0)
normal_mean = np.mean(normal_spectra, axis=0)

print("\nCounts:")
for p in patients:
    print(f"  {p}: {counts[p]['T']}T  {counts[p]['NT']}NT")

# ----------------------------------------------------------------
# 2. class_distribution.png
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
x      = np.arange(len(patients))
width  = 0.35
t_vals = [counts[p]["T"]  for p in patients]
n_vals = [counts[p]["NT"] for p in patients]

bars_t = ax.bar(x - width/2, t_vals, width, label="Tumor (T)",         color="#E05555")
bars_n = ax.bar(x + width/2, n_vals, width, label="Normal Tissue (NT)", color="#4CAF50")

for bar in bars_t:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)),
                ha="center", va="bottom", fontsize=10)
for bar in bars_n:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)),
                ha="center", va="bottom", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(patients)
ax.set_xlabel("Patient")
ax.set_ylabel("Number of ROIs")
ax.set_title("HistologyHSI-GB: ROI Distribution by Patient and Class")
ax.legend()
ax.set_ylim(0, max(n_vals) + 8)
plt.tight_layout()
plt.savefig(SUMMARY_DIR / "class_distribution.png", dpi=150)
plt.close()
print("Saved class_distribution.png")

# ----------------------------------------------------------------
# 3. spectral_signatures.png
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(wavelengths, tumor_mean,  color="#E05555", label="Tumor",  linewidth=2)
ax.plot(wavelengths, normal_mean, color="#4CAF50", label="Normal", linewidth=2)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Normalized Reflectance")
ax.set_title("Mean Spectral Signatures: Tumor vs Normal Tissue\n"
             "(centre 50x50 pixels, after calibration and normalization)")
ax.legend()
ax.set_xlim(wavelengths[0], wavelengths[-1])
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(SUMMARY_DIR / "spectral_signatures.png", dpi=150)
plt.close()
print("Saved spectral_signatures.png")

# ----------------------------------------------------------------
# 4. spectral_signatures.json
# ----------------------------------------------------------------
sig_data = {
    "wavelengths_nm":  wavelengths.tolist(),
    "tumor_mean":      tumor_mean.tolist(),
    "normal_mean":     normal_mean.tolist(),
    "n_tumor_rois":    len(tumor_spectra),
    "n_normal_rois":   len(normal_spectra),
    "description":     "Mean spectrum per class from centre 50x50 patch of each ROI"
}
with open(SUMMARY_DIR / "spectral_signatures.json", "w") as jf:
    json.dump(sig_data, jf, indent=2)
print("Saved spectral_signatures.json")

# ----------------------------------------------------------------
# 5. dataset_summary.txt
# ----------------------------------------------------------------
grand_t  = sum(counts[p]["T"]  for p in patients)
grand_nt = sum(counts[p]["NT"] for p in patients)
ratio    = grand_t / (grand_t + grand_nt)

lines = [
    "HistologyHSI-GB Dataset Summary",
    "================================",
    "",
    "Source: The Cancer Imaging Archive (TCIA)",
    "Collection: HistologyHSI-GB",
    "Cancer type: Glioblastoma (WHO Grade IV)",
    "Imaging type: Hyperspectral histology slides",
    "",
    "RAW DATA SPECIFICATIONS",
    "-----------------------",
    "Image dimensions : 800 x 1004 pixels per ROI",
    "Spectral bands   : 826 bands (400.5 nm to 1000.7 nm)",
    "Spectral step    : ~0.73 nm per band",
    "Data type        : uint16 (raw sensor counts 65-3106)",
    "Storage format   : ENVI BIL binary + .hdr header",
    "Calibration      : darkReference + whiteReference per ROI",
    "",
    "DATASET STRUCTURE",
    "-----------------",
    "Total patients   : 3 (P1, P2, P3) + top-level ROIs",
    f"Total ROI folders: {grand_t + grand_nt}",
    f"Tumor ROIs   (T) : {grand_t}",
    f"Normal ROIs (NT) : {grand_nt}",
    f"T/NT ratio       : {grand_t/grand_nt:.2f} (imbalanced)",
    "Total raw size   : ~177.8 GB",
    "",
    "PREPROCESSING APPLIED",
    "---------------------",
    "Step 1: Reflectance calibration",
    "  Formula: Reflection = (Raw - Dark) / (White - Dark)",
    "  Values clipped to [0.0, 1.0]",
    "Step 2: Band selection",
    "  Removed bands above 909 nm (noisy region)",
    "  826 bands -> 699 bands kept (400.5 nm to 909.0 nm)",
    "Step 3: Per-pixel normalization",
    "  Each pixel spectrum divided by its own maximum value",
    "  Removes brightness variation, preserves spectral shape",
    "Step 4: Saved as HDF5 (.h5)",
    "  Format: float32, gzip compressed",
    "  Final shape per ROI: (800, 1004, 699)",
    "",
    "VERIFIED FILES PER PATIENT",
    "--------------------------",
]
for p in patients:
    lines.append(f"  {p}: {counts[p]['T']} tumor, {counts[p]['NT']} normal")

lines += [
    "",
    "SPECTRAL INFORMATION (after preprocessing)",
    "------------------------------------------",
    f"Wavelength range : {wavelengths[0]:.1f} nm to {wavelengths[-1]:.1f} nm",
    f"Number of bands  : {len(wavelengths)}",
    "Value range      : [0.0, 1.0] float32",
    f"Tumor mean spectrum peak  : {tumor_mean.max():.4f} at {wavelengths[tumor_mean.argmax()]:.1f} nm",
    f"Normal mean spectrum peak : {normal_mean.max():.4f} at {wavelengths[normal_mean.argmax()]:.1f} nm",
    f"Most discriminative region: ~520-560 nm (green absorption trough)",
]

with open(SUMMARY_DIR / "dataset_summary.txt", "w") as tf:
    tf.write("\n".join(lines) + "\n")
print("Saved dataset_summary.txt")

print("\nAll done. Summary files regenerated with full 134 ROIs.")
