# coding: utf-8
# HistologyHSI-GB Dataset Inspector
# Run: python inspect_dataset.py

import os
import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

from utils.config import DATA_ROOT
root = DATA_ROOT
if not root.exists():
    print("[ERROR] Path not found: " + str(DATA_ROOT))
    print("Fix: open Windows Explorer, go to HSI folder,")
    print("click the address bar, copy the path, paste it into DATA_ROOT above.")
    sys.exit(1)

print("")
print("============================================================")
print("  HistologyHSI-GB Dataset Inspector")
print("============================================================")
print("  Root: " + str(root))
print("")


# --- Find all ROI folders ---
print("Scanning for ROI folders ...")

def find_roi_folders(root):
    results = []
    for folder in sorted(root.rglob("*")):
        if not folder.is_dir():
            continue
        if not (folder / "raw").exists():
            continue
        if not (folder / "raw.hdr").exists():
            continue

        name  = folder.name
        parts = name.split("_")

        if len(parts) >= 4 and parts[0] == "ROI":
            roi_type = parts[1]
            cut      = parts[2]
            label    = parts[-1]
        else:
            roi_type, cut, label = "?", "?", "?"

        patient = "top-level"
        for p in folder.parents:
            if p == root:
                break
            if re.match(r"^P\d+$", p.name):
                patient = p.name
                break

        size_bytes = (folder / "raw").stat().st_size
        results.append({
            "path":     folder,
            "patient":  patient,
            "roi_type": roi_type,
            "cut":      cut,
            "label":    label,
            "size_gb":  size_bytes / 1e9,
        })
    return results


rois = find_roi_folders(root)

if not rois:
    print("[ERROR] No ROI folders found.")
    print("Expected subfolders like ROI_01_C01_T containing 'raw' and 'raw.hdr'.")
    sys.exit(1)

print("  Found " + str(len(rois)) + " ROI folders")
print("")


# --- Summary by patient ---
by_patient = defaultdict(lambda: {"T": 0, "NT": 0, "size_gb": 0.0})

for r in rois:
    p   = r["patient"]
    lbl = r["label"]
    if lbl in ("T", "NT"):
        by_patient[p][lbl] += 1
    by_patient[p]["size_gb"] += r["size_gb"]

total_T  = sum(v.get("T",  0) for v in by_patient.values())
total_NT = sum(v.get("NT", 0) for v in by_patient.values())
total_gb = sum(v["size_gb"]   for v in by_patient.values())

print("------------------------------------------------------------")
print("  Patient          Tumor(T)    Normal(NT)    Size(GB)")
print("------------------------------------------------------------")
for patient in sorted(by_patient.keys()):
    v = by_patient[patient]
    print("  {:<16} {:>8} {:>12} {:>12.1f}".format(
        patient, v.get("T",0), v.get("NT",0), v["size_gb"]))
print("------------------------------------------------------------")
print("  {:<16} {:>8} {:>12} {:>12.1f}".format(
    "TOTAL", total_T, total_NT, total_gb))
print("------------------------------------------------------------")
print("")


# --- Parse ENVI header ---
def parse_hdr(hdr_path):
    with open(hdr_path, "r", errors="ignore") as f:
        text = f.read()

    def get_int(key):
        m = re.search(rf"{key}\s*=\s*(\d+)", text, re.IGNORECASE)
        return int(m.group(1)) if m else None

    def get_str(key):
        m = re.search(rf"{key}\s*=\s*(.+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else "unknown"

    def get_float_list(key):
        m = re.search(
            rf"{key}\s*=\s*\{{([^}}]+)\}}",
            text, re.IGNORECASE | re.DOTALL)
        if not m:
            return []
        return [float(x) for x in re.findall(r"[\d.]+", m.group(1))]

    dtype_map = {
        1: "uint8", 2: "int16", 3: "int32", 4: "float32",
        5: "float64", 12: "uint16", 13: "uint32",
    }
    dtype_id = get_int("data type") or get_int("data_type") or 4

    return {
        "lines":       get_int("lines"),
        "samples":     get_int("samples"),
        "bands":       get_int("bands"),
        "dtype_id":    dtype_id,
        "dtype":       dtype_map.get(dtype_id, "type_" + str(dtype_id)),
        "interleave":  get_str("interleave"),
        "wavelengths": get_float_list("wavelength"),
    }


sample_roi = sorted(rois, key=lambda x: x["size_gb"], reverse=True)[0]
hdr_path   = sample_roi["path"] / "raw.hdr"

print("Reading header from sample ROI:")
print("  " + sample_roi["path"].name +
      "  (patient=" + sample_roi["patient"] +
      ", label="   + sample_roi["label"] + ")")
print("")

hdr   = parse_hdr(hdr_path)
rows  = hdr["lines"]
cols  = hdr["samples"]
bands = hdr["bands"]
dtype = hdr["dtype"]
il    = hdr["interleave"]
wl    = hdr["wavelengths"]

print("  Image dimensions : " + str(rows) + " rows x " + str(cols) +
      " cols x " + str(bands) + " bands")
print("  Data type        : " + dtype +
      "  (ENVI id=" + str(hdr["dtype_id"]) + ")")
print("  Storage format   : " + il + "  (bsq/bil/bip)")

if wl:
    step = (wl[-1] - wl[0]) / max(len(wl) - 1, 1)
    print("  Spectral range   : {:.1f} nm  to  {:.1f} nm".format(wl[0], wl[-1]))
    print("  Wavelength count : " + str(len(wl)))
    print("  Approx step      : {:.2f} nm per band".format(step))
else:
    print("  Wavelengths      : not listed in header")

expected = rows * cols * bands * np.dtype(dtype).itemsize
actual   = (sample_roi["path"] / "raw").stat().st_size
mismatch = abs(expected - actual) / max(expected, 1)

print("")
print("  Expected file size : {:.3f} GB  (from header)".format(expected / 1e9))
print("  Actual file size   : {:.3f} GB  (on disk)".format(actual / 1e9))
if mismatch < 0.01:
    print("  Integrity check   : PASS")
else:
    print("  Integrity check   : WARNING - {:.1f}% mismatch".format(mismatch * 100))


# --- Check calibration files ---
print("")
print("Checking calibration files ...")
for ref in ["darkReference", "whiteReference"]:
    r_bin = sample_roi["path"] / ref
    r_hdr = sample_roi["path"] / (ref + ".hdr")
    if r_bin.exists() and r_hdr.exists():
        kb = r_bin.stat().st_size / 1024
        print("  {:<22}: present  ({:.0f} KB)".format(ref, kb))
    else:
        print("  {:<22}: MISSING".format(ref))


# --- Try reading a patch ---
print("")
print("Attempting to load a 20x20 patch via spectral ...")
try:
    import spectral
    img   = spectral.open_image(str(hdr_path))
    patch = img[0:20, 0:20, :]
    print("  Patch shape  : " + str(patch.shape))
    print("  Value range  : [{:.1f}, {:.1f}]".format(
        float(patch.min()), float(patch.max())))
    print("  Mean value   : {:.2f}".format(float(patch.mean())))
    print("  READ         : SUCCESS")
except ImportError:
    print("  [SKIP] spectral not installed - run: python -m pip install spectral")
except Exception as e:
    print("  [ERROR] " + str(e))


# --- Final summary ---
print("")
print("============================================================")
print("  DATASET SUMMARY")
print("============================================================")
print("  Patients found      : " + str(len(by_patient)))
print("  Total ROI folders   : " + str(len(rois)))
print("  Tumor ROIs     (T)  : " + str(total_T))
print("  Normal ROIs   (NT)  : " + str(total_NT))

ratio = total_T / max(total_NT, 1)
balance = "balanced" if 0.7 < ratio < 1.4 else "IMBALANCED - note for training"
print("  T / NT ratio        : {:.2f}  ({})".format(ratio, balance))
print("  Total size on disk  : {:.1f} GB".format(total_gb))
print("  Raw format          : ENVI binary + .hdr header")
print("  Calibration refs    : darkReference + whiteReference (per ROI)")
if wl:
    print("  Spectral range      : {:.0f} - {:.0f} nm  ({} bands)".format(
        wl[0], wl[-1], len(wl)))
print("============================================================")
print("")
print("Ready for Step 2: preprocessing pipeline.")
print("  Formula: Reflection = (Raw - Dark) / (White - Dark)")
print("")
