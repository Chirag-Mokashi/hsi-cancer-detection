import h5py
import os
from pathlib import Path

preprocessed = Path(r"C:\Users\mokas\OneDrive\Desktop\HSI\preprocessed")

# Delete the corrupt stub
stub = preprocessed / "P1_ROI_03_C13_NT.h5"
if stub.exists():
    stub.unlink()
    print(f"Deleted stub: {stub.name}")

# Rename and fix patient attribute
rois = [
    "ROI_03_C13_NT", "ROI_04_C01_NT", "ROI_04_C02_NT", "ROI_04_C03_NT",
    "ROI_04_C04_NT", "ROI_04_C05_NT", "ROI_04_C06_NT", "ROI_04_C07_NT",
    "ROI_04_C08_NT"
]

for roi in rois:
    old_path = preprocessed / f"top-level_{roi}.h5"
    new_path = preprocessed / f"P1_{roi}.h5"
    if old_path.exists():
        with h5py.File(old_path, "a") as f:
            f.attrs["patient"] = "P1"
        old_path.rename(new_path)
        print(f"Fixed: top-level_{roi}.h5 -> P1_{roi}.h5  (patient=P1)")
    else:
        print(f"NOT FOUND: {old_path.name}")
