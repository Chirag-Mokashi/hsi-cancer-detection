import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Path: works both locally and in Colab ────────────────────
try:
    from google.colab import drive
    drive.mount('/content/drive')
    preprocessed = Path('/content/drive/MyDrive/HSI/preprocessed')
except ImportError:
    preprocessed = Path(r"C:\Users\mokas\OneDrive\Desktop\HSI\preprocessed")

files = sorted(f for f in preprocessed.glob("*.h5") if f.name != 'samples.h5')

patients = defaultdict(lambda: {"T": 0, "NT": 0, "files": []})
bad = []

print("=" * 60)
print("FULL DATASET AUDIT")
print(f"Path : {preprocessed}")
print(f"Files: {len(files)} ROI files (samples.h5 excluded)")
print("=" * 60)

for i, f in enumerate(files, 1):
    print(f"  [{i:3d}/{len(files)}] {f.name} ...", end=" ", flush=True)
    try:
        with h5py.File(f, "r") as hf:
            cube = hf["cube"][:]
            wl   = hf["wavelengths"][:]
            label   = str(hf.attrs["label"])
            patient = str(hf.attrs["patient"])

            # Shape
            assert cube.shape == (800, 1004, 699), \
                f"Bad shape: {cube.shape}"

            # Label
            assert label in ("T", "NT"), \
                f"Bad label: '{label}'"

            # Patient — top-level should have been patched to P1
            assert patient in ("P1", "P2", "P3"), \
                f"Bad/unpatched patient: '{patient}' (fix_patient.py may not have run)"

            # Wavelengths
            assert len(wl) == 699, \
                f"Bad wavelength count: {len(wl)}"
            assert 400.0 <= wl[0] <= 401.0, \
                f"Bad wl start: {wl[0]:.2f}"
            assert 908.0 <= wl[-1] <= 910.0, \
                f"Bad wl end: {wl[-1]:.2f}"

            # Value integrity
            assert cube.min() >= 0.0, \
                f"Values below 0: min={cube.min():.4f}"
            assert cube.max() <= 1.0, \
                f"Values above 1: max={cube.max():.4f}"
            assert not np.any(np.isnan(cube)), "NaN values found"
            assert not np.any(np.isinf(cube)), "Inf values found"

            patients[patient]["files"].append(f.name)
            patients[patient][label] += 1
            print("OK")

    except Exception as e:
        print(f"FAIL — {e}")
        bad.append((f.name, str(e)))

# ── Summary ──────────────────────────────────────────────────
print()
print(f"{'Patient':<12} {'T':>5} {'NT':>5} {'Total':>7}")
print("-" * 32)
grand_t = grand_nt = 0
for p in ("P1", "P2", "P3"):
    t  = patients[p]["T"]
    nt = patients[p]["NT"]
    grand_t  += t
    grand_nt += nt
    print(f"{p:<12} {t:>5} {nt:>5} {t+nt:>7}")
print("-" * 32)
print(f"{'TOTAL':<12} {grand_t:>5} {grand_nt:>5} {grand_t+grand_nt:>7}")

print(f"\nTotal ROI files : {len(files)}")
print(f"T/NT ratio      : {grand_t/(grand_nt+grand_t):.3f}  "
      f"(expected ~0.291 for 39T / 95NT)")
print(f"Bad files       : {len(bad)}")

if bad:
    print("\nBAD FILES — fix before training:")
    for name, err in bad:
        print(f"  {name}: {err}")
else:
    print("\nAll files passed all checks. Safe to train.")
print("=" * 60)
