"""
Run this in Colab BEFORE restarting training.
Steps:
  1. Deletes the 2 duplicate (1) files from Drive
  2. Patches patient attribute on 6 top-level files from 'top-level' -> 'P1'
  3. Re-runs audit to confirm 134 files, 0 bad
"""
import h5py
import numpy as np
from pathlib import Path

preprocessed = Path('/content/drive/MyDrive/HSI/preprocessed')

# ── Step 1: Delete duplicate (1) files ───────────────────────
print("=" * 60)
print("STEP 1: Delete duplicate files")
print("=" * 60)

duplicates = [
    preprocessed / 'P3_ROI_02_C28_NT (1).h5',
    preprocessed / 'P3_ROI_02_C29_NT (1).h5',
]

for f in duplicates:
    if f.exists():
        f.unlink()
        print(f"  Deleted: {f.name}")
    else:
        print(f"  Not found (already gone?): {f.name}")

# ── Step 2: Patch top-level patient attribute to P1 ──────────
print()
print("=" * 60)
print("STEP 2: Patch top-level patient attribute -> P1")
print("=" * 60)

toplevel_files = sorted(preprocessed.glob('top-level_*.h5'))
print(f"  Found {len(toplevel_files)} top-level files to patch")

for f in toplevel_files:
    with h5py.File(f, 'a') as hf:
        old = str(hf.attrs.get('patient', 'MISSING'))
        hf.attrs['patient'] = 'P1'
        new = str(hf.attrs['patient'])
    print(f"  {f.name}: '{old}' -> '{new}'")

# ── Step 3: Re-run audit ──────────────────────────────────────
print()
print("=" * 60)
print("STEP 3: Re-run audit")
print("=" * 60)

from collections import defaultdict
files = sorted(f for f in preprocessed.glob("*.h5") if f.name != 'samples.h5')
patients = defaultdict(lambda: {"T": 0, "NT": 0})
bad = []

print(f"Files found: {len(files)}  (expected 134)")
print()

for i, f in enumerate(files, 1):
    print(f"  [{i:3d}/{len(files)}] {f.name} ...", end=" ", flush=True)
    try:
        with h5py.File(f, "r") as hf:
            cube    = hf["cube"][:]
            wl      = hf["wavelengths"][:]
            label   = str(hf.attrs["label"])
            patient = str(hf.attrs["patient"])
            assert cube.shape == (800, 1004, 699), f"Bad shape: {cube.shape}"
            assert label in ("T", "NT"),           f"Bad label: '{label}'"
            assert patient in ("P1", "P2", "P3"),  f"Bad patient: '{patient}'"
            assert len(wl) == 699,                 f"Bad wl count: {len(wl)}"
            assert 400.0 <= wl[0] <= 401.0,        f"Bad wl start: {wl[0]:.2f}"
            assert 908.0 <= wl[-1] <= 910.0,       f"Bad wl end: {wl[-1]:.2f}"
            assert cube.min() >= 0.0,              f"Values below 0: {cube.min():.4f}"
            assert cube.max() <= 1.0,              f"Values above 1: {cube.max():.4f}"
            assert not np.any(np.isnan(cube)),     "NaN found"
            assert not np.any(np.isinf(cube)),     "Inf found"
            patients[patient][label] += 1
            print("OK")
    except Exception as e:
        print(f"FAIL -- {e}")
        bad.append((f.name, str(e)))

print()
print("-" * 40)
for p in ("P1", "P2", "P3"):
    t, nt = patients[p]["T"], patients[p]["NT"]
    print(f"  {p}: {t}T  {nt}NT  ({t+nt} total)")
grand_t  = sum(patients[p]["T"]  for p in ("P1","P2","P3"))
grand_nt = sum(patients[p]["NT"] for p in ("P1","P2","P3"))
print(f"  TOTAL: {grand_t}T  {grand_nt}NT  = {grand_t+grand_nt} files")
print(f"  Expected:  39T   95NT  = 134 files")
print()
if bad:
    print(f"BAD FILES ({len(bad)}):")
    for name, err in bad:
        print(f"  {name}: {err}")
else:
    print("All files passed. SAFE TO RESTART TRAINING.")
print("=" * 60)
