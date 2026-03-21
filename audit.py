import h5py
from pathlib import Path
from collections import defaultdict

preprocessed = Path(r"C:\Users\mokas\OneDrive\Desktop\HSI\preprocessed")
files = sorted(preprocessed.glob("*.h5"))

patients = defaultdict(lambda: {"T": 0, "NT": 0, "files": []})
bad = []
wl_issues = []
value_issues = []

print("=" * 60)
print("FULL DATASET AUDIT")
print("=" * 60)

for f in files:
    try:
        with h5py.File(f, "r") as hf:
            cube = hf["cube"][:]
            wl = hf["wavelengths"][:]
            label = str(hf.attrs["label"])
            patient = str(hf.attrs["patient"])

            # Shape check
            assert cube.shape == (800, 1004, 699), f"Bad shape: {cube.shape}"

            # Label check
            assert label in ["T", "NT"], f"Bad label: {label}"

            # Patient check
            assert patient in ["P1", "P2", "P3", "top-level"], f"Bad patient: {patient}"

            # Wavelength check
            assert len(wl) == 699, f"Bad wavelength count: {len(wl)}"
            assert 400.0 <= wl[0] <= 401.0, f"Bad wl start: {wl[0]}"
            assert 908.0 <= wl[-1] <= 910.0, f"Bad wl end: {wl[-1]}"

            # Value range check
            import numpy as np
            assert cube.min() >= 0.0, f"Values below 0: {cube.min()}"
            assert cube.max() <= 1.0, f"Values above 1: {cube.max()}"
            assert not np.any(np.isnan(cube)), "NaN values found"
            assert not np.any(np.isinf(cube)), "Inf values found"

            patients[patient]["files"].append(f.name)
            patients[patient][label] += 1

    except Exception as e:
        bad.append((f.name, str(e)))

# Per-patient summary
print(f"\n{'Patient':<12} {'T':>5} {'NT':>5} {'Total':>7}")
print("-" * 32)
grand_t = grand_nt = 0
for p in ["P1", "P2", "P3", "top-level"]:
    t = patients[p]["T"]
    nt = patients[p]["NT"]
    grand_t += t
    grand_nt += nt
    print(f"{p:<12} {t:>5} {nt:>5} {t+nt:>7}")
print("-" * 32)
print(f"{'TOTAL':<12} {grand_t:>5} {grand_nt:>5} {grand_t+grand_nt:>7}")

# Overall stats
print(f"\nTotal files   : {len(files)}")
print(f"T/NT ratio    : {grand_t/(grand_nt+grand_t):.3f}")
print(f"Bad files     : {len(bad)}")
if bad:
    print("\nBAD FILES:")
    for name, err in bad:
        print(f"  {name}: {err}")
else:
    print("\nAll files passed all checks.")
print("=" * 60)
