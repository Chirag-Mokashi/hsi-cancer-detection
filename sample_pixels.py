# sample_pixels.py v2
# Step 3 - Pixel Sampling for Band Selection
# FAST version: loads one contiguous row block per file (1 read = 1 decompression)
# Outputs: preprocessed/samples.h5
#   X: (N, 699) float32
#   y: (N,)     int8    0=NT  1=T
#   wavelengths: (699,) float32

import h5py
import numpy as np
from pathlib import Path
import time

# ---- Config ----
PREPROCESSED_DIR = Path(r"C:\Users\mokas\OneDrive\Desktop\HSI\preprocessed")
OUTPUT_FILE      = PREPROCESSED_DIR / "samples.h5"
PIXELS_PER_ROI   = 500
BLOCK_ROWS       = 10    # load 10 consecutive rows per file: 10 x 1004 = 10040 pixels
                         # pick 500 randomly from those - 1 single h5py read per file
RANDOM_SEED      = 42
# ----------------

rng = np.random.default_rng(RANDOM_SEED)

files = sorted(PREPROCESSED_DIR.glob("*.h5"))
files = [f for f in files if f.name != "samples.h5"]

print(f"Found {len(files)} HDF5 files")
print(f"Strategy: 1 contiguous block of {BLOCK_ROWS} rows per file -> pick {PIXELS_PER_ROI} pixels")
print(f"Expected total samples: {len(files) * PIXELS_PER_ROI}")
print(f"Expected output size: ~{len(files) * PIXELS_PER_ROI * 699 * 4 / 1e6:.1f} MB")
print()

all_X = []
all_y = []
wavelengths = None

t_start = time.time()

for i, fpath in enumerate(files):
    t_file = time.time()

    with h5py.File(fpath, "r") as f:
        n_rows, n_cols, n_bands = f["cube"].shape
        label_str = str(f.attrs["label"])
        patient   = str(f.attrs["patient"])
        label_int = 1 if label_str == "T" else 0

        if wavelengths is None:
            wavelengths = f["wavelengths"][:]

        # Pick random starting row so block fits within cube
        max_start = n_rows - BLOCK_ROWS
        start_row = int(rng.integers(0, max_start))
        end_row   = start_row + BLOCK_ROWS

        # ONE read: shape (BLOCK_ROWS, n_cols, n_bands)
        block = f["cube"][start_row:end_row, :, :]

    # Flatten block to (BLOCK_ROWS * n_cols, n_bands)
    flat = block.reshape(-1, n_bands)

    # Randomly pick PIXELS_PER_ROI from the flat block
    chosen = rng.choice(flat.shape[0], size=PIXELS_PER_ROI, replace=False)
    sampled = flat[chosen]   # (PIXELS_PER_ROI, n_bands)

    all_X.append(sampled.astype(np.float32))
    all_y.append(np.full(PIXELS_PER_ROI, label_int, dtype=np.int8))

    elapsed = time.time() - t_file
    eta = (time.time() - t_start) / (i + 1) * (len(files) - i - 1)
    print(f"[{i+1:3d}/{len(files)}] {fpath.name:42s} {label_str:2s}  {patient:10s} ({elapsed:.1f}s)  ETA: {eta/60:.1f}min")

print()
print("Stacking arrays...")
X = np.vstack(all_X)
y = np.concatenate(all_y)

print(f"X shape: {X.shape}  dtype: {X.dtype}")
print(f"y shape: {y.shape}  dtype: {y.dtype}")
print(f"Class balance  Tumor(1): {int(y.sum())}  Normal(0): {int((y==0).sum())}")
print(f"Wavelengths: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm, count: {len(wavelengths)}")
print()

print(f"Saving to: {OUTPUT_FILE}")
with h5py.File(OUTPUT_FILE, "w") as f:
    f.create_dataset("X",           data=X,           compression="gzip", compression_opts=4)
    f.create_dataset("y",           data=y,           compression="gzip", compression_opts=4)
    f.create_dataset("wavelengths", data=wavelengths, compression="gzip", compression_opts=4)
    f.attrs["pixels_per_roi"] = PIXELS_PER_ROI
    f.attrs["n_files"]        = len(files)
    f.attrs["random_seed"]    = RANDOM_SEED
    f.attrs["block_rows"]     = BLOCK_ROWS
    f.attrs["description"]    = "Pixel samples for band selection. X=(N,699) float32, y=(N,) int8 0=NT 1=T"

size_mb = OUTPUT_FILE.stat().st_size / 1e6
total_time = time.time() - t_start

print(f"Saved. File size: {size_mb:.1f} MB")
print(f"Total time: {total_time:.1f}s  ({total_time/60:.1f} min)")
print()
print("Next: upload preprocessed/samples.h5 to Google Drive for Colab band selection")
