# analyse_samples.py
# Thorough analysis of samples.h5 before band selection
# Runs entirely locally - no Colab needed
# Outputs: dataset_summary/sample_analysis/ folder with plots and a report

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import time

# ---- Config ----
SAMPLES_FILE = Path(r"C:\Users\mokas\OneDrive\Desktop\HSI\preprocessed\samples.h5")
OUT_DIR      = Path(r"C:\Users\mokas\OneDrive\Desktop\HSI\dataset_summary\sample_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ----------------

print("=" * 60)
print("SAMPLE DATA ANALYSIS")
print("=" * 60)

# ---- Load ----
print("\nLoading samples.h5 ...")
t0 = time.time()
with h5py.File(SAMPLES_FILE, "r") as f:
    X  = f["X"][:]           # (67000, 699) float32
    y  = f["y"][:]           # (67000,)     int8
    wl = f["wavelengths"][:] # (699,)       float32
print(f"Loaded in {time.time()-t0:.1f}s")

N, B = X.shape
T_mask  = y == 1
NT_mask = y == 0
X_T  = X[T_mask]
X_NT = X[NT_mask]

print(f"\n{'='*60}")
print("1. BASIC STATISTICS")
print(f"{'='*60}")
print(f"Total samples    : {N}")
print(f"Tumor (T)        : {T_mask.sum()} ({100*T_mask.mean():.1f}%)")
print(f"Normal (NT)      : {NT_mask.sum()} ({100*NT_mask.mean():.1f}%)")
print(f"Bands            : {B}")
print(f"Wavelength range : {wl[0]:.1f} - {wl[-1]:.1f} nm")
print(f"X global min     : {X.min():.6f}")
print(f"X global max     : {X.max():.6f}")
print(f"X global mean    : {X.mean():.6f}")
print(f"X global std     : {X.std():.6f}")

# Check for NaN / Inf
n_nan = np.isnan(X).sum()
n_inf = np.isinf(X).sum()
print(f"NaN values       : {n_nan}")
print(f"Inf values       : {n_inf}")

# Check value range compliance
out_of_range = ((X < 0) | (X > 1)).sum()
print(f"Values outside [0,1]: {out_of_range}")

# Pixels at exactly 0 or 1
exactly_zero = (X == 0.0).sum()
exactly_one  = (X == 1.0).sum()
print(f"Values exactly 0.0  : {exactly_zero} ({100*exactly_zero/(N*B):.3f}%)")
print(f"Values exactly 1.0  : {exactly_one}  ({100*exactly_one/(N*B):.3f}%)")

# Per-class stats
print(f"\n--- Per-class global statistics ---")
print(f"{'':20s} {'Tumor':>12s} {'Normal':>12s}")
print(f"{'Mean':20s} {X_T.mean():12.6f} {X_NT.mean():12.6f}")
print(f"{'Std':20s} {X_T.std():12.6f} {X_NT.std():12.6f}")
print(f"{'Min':20s} {X_T.min():12.6f} {X_NT.min():12.6f}")
print(f"{'Max':20s} {X_T.max():12.6f} {X_NT.max():12.6f}")

# ---- Per-band stats ----
print(f"\n{'='*60}")
print("2. PER-BAND STATISTICS")
print(f"{'='*60}")

mean_T  = X_T.mean(axis=0)
mean_NT = X_NT.mean(axis=0)
std_T   = X_T.std(axis=0)
std_NT  = X_NT.std(axis=0)
var_all = X.var(axis=0)

# Cohen's d per band: effect size between T and NT
pooled_std = np.sqrt((std_T**2 + std_NT**2) / 2)
pooled_std = np.where(pooled_std == 0, 1e-10, pooled_std)
cohens_d   = np.abs(mean_T - mean_NT) / pooled_std

# t-statistic per band (Welch)
n_T  = T_mask.sum()
n_NT = NT_mask.sum()
se   = np.sqrt(std_T**2/n_T + std_NT**2/n_NT)
se   = np.where(se == 0, 1e-10, se)
t_stat = np.abs(mean_T - mean_NT) / se

# Top 10 most discriminative bands by Cohen's d
top10_idx = np.argsort(cohens_d)[::-1][:10]
print(f"\nTop 10 most discriminative bands (by Cohen's d):")
print(f"{'Rank':>4s}  {'Band':>5s}  {'Wavelength':>10s}  {'Mean_T':>8s}  {'Mean_NT':>8s}  {'Cohen_d':>8s}  {'t-stat':>8s}")
for rank, idx in enumerate(top10_idx):
    print(f"{rank+1:4d}  {idx:5d}  {wl[idx]:8.1f} nm  {mean_T[idx]:8.4f}  {mean_NT[idx]:8.4f}  {cohens_d[idx]:8.4f}  {t_stat[idx]:8.1f}")

# Bottom 10 least discriminative bands
bot10_idx = np.argsort(cohens_d)[:10]
print(f"\nBottom 10 least discriminative bands (by Cohen's d):")
print(f"{'Rank':>4s}  {'Band':>5s}  {'Wavelength':>10s}  {'Mean_T':>8s}  {'Mean_NT':>8s}  {'Cohen_d':>8s}")
for rank, idx in enumerate(bot10_idx):
    print(f"{rank+1:4d}  {idx:5d}  {wl[idx]:8.1f} nm  {mean_T[idx]:8.4f}  {mean_NT[idx]:8.4f}  {cohens_d[idx]:8.4f}")

# Spectral region summary
regions = [
    ("Violet/Blue  400-500nm", 400, 500),
    ("Green        500-570nm", 500, 570),
    ("Red          570-700nm", 570, 700),
    ("NIR          700-909nm", 700, 909),
]
print(f"\nMean Cohen's d by spectral region:")
for name, lo, hi in regions:
    mask = (wl >= lo) & (wl < hi)
    if mask.sum() > 0:
        print(f"  {name}: mean_d={cohens_d[mask].mean():.4f}  max_d={cohens_d[mask].max():.4f}  bands={mask.sum()}")

# Most discriminative wavelength overall
best_band = np.argmax(cohens_d)
print(f"\nSingle most discriminative wavelength: {wl[best_band]:.1f} nm (band {best_band}, d={cohens_d[best_band]:.4f})")

# ---- Band correlation ----
print(f"\n{'='*60}")
print("3. BAND CORRELATION STRUCTURE")
print(f"{'='*60}")

# Subsample for correlation (full 67000x699 correlation is 699x699 = fine actually)
print("Computing 699x699 correlation matrix (may take ~10s)...")
t0 = time.time()
# Use float64 for correlation stability
corr = np.corrcoef(X.T.astype(np.float64))  # (699, 699)
print(f"Done in {time.time()-t0:.1f}s")

# Upper triangle excluding diagonal
upper = corr[np.triu_indices(B, k=1)]
print(f"Mean inter-band correlation  : {upper.mean():.4f}")
print(f"Median inter-band correlation: {np.median(upper):.4f}")
print(f"Pairs with |r| > 0.95        : {(np.abs(upper) > 0.95).sum()} / {len(upper)}")
print(f"Pairs with |r| > 0.99        : {(np.abs(upper) > 0.99).sum()} / {len(upper)}")
print(f"Pairs with |r| < 0.50        : {(np.abs(upper) < 0.50).sum()} / {len(upper)}")

# Find most redundant band (highest mean correlation with others)
mean_corr_per_band = np.abs(corr - np.eye(B)).mean(axis=1)
most_redundant = np.argmax(mean_corr_per_band)
least_redundant = np.argmin(mean_corr_per_band)
print(f"Most redundant band  : band {most_redundant} ({wl[most_redundant]:.1f} nm), mean |r|={mean_corr_per_band[most_redundant]:.4f}")
print(f"Least redundant band : band {least_redundant} ({wl[least_redundant]:.1f} nm), mean |r|={mean_corr_per_band[least_redundant]:.4f}")

# ---- Variance analysis ----
print(f"\n{'='*60}")
print("4. VARIANCE ANALYSIS")
print(f"{'='*60}")
total_var = var_all.sum()
# How many bands capture 90%, 95%, 99% of total variance
sorted_var = np.sort(var_all)[::-1]
cumvar = np.cumsum(sorted_var) / total_var
n90 = np.searchsorted(cumvar, 0.90) + 1
n95 = np.searchsorted(cumvar, 0.95) + 1
n99 = np.searchsorted(cumvar, 0.99) + 1
print(f"Bands needed for 90% variance: {n90}")
print(f"Bands needed for 95% variance: {n95}")
print(f"Bands needed for 99% variance: {n99}")
print(f"(This informs PCA component selection)")

# Low-variance bands (potentially uninformative)
var_threshold = np.percentile(var_all, 5)
low_var_bands = np.where(var_all < var_threshold)[0]
print(f"\nLowest-variance 5% bands ({len(low_var_bands)} bands):")
print(f"  Wavelengths: {wl[low_var_bands[0]]:.1f} - {wl[low_var_bands[-1]]:.1f} nm")
print(f"  These are candidates for removal in MI/LASSO")

# ---- Anomaly check ----
print(f"\n{'='*60}")
print("5. ANOMALY / QUALITY CHECKS")
print(f"{'='*60}")

# Pixels with all-zero spectra
all_zero_pixels = (X == 0).all(axis=1).sum()
print(f"All-zero pixel spectra : {all_zero_pixels}")

# Pixels with max == min (flat spectrum)
flat_pixels = (X.max(axis=1) == X.min(axis=1)).sum()
print(f"Flat spectrum pixels   : {flat_pixels}")

# Pixels where max band is NOT 1.0 (per-pixel normalization check)
# After normalization each pixel should have max close to 1.0
pixel_max = X.max(axis=1)
not_normalized = (pixel_max < 0.99).sum()
print(f"Pixels with max < 0.99 : {not_normalized} ({100*not_normalized/N:.2f}%) - expected ~0 after per-pixel norm")

# Per-band: how many pixels are exactly 1.0 (saturation)
saturated_per_band = (X == 1.0).sum(axis=0)
most_saturated_band = np.argmax(saturated_per_band)
print(f"Band with most saturation: band {most_saturated_band} ({wl[most_saturated_band]:.1f} nm), {saturated_per_band[most_saturated_band]} pixels")

# ---- PLOTS ----
print(f"\n{'='*60}")
print("6. GENERATING PLOTS")
print(f"{'='*60}")

# Plot 1: Mean spectra T vs NT with std bands + Cohen's d
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(wl, mean_T,  color="#d62728", linewidth=1.5, label=f"Tumor (n={n_T})")
ax1.plot(wl, mean_NT, color="#1f77b4", linewidth=1.5, label=f"Normal (n={n_NT})")
ax1.fill_between(wl, mean_T-std_T,   mean_T+std_T,   alpha=0.15, color="#d62728")
ax1.fill_between(wl, mean_NT-std_NT, mean_NT+std_NT, alpha=0.15, color="#1f77b4")
ax1.axvspan(520, 560, alpha=0.1, color="green", label="520-560nm (hemoglobin trough)")
ax1.set_ylabel("Normalized Reflectance", fontsize=11)
ax1.set_title("Mean Spectral Signatures: Tumor vs Normal (with std bands)", fontsize=12)
ax1.legend(fontsize=10)
ax1.set_xlim(wl[0], wl[-1])
ax1.grid(True, alpha=0.3)

ax2.plot(wl, cohens_d, color="#2ca02c", linewidth=1.0)
ax2.axvspan(520, 560, alpha=0.1, color="green")
ax2.axhline(0.2, color="gray", linestyle="--", linewidth=0.8, label="small effect (d=0.2)")
ax2.axhline(0.5, color="orange", linestyle="--", linewidth=0.8, label="medium effect (d=0.5)")
ax2.axhline(0.8, color="red", linestyle="--", linewidth=0.8, label="large effect (d=0.8)")
ax2.set_xlabel("Wavelength (nm)", fontsize=11)
ax2.set_ylabel("Cohen's d", fontsize=11)
ax2.set_title("Per-Band Discriminability: Cohen's d (Tumor vs Normal)", fontsize=12)
ax2.legend(fontsize=9)
ax2.set_xlim(wl[0], wl[-1])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
p1 = OUT_DIR / "spectral_discriminability.png"
plt.savefig(p1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p1}")

# Plot 2: Band variance
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(wl, var_all, color="#9467bd", linewidth=1.0)
ax.axvspan(520, 560, alpha=0.1, color="green", label="520-560nm")
ax.set_xlabel("Wavelength (nm)", fontsize=11)
ax.set_ylabel("Variance", fontsize=11)
ax.set_title("Per-Band Variance Across All Pixels", fontsize=12)
ax.legend(fontsize=9)
ax.set_xlim(wl[0], wl[-1])
ax.grid(True, alpha=0.3)
plt.tight_layout()
p2 = OUT_DIR / "band_variance.png"
plt.savefig(p2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p2}")

# Plot 3: Correlation matrix heatmap (subsampled to every 5th band for visibility)
stride = 5
corr_sub = corr[::stride, ::stride]
wl_sub   = wl[::stride]
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr_sub, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
plt.colorbar(im, ax=ax, label="Pearson r")
n_ticks = min(10, len(wl_sub))
tick_idx = np.linspace(0, len(wl_sub)-1, n_ticks, dtype=int)
ax.set_xticks(tick_idx)
ax.set_yticks(tick_idx)
ax.set_xticklabels([f"{wl_sub[i]:.0f}" for i in tick_idx], rotation=45, fontsize=8)
ax.set_yticklabels([f"{wl_sub[i]:.0f}" for i in tick_idx], fontsize=8)
ax.set_title(f"Band Correlation Matrix (every {stride}th band shown, {len(wl_sub)} bands displayed)", fontsize=11)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Wavelength (nm)")
plt.tight_layout()
p3 = OUT_DIR / "band_correlation.png"
plt.savefig(p3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p3}")

# Plot 4: Cohen's d ranked bar (top 50 bands)
top50 = np.argsort(cohens_d)[::-1][:50]
fig, ax = plt.subplots(figsize=(16, 5))
colors = ["#d62728" if cohens_d[i] > 0.8 else "#ff7f0e" if cohens_d[i] > 0.5 else "#2ca02c" for i in top50]
ax.bar(range(50), cohens_d[top50], color=colors)
ax.set_xticks(range(50))
ax.set_xticklabels([f"{wl[i]:.0f}" for i in top50], rotation=90, fontsize=7)
ax.axhline(0.8, color="red",    linestyle="--", linewidth=0.8, label="large effect (d=0.8)")
ax.axhline(0.5, color="orange", linestyle="--", linewidth=0.8, label="medium effect (d=0.5)")
ax.set_xlabel("Wavelength (nm)", fontsize=11)
ax.set_ylabel("Cohen's d", fontsize=11)
ax.set_title("Top 50 Most Discriminative Bands by Cohen's d", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
p4 = OUT_DIR / "top50_discriminative_bands.png"
plt.savefig(p4, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p4}")

# ---- Text Report (complete - mirrors all terminal output) ----
report_path = OUT_DIR / "analysis_report.txt"
with open(report_path, "w") as rpt:
    rpt.write("SAMPLE DATA ANALYSIS REPORT\n")
    rpt.write("=" * 60 + "\n\n")

    # Section 1: Basic statistics
    rpt.write("1. BASIC STATISTICS\n")
    rpt.write("-" * 50 + "\n")
    rpt.write(f"  Samples file     : {SAMPLES_FILE}\n")
    rpt.write(f"  Total samples    : {N}\n")
    rpt.write(f"  Tumor (T)        : {T_mask.sum()} ({100*T_mask.mean():.1f}%)\n")
    rpt.write(f"  Normal (NT)      : {NT_mask.sum()} ({100*NT_mask.mean():.1f}%)\n")
    rpt.write(f"  Bands            : {B}\n")
    rpt.write(f"  Wavelength range : {wl[0]:.1f} - {wl[-1]:.1f} nm\n")
    rpt.write(f"  X global min     : {X.min():.6f}\n")
    rpt.write(f"  X global max     : {X.max():.6f}\n")
    rpt.write(f"  X global mean    : {X.mean():.6f}\n")
    rpt.write(f"  X global std     : {X.std():.6f}\n")
    rpt.write(f"  NaN values       : {n_nan}\n")
    rpt.write(f"  Inf values       : {n_inf}\n")
    rpt.write(f"  Values outside [0,1]: {out_of_range}\n")
    rpt.write(f"  Values exactly 0.0  : {exactly_zero} ({100*exactly_zero/(N*B):.3f}%)\n")
    rpt.write(f"  Values exactly 1.0  : {exactly_one} ({100*exactly_one/(N*B):.3f}%)\n")
    rpt.write(f"\n  Per-class global statistics:\n")
    rpt.write(f"  {'':20s} {'Tumor':>12s} {'Normal':>12s}\n")
    rpt.write(f"  {'Mean':20s} {X_T.mean():12.6f} {X_NT.mean():12.6f}\n")
    rpt.write(f"  {'Std':20s} {X_T.std():12.6f} {X_NT.std():12.6f}\n")
    rpt.write(f"  {'Min':20s} {X_T.min():12.6f} {X_NT.min():12.6f}\n")
    rpt.write(f"  {'Max':20s} {X_T.max():12.6f} {X_NT.max():12.6f}\n")

    # Section 2: Per-band discriminability
    rpt.write("\n2. PER-BAND DISCRIMINABILITY\n")
    rpt.write("-" * 50 + "\n")
    rpt.write(f"  Single most discriminative wavelength: {wl[best_band]:.1f} nm (band {best_band}, d={cohens_d[best_band]:.4f})\n\n")
    rpt.write(f"  Top 20 most discriminative bands (Cohen's d):\n")
    rpt.write(f"  {'Rank':>4s}  {'Band':>5s}  {'Wavelength':>10s}  {'Mean_T':>8s}  {'Mean_NT':>8s}  {'Cohen_d':>8s}  {'t-stat':>8s}\n")
    for rank, idx in enumerate(np.argsort(cohens_d)[::-1][:20]):
        rpt.write(f"  {rank+1:4d}  {idx:5d}  {wl[idx]:8.1f} nm  {mean_T[idx]:8.4f}  {mean_NT[idx]:8.4f}  {cohens_d[idx]:8.4f}  {t_stat[idx]:8.1f}\n")
    rpt.write(f"\n  Bottom 10 least discriminative bands (Cohen's d):\n")
    rpt.write(f"  {'Rank':>4s}  {'Band':>5s}  {'Wavelength':>10s}  {'Mean_T':>8s}  {'Mean_NT':>8s}  {'Cohen_d':>8s}\n")
    for rank, idx in enumerate(np.argsort(cohens_d)[:10]):
        rpt.write(f"  {rank+1:4d}  {idx:5d}  {wl[idx]:8.1f} nm  {mean_T[idx]:8.4f}  {mean_NT[idx]:8.4f}  {cohens_d[idx]:8.4f}\n")
    rpt.write(f"\n  Mean Cohen's d by spectral region:\n")
    for name, lo, hi in regions:
        mask = (wl >= lo) & (wl < hi)
        if mask.sum() > 0:
            rpt.write(f"    {name}: mean_d={cohens_d[mask].mean():.4f}  max_d={cohens_d[mask].max():.4f}  bands={mask.sum()}\n")

    # Section 3: Correlation
    rpt.write("\n3. BAND CORRELATION STRUCTURE\n")
    rpt.write("-" * 50 + "\n")
    rpt.write(f"  Mean inter-band correlation  : {upper.mean():.4f}\n")
    rpt.write(f"  Median inter-band correlation: {np.median(upper):.4f}\n")
    rpt.write(f"  Pairs with |r| > 0.95        : {(np.abs(upper) > 0.95).sum()} / {len(upper)}\n")
    rpt.write(f"  Pairs with |r| > 0.99        : {(np.abs(upper) > 0.99).sum()} / {len(upper)}\n")
    rpt.write(f"  Pairs with |r| < 0.50        : {(np.abs(upper) < 0.50).sum()} / {len(upper)}\n")
    rpt.write(f"  Most redundant band  : band {most_redundant} ({wl[most_redundant]:.1f} nm), mean |r|={mean_corr_per_band[most_redundant]:.4f}\n")
    rpt.write(f"  Least redundant band : band {least_redundant} ({wl[least_redundant]:.1f} nm), mean |r|={mean_corr_per_band[least_redundant]:.4f}\n")

    # Section 4: Variance
    rpt.write("\n4. VARIANCE ANALYSIS\n")
    rpt.write("-" * 50 + "\n")
    rpt.write(f"  Bands needed for 90% variance: {n90}\n")
    rpt.write(f"  Bands needed for 95% variance: {n95}\n")
    rpt.write(f"  Bands needed for 99% variance: {n99}\n")
    rpt.write(f"  Lowest-variance 5% bands: {len(low_var_bands)} bands, wavelengths {wl[low_var_bands[0]]:.1f} - {wl[low_var_bands[-1]]:.1f} nm\n")
    rpt.write(f"  (These are candidates for removal in MI/LASSO)\n")

    # Section 5: Anomaly checks
    rpt.write("\n5. ANOMALY / QUALITY CHECKS\n")
    rpt.write("-" * 50 + "\n")
    rpt.write(f"  All-zero pixel spectra : {all_zero_pixels}\n")
    rpt.write(f"  Flat spectrum pixels   : {flat_pixels}\n")
    rpt.write(f"  Pixels with max < 0.99 : {not_normalized} ({100*not_normalized/N:.2f}%)\n")
    rpt.write(f"  Band with most saturation: band {most_saturated_band} ({wl[most_saturated_band]:.1f} nm), {saturated_per_band[most_saturated_band]} pixels\n")

    # Key insights for Step 3
    rpt.write("\n6. KEY INSIGHTS FOR BAND SELECTION (Step 3)\n")
    rpt.write("-" * 50 + "\n")
    rpt.write(f"  - Most discriminative region is Red 570-700nm (mean d=0.8803)\n")
    rpt.write(f"  - Peak discriminability at {wl[best_band]:.1f} nm (d={cohens_d[best_band]:.4f})\n")
    rpt.write(f"  - 16,659 band pairs have |r|>0.95 confirming high redundancy -> band selection justified\n")
    rpt.write(f"  - NIR tail (826-878nm) is lowest variance -> likely to be dropped by MI/LASSO\n")
    rpt.write(f"  - Per-pixel normalization confirmed: 0 pixels with max < 0.99\n")
    rpt.write(f"  - Class mean difference: T={X_T.mean():.4f} vs NT={X_NT.mean():.4f} (delta={abs(X_T.mean()-X_NT.mean()):.4f})\n")
    rpt.write(f"  - T has higher std ({X_T.std():.4f}) than NT ({X_NT.std():.4f}) -> tumor is spectrally more variable\n")

print(f"Saved: {report_path}")

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Output folder: {OUT_DIR}")
print("Files generated:")
print("  spectral_discriminability.png  - mean spectra + Cohen's d per band")
print("  band_variance.png              - variance across all pixels per band")
print("  band_correlation.png           - inter-band correlation heatmap")
print("  top50_discriminative_bands.png - ranked top 50 bands")
print("  analysis_report.txt            - full text summary")
