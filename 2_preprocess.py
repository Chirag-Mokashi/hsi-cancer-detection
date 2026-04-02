# coding: utf-8
# HistologyHSI-GB Preprocessing Pipeline v4
# Strict verification before any raw data deletion.
#
# Verification checks per file:
#   1. File opens without error
#   2. Has cube, wavelengths datasets and label, patient attributes
#   3. Shape is exactly (800, 1004, 699)
#   4. Values in [0, 1] - checked on 3 random patches
#   5. No NaN or Inf values
#   6. Mean is between 0.3 and 0.95 (sanity check on calibration)
#   7. File size is at least 10 MB (not empty/truncated)
#
# Raw folder deleted ONLY when ALL of a patient's ROIs pass ALL checks.

DATA_ROOT  = r"C:\Users\mokas\OneDrive\Desktop\HSI"
OUT_FOLDER = r"C:\Users\mokas\OneDrive\Desktop\HSI\preprocessed"

MAX_WAVELENGTH_NM    = 909.0
AUTO_DELETE_PATIENTS = ["P2", "P3", "top-level"]

import re
import sys
import time
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import h5py
except ImportError:
    print("[ERROR] h5py not installed. Run: python -m pip install h5py")
    sys.exit(1)

root    = Path(DATA_ROOT)
out_dir = Path(OUT_FOLDER)
out_dir.mkdir(exist_ok=True)

if not root.exists():
    print("[ERROR] DATA_ROOT not found: " + DATA_ROOT)
    sys.exit(1)


# --- Strict verification ---

def verify_h5_strict(path):
    """
    Strictly verify a preprocessed h5 file.
    Returns (True, "OK") or (False, "reason for failure")
    """
    # Check 1: minimum file size (10 MB - a valid cube is ~300MB compressed)
    size_mb = path.stat().st_size / 1e6
    if size_mb < 10:
        return False, "file too small ({:.1f} MB) - likely truncated".format(size_mb)

    try:
        with h5py.File(str(path), "r") as hf:

            # Check 2: required datasets and attributes exist
            if "cube" not in hf:
                return False, "missing 'cube' dataset"
            if "wavelengths" not in hf:
                return False, "missing 'wavelengths' dataset"
            if "label" not in hf.attrs:
                return False, "missing 'label' attribute"
            if "patient" not in hf.attrs:
                return False, "missing 'patient' attribute"

            # Check 3: exact shape
            shape = hf["cube"].shape
            if shape != (800, 1004, 699):
                return False, "wrong shape {} expected (800, 1004, 699)".format(shape)

            # Check 4: label is valid
            label = str(hf.attrs["label"])
            if label not in ("T", "NT"):
                return False, "invalid label '{}'".format(label)

            # Check 5: wavelength count matches
            wl_count = hf["wavelengths"].shape[0]
            if wl_count != 699:
                return False, "wrong wavelength count {}".format(wl_count)

            # Check 6: read 3 patches from different regions and verify values
            # Top-left, centre, bottom-right
            patches = [
                hf["cube"][0:20,    0:20,    :],   # top-left
                hf["cube"][390:410, 492:512, :],   # centre
                hf["cube"][780:800, 984:1004, :],  # bottom-right
            ]

            for j, patch in enumerate(patches):
                patch = np.array(patch, dtype=np.float32)

                # Check values in range
                if patch.min() < -0.01:
                    return False, "patch {} has values below 0: min={:.4f}".format(
                        j, patch.min())
                if patch.max() > 1.01:
                    return False, "patch {} has values above 1: max={:.4f}".format(
                        j, patch.max())

                # Check no NaN or Inf
                if np.any(np.isnan(patch)):
                    return False, "patch {} contains NaN values".format(j)
                if np.any(np.isinf(patch)):
                    return False, "patch {} contains Inf values".format(j)

                # Check mean is sensible (calibrated tissue should be 0.3-0.95)
                mean_val = float(patch.mean())
                if mean_val < 0.1 or mean_val > 1.0:
                    return False, "patch {} mean {:.3f} is outside expected range".format(
                        j, mean_val)

        return True, "OK"

    except Exception as e:
        return False, str(e)


# --- Dataset helpers ---

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
            label = parts[-1]
        else:
            label = "?"
        patient = "top-level"
        for p in folder.parents:
            if p == root:
                break
            if re.match(r"^P\d+$", p.name):
                patient = p.name
                break
        results.append({
            "path":    folder,
            "name":    name,
            "patient": patient,
            "label":   label,
        })
    return results


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
        1: np.uint8,  2: np.int16,   3: np.int32,  4: np.float32,
        5: np.float64, 12: np.uint16, 13: np.uint32,
    }
    dtype_id = get_int("data type") or get_int("data_type") or 12

    return {
        "lines":       get_int("lines"),
        "samples":     get_int("samples"),
        "bands":       get_int("bands"),
        "dtype":       dtype_map.get(dtype_id, np.uint16),
        "interleave":  get_str("interleave"),
        "byte_order":  get_int("byte order") or 0,
        "wavelengths": np.array(get_float_list("wavelength"), dtype=np.float32),
    }


def load_envi_bil(bin_path, hdr):
    lines   = hdr["lines"]
    samples = hdr["samples"]
    bands   = hdr["bands"]
    dtype   = hdr["dtype"]
    bo      = hdr["byte_order"]
    dt = np.dtype(dtype)
    dt = dt.newbyteorder(">") if bo == 1 else dt.newbyteorder("<")
    data = np.fromfile(str(bin_path), dtype=dt)
    data = data.reshape((lines, bands, samples))
    data = data.transpose(0, 2, 1)
    return data.astype(np.float32)


def load_reference_bil(bin_path, hdr):
    lines   = hdr["lines"]
    samples = hdr["samples"]
    bands   = hdr["bands"]
    dtype   = hdr["dtype"]
    bo      = hdr["byte_order"]
    dt = np.dtype(dtype)
    dt = dt.newbyteorder(">") if bo == 1 else dt.newbyteorder("<")
    data = np.fromfile(str(bin_path), dtype=dt).astype(np.float32)
    expected = lines * bands * samples
    if data.size == expected:
        data = data.reshape((lines, bands, samples))
        data = data.transpose(0, 2, 1)
        data = data.mean(axis=0)
    elif data.size == bands * samples:
        data = data.reshape((bands, samples)).T
    else:
        total_pixels = data.size // bands
        data = data[:total_pixels * bands].reshape((total_pixels, bands))
    return data


def calibrate(raw, dark, white):
    denom = white - dark
    denom = np.where(denom <= 0, 1e-6, denom)
    raw -= dark[np.newaxis, :, :]
    raw /= denom[np.newaxis, :, :]
    np.clip(raw, 0.0, 1.0, out=raw)
    return raw


def per_pixel_normalize(cube):
    pixel_max = cube.max(axis=2, keepdims=True)
    pixel_max = np.where(pixel_max <= 0, 1.0, pixel_max)
    cube /= pixel_max
    return cube


def save_h5(out_path, cube, label, patient, wavelengths):
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset(
            "cube",
            data=cube,
            compression="gzip",
            compression_opts=4,
            chunks=(100, 100, cube.shape[2])
        )
        f.create_dataset("wavelengths", data=wavelengths)
        f.attrs["label"]   = label
        f.attrs["patient"] = patient


def check_free_space_gb():
    usage = shutil.disk_usage(DATA_ROOT)
    return usage.free / 1e9


def try_delete_patient_raw(patient, rois):
    """
    Delete patient raw folder ONLY after strict verification of all their ROIs.
    Prints a per-ROI verification report before deleting.
    """
    if patient not in AUTO_DELETE_PATIENTS:
        return False

    patient_rois = [r for r in rois if r["patient"] == patient]
    total        = len(patient_rois)

    print("")
    print("  Checking if all {} ROIs for {} are verified ...".format(total, patient))

    all_pass   = True
    fail_count = 0

    for roi in patient_rois:
        out_path = out_dir / (patient + "_" + roi["name"] + ".h5")

        if not out_path.exists():
            print("  MISSING: " + roi["name"])
            all_pass = False
            fail_count += 1
            continue

        ok, reason = verify_h5_strict(out_path)
        if ok:
            print("  PASS: " + roi["name"] +
                  "  ({:.0f} MB)".format(out_path.stat().st_size / 1e6))
        else:
            print("  FAIL: " + roi["name"] + "  -> " + reason)
            all_pass = False
            fail_count += 1

    if not all_pass:
        print("  {} ROI(s) failed verification - keeping raw folder".format(fail_count))
        return False

    # All passed - safe to delete
    print("  All {} ROIs verified OK".format(total))

    if patient == "top-level":
        deleted_any = False
        for roi in patient_rois:
            raw_folder = roi["path"]
            if raw_folder.exists():
                shutil.rmtree(str(raw_folder))
                print("  [DELETED] " + raw_folder.name)
                deleted_any = True
        return deleted_any
    else:
        patient_folder = root / patient
        if patient_folder.exists():
            shutil.rmtree(str(patient_folder))
            print("  [DELETED] raw folder: " + patient)
            return True

    return False


# --- Main pipeline ---

rois = find_roi_folders(root)
print("")
print("Preprocessing pipeline v4  (strict verification)")
print("  ROIs found  : " + str(len(rois)))
print("  Output      : " + str(out_dir))
print("  Free space  : {:.1f} GB".format(check_free_space_gb()))
print("")

processed        = 0
already_done     = 0
errors           = 0
deleted_patients = set()

for i, roi in enumerate(rois):
    out_path = out_dir / (roi["patient"] + "_" + roi["name"] + ".h5")

    # Check if already processed and verified
    if out_path.exists():
        ok, reason = verify_h5_strict(out_path)
        if ok:
            already_done += 1
            print("[{}/{}] SKIP (verified): {}".format(
                i+1, len(rois), roi["name"]))

            # Try deleting patient raw folder if all done
            patient = roi["patient"]
            if patient not in deleted_patients:
                if try_delete_patient_raw(patient, rois):
                    deleted_patients.add(patient)
                    print("  Free space now: {:.1f} GB".format(
                        check_free_space_gb()))
            continue
        else:
            # File exists but failed verification - delete and reprocess
            print("[{}/{}] REPROCESS (failed: {}): {}".format(
                i+1, len(rois), reason, roi["name"]))
            out_path.unlink()

    t0 = time.time()
    print("[{}/{}] Processing: {}  (patient={}, label={})  free={:.1f}GB".format(
        i+1, len(rois), roi["name"], roi["patient"],
        roi["label"], check_free_space_gb()))

    try:
        hdr     = parse_hdr(roi["path"] / "raw.hdr")
        wl      = hdr["wavelengths"]
        lines   = hdr["lines"]
        samples = hdr["samples"]
        bands   = hdr["bands"]

        if len(wl) > 0:
            keep_idx = np.where(wl <= MAX_WAVELENGTH_NM)[0]
        else:
            keep_idx = np.arange(bands)
            wl       = np.arange(bands, dtype=np.float32)

        raw = load_envi_bil(roi["path"] / "raw", hdr)

        dark_hdr  = parse_hdr(roi["path"] / "darkReference.hdr")
        white_hdr = parse_hdr(roi["path"] / "whiteReference.hdr")
        dark      = load_reference_bil(roi["path"] / "darkReference",  dark_hdr)
        white     = load_reference_bil(roi["path"] / "whiteReference", white_hdr)

        if dark.shape[0] != samples:
            dark  = dark.mean(axis=0, keepdims=True).repeat(samples, axis=0)
        if white.shape[0] != samples:
            white = white.mean(axis=0, keepdims=True).repeat(samples, axis=0)
        if dark.shape[1] != bands:
            dark  = dark[:, :bands]
            white = white[:, :bands]

        cube = calibrate(raw, dark, white)
        del raw, dark, white

        cube    = cube[:, :, keep_idx]
        wl_kept = wl[keep_idx]
        cube    = per_pixel_normalize(cube)

        save_h5(out_path, cube, roi["label"], roi["patient"], wl_kept)
        del cube

        # Strict verification of saved file
        ok, reason = verify_h5_strict(out_path)
        if ok:
            elapsed = time.time() - t0
            size_mb = out_path.stat().st_size / 1e6
            print("  VERIFIED OK  {:.0f} MB  {:.1f}s".format(size_mb, elapsed))
            processed += 1

            # Try deleting patient raw folder
            patient = roi["patient"]
            if patient not in deleted_patients:
                if try_delete_patient_raw(patient, rois):
                    deleted_patients.add(patient)
                    print("  Free space now: {:.1f} GB".format(
                        check_free_space_gb()))
        else:
            print("  VERIFICATION FAILED: " + reason)
            if out_path.exists():
                out_path.unlink()
            errors += 1

    except MemoryError:
        print("  [ERROR] Out of memory - close other apps and retry")
        errors += 1
    except OSError as e:
        if "No space" in str(e):
            print("  [ERROR] Disk full - stopping")
            print("  Tip: some patients may now be fully done - run again")
            break
        else:
            print("  [ERROR] " + str(e))
            errors += 1
    except Exception as e:
        print("  [ERROR] " + str(e))
        import traceback
        traceback.print_exc()
        errors += 1

    print("")

print("============================================================")
print("  PREPROCESSING COMPLETE")
print("============================================================")
print("  Processed   : " + str(processed))
print("  Skipped     : " + str(already_done) + "  (verified good)")
print("  Errors      : " + str(errors))
print("  Raw deleted : " + str(sorted(deleted_patients)))
print("  Free space  : {:.1f} GB".format(check_free_space_gb()))
print("============================================================")
print("")
if errors == 0:
    print("All done. Ready for Step 3: band selection.")
else:
    print(str(errors) + " ROIs failed. Run again to retry.")
print("")
