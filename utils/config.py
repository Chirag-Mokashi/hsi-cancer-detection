# coding: utf-8
# utils/config.py
# Central path configuration for the HSI pipeline.
#
# Set HSI_DATA_ROOT before running any pipeline script:
#   Windows:  set HSI_DATA_ROOT=C:\path\to\HSI
#   Unix/Mac: export HSI_DATA_ROOT=/path/to/HSI
#
# Defaults to C:\Users\mokas\OneDrive\Desktop\HSI if the env var is not set.

import os
from pathlib import Path

DATA_ROOT        = Path(os.environ.get(
    "HSI_DATA_ROOT",
    r"C:\Users\mokas\OneDrive\Desktop\HSI"
)).expanduser().resolve()

PREPROCESSED_DIR = DATA_ROOT / "preprocessed"
BAND_SEL_DIR     = DATA_ROOT / "band_selection"
