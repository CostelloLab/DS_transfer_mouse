"""
Contains configuration about input data.
"""

import os
from pathlib import Path

from pvae.conf.common import CODE_DIR as __CODE_DIR
from pvae.conf.common import ENV_PREFIX as __ENV_PREFIX

__ENV_PREFIX += f"{Path(__file__).stem.upper()}_"

# ROOT_DIR points to the base directory where all the data and results are
ROOT_DIR = os.environ.get(f"{__ENV_PREFIX}ROOT_DIR", __CODE_DIR / "base")
ROOT_DIR = Path(ROOT_DIR).resolve()

# DATA_DIR stores input data
INPUT_DIR = os.environ.get(f"{__ENV_PREFIX}INPUT_DIR", ROOT_DIR / "input")
INPUT_DIR = Path(INPUT_DIR).resolve()

# RESULTS_DIR stores newly generated data
OUTPUT_DIR = os.environ.get(f"{__ENV_PREFIX}OUTPUT_DIR", ROOT_DIR / "output")
OUTPUT_DIR = Path(OUTPUT_DIR).resolve()

# TMP_DIR stores temporary data that can be safely deleted
TMP_DIR = os.environ.get(f"{__ENV_PREFIX}TMP_DIR", INPUT_DIR / "tmp")
TMP_DIR = Path(TMP_DIR).resolve()

# GTEx
GTEX_DIR = os.environ.get(f"{__ENV_PREFIX}GTEX_DIR", INPUT_DIR / "gtex_v8")
GTEX_DIR = Path(GTEX_DIR).resolve()

GTEX_ORIG_DIR = os.environ.get(f"{__ENV_PREFIX}GTEX_ORIG_DIR", GTEX_DIR / "orig")
GTEX_ORIG_DIR = Path(GTEX_ORIG_DIR).resolve()

GTEX_PROCESSED_DIR = os.environ.get(
    f"{__ENV_PREFIX}GTEX_PROCESSED_DIR", GTEX_DIR / "processed"
)
GTEX_PROCESSED_DIR = Path(GTEX_PROCESSED_DIR).resolve()

# MICE
MICE_DIR = os.environ.get(f"{__ENV_PREFIX}MICE_DIR", INPUT_DIR / "mice_data")
MICE_DIR = Path(MICE_DIR).resolve()

MICE_ORIG_DIR = os.environ.get(f"{__ENV_PREFIX}MICE_ORIG_DIR", MICE_DIR / "orig")
MICE_ORIG_DIR = Path(MICE_ORIG_DIR).resolve()

MICE_PROCESSED_DIR = os.environ.get(
    f"{__ENV_PREFIX}MICE_PROCESSED_DIR", MICE_DIR / "processed"
)
MICE_PROCESSED_DIR = Path(MICE_PROCESSED_DIR).resolve()

