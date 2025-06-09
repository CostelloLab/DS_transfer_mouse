"""
Contains configuration regarding the manuscript.
"""

import os
import tempfile
from pathlib import Path

from pvae.conf.common import ENV_PREFIX as __ENV_PREFIX

__ENV_PREFIX += f"{Path(__file__).stem.upper()}_"

# DIR points to the top directory of a Manubot-based manuscript
DIR = os.environ.get(f"{__ENV_PREFIX}DIR", tempfile.TemporaryDirectory().name)
DIR = Path(DIR).resolve()

# FIGURES_DIR is the directory where figures are stored
FIGURES_DIR = DIR / "content" / "images"

# SUPPLEMENTARY_MATERIAL_DIR is the directory where supplementary files are stored
SUPPLEMENTARY_MATERIAL_DIR = DIR / "content" / "supplementary_files"
