"""
Contains configuration entries related to Jupyter Notebooks.
"""

import os
from pathlib import Path

from pvae.conf.common import ENV_PREFIX as __ENV_PREFIX

__ENV_PREFIX += f"{Path(__file__).stem.upper()}_"

# Jupyter server port
JUPYTER_SERVER_PORT = os.environ.get(f"{__ENV_PREFIX}JUPYTER_SERVER_PORT", 8899)

# when a notebook is run from the command line and it finished successfully, it
# indicates whether the original notebook should be overridden with the new one
RUN_OVERRIDE = os.environ.get(f"{__ENV_PREFIX}RUN_OVERRIDE", 0)
