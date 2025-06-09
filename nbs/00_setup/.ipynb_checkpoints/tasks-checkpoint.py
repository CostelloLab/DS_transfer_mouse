"""
This file defines the tasks for pytask.
"""

from pathlib import Path

import pytask

from pvae import conf
from pvae.pytask import Notebook
from pvae.pytask.utils import get_task_data

TASKS = {
    "download_gtex_v8_samples_metadata": (
        conf.common.CODE_DIR / "nbs/00_setup/gtex_v8/py/01-download-sample_metadata.py",
        pytask.mark.setup_data,
    ),
    "download_gtex_v8_gene_expr_data": (
        conf.common.CODE_DIR / "nbs/00_setup/gtex_v8/py/01-download-gene_expr_data.py",
        pytask.mark.setup_data,
    ),
    "process_gtex_v8_genes_ids": (
        conf.common.CODE_DIR / "nbs/00_setup/gtex_v8/py/02-process-gene_ids.py",
        pytask.mark.setup_data,
    ),
    "process_gtex_v8_samples_metadata": (
        conf.common.CODE_DIR / "nbs/00_setup/gtex_v8/py/02-process-sample_metadata.py",
        pytask.mark.setup_data,
    ),
    "process_gtex_v8_gene_expr_data": (
        conf.common.CODE_DIR / "nbs/00_setup/gtex_v8/py/02-process-gene_expr_data.py",
        pytask.mark.setup_data,
    ),
    "standardize_gtex_v8_gene_expr_data": (
        conf.common.CODE_DIR
        / "nbs/00_setup/gtex_v8/py/03-standardize-gene_expr_data.py",
        pytask.mark.setup_data,
    ),
}

for task_name, task_data in TASKS.items():
    nb_path, task_markers = get_task_data(task_data)

    nb = Notebook(nb_path)
    nb_inputs = nb.get_inputs()
    nb_outputs = nb.get_outputs()

    @pytask.task(name=task_name)
    def _task(
        nb_path: Path = nb_path,
        nb_obj: Notebook = nb,
        input_data: dict[str, Path] = nb_inputs,
        produces: dict[str, Path] = nb_outputs,
    ) -> None:
        nb_obj.run()

    for task_marker in task_markers:
        _task = task_marker(_task)
