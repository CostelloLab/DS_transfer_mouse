"""
This file defines the tasks for pytask.
"""

from pathlib import Path

import pytask

from pvae import conf
from pvae.pytask import Notebook

# fmt: off

nb0_path = (
    conf.common.CODE_DIR / "nbs/10_pvae/py/00-split_pathways_labels.py"
)
nb0 = Notebook(nb0_path)
@pytask.task
@pytask.mark.setup_data
def split_pathways_train_test_sets(
    nb_path: Path = nb0_path,
    input_data: dict[str, Path] = nb0.get_inputs(),
    produces: dict[str, Path] = nb0.get_outputs(),
) -> None:
    nb0.run()


for lr in (1e-4,):
    for batch_size in (50, 100, 200):
        for pred_l in (0.0, 2.0, 4.0, 8.0):
            prefix = f"lr_{lr:.0e}-batch_size_{batch_size:03d}-pred_l_{pred_l}"

            # pVAE training
            nb1_path = (
                conf.common.CODE_DIR / "nbs/10_pvae/py/01-train-pvae-gtex.py"
            )
            nb1 = Notebook(
                input_path=nb1_path,
                output_path=f"output/01-train-vae-gtex-runs/{prefix}.ipynb",
                parameters=[f"LEARNING_RATE {lr}", f"BATCH_SIZE {batch_size}", f"PRED_L {pred_l}"],
            )

            @pytask.task(id=f"{prefix}")
            @pytask.mark.run_pvae
            def train_vae_gtex(
                nb_path: Path = nb1_path,
                notebook_obj=nb1,
                input_data: dict[str, Path] = nb1.get_inputs(),
                produces: dict[str, Path] = nb1.get_outputs()
            ) -> None:
                notebook_obj.run()

            # pVAE pathway prediction evaluation
            nb2_path = (
                conf.common.CODE_DIR / "nbs/10_pvae/py/02-pathway_predictor.py"
            )
            nb2 = Notebook(
                input_path=nb2_path,
                output_path=f"output/02-pathway_predictor/{prefix}.ipynb",
                parameters=[f"LEARNING_RATE {lr}", f"BATCH_SIZE {batch_size}", f"PRED_L {pred_l}"],
            )

            @pytask.task(id=f"{prefix}")
            @pytask.mark.run_pvae
            def eval_pvae_gtex(
                nb_path: Path = nb2_path,
                notebook_obj=nb2,
                input_data: dict[str, Path] = nb2.get_inputs(),
                produces: dict[str, Path] = nb2.get_outputs()
            ) -> None:
                notebook_obj.run()
