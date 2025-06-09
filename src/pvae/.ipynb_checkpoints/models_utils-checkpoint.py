from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from tqdm.notebook import trange

from pvae.data import PathwaySplitter, StandardizedDataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _weight_init(w):
    nn.init.xavier_uniform_(w)


def train_vae(
    model_func,
    dataset,
    pathways,
    pathways_training_perc,
    models_output_dir: Path = None,
    vae_output_file_template: str = None,
    scaler_output_files_template: str = None,
    k_folds=5,
    n_folds_to_run=None,
    epochs=50,
    batch_size=250,
    lr=1e-5,
    wd=1e-5,
    random_state=None,
):
    """
    Train standard VAE with no pathway information.

    Args:
        model_func (function): The function to create the model.
        dataset (torch.utils.data.Dataset): The training dataset.
        pathways (pd.DataFrame): The pathways to predict (binary matrix).
        pathways_training_perc (float): The percentage of pathways to use for training.
        models_output_dir (Path, optional): The directory to save the models. Defaults to None
            (in that case, models are not saved).
        vae_output_file_template: a file path template (with a key "fold") to save the
            pVAE models for each fold. If None (default), the models are not saved.
        scaler_output_files_template: a file path template (with a key "fold") to save the
            StandardScaler model for each fold. If None (default), the models are not saved.
        k_folds (int, optional): The number of folds to use for cross-validation. Defaults to 5.
        n_folds_to_run (int, optional): The number of folds to run. Defaults to None.
        epochs (int, optional): The number of epochs to train for. Defaults to 50.
        batch_size (int, optional): The batch size. Defaults to 250.
        lr (float, optional): The learning rate. Defaults to 1e-5.
        wd (float, optional): The weight decay. Defaults to 1e-5.
        random_state ([type], optional): The random state. Defaults to None.

    Returns:
        A tuple with two elements: the training losses and the validation losses.
    """
    kfold_splits = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    if n_folds_to_run is None:
        n_folds_to_run = kfold_splits.get_n_splits()

    # pathways_splitter = PathwaySplitter(
    #     pathways, training_perc=pathways_training_perc, random_state=random_state
    # )

    train_losses = []
    val_losses = []

    total_n_folds = min(n_folds_to_run, kfold_splits.get_n_splits())
    folds_iter = kfold_splits.split(np.arange(len(dataset)))

    for _fold in trange(total_n_folds, desc="Fold"):
        train_idx, val_idx = next(folds_iter)

        model = model_func()
        model.to(DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # FIXME: add generator/random_state to samplers?
        train_sampler = SubsetRandomSampler(train_idx)
        train_loader = StandardizedDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=train_sampler,
        )

        val_sampler = SubsetRandomSampler(val_idx)
        val_loader = StandardizedDataLoader(
            dataset,
            scaler=train_loader.scaler,
            batch_size=batch_size,
            sampler=val_sampler,
        )


        fold_train_losses = defaultdict(list)
        fold_val_losses = defaultdict(list)

        for _epoch in trange(epochs, desc="Epoch"):
            # Training
            model.train()

            train_loss = dict.fromkeys(("full", "mse", "kl"), 0.0)
            val_loss = dict.fromkeys(("full", "mse", "kl"), 0.0)

            for batch_idx, (batch_data, _, batch_data_idxs) in enumerate(train_loader):
                batch_data = batch_data.to(DEVICE)

                # during training, we do validation of pathways prediction using the validation
                # set of pathway labels but on the training set of data since we need to use
                # the same features
                loss, losses = model.training_step(
                    batch_data,
                    batch_idx,
                    batch_data_idxs=batch_data_idxs,
                    pathways_train=[],
                    pathways_val=[],
                )

                # clear gradients
                optim.zero_grad()

                # backward
                loss.backward()

                # update parameters
                optim.step()

                for k, v in losses.items():
                    train_loss[k] += v.item()

                # val_loss["pathway"] += pathway_val_loss.item()

            for k, v in train_loss.items():
                train_loss[k] = v / len(train_loader.sampler)
                fold_train_losses[k].append(train_loss[k])

            # Validation
            model.eval()

            with torch.no_grad():
                for batch_idx, (batch_data, _, _batch_data_idxs) in enumerate(
                    val_loader
                ):
                    batch_data = batch_data.to(DEVICE)

                    losses = model.validation_step(
                        batch_data,
                        batch_idx,
                    )[1]

                    for k, v in losses.items():
                        val_loss[k] += v.item()

                for k, v in val_loss.items():
                    val_loss[k] = v / len(val_loader.sampler)
                    fold_val_losses[k].append(val_loss[k])

            print(
                f"Epoch {_epoch}: "
                f"{train_loss['full']:,.0f} / {val_loss['full']:,.0f} ="
                f"\t{train_loss['mse']:,.0f} / {val_loss['mse']:,.0f} +"
                f"\t{train_loss['kl']:,.0f} / {val_loss['kl']:,.0f} "
            )

        train_losses.append(fold_train_losses)
        val_losses.append(fold_val_losses)

        # save models for this fold
        if models_output_dir is not None:
            models_output_dir.mkdir(parents=True, exist_ok=True)

            # save VAE
            if vae_output_file_template is not None:
                assert "{fold}" in vae_output_file_template, (
                    "vae_output_file_template must contain '{fold}'"
                )
                output_filename = vae_output_file_template.format(fold=_fold)

                output_file = models_output_dir / output_filename
                torch.save(model.state_dict(), output_file)

            # save StandardScaler
            if scaler_output_files_template is not None:
                assert "{fold}" in scaler_output_files_template, (
                    "scaler_output_files_template must contain '{fold}'"
                )
                output_filename = scaler_output_files_template.format(fold=_fold)

                output_file = models_output_dir / output_filename
                joblib.dump(train_loader.scaler, output_file)

    return train_losses, val_losses
