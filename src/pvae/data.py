from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class FeatherDataset(Dataset):
    """
    Dataset that reads a feather binary data file. It assumes there is one column named "index" with the row names,
    and all the rest is data.

    Args:
        feather_file_path: Path to the Feather file. It assumes there is one column named "index" with
            the row names, and all the rest is data.
        read_full_data: If True, read the full data into memory.
    """

    def __init__(self, feather_file_path: Path | str, read_full_data=True):
        if not read_full_data:
            raise NotImplementedError("read_full_data=False is not implemented yet.")

        self.data = pd.read_feather(feather_file_path).drop(columns=["index"])
        self.length = self.data.shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.data.iloc[:, idx].to_numpy()
        data = torch.tensor(data, dtype=torch.float32)

        return data, torch.tensor([]), idx


class StandardizedDataLoader(DataLoader):
    """
    DataLoader that standardizes the data using sklearn's StandardScaler. It learns the mean and std of the data using
    the partial_fit method (so it can process large datasets) with batches of data (same as the data loader).

    It has the same arguments as a DataLoader, except for scaler, which is a StandardScaler object. If not given, it
    will be created and fitted with the data.
    """

    def __init__(self, *args, scaler=None, **kwargs):
        super().__init__(*args, **kwargs)

        # learn the mean and std of the data
        self.scaler = scaler

        if self.scaler is None:
            self.scaler = StandardScaler()

            loader = DataLoader(*args, **kwargs)
            for data, _, _data_idx in loader:
                self.scaler.partial_fit(data.numpy())

    def __iter__(self):
        for batch_data, *_ in super().__iter__():
            yield (
                torch.as_tensor(self.scaler.transform(batch_data), dtype=torch.float32),
                *_,
            )


class PathwaySplitter:
    """
    It generates training and test matrices by sampling from a binary matrix with multiple labels
    (pathways) in columns and objects (genes) in rows. It is used to split the pathways matrix
    into training and test sets.

    Args:
        pathways: Binary matrix with pathways in columns and genes in rows.
        training_perc: Percentage of genes that are left for the training matrix.
        random_state: Random state for reproducibility.
    """

    def __init__(self, pathways: pd.DataFrame, training_perc: float, random_state=None):
        self.pathways = pathways

        assert 0.0 < training_perc < 1.0, "training_perc must be between 0 and 1"
        self.training_perc = training_perc

        self.random_state = random_state

    def sample(self, x: pd.Series) -> pd.Series:
        """
        Samples 20% of genes that are part of a pathway and removes them (assigns zero).
        """
        # FIXME: == 1 part is not necessary:
        x_pos = x[x == 1]

        genes_holdout = x_pos.sample(
            frac=1.0 - self.training_perc, random_state=self.random_state
        ).index
        x = x.copy()
        x.loc[genes_holdout] = False
        return x

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the pathways matrix into training and test sets. The training set matrix will have
        20% of the genes removed from each pathway, and those will be assigned to the test set.
        :return:
        """
        # TODO: return torch.long dtype

        pathways_training = self.pathways.apply(self.sample)
        pathways_test = (
            self.pathways.astype(int) - pathways_training.astype(int)
        ).astype(bool)
        assert (pathways_training | pathways_test).equals(self.pathways), (
            "error in splitting pathways"
        )

        return pathways_training, pathways_test
