"""
A file full of useful functions.

Taken from: https://github.com/greenelab/linear_signal/blob/master/src/utils.py
"""

import collections
import copy
import functools
import json
import pickle
import random
from pathlib import Path
from typing import Any

# import h5py as h5
# import neptune.new as neptune
import numpy as np
import pandas as pd
import torch
import yaml
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

# from sklearn.metrics import accuracy_score

# import datasets


BLOOD_KEYS = [
    "blood",
    "blood (buffy coat)",
    "blood cells",
    "blood monocytes",
    "blood sample",
    "cells from whole blood",
    "fresh venous blood anticoagulated with 50 g/ml thrombin-inhibitor lepirudin",
    "healthy human blood",
    "host peripheral blood",
    "leukemic peripheral blood",
    "monocytes isolated from pbmc",
    "normal peripheral blood cells",
    "pbmc",
    "pbmcs",
    "peripheral blood",
    "peripheral blood (pb)",
    "peripheral blood mononuclear cell",
    "peripheral blood mononuclear cell (pbmc)",
    "peripheral blood mononuclear cells",
    "peripheral blood mononuclear cells (pbmc)",
    "peripheral blood mononuclear cells (pbmcs)",
    "peripheral blood mononuclear cells (pbmcs) from healthy donors",
    "peripheral maternal blood",
    "peripheral whole blood",
    "periphral blood",
    "pheripheral blood",
    "whole blood",
    "whole blood (wb)",
    "whole blood, maternal peripheral",
    "whole venous blood",
]


def get_mutation_labels(mutation_file_path: str, gene: str) -> dict[str, str]:
    """
    Get the mutation status of a gene in all tcga samples

    Arguments
    ---------
    mutation_file_path: The path to the tcga mutation file (usually called data/mutations.tsv)
    gene: The id of gene whose mutations should be parsed

    Returns
    -------
    sample_to_label: The mapping between sample id and binary mutation status
    """
    mutation_df = pd.read_csv(mutation_file_path, sep="\t", index_col=0)

    sample_to_label = dict(mutation_df[gene])

    return sample_to_label


def get_gtex_sample_to_study(metadata_path: str) -> dict[str, str]:
    """
    Parse the GTEx metadata file to map samples to studies

    Arguments
    ---------
    metadata_path: The path to the GTEx sample attributes file

    Returns
    -------
    sample_to_study: A dict mapping each sample to its corresponding study
    """
    metadata_df = pd.read_csv(metadata_path, sep="\t", index_col=0)

    samples = metadata_df.index
    donors = [s.split("-")[1] for s in samples]

    sample_to_study = dict(zip(samples, donors, strict=False))

    return sample_to_study


def get_gtex_sample_to_label(metadata_path: str) -> dict[str, str]:
    """
    Parse the GTEx metadata file to map samples to tissues

    Arguments
    ---------
    metadata_path: The path to the GTEx sample attributes file

    Returns
    -------
    sample_to_study: A dict mapping each sample to its corresponding tissue
    """
    metadata_df = pd.read_csv(metadata_path, sep="\t", index_col=0)

    sample_to_label = dict(zip(metadata_df.index, metadata_df["SMTS"], strict=False))

    return sample_to_label


def remove_study_samples(
    dataset: "datasets.ExpressionDataset", studies_to_remove: set[str]
) -> "datasets.ExpressionDataset":
    """
    Remove the samples corresponding to the given studies from a dataset

    Arguments
    ---------
    dataset - The dataset samples will be removed from
    studies_to_remove - The studies whose samples should be removed from the dataset

    Returns
    -------
    dataset - The dataset after samples have been removed
    """
    samples = dataset.get_samples()
    sample_to_study = dataset.get_samples_to_studies()
    samples_to_remove = get_samples_in_studies(
        samples, studies_to_remove, sample_to_study
    )
    dataset.remove_samples(samples_to_remove)

    return dataset


def split_by_tissue(
    data: "datasets.ExpressionDataset", tissues: list[str], num_splits: int = 5
) -> list["datasets.ExpressionDataset"]:
    """
    Split a dataset into sections, with each split containing distinct tissues

    Arguments
    ---------
    data: The dataset to split
    tissues: The list of tissues to keep
    num_splits: The number of partitions to divide the dataset into

    Returns
    -------
    splits: A list of datasets split by tissue
    """
    if num_splits <= 1:
        return data

    splits = []

    # Split labels into lists
    all_subsets = []
    for i in range(num_splits):
        subset_tissues = [
            t for t_index, t in enumerate(tissues) if t_index % num_splits == i
        ]
        all_subsets.append(subset_tissues)

    # Use subset_to_label to create five datasets
    for subset in all_subsets:
        data = data.subset_samples_to_labels(subset)
        # This is very memory intensive since the full dataset gets stored behind the scenes after
        # subsetting. It could be made more efficient with a special subset function that
        # removes the old data, but that may make the API too messy
        data_copy = copy.deepcopy(data)
        data.reset_filters()

        splits.append(data_copy)

    # Ensure no studies are shared between the datasets
    for i, data_one in enumerate(splits):
        for j, data_two in enumerate(splits):
            # We need to get the studies each time because they'll updated when samples change
            set_one = data_one.get_studies()
            set_two = data_two.get_studies()

            if set_one == set_two:
                continue

            shared_studies = set_one.intersection(set_two)
            if len(shared_studies) > 0:
                pass
                # If there are more studies in set one, remove samples from set one
                if len(set_one) >= len(set_two):
                    data_one = remove_study_samples(data_one, shared_studies)
                    splits[i] = data_one
                # If there are more studies in set two, remove samples from set two
                else:
                    data_two = remove_study_samples(data_two, shared_studies)
                    splits[j] = data_two

    return splits


def split_sample_names(df_row: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Get a list of sample names from a dataframe row containing comma separated names

    Arguments
    ---------
    df_row: The current dataframe row to process. Should have a "train samples"
            and a "val samples" column

    Returns
    -------
    train_samples: The sample ids from the "train samples" row
    val__samples: The sample ids from the "val samples" row
    """
    train_samples = df_row["train samples"].split(",")
    val_samples = df_row["val samples"].split(",")

    return train_samples, val_samples


def create_dataset_stat_df(
    metrics_df: pd.DataFrame,
    sample_to_study: dict[str, str],
    sample_metadata: dict,
    sample_to_label: dict[str, str],
    disease: str,
) -> pd.DataFrame:
    """Create a dataframe storing stats about model training runs using information
    from a dataframe created from their result file

    Arguments
    ---------
    metrics_df: A dataframe containing the dataframe form of a results file
    sample_to_study: A mapping from sample ids to study ids
    sample_metadata: A dictionary containing metadata about samples
    sample_to_label: a mapping from sample ids to disease labels
    disease: The disease of interest for the current results file

    Returns
    -------
    A dataframe with the statistics for the current dataset
    """

    data_dict = {
        "train_disease_count": [],
        "train_healthy_count": [],
        "val_disease_count": [],
        "val_healthy_count": [],
        "accuracy": [],
        "balanced_accuracy": [],
        "subset_fraction": [],
        "seed": [],
        "model": [],
    }
    for _, row in metrics_df.iterrows():
        # Keep analysis simple for now
        data_dict["seed"].append(row["seed"])
        data_dict["subset_fraction"].append(row["healthy_used"])
        data_dict["accuracy"].append(row["accuracy"])
        data_dict["model"].append(row["supervised"])
        if "balanced_accuracy" in row:
            data_dict["balanced_accuracy"].append(row["balanced_accuracy"])

        train_samples, val_samples = split_sample_names(row)

        (
            train_studies,
            train_platforms,
            train_diseases,
            train_disease_counts,
        ) = get_dataset_stats(
            train_samples, sample_to_study, sample_metadata, sample_to_label
        )
        data_dict["train_disease_count"].append(train_diseases[disease])
        data_dict["train_healthy_count"].append(train_diseases["healthy"])

        (
            val_studies,
            val_platforms,
            val_diseases,
            val_disease_counts,
        ) = get_dataset_stats(
            val_samples, sample_to_study, sample_metadata, sample_to_label
        )
        data_dict["val_disease_count"].append(val_diseases[disease])
        data_dict["val_healthy_count"].append(val_diseases["healthy"])

    stat_df = pd.DataFrame.from_dict(data_dict)

    stat_df["train_disease_percent"] = stat_df["train_disease_count"] / (
        stat_df["train_disease_count"] + stat_df["train_healthy_count"]
    )

    stat_df["val_disease_percent"] = stat_df["val_disease_count"] / (
        stat_df["val_disease_count"] + stat_df["val_healthy_count"]
    )

    stat_df["train_val_diff"] = abs(
        stat_df["train_disease_percent"] - stat_df["val_disease_percent"]
    )
    stat_df["train_count"] = (
        stat_df["train_disease_count"] + stat_df["train_healthy_count"]
    )

    return stat_df


def get_dataset_stats(
    sample_list: list[str],
    sample_to_study: dict[str, str],
    sample_metadata: dict,
    sample_to_label: dict[str, str],
) -> tuple[collections.Counter, collections.Counter, collections.Counter, dict]:
    """
    Calculate statistics about a list of samples

    Arguments
    ---------
    sample_list: A list of sample ids to calculate statistics for
    sample_to_study: A mapping from sample ids to study ids
    sample_metadata: A dictionary containing metadata about samples
    sample_to_label: a mapping from sample ids to disease labels

    Returns
    -------
    studies: The number of samples in each study id
    platforms: The number of samples using each expression quantification platform
    diseases: The number of samples labeled with each disease
    study_disease_counts: The number of samples corresponding to each disease in each study
    """
    studies = []
    platforms = []
    diseases = []
    study_disease_counts = {}

    for sample in sample_list:
        study = sample_to_study[sample]
        studies.append(study)
        platform = sample_metadata[sample]["refinebio_platform"].lower()
        platforms.append(platform)

        disease = sample_to_label[sample]
        diseases.append(disease)

        if study in study_disease_counts:
            study_disease_counts[study][disease] = (
                study_disease_counts[study].get(disease, 0) + 1
            )
        else:
            study_disease_counts[study] = {disease: 1}

    studies = collections.Counter(studies)
    platforms = collections.Counter(platforms)
    diseases = collections.Counter(diseases)

    return studies, platforms, diseases, study_disease_counts


def generate_mask(shape: torch.Size, fraction_zeros: float) -> torch.FloatTensor:
    """Generate a mask tensor marking input data for dropout

    Arguments
    ---------
    shape: The shape of the resulting mask tensor
    fraction_zeros: The probability of setting each element in the resulting tensor to
                    zero (the rest will be set to ones)

    Returns
    -------
    mask: The mask tensor of shape `shape`
    """
    # Get [0,1] random numbers, and set all greater than fraction_zeros to 1
    # (and set all others to 0)
    return torch.rand(shape) > fraction_zeros


def parse_map_file(map_file_path: str) -> dict[str, str]:
    """Create a sample: label mapping from the pickled file output by label_samples.py

    Arguments
    ---------
    map_file_path: The path to a pickled file created by label_samples.py

    Returns
    -------
    sample_to_label: A string to string dict mapping sample ids to their corresponding label string
        E.g. {'GSM297791': 'sepsis'}
    """
    sample_to_label = {}
    label_to_sample = None
    with open(map_file_path, "rb") as map_file:
        label_to_sample, _ = pickle.load(map_file)

    for label in label_to_sample:
        for sample in label_to_sample[label]:
            sample_to_label[sample] = label

    return sample_to_label


def get_tissue(sample_metadata: dict, sample: str) -> str | None:
    """Extract the tissue type for the given sample from the metadata

    Arguments
    ---------
    sample_metadata: A dictionary containing metadata about all samples in the dataset
    sample: The sample id

    Returns
    -------
    tissue: The tissue name, if present. Otherwise returns None
    """
    try:
        ch1 = sample_metadata[sample]["refinebio_annotations"][0]["characteristics_ch1"]
        for characteristic in ch1:
            if "tissue:" in characteristic:
                tissue = characteristic.split(":")[1]
                tissue = tissue.strip().lower()
                return tissue
    # Catch exceptions caused by a field not being present
    except KeyError:
        return None

    # 'refinebio_annotations' is usually a length 1 list containing a dictionary.
    # Sometimes it's a length 0 list indicating there aren't annotations
    except IndexError:
        return None
    return None


def get_blood_sample_ids(metadata: dict, sample_to_label: dict[str, str]) -> set[str]:
    """Retrieve the sample identifiers for all samples in the datset that are from blood

    Arguments
    ---------
    metadata: A dictionary containing metadata about the dataset. Usually found in a file called
        aggregated_metadata.json
    sample_to_label: A mapping between sample identifiers and their disease label

    Returns
    -------
    sample_ids: The identifiers for all blood samples
    """
    sample_metadata = metadata["samples"]

    # Find labeled and unlabeled blood samples
    labeled_samples = set(sample_to_label.keys())

    unlabeled_samples = set()
    for sample in sample_metadata:
        tissue = get_tissue(sample_metadata, sample)
        if tissue in BLOOD_KEYS and sample not in labeled_samples:
            unlabeled_samples.add(sample)

    sample_ids = labeled_samples.union(unlabeled_samples)

    return sample_ids


def run_combat(expression_values: np.array, batches: list[str]) -> np.array:
    """Use ComBat to correct for batch effects

    Arguments
    ---------
    expression_values: A genes x samples matrix of expression values to be corrected
    batches: The batch e.g. platform, study, or experiment that each sample came from, in order

    Returns
    -------
    corrected_expression: A genes x samples matrix of batch corrected expression
    """
    sva = importr("sva")

    pandas2ri.activate()

    corrected_expression = sva.ComBat(expression_values, batches)

    return corrected_expression


def run_limma(
    expression_values: np.array, batches: list[str], second_batch: list[str] = None
) -> np.array:
    """Use limma to correct for batch effects

    Arguments
    ---------
    expression_values: A genes x samples matrix of expression values to be corrected
    batches: The batch e.g. platform, study, or experiment that each sample came from, in order
    second_batch: Another list of batch information to account for

    Returns
    -------
    corrected_expression: A genes x samples matrix of batch corrected expression
    """
    limma = importr("limma")

    pandas2ri.activate()

    if second_batch is None:
        return limma.removeBatchEffect(expression_values, batches)
    else:
        return limma.removeBatchEffect(expression_values, batches, second_batch)


def parse_label_file(label_file_path: str | Path) -> dict[str, str]:
    """
    Create a sample to label mapping from the pickled file output by label_samples.py

    Arguments
    ---------
    map_file_path: The path to a pickled file created by label_samples.py

    Returns
    -------
    sample_to_label: A string to string dict mapping sample ids to their
        corresponding label string. E.g. {'GSM297791': 'sepsis'}
    """
    sample_to_label = {}
    label_to_sample = None
    with open(label_file_path, "rb") as map_file:
        label_to_sample, _ = pickle.load(map_file)

    for label in label_to_sample:
        for sample in label_to_sample[label]:
            assert sample not in sample_to_label
            sample_to_label[sample] = label

    return sample_to_label


def parse_metadata_file(metadata_path: str | Path) -> dict:
    """
    Parse a json file containing metadata about a compendium's samples

    Arguments
    ---------
    metadata_path: The file containing metadata for all samples in the compendium

    Returns
    -------
    metadata: The json object stored at metadata_path
    """
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
        return metadata


def calculate_loss_weights(dataset: "datasets.RefineBioLabeledDataset") -> torch.Tensor:
    """
    Calculate the weights to use in training based on the inverse of their class frequency

    Arguments
    ---------
    dataset: The object containing data to calculate the class frequencies from

    Returns
    -------
    weights: A 1-d tensor compatible with pytorch weighted loss functions
    """
    classes = dataset.get_label_encoder().classes_
    counts = torch.zeros(len(classes))

    for _, label in dataset:
        counts[label] += 1

    # Weight classes based on the inverse of their frequencies
    weights = 1 / (counts + 1)
    return weights


def calculate_skl_class_weights(
    dataset: "datasets.RefineBioLabeledDataset",
) -> dict[int, float]:
    """
    Calculate the weights to use in training based on the inverse of their class frequency
    and return a dict for use by skl models

    Arguments
    ---------
    dataset: The object containing data to calculate the class frequencies from

    Returns
    -------
    class_weights: A mapping between encoded class labels and their inverse frequencies
    """
    classes = dataset.get_label_encoder().classes_
    counts = torch.zeros(len(classes))

    for _, label in dataset:
        counts[label] += 1

    class_weights = {}
    for i, count in enumerate(counts):
        class_weights[i] = 1 / (count + 1)

    return class_weights


def parse_flynn_labels(label_path: str) -> dict[str, str]:
    """
    Create a sample to label mapping from the Flynn et al. sex labels

    Arguments
    ---------
    label_path: The path to the metadata file from Flynn et al

    Returns
    -------
    sample_to_label: A sample to label mapping
    """
    metadata_df = pd.read_csv(label_path)

    sample_to_label = {}

    for _, row in metadata_df.iterrows():
        if row["organism"] != "human" or row["data_type"] != "rnaseq":
            continue

        # Remove samples without known sex or from studies with male and female samples
        # where the given sample's sex is unknown
        if row["metadata_sex"] == "male" or row["metadata_sex"] == "female":
            sample_to_label[row["sample_acc"]] = row["metadata_sex"]

    return sample_to_label


@functools.lru_cache
def load_compendium_file(compendium_path: str | Path) -> pd.DataFrame:
    """
    Load refine.bio compendium data from a tsv file

    Arguments
    ---------
    compendium_path: The path to the file containing the compendium of gene expression data

    Returns
    -------
    expression_df: A dataframe where the rows are genes anbd the columns are samples
    """
    # Assume the expression is a pickle file. If not, try opening it as a tsv
    try:
        expression_df = pd.read_pickle(compendium_path)
    except pickle.UnpicklingError:
        expression_df = pd.read_csv(compendium_path, sep="\t", index_col=0)

    return expression_df


@functools.lru_cache
def load_recount_data(
    config_file: str,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """
    Read the dataset config file for the recount dataset and parse the files it points to

    Arguments
    ---------
    config_file: The file containing paths to the metadata file and counts file

    Returns
    -------
    expression_df - A dataframe containing expression data where rows are genes and columns
                    are samples
    sample_to_label - A dict mapping samples to labels
    sample_to_study - A dict mapping samples to studies
    """

    with open(config_file) as in_file:
        dataset_config = yaml.safe_load(in_file)

    compendium_path = dataset_config.pop("compendium_path")
    metadata_path = dataset_config.pop("metadata_path")

    expression_df = load_compendium_file(compendium_path).T
    sample_to_study = recount_map_sample_to_study(metadata_path)

    sample_to_label = None
    with open(dataset_config.pop("label_path"), "rb") as in_file:
        sample_to_label = pickle.load(in_file)

    return expression_df, sample_to_label, sample_to_study


def recount_map_sample_to_study(metadata_file: str) -> dict[str, str]:
    """
    Parse the recount3 metadata file and extract the sample to study mappings

    Arguments
    ---------
    metadata_file: The path to where the metadata is stored

    Returns
    -------
    sample_to_study: A mapping between samples and studies
    """
    with open(metadata_file) as in_file:
        header = in_file.readline()
        header = header.replace('"', "")
        header = header.strip().split("\t")

        # Add one to the indices to account for the index column in metadata not present in the
        # header
        sample_index = header.index("external_id") + 1
        study_index = header.index("study") + 1

        sample_to_study = {}
        for line in in_file:
            line = line.strip().split("\t")
            sample = line[sample_index]
            sample = sample.replace('"', "")
            study = line[study_index]
            study = study.replace('"', "")
            sample_to_study[sample] = study

    return sample_to_study


def map_sample_to_study(metadata_json: dict, sample_ids: list[str]) -> dict[str, str]:
    """
    Map each sample id to the study that generated it

    Arguments
    ---------
    metadata_json: The metadata for the whole compendium. This metadata is structured by the
        refine.bio pipeline, and will typically be found in a file called aggregated_metadata.json
    sample_ids:
        The accessions for each sample

    Returns
    -------
    sample_to_study:
        The mapping from sample accessions to the study they are a member of
    """
    experiments = metadata_json["experiments"]
    id_set = set(sample_ids)

    sample_to_study = {}
    for study in experiments:
        for accession in experiments[study]["sample_accession_codes"]:
            if accession in id_set:
                sample_to_study[accession] = study

    return sample_to_study


def get_samples_in_studies(
    samples: list[str], studies: set[str], sample_to_study: dict[str, str]
) -> list[str]:
    """
    Find which samples from the list were generated by the given studies

    Arguments
    ---------
    samples: The accessions of all samples
    studies: The studies of interest that generated a subset of samples in the list
    sample_to_study: A mapping between each sample and the study that generated it

    Returns
    -------
    subset_samples: The samples that were generated by a study in `studies`
    """
    subset_samples = [
        sample for sample in samples if sample_to_study[sample] in studies
    ]
    return subset_samples


def sigmoid_to_predictions(model_output: np.ndarray) -> torch.Tensor:
    """
    Convert the sigmoid output of a model to integer labels

    Arguments
    ---------
    predictions: The labels the model predicted

    Returns
    -------
    The integer labels predicted by the model
    """
    return torch.argmax(model_output, dim=-1)


def deterministic_shuffle_set(set_: set) -> list[Any]:
    """
    random.choice does not behave deterministically when used on sets, even if a seed is set.
    This function sorts the list representation of the set and samples from it, preventing
    determinism bugs

    Arguments
    ---------
    set_: The set to shuffle

    Returns
    -------
    shuffled_list: The shuffled list representation of the original set
    """
    shuffled_list = random.sample(sorted(set_), len(set_))

    return shuffled_list


def determine_subset_fraction(
    train_positive: int, train_negative: int, val_positive: int, val_negative: int
) -> int:
    """
    Determine the correct fraction of samples to remove from the training positive or negative
    sample pool to match the fraction of positive samples in the validation set

    Arguments
    ---------
    train_positive: The number of positive training samples
    train_negative: The number of negative training samples
    val_positive: The number of positive validation samples
    val_negative: The number of negative validation samples

    Returns
    -------
    subset_fraction: The fraction of positive or negative (determined by the calling code) samples
                     to remove
    """
    train_disease_fraction = train_positive / (train_negative + train_positive)
    val_disease_fraction = val_positive / (val_positive + val_negative)

    # If train ratio is too high, remove positive samples
    if train_disease_fraction > val_disease_fraction:
        # X / (negative + X) = val_frac. Solve for X
        target = (val_disease_fraction * train_negative) / (1 - val_disease_fraction)
        subset_fraction = target / train_positive
    # If the train ratio is too low, remove negative samples
    elif train_disease_fraction < val_disease_fraction:
        # positive / (positive + X) = val_frac. Solve for X
        target = (
            train_positive - (val_disease_fraction * train_positive)
        ) / val_disease_fraction
        subset_fraction = target / train_negative
    # If the train and val ratios are balanced, then don't remove any samples
    else:
        return 0

    return subset_fraction


def subset_to_equal_ratio(
    train_data: "datasets.LabeledDataset",
    val_data: "datasets.LabeledDataset",
    label: str,
    negative_class: str,
    seed: int,
) -> "datasets.LabeledDataset":
    """
    Subset the training dataset to match the ratio of positive to negative expression samples in
    the validation dataset

    Arguments
    ---------
    train_data: The train expression dataset
    val_data: The validation expression dataset

    Returns
    -------
    train_data: The subsetted expression dataset
    """

    train_disease_counts = train_data.map_labels_to_counts()
    val_disease_counts = val_data.map_labels_to_counts()

    train_positive = train_disease_counts.get(label, 0)
    train_negative = train_disease_counts.get(negative_class, 0)
    val_positive = val_disease_counts.get(label, 0)
    val_negative = val_disease_counts.get(negative_class, 0)

    train_disease_fraction = train_positive / (train_positive + train_negative)
    val_disease_fraction = val_positive / (val_positive + val_negative)

    subset_fraction = determine_subset_fraction(
        train_positive, train_negative, val_positive, val_negative
    )

    # If train ratio is too high, remove positive samples
    if train_disease_fraction > val_disease_fraction:
        train_data = train_data.subset_samples_for_label(subset_fraction, label, seed)
    # If train ratio is too low, remove negative samples
    elif train_disease_fraction < val_disease_fraction:
        train_data = train_data.subset_samples_for_label(
            subset_fraction, negative_class, seed
        )
    return train_data
