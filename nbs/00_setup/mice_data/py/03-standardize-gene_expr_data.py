# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # Description

# %% [markdown] tags=[]
# It reads the entire Mice data set, performs some preprocessing and then standardizes the data to have mean 0 and std 1.

# %% [markdown] tags=[]
# # Modules

# %% tags=[]
# reload imported modules if changed
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
from IPython.display import display

# from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pvae import conf

# %% [markdown] tags=[]
# # Settings and paths

# %% tags=[]
conf.data.MICE_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
display(conf.data.MICE_PROCESSED_DIR)

# %% tags=["inout_defs"]
# INPUT_FILES and OUTPUT_FILES must be provided if running with pytask
INPUT_FILES = {
    "mice_gene_expr_data_feather": conf.data.MICE_PROCESSED_DIR
    / "no_scrna_rpkm.ftr",
}

OUTPUT_FILES = {
    "mice_gene_expr_data_feather_prep": conf.data.MICE_PROCESSED_DIR
    / "mice_data-full-prep.ftr",
    "mice_gene_expr_data_feather_log": conf.data.MICE_PROCESSED_DIR
    / "mice_data-full-log.ftr",
    "mice_gene_expr_data_feather_std": conf.data.MICE_PROCESSED_DIR
    / "mice_data-full-std.ftr",
}

# %% [markdown] tags=[]
# # Load data

# %% tags=[]
full_dataset = pd.read_feather(INPUT_FILES["mice_gene_expr_data_feather"])

# %% tags=[]
full_dataset.shape

# %% tags=[]
full_dataset.head()

# %% [markdown] tags=[]
# # Prepare data

# %% tags=[]
full_dataset.isna().any(axis=None)

# %% tags=[]
data = full_dataset.set_index("index").rename_axis(index=None)

# %% tags=[]
data = data.iloc[:, 1:].T

# %% tags=[]
data.shape

# %% tags=[]
data.head()

# %% [markdown] tags=[]
# # Save original data

# %% tags=[]
data_output = data.T.reset_index()
# print(data_output.drop('index', axis=1).min(axis=0).min())
display(data_output.isna().any(axis=None))
data_output.head()

# %% tags=[]
data_output.to_feather(OUTPUT_FILES["mice_gene_expr_data_feather_prep"])

# %% [markdown] tags=[]
# # Save log2(data + 1)

# %% tags=[]
data_log = np.log2(data.T + 3).reset_index()
display(data_log.isna().any(axis=None))
data_log.head()

# %% tags=[]
data_log.to_feather(OUTPUT_FILES["mice_gene_expr_data_feather_log"])

# %% [markdown] tags=[]
# # Save data log2 z-scaled

# %% tags=[]
# Standardize the features
scaler = StandardScaler().set_output(transform="pandas")
data_scaled = scaler.fit_transform(data_log.set_index("index").T)

# %% tags=[]
data_scaled.head()

# %% tags=[]
data_scaled.iloc[:, 1:10].describe()

# %% tags=[]
data_output = data_scaled.T.reset_index()
display(data_output.isna().any(axis=None))
data_output.head()

# %% tags=[]
data_output.to_feather(OUTPUT_FILES["mice_gene_expr_data_feather_std"])

# %%
