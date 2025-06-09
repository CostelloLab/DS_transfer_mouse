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
#     display_name: pvae
#     language: python
#     name: pvae
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
import preprocess_expression as pe
from gprofiler import GProfiler
# import pymart 
# import biomart 
# from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pvae import conf

# %% [markdown] tags=[]
# # Settings and paths

# %% tags=[]
conf.data.MICE_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
conf.data.MICE_ORIG_DIR.mkdir(parents=True, exist_ok=True)
display(conf.data.MICE_PROCESSED_DIR)
display(conf.data.MICE_ORIG_DIR)

# %% tags=["inout_defs"]
# INPUT_FILES and OUTPUT_FILES must be provided if running with pytask
INPUT_FILES = {
    "gene_count": conf.data.MICE_ORIG_DIR/
    "GSE218883_Dp16_Norm_Adult_heart_GencodevM24_counts.txt",
    "gene_length": conf.data.MICE_ORIG_DIR/
    "gene_lengths.tsv",
    "extended_gene_pathway": conf.data.MICE_ORIG_DIR/
    "extended_plier_pathways.tsv",
}

OUTPUT_FILES = {
    "mice_gene_expr_norm_adult_heart_rpkm_std": conf.data.MICE_PROCESSED_DIR
    / "mice_gene_expr_norm_adult_heart_rpkm_std.ftr",
    "mice_gene_expr_norm_adult_heart_rpkm_std_log": conf.data.MICE_PROCESSED_DIR
    / "mice_gene_expr_norm_adult_heart_rpkm_std_log.ftr",
}

# %% [markdown]
# # Save Original data with rkpm normalized #

# %%
rpkm_normalized_df = pe.calculate_rkpm_normalization(INPUT_FILES["gene_count"], INPUT_FILES["gene_length"], INPUT_FILES["extended_gene_pathway"],
                 OUTPUT_FILES["mice_gene_expr_norm_adult_heart_rpkm_std"])  

# %%
rpkm_normalized_df.shape

# %% [markdown]
# # Save log2(data + 1)

# %%
rpkm_normalized_df_log = np.log2(rpkm_normalized_df + 2)
display(rpkm_normalized_df_log.isna().any(axis=None))
print("Number of nan values :", rpkm_normalized_df_log.isna().sum().sum())
rpkm_normalized_df_log = rpkm_normalized_df_log.fillna(0)
rpkm_normalized_df_log.head()
rpkm_normalized_df_log.to_feather(OUTPUT_FILES["mice_gene_expr_norm_adult_heart_rpkm_std_log"])
