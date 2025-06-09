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
# 1. Collect the mice data from here https://www.nature.com/articles/s41588-023-01399-7#data-availability
# 2. Use that as outoff sample data (genes x samples) to generate B matrix (lvs x samples) from VAE and PLIER models
# 3. Run permutation multiple group analysis on generate B matrices from VAE and PLIER.

# %% [markdown] tags=[]
# # Modules

# %% tags=["modules_imports"]
# reload imported modules if changed
# %load_ext autoreload
# %autoreload 2

import pprint

import joblib
import matplotlib.pyplot as plt
import numpy as np
import random
import pyreadr
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr, permutation_test, f_oneway, ranksums 
from statsmodels.stats.multitest import multipletests
import dcor

from pvae import conf
from pvae.data import FeatherDataset
from pvae.models import VariationalAutoencoder
from pvae.models_utils import train_vae

# %% [markdown] tags=[]
# # Settings and paths

# %% tags=[]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
display(DEVICE)

# %% tags=["parameters"]
# Model parameters
K = 168
KL_L = 1.0
PRED_L = 2.0

# Others parameters
K_FOLDS = 3
N_FOLDS_TO_RUN = None  # None to run all K_FOLDS (used for testing)
N_EPOCS = 50
BATCH_SIZE = 50
PATHWAYS_TRAINING_PERC = 0.80
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
RANDOM_SEED = 0

# %% tags=["parameters_extra"]
# Paths
OUTPUT_FOLDER = (
    conf.data.OUTPUT_DIR
    / "models"
    / "mice_vae"
    / f"k_{K}-kl_l_{KL_L}-pred_l_{PRED_L}-batch_size_{BATCH_SIZE}-lr_{LEARNING_RATE:.0e}-n_epochs_{N_EPOCS}"
)
VAE_MODEL_OUTPUT_FILE = "fold_{fold}-mice_vae_model.pt"
DATA_SCALER_OUTPUT_FILE = "fold_{fold}-data_scaler.joblib"

# %% tags=["inout_defs"]
# INPUT_FILES and OUTPUT_FILES must be provided if running with pytask
INPUT_FILES = {
    "mice_gene_expr_data_input": conf.data.MICE_PROCESSED_DIR
    / "mice_data-full-log.ftr",
    "mice_gene_expr_norm_adult_heart_input": conf.data.MICE_PROCESSED_DIR
    / "mice_gene_expr_norm_adult_heart_rpkm_std.ftr",
    "mice_gene_expr_norm_brain_input": conf.data.MICE_PROCESSED_DIR
    / "mice_gene_expr_norm_brain_rpkm_std.ftr",
    "mice_gene_expr_mesenteric_lymph_nodes_input": conf.data.MICE_PROCESSED_DIR
    / "mice_gene_expr_mesenteric_lymph_nodes_rpkm_std.ftr",
    "mice_gene_expr_facial_mesenchyme_input": conf.data.MICE_PROCESSED_DIR
    / "mice_gene_expr_facial_mesenchyme_rpkm_std.ftr",
    "mice_plier_model_Z": conf.data.MICE_PROCESSED_DIR
    / "mice_model_Z_std.csv",
    "mice_plier_model_B": conf.data.MICE_PROCESSED_DIR
    / "mice_model_B_std.csv"}

OUTPUT_FILES = {
    f"mice_vae_model_fold{i}": OUTPUT_FOLDER / VAE_MODEL_OUTPUT_FILE.format(fold=i)
    for i in range(min(K_FOLDS, N_FOLDS_TO_RUN or K_FOLDS))
}
OUTPUT_FILES.update(
    {
        f"data_scaler_fold{i}": OUTPUT_FOLDER / DATA_SCALER_OUTPUT_FILE.format(fold=i)
        for i in range(min(K_FOLDS, N_FOLDS_TO_RUN or K_FOLDS))
    }
)
OUTPUT_FILES.update({"performance_stats": OUTPUT_FOLDER / "performance_stats.pkl"})
OUTPUT_FILES.update({"mice_gene_expr_norm_adult_heart_vae_B_matrix": OUTPUT_FOLDER / "mice_gene_expr_norm_adult_heart_vae_B_matrix.ftr"})
OUTPUT_FILES.update({"mice_gene_expr_norm_adult_heart_plier_B_matrix": OUTPUT_FOLDER / "mice_gene_expr_norm_adult_heart_plier_B_matrix.ftr"})

# %% tags=[]
pprint.pprint(INPUT_FILES)

# %% tags=[]
pprint.pprint(OUTPUT_FILES)

# %% tags=[]
for _k, v in OUTPUT_FILES.items():
    v.parent.mkdir(parents=True, exist_ok=True)

# %% [markdown] tags=[]
# # Load data

# %% [markdown] tags=[]
# ## FeatherDataset

# %% tags=[]
orig_dataset = pd.read_feather(INPUT_FILES["mice_gene_expr_data_input"]).set_index("index")
new_dataset_adult_heart = pd.read_feather(INPUT_FILES["mice_gene_expr_norm_adult_heart_input"])
new_dataset_brain = pd.read_feather(INPUT_FILES["mice_gene_expr_norm_brain_input"])
new_dataset_mesenteric_lymph_nodes = pd.read_feather(INPUT_FILES["mice_gene_expr_mesenteric_lymph_nodes_input"])
new_dataset_facial_mesenchyme = pd.read_feather(INPUT_FILES["mice_gene_expr_facial_mesenchyme_input"])

# %%
orig_dataset.shape


# %% [markdown]
# # Make new dataset gene same as original set #

# %%
def new_dataset_with_gene_match_from_old_data(new_dataset, orig_dataset):
    new_set = set(new_dataset.index.tolist())
    orig_set = set(orig_dataset.index.tolist())
    common_set = new_set.intersection(orig_set)
    # print("common gene set :", len(common_set))
    gene_not_in_new_set = orig_set - common_set
    gene_not_in_orig_set = new_set - common_set
    
    for gene in gene_not_in_new_set:
        new_dataset.loc[gene] = [0]* len(new_dataset.columns)
    new_dataset = new_dataset.drop(gene_not_in_orig_set)
    assert len(new_dataset.index.tolist()) == len(orig_dataset.index.tolist())

    assert orig_dataset.isnull().any().any() == False
    assert new_dataset.isnull().any().any() == False 
    assert orig_dataset.isna().any().any() == False
    assert new_dataset.isna().any().any() == False
    return new_dataset 


# %% [markdown]
# # Join new labeled data based on Tissue type # 

# %%
new_dataset_label_sample_dict = {}
new_dataset_adult_heart_matched_genes = new_dataset_with_gene_match_from_old_data(new_dataset_adult_heart, orig_dataset)
new_dataset_adult_heart_matched_genes = new_dataset_adult_heart_matched_genes.T
print(new_dataset_adult_heart_matched_genes.index.tolist())
heart_labels = ["heart"] * len(new_dataset_adult_heart_matched_genes.index.tolist())
new_dataset_label_sample_dict["heart"] = new_dataset_adult_heart_matched_genes.index.tolist()

new_dataset_brain_genes = new_dataset_with_gene_match_from_old_data(new_dataset_brain, orig_dataset)
new_dataset_brain_genes = new_dataset_brain_genes.T
brain_labels = ["brain"] * len(new_dataset_brain_genes.index.tolist())
print(new_dataset_brain_genes.index.tolist())
new_dataset_label_sample_dict["brain"] = new_dataset_brain_genes.index.tolist()

new_dataset_mesenteric_lymph_nodes_genes = new_dataset_with_gene_match_from_old_data(new_dataset_mesenteric_lymph_nodes, orig_dataset)
new_dataset_mesenteric_lymph_nodes_genes = new_dataset_mesenteric_lymph_nodes_genes.T
mesenteric_labels = ["mesenteric_lymph_nodes"] * len(new_dataset_mesenteric_lymph_nodes_genes.index.tolist())
print(new_dataset_mesenteric_lymph_nodes_genes.index.tolist())
new_dataset_label_sample_dict["mesenteric_lymph_nodes"] = new_dataset_mesenteric_lymph_nodes_genes.index.tolist()

new_dataset_facial_mesenchyme_genes = new_dataset_with_gene_match_from_old_data(new_dataset_facial_mesenchyme, orig_dataset)
new_dataset_facial_mesenchyme_genes = new_dataset_facial_mesenchyme_genes.T
facial_mesenchyme_labels = ["facial_mesenchyme"] * len(new_dataset_facial_mesenchyme_genes.index.tolist())
print(new_dataset_facial_mesenchyme_genes.index.tolist())
new_dataset_label_sample_dict["facial_mesenchyme"] = new_dataset_facial_mesenchyme_genes.index.tolist()

new_dataset = pd.concat([new_dataset_adult_heart_matched_genes, new_dataset_brain_genes, 
                         new_dataset_mesenteric_lymph_nodes_genes, new_dataset_facial_mesenchyme_genes])
new_dataset_labels = heart_labels + brain_labels + mesenteric_labels + facial_mesenchyme_labels
# new_dataset_label_sample_dict = dict(zip(new_dataset.index.tolist(), new_dataset_labels))
labels_set = set(new_dataset_labels)
display(new_dataset_label_sample_dict)


# %%
# dataset[:][0][torch.isnan(dataset[:][0])] = 0
# dataset.data.shape
# nan_mask = torch.isnan(dataset[:][0])
# has_nan = torch.any(nan_mask)
# print(nan_mask)
# print(has_nan)
# nan_indices = torch.nonzero(nan_mask)
# print(nan_indices)

# %% tags=[]
# display(dataset.length)
# assert dataset.length == 190110

# %% tags=[]
# dataset[0]

# %% [markdown] tags=[]
# ## Get dataset dimensions

# %% tags=[]
data_n_samples = orig_dataset.shape[1]
display(data_n_samples)
data_n_genes = orig_dataset.shape[0]
display(data_n_genes)

# %% [markdown] tags=[]
# # VAE


# %%
random_state = np.random.RandomState(RANDOM_SEED)


# %% tags=[]
def create_model():
    return VariationalAutoencoder(
        k=K,
        n_features=data_n_genes,
        n_samples=data_n_samples,
        n_pathways=0,
        kl_l=KL_L,
        pred_l=PRED_L,
    )


# %% tags=[]
display(create_model())

# %% [markdown] tags=[]
# # Load a model

# %%
_fold = 0

# %%
train_data_scaler = joblib.load(OUTPUT_FILES[f"data_scaler_fold{_fold}"])
display(train_data_scaler)

# %%
model = create_model()
model.load_state_dict(torch.load(OUTPUT_FILES[f"mice_vae_model_fold{_fold}"]))
model.to("cpu")
model.eval()

# %% [markdown] tags=[]
# # VAE B matrix 

# %% tags=[]
new_dataset_torch = torch.as_tensor(
    new_dataset.to_numpy(), dtype=torch.float32
).detach()
B_vae = model.encoder(new_dataset_torch).detach().numpy()

# %% tags=[]
B_vae.shape

# %% tags=[]
B_vae = pd.DataFrame(
    data=B_vae, index=new_dataset.index.tolist(), columns=[f"LV{i}" for i in range(K)]
)

# %% tags=[]
B_vae.head()

# %% tags=[]
B_vae.to_pickle(OUTPUT_FILES["mice_gene_expr_norm_adult_heart_vae_B_matrix"])

# %% [markdown]
# # PLIER B matrix #

# %%
lambda_2 = 25.2449085603066
df_Z = pd.read_csv(INPUT_FILES['mice_plier_model_Z'])
df_Z = df_Z.set_index("Unnamed: 0").rename_axis(index=None)
df_Z.head()

# %%
display(df_Z.shape)
gene_set_plier = set(df_Z.index.tolist())
gene_set_orig_dataset  = set(new_dataset.T.index.tolist())
common_set = gene_set_plier.intersection(gene_set_orig_dataset)
display(len(common_set))

# %%
df_Z_common_genes = df_Z.loc[list(common_set)] 
display(df_Z_common_genes.shape)

# %%
Plier_Z_tensor = torch.tensor(df_Z_common_genes.values, dtype=torch.float32)
Plier_Z_tensor_gpu = Plier_Z_tensor.to(DEVICE)
Plier_gene_gene_coexp_gpu = torch.matmul(Plier_Z_tensor_gpu.T, Plier_Z_tensor_gpu)
Identity_matrix_gpu = torch.eye(Plier_gene_gene_coexp_gpu.shape[0]).to(DEVICE)
temp_1_gpu = torch.linalg.inv(Plier_gene_gene_coexp_gpu + lambda_2 * Identity_matrix_gpu)
temp_2 = torch.matmul(temp_1_gpu, Plier_Z_tensor_gpu.T).cpu()
B_plier = np.matmul(temp_2, new_dataset.T.loc[list(common_set)].to_numpy())

# %%
temp_2.shape

# %%
B_plier.shape

# %%
B_plier = pd.DataFrame(
    data=B_plier.T, index=new_dataset.index.tolist(), columns=[f"LV{i}" for i in range(B_plier.shape[0])]
)

# %%
B_plier.to_pickle(OUTPUT_FILES["mice_gene_expr_norm_adult_heart_plier_B_matrix"])

# %% [markdown]
# # LV's Correlation

# %%
B_plier_df = pd.read_pickle(OUTPUT_FILES["mice_gene_expr_norm_adult_heart_plier_B_matrix"])
B_plier_df.head()

# %%
correlation_matrix_pearson_plier = B_plier_df.corr(method='pearson')
correlation_matrix_spearman_plier = B_plier_df.corr(method='spearman')
correlation_matrix_pearson_plier_df = pd.DataFrame(data = correlation_matrix_pearson_plier, index=B_plier_df.columns.tolist(), columns=B_plier_df.columns.tolist(),)
correlation_matrix_spearman_plier_df = pd.DataFrame(data = correlation_matrix_spearman_plier, index=B_plier_df.columns.tolist(), columns=B_plier_df.columns.tolist(),)

# print(correlation_matrix_plier)

# %%
B_vae_df = pd.read_pickle(OUTPUT_FILES["mice_gene_expr_norm_adult_heart_vae_B_matrix"])
B_vae_df.head()

# %%
correlation_matrix_pearson_vae = B_plier_df.corr(method='pearson')
correlation_matrix_spearman_vae = B_plier_df.corr(method='spearman')
correlation_matrix_pearson_vae_df = pd.DataFrame(data = correlation_matrix_pearson_vae, index=B_vae_df.columns.tolist(), columns=B_vae_df.columns.tolist(),)
correlation_matrix_spearman_vae_df = pd.DataFrame(data = correlation_matrix_spearman_vae, index=B_vae_df.columns.tolist(), columns=B_vae_df.columns.tolist(),)

# %% [markdown]
# # Correlation between LV's from VAE and PLIER

# %%
correlation_matrix_pearson_plier_vae = np.zeros((B_plier_df.shape[1], B_vae_df.shape[1]))
B_vae_df_lv_list = B_vae_df.columns.tolist()
B_plier_df_lv_list = B_plier_df.columns.tolist()
for idx1, lv_vae in enumerate(B_vae_df_lv_list):
    for idx2, lv_plier in enumerate(B_plier_df_lv_list):
        correlation_matrix_pearson_plier_vae[idx2, idx1] = B_plier_df[lv_plier].corr(B_vae_df[lv_vae], method = "pearson")

correlation_matrix_spearman_plier_vae = np.zeros((B_plier_df.shape[1], B_vae_df.shape[1]))
B_vae_df_lv_list = B_vae_df.columns.tolist()
B_plier_df_lv_list = B_plier_df.columns.tolist()
for idx1, lv_vae in enumerate(B_vae_df_lv_list):
    for idx2, lv_plier in enumerate(B_plier_df_lv_list):
        correlation_matrix_spearman_plier_vae[idx2, idx1] = B_plier_df[lv_plier].corr(B_vae_df[lv_vae], method = "spearman")



correlation_matrix_pearson_plier_vae_df = pd.DataFrame(data = correlation_matrix_pearson_plier_vae, index=B_plier_df_lv_list, columns=B_plier_df_lv_list,)
correlation_matrix_spearman_plier_vae_df = pd.DataFrame(data = correlation_matrix_spearman_plier_vae, index=B_plier_df_lv_list, columns=B_plier_df_lv_list,)

# %%
correlation_matrix_pearson_plier_vae_df.head()

# %%
correlation_matrix_spearman_plier_vae_df.head()

# %%
plt.figure(figsize=(12, 9))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix_pearson_vae_df, cmap="rocket", linewidths=.5)

plt.xlabel("VAE LV's") # Customize the X-axis label
plt.ylabel("VAE LV's") # Customize the Y-axis label

plt.title("Correlation (Pearson) Plot between VAE LV's") # Customize the title
plt.show()

# %%
plt.figure(figsize=(12, 9))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix_spearman_vae_df, cmap="rocket",linewidths=.5)

plt.xlabel("VAE LV's") # Customize the X-axis label
plt.ylabel("VAE LV's") # Customize the Y-axis label

plt.title("Correlation (Spearman) Plot between VAE LV's") # Customize the title
plt.show()

# %%
plt.figure(figsize=(12, 9))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix_pearson_plier_df, cmap="rocket",linewidths=.5)

plt.xlabel("Plier LV's") # Customize the X-axis label
plt.ylabel("Plier LV's") # Customize the Y-axis label

plt.title("Correlation (Pearson) Plot between Plier LV's") # Customize the title
plt.show()

# %%
plt.figure(figsize=(12, 9))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix_spearman_plier_df, cmap="rocket",linewidths=.5)

plt.xlabel("Plier LV's") # Customize the X-axis label
plt.ylabel("Plier LV's") # Customize the Y-axis label

plt.title("Correlation (Spearman) Plot between Plier LV's") # Customize the title
plt.show()

# %%
correlation_matrix_pearson_plier_vae_max = correlation_matrix_pearson_plier_vae_df.max(axis=0).sort_values()
key_list_sorted = correlation_matrix_pearson_plier_vae_max.index.tolist()
correlation_matrix_pearson_plier_vae_sort_df = correlation_matrix_pearson_plier_vae_df.reindex(columns=key_list_sorted)
for col in correlation_matrix_pearson_plier_vae_sort_df:
    correlation_matrix_pearson_plier_vae_sort_df[col] = [correlation_matrix_pearson_plier_vae_max[col]]*correlation_matrix_pearson_plier_vae_sort_df.shape[1] 


plt.figure(figsize=(12, 9))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix_pearson_plier_vae_sort_df, cmap="rocket",linewidths=0.5)
plt.xlabel("VAE LV's") # Customize the X-axis label
plt.ylabel("Plier LV's") # Customize the Y-axis label

plt.title("Correlation (Pearson) Plot between Plier and VAE LV's") # Customize the title
plt.show()

# %%
correlation_matrix_spearman_plier_vae_max = correlation_matrix_spearman_plier_vae_df.max(axis=0).sort_values()
key_list_sorted = correlation_matrix_spearman_plier_vae_max.index.tolist()
correlation_matrix_spearman_plier_vae_sort_df = correlation_matrix_spearman_plier_vae_df.reindex(columns=key_list_sorted)
for col in correlation_matrix_spearman_plier_vae_sort_df:
    correlation_matrix_spearman_plier_vae_sort_df[col] = [correlation_matrix_spearman_plier_vae_max[col]]*correlation_matrix_spearman_plier_vae_sort_df.shape[1] 


plt.figure(figsize=(12, 9))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix_spearman_plier_vae_sort_df, cmap="rocket",linewidths=0.5)
plt.xlabel("VAE LV's") # Customize the X-axis label
plt.ylabel("Plier LV's") # Customize the Y-axis label

plt.title("Correlation (Spearman) Plot between Plier and VAE LV's") # Customize the title
plt.show()


# %% [markdown]
# # Two Group Permutation Test

# %%
def plot_null_distribution(null_dist, observed, p_value, model_type = "VAE"):
    plt.figure(figsize=(7, 4))
    sns.histplot(null_dist, bins=100, kde=False, color='green')
    plt.axvline(observed, color='red', linestyle='--', label='Observed')
    plt.title(f'Null distribution for significant lv count ({model_type})')
    plt.xlabel('null dist of significant lvs count')
    plt.ylabel('Frequency')
    plt.legend()
    plt.text(90, 80, f'p-value :{p_value}', fontsize=9)
    plt.show()
    
def plot_null_distribution_per_lv(null_dist, observed, lv_list, lv_idx, model_type = "VAE"):
    plt.figure(figsize=(7, 4))
    sns.histplot(null_dist[:, lv_idx], bins=30, kde=True, color='lightblue')
    plt.axvline(observed[lv_idx], color='red', linestyle='--', label='Observed')
    plt.title(f'Null distribution for  {lv_list[lv_idx]} ({model_type})')
    plt.xlabel('Mean difference (perm)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
def incremental_shuffles_samples(label_sample_dict):
    label_sample_dict_temp = {} 
    sample_list = [] 
    for label, samples in label_sample_dict.items():
        sample_list.extend(samples)

    for label, samples in label_sample_dict.items():
        temp = random.sample(sample_list, len(samples))
        label_sample_dict_temp[label] = temp
        for sample in temp:
            sample_list.remove(sample)

    return label_sample_dict_temp

def single_group_test(B_df, group, label_sample_dict, seed=42):
    """
    Parameters:
        B_df: (n_samples, n_lvs) lv's expression matrix
        label_sample_dict,: (n_samples,) array of group labels
        seed: random seed
        
    Returns:
        DataFrame with gene index, observed F, p-value, adjusted p-value
    """
    np.random.seed(seed)
    n_samples, n_lvs = B_df.shape
    lv_list = B_df.columns.tolist()
    
    # Compute observed F-statistics
    group_0 = B_df.loc[label_sample_dict[group]]
    # observed_mean = (group_0.mean(axis=0)).to_numpy()
    group_rest = B_df.loc[~B_df.index.isin(label_sample_dict[group])]
    p_vals = np.zeros(n_lvs)
    for idx, lv in enumerate(lv_list):
        _, p_vals[idx] = ranksums(group_0[lv], group_rest[lv])
    
    # FDR correction (Benjamini-Hochberg)
    _, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')

    return pd.DataFrame({
        'lv': lv_list,
        'p_value': p_vals,
        'p_adj': p_adj,
    }).sort_values('p_adj')

def permutation_test_multigroup_lv_count(B_df, label_sample_dict, n_permutations=1000, seed=42):
    np.random.seed(seed)
    sign_lv_set = set()
    for group in label_sample_dict:
        test = single_group_test(B_df, group, label_sample_dict, seed=42)
        sign_lv_set = sign_lv_set.union(test[test['p_adj'] < 0.05]['lv'].tolist())
    
    observed_sign_lvs_count = len(sign_lv_set)

    null_lvs_count_distribution = np.zeros((n_permutations,))
    for p in range(n_permutations):
        label_sample_dict_perm = incremental_shuffles_samples(label_sample_dict)
        sign_lv_set_perm = set()
        for group in label_sample_dict_perm:
            test_perm = single_group_test(B_df, group, label_sample_dict_perm, seed=42)
            sign_lv_set_perm = sign_lv_set_perm.union(test_perm[test_perm['p_adj'] < 0.05]['lv'].tolist())

        null_lvs_count_distribution[p] = len(sign_lv_set_perm)
        
    p_vals = np.mean(null_lvs_count_distribution >=  observed_sign_lvs_count, axis=0)
    
    # FDR correction (Benjamini-Hochberg)
    _, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')

    return pd.DataFrame({
        'lvs_count': observed_sign_lvs_count,
        'p_value': p_vals,
        'p_adj': p_adj,
    }).sort_values('p_adj'), null_lvs_count_distribution, sign_lv_set 

    
        
def permutation_test_twogroup(B_df, two_groups, label_sample_dict, n_permutations=1000, seed=42):

    """
    Permutation-based one-way ANOVA for multiple groups (one test per gene).
    
    Parameters:
        B_df: (n_samples, n_lvs) lv's expression matrix
        label_sample_dict,: (n_samples,) array of group labels
        n_permutations: number of permutations
        seed: random seed
        
    Returns:
        DataFrame with gene index, observed F, p-value, adjusted p-value
    """
    np.random.seed(seed)
    n_samples, n_lvs = B_df.shape
    lv_list = B_df.columns.tolist()
    
    # Compute observed F-statistics
    group_0 = B_df.loc[label_sample_dict[two_groups[0]]]
    group_1 = B_df.loc[label_sample_dict[two_groups[1]]]
    observed_mean_diff = (group_0.mean(axis=0) - group_1.mean(axis=0)).to_numpy()
    

    # Permute group labels and compute mean diff 
    null_mean_diff_distribution = np.zeros((n_permutations, n_lvs))
    for p in range(n_permutations):
        label_sample_dict_perm = incremental_shuffles_samples(label_sample_dict)
        group_0 = B_df.loc[label_sample_dict_perm[two_groups[0]]]
        group_1 = B_df.loc[label_sample_dict_perm[two_groups[1]]]
        null_mean_diff_distribution[p, :] = (group_0.mean(axis=0) - group_1.mean(axis=0)).to_numpy()
        
    # Compute Two-sided p-values
    p_vals = np.mean(np.abs(null_mean_diff_distribution) >= np.abs(observed_mean_diff), axis=0)
    
    # FDR correction (Benjamini-Hochberg)
    _, p_adj, _, _ = multipletests(p_vals, alpha=0.05, method='fdr_bh')

    return pd.DataFrame({
        'lv': lv_list,
        'mean_diff': observed_mean_diff,
        'p_value': p_vals,
        'p_adj': p_adj,
    }).sort_values('p_adj'), null_mean_diff_distribution
    
    
    
def permutation_test_multigroup(B_df, label_sample_dict, n_permutations=1000, seed=42):
    """
    Permutation-based one-way ANOVA for multiple groups (one test per gene).
    
    Parameters:
        B_df: (n_samples, n_lvs) lv's expression matrix
        label_sample_dict,: (n_samples,) array of group labels
        n_permutations: number of permutations
        seed: random seed
        
    Returns:
        DataFrame with gene index, observed F, p-value, adjusted p-value
    """
    np.random.seed(seed)
    n_samples, n_lvs = B_df.shape
    groups = list(label_sample_dict.keys())
    lv_list = B_df.columns.tolist()
    
    # Compute observed F-statistics
    observed_f = np.zeros(n_lvs)
    for idx, lv in enumerate(lv_list):
        data_by_group = [B_df.loc[label_sample_dict[g]][lv].to_numpy() for g in groups]
        observed_f[idx], _ = f_oneway(*data_by_group)
    
    # Permute group labels and compute F-statistics
    perm_f = np.zeros((n_permutations, n_lvs))
    for p in range(n_permutations):
        # y_perm = np.random.permutation(y)
        label_sample_dict_perm = incremental_shuffles_samples(label_sample_dict)
        for idx, lv in enumerate(lv_list):
            data_by_group = [B_df.loc[label_sample_dict_perm[g]][lv].to_numpy() for g in groups]
            f_stat, _ = f_oneway(*data_by_group)
            perm_f[p, idx] = f_stat
    
    # Compute empirical p-values
    print(perm_f.shape, observed_f.shape)
    p_vals = np.mean(perm_f >= observed_f, axis=0)
    
    # FDR correction (Benjamini-Hochberg)
    _, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')
    
    return pd.DataFrame({
        'lv_index': lv_list,
        'F_stat': observed_f,
        'p_value': p_vals,
        'p_adj': p_adj
    }).sort_values('p_adj')


# %%
test_vae, null_mean_diff_distribution_vae = permutation_test_twogroup(B_vae, ["heart", "brain"], new_dataset_label_sample_dict, n_permutations=100, seed=42)

# %%
print(test_vae[test_vae['p_adj'] < 0.05])

# %%
test_plier, null_mean_diff_distribution_plier = permutation_test_twogroup(B_plier, ["heart", "brain"], new_dataset_label_sample_dict, n_permutations=100, seed=42)

# %%
print(test_plier[test_plier['p_adj'] < 0.05])

# %%
plot_null_distribution_per_lv(null_mean_diff_distribution_vae, test_vae["mean_diff"], test_vae["lv"], 3, model_type="VAE")

# %%
plot_null_distribution_per_lv(null_mean_diff_distribution_plier, test_plier["mean_diff"], test_plier["lv"], 0, model_type="Plier")

# %%
correlation_matrix_pearson_plier_vae_subset_df = correlation_matrix_pearson_plier_vae_df[test_vae[test_vae['p_adj'] < 0.05]["lv"].tolist()]
correlation_matrix_pearson_plier_vae_subset_df = correlation_matrix_pearson_plier_vae_subset_df.loc[test_plier[test_plier['p_adj'] < 0.05]["lv"].tolist()]
correlation_matrix_pearson_plier_vae_max = correlation_matrix_pearson_plier_vae_subset_df.max(axis=0).sort_values()
key_list_sorted = correlation_matrix_pearson_plier_vae_max.index.tolist()
correlation_matrix_pearson_plier_vae_sort_df = correlation_matrix_pearson_plier_vae_subset_df.reindex(columns=key_list_sorted)
# print(correlation_matrix_pearson_plier_vae_sort_df.columns)

for col in correlation_matrix_pearson_plier_vae_sort_df:
    correlation_matrix_pearson_plier_vae_sort_df[col] = [correlation_matrix_pearson_plier_vae_max[col]]*correlation_matrix_pearson_plier_vae_sort_df.shape[0] 


plt.figure(figsize=(12, 9))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix_pearson_plier_vae_sort_df, cmap="rocket",linewidths=0.5)
plt.xlabel("VAE LV's") # Customize the X-axis label
plt.ylabel("Plier LV's") # Customize the Y-axis label

plt.title("Correlation (Pearson) Plot between Plier and VAE LV's") # Customize the title
plt.show()

# %%
test_vae_ranksum = single_group_test(B_vae, "brain", new_dataset_label_sample_dict)

# %%
test_vae_ranksum[test_vae_ranksum['p_adj'] < 0.5]

# %%
test_plier_ranksum = single_group_test(B_plier, "heart", new_dataset_label_sample_dict)

# %%
test_plier_ranksum[test_plier_ranksum['p_adj'] < 0.5]

# %%
test_vae, null_lvs_count_distribution_vae, sign_lv_set_vae = permutation_test_multigroup_lv_count(B_vae, new_dataset_label_sample_dict, n_permutations=100, seed=42)

# %%
test_vae[test_vae['p_adj'] < 0.05]

# %%
plot_null_distribution(null_lvs_count_distribution_vae, test_vae["lvs_count"].tolist()[0], test_vae['p_value'][0], model_type="VAE")

# %%
test_plier, null_lvs_count_distribution_plier, sign_lv_set_plier = permutation_test_multigroup_lv_count(B_plier, new_dataset_label_sample_dict, n_permutations=100, seed=42)

# %%
plot_null_distribution(null_lvs_count_distribution_plier, test_plier["lvs_count"].tolist()[0], test_plier['p_value'][0], model_type="Plier")

# %%
correlation_matrix_pearson_plier_vae_subset_df = correlation_matrix_pearson_plier_vae_df[list(sign_lv_set_vae)]
correlation_matrix_pearson_plier_vae_subset_df = correlation_matrix_pearson_plier_vae_subset_df.loc[list(sign_lv_set_plier)]
correlation_matrix_pearson_plier_vae_max = correlation_matrix_pearson_plier_vae_subset_df.max(axis=0).sort_values()
key_list_sorted = correlation_matrix_pearson_plier_vae_max.index.tolist()
correlation_matrix_pearson_plier_vae_sort_df = correlation_matrix_pearson_plier_vae_subset_df.reindex(columns=key_list_sorted)
# for col in correlation_matrix_pearson_plier_vae_sort_df:
#     correlation_matrix_pearson_plier_vae_sort_df[col] = [correlation_matrix_pearson_plier_vae_max[col]]*correlation_matrix_pearson_plier_vae_sort_df.shape[0] 


plt.figure(figsize=(12, 9))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix_pearson_plier_vae_sort_df, vmin=0, vmax=1, cmap="rocket",linewidths=0.5)
plt.xlabel("VAE LV's") # Customize the X-axis label
plt.ylabel("Plier LV's") # Customize the Y-axis label

plt.title("Correlation (Pearson) Plot between Plier and VAE LV's") # Customize the title
plt.show()

# %%
correlation_matrix_pearson_plier_vae_subset_df = correlation_matrix_pearson_plier_vae_df[list(sign_lv_set_vae)]
correlation_matrix_pearson_plier_vae_subset_df = correlation_matrix_pearson_plier_vae_subset_df.loc[list(sign_lv_set_plier)]
correlation_matrix_pearson_plier_vae_max = correlation_matrix_pearson_plier_vae_subset_df.max(axis=0).sort_values()
key_list_sorted = correlation_matrix_pearson_plier_vae_max.index.tolist()
correlation_matrix_pearson_plier_vae_sort_df = correlation_matrix_pearson_plier_vae_subset_df.reindex(columns=key_list_sorted)
correlation_matrix_pearson_plier_vae_max_sort_df = correlation_matrix_pearson_plier_vae_sort_df.copy()
for col in correlation_matrix_pearson_plier_vae_sort_df:
    correlation_matrix_pearson_plier_vae_max_sort_df[col] = [correlation_matrix_pearson_plier_vae_max[col]]*correlation_matrix_pearson_plier_vae_max_sort_df.shape[0] 


plt.figure(figsize=(12, 9))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix_pearson_plier_vae_max_sort_df, vmin=0, vmax=1, cmap="rocket",linewidths=0.5)
plt.xlabel("VAE LV's") # Customize the X-axis label
plt.ylabel("Plier LV's") # Customize the Y-axis label

plt.title("Max Correlation (Pearson) Plot between Plier and VAE LV's") # Customize the title
plt.show()

# %%
g = sns.clustermap(correlation_matrix_pearson_plier_vae_sort_df, vmin=0, vmax=1, cmap='coolwarm', figsize=(20, 18))
plt.title('Clustered Heatmap of Correlation (Pearson) between Plier and VAE LVs')
g.ax_heatmap.set_xlabel("VAE LV's")
g.ax_heatmap.set_ylabel("Plier LV's")
plt.show()

# %%
