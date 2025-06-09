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
# Runs VAE with standard parameters. It saves the model and generates plots.

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
from scipy.stats import pearsonr, spearmanr
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
    / "mice_gene_expr_norm_adult_heart_full_log.ftr",
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
dataset = FeatherDataset(INPUT_FILES["mice_gene_expr_data_input"], read_full_data=True)

# %%
dataset[:][0][torch.isnan(dataset[:][0])] = 0
dataset.data.shape
nan_mask = torch.isnan(dataset[:][0])
has_nan = torch.any(nan_mask)
print(nan_mask)
print(has_nan)
nan_indices = torch.nonzero(nan_mask)
print(nan_indices)

# %% tags=[]
display(dataset.length)
assert dataset.length == 190110

# %% tags=[]
dataset[0]

# %% [markdown] tags=[]
# ## Get dataset dimensions

# %% tags=[]
data_n_samples = dataset.length
display(data_n_samples)

data_n_genes = dataset[0][0].shape[0]
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
# # Save

# %% tags=[]
data = pd.read_feather(INPUT_FILES["mice_gene_expr_data_input"]).set_index("index")

# %% tags=[]
orig_data = torch.as_tensor(
    train_data_scaler.transform(data.T.to_numpy()), dtype=torch.float32
).detach()

# %% tags=[]
orig_data.shape

# %% [markdown] tags=[]
# ## Latent space

# %% tags=[]
z = model.encoder(orig_data).detach().numpy()

# %% tags=[]
z.shape

# %% tags=[]
z = pd.DataFrame(
    data=z, index=data.columns.tolist(), columns=[f"lv{i}" for i in range(K)]
)

# %% tags=[]
z.head()

# %% tags=[]
# z.to_pickle(OUTPUT_FILES["pvae_latent_space"])

# %% [markdown] tags=[]
# ## Reconstructed data

# %% tags=[]
data_recon = model.forward(orig_data).detach()

# %% tags=[]
data_recon.shape

# %% tags=[]
data_recon

# %% tags=[]
data_recon_df = pd.DataFrame(
    data=data_recon.T.detach().numpy(),
    index=data.index.tolist(),
    columns=data.columns.tolist(),
)

# %% tags=[]
data_recon_df.shape

# %% tags=[]
data_recon_df.head()

# %% tags=[]
# data_recon_df.to_pickle(OUTPUT_FILES["pvae_data_recon"])

# %% [markdown] tags=[]
# # MSE

# %% tags=[]
_l = nn.MSELoss()
display(_l(data_recon, orig_data))

# %% tags=[]
_l = nn.MSELoss(reduction="sum")
display(_l(data_recon, orig_data))

# %% [markdown] tags=[]
# # Plot

# %% tags=[]
scaler = StandardScaler().set_output(transform="pandas")
z_scaled = scaler.fit_transform(z)

# %% tags=[]
z_scaled.head()

# %% [markdown] tags=[]
# ## Latent space

# %% [markdown] tags=[]
# ### PCA

# %% tags=[]
pca = PCA(n_components=4, random_state=0).set_output(transform="pandas")
z_pca = pca.fit_transform(z_scaled)

# %% tags=[]
z_pca.shape

# %% tags=[]
z_pca.head()

# %% tags=[]
# df_plot = z_pca.assign(tissue=sample_tissues["simplified_tissue"])

# %% tags=[]
# df_plot

# %% tags=[]
# sns.set_theme(style="ticks", palette="tab10")
sns.pairplot(data=z_pca)

# %% [markdown]
# # PLIER B Matrix (Mice)

# %% scrolled=true
df_Z = pd.read_csv(INPUT_FILES['mice_plier_model_Z'])
df_Z = df_Z.set_index("Unnamed: 0").rename_axis(index=None)
df_Z.head()

df_B = pd.read_csv(INPUT_FILES['mice_plier_model_B'])
df_B = df_B.set_index("Unnamed: 0").rename_axis(index=None)
df_B = df_B.drop('SRR1557112', axis=1)
df_B.head()


# %%
scaler = StandardScaler().set_output(transform="pandas")
z_B_scaled = scaler.fit_transform(df_B.T)

# %%
pca = PCA(n_components=4, random_state=0).set_output(transform="pandas")
z_B_pca = pca.fit_transform(z_B_scaled)

# %%
z_B_pca.shape

# %%
z_B_pca.head()

# %%
sns.pairplot(data=z_B_pca)

# %%
df_B.shape
df_B.head()
# df_B.columns

# %%
z_scaled.shape
z_scaled_T = z_scaled.T 
z_scaled_T.head()

# %%
coexp_save_folder = '/home/prashant/Documents/milton_lab/pvae/base/output/mice_gene_gene_coexp/'

# %% [markdown]
# # Gene Gene coexpression matrix (PLIER) #

# %%
Plier_Z_tensor = torch.tensor(df_Z.values, dtype=torch.float32)
Plier_Z_norm_tensor = F.normalize(Plier_Z_tensor, dim=1)
Plier_Z_norm_tensor_gpu = Plier_Z_norm_tensor.to(DEVICE)
Plier_gene_gene_coexp_gpu = torch.matmul(Plier_Z_norm_tensor_gpu, Plier_Z_norm_tensor_gpu.T)
Plier_gene_gene_coexp_cpu = Plier_gene_gene_coexp_gpu.cpu()
Plier_gene_gene_coexp_df = pd.DataFrame(Plier_gene_gene_coexp_cpu.numpy())
Plier_gene_gene_coexp_df.index = df_Z.index
Plier_gene_gene_coexp_df.columns = df_Z.index
Plier_gene_gene_coexp_df.head()
display(Plier_gene_gene_coexp_df.shape)
Plier_gene_gene_coexp_df.to_csv(coexp_save_folder + "Plier_gene_gene_coexp.csv")

# %% [markdown]
# # Gene Gene coexpression matrix (VAE) #

# %%
data = pd.read_feather(INPUT_FILES["mice_gene_expr_data_input"]).set_index("index")
orig_data = torch.as_tensor(
    train_data_scaler.transform(data.T.to_numpy()), dtype=torch.float32
).detach()
orig_data_df = pd.DataFrame(
    data=orig_data.T, index=data.index.tolist(), columns=data.columns.tolist())
orig_data_df.head()
orig_data_tensor = torch.tensor(orig_data_df.values, dtype=torch.float32)
orig_data_tensor_gpu = orig_data_tensor.to(DEVICE)

z_gpu = torch.tensor(z.values, dtype=torch.float32).to(DEVICE)

orig_data_z_gpu = torch.matmul(orig_data_tensor_gpu, z_gpu)
z_z_T_gpu = torch.matmul(z_gpu.T, z_gpu)
z_z_T_inv_gpu = torch.linalg.inv(z_z_T_gpu)
Vae_Z_gpu = torch.matmul(orig_data_z_gpu, z_z_T_inv_gpu)
Vae_Z_cpu = Vae_Z_gpu.cpu().numpy()
Vae_Z_df = pd.DataFrame(
    data=Vae_Z_cpu, index=data.index.tolist(), columns=[f"LV{i}" for i in range(K)])


Vae_Z_tensor = torch.tensor(Vae_Z_df.values, dtype=torch.float32)
Vae_Z_norm_tensor = F.normalize(Vae_Z_tensor, dim=1)
Vae_Z_norm_tensor_gpu = Vae_Z_norm_tensor.to(DEVICE)
Vae_gene_gene_coexp_gpu = torch.matmul(Vae_Z_norm_tensor_gpu, Vae_Z_norm_tensor_gpu.T)
Vae_gene_gene_coexp_cpu = Vae_gene_gene_coexp_gpu.cpu()
Vae_gene_gene_coexp_df = pd.DataFrame(Vae_gene_gene_coexp_cpu.numpy())
Vae_gene_gene_coexp_df.index = Vae_Z_df.index
Vae_gene_gene_coexp_df.columns = Vae_Z_df.index
Vae_gene_gene_coexp_df.head()
display(Vae_gene_gene_coexp_df.shape)
Vae_gene_gene_coexp_df.to_csv(coexp_save_folder + "Vae_gene_gene_coexp.csv")

# %% [markdown]
# # Not all genes are in Plie and VAE. Take Common Genes #

# %%
set1 = set(Vae_gene_gene_coexp_df.columns.tolist())
set2 = set(Plier_gene_gene_coexp_df.columns.tolist())
common = list(set1.intersection(set2))
display(len(common))
Vae_gene_gene_coexp_common_df = Vae_gene_gene_coexp_df.loc[common, common]
Plier_gene_gene_coexp_common_df = Plier_gene_gene_coexp_df.loc[common, common] 
display(Vae_gene_gene_coexp_common_df.shape, Plier_gene_gene_coexp_common_df.shape)

Vae_gene_gene_coexp_common_df.to_csv(coexp_save_folder + "Vae_gene_gene_coexp_common_genes.csv")
Plier_gene_gene_coexp_common_df.to_csv(coexp_save_folder + "Plier_gene_gene_coexp_common_genes.csv")

# %%
Vae_gene_gene_coexp_common_df.head()

# %%
Plier_gene_gene_coexp_common_df.head()

# %% [markdown]
# # Gene Gene Coexpression Matrix (Original data) #

# %%
orig_data_df.head()

# %%
ori_Z_tensor = torch.tensor(orig_data_df.values, dtype=torch.float32)
ori_Z_norm_tensor = F.normalize(ori_Z_tensor, dim=1)
ori_Z_norm_tensor_gpu = ori_Z_norm_tensor.to(DEVICE)
ori_gene_gene_coexp_gpu = torch.matmul(ori_Z_norm_tensor_gpu, ori_Z_norm_tensor_gpu.T)
ori_gene_gene_coexp_cpu = ori_gene_gene_coexp_gpu.cpu()
ori_gene_gene_coexp_df = pd.DataFrame(ori_gene_gene_coexp_cpu.numpy())
ori_gene_gene_coexp_df.index = orig_data_df.index
ori_gene_gene_coexp_df.columns = orig_data_df.index
ori_gene_gene_coexp_df.head()
display(ori_gene_gene_coexp_df.shape)
ori_gene_gene_coexp_df.to_csv(coexp_save_folder + "ori_gene_gene_coexp.csv")
ori_gene_gene_coexp_common_df = ori_gene_gene_coexp_df.loc[common, common]
display(ori_gene_gene_coexp_common_df.shape)
ori_gene_gene_coexp_common_df.to_csv(coexp_save_folder + "ori_gene_gene_coexp_common_genes.csv")

# %%
# pca = PCA(n_components=0.80, random_state=0).set_output(transform="pandas")
# df_B_pca = pca.fit_transform(df_B)
# display(df_B_pca.shape)

# pca = PCA(n_components=0.80, random_state=0).set_output(transform="pandas")
# z_scaledT_pca = pca.fit_transform(z_scaled.T)
# display(z_scaledT_pca.shape)

# %%
# corr_dict_pcorr = {}
# corr_dict_scorr = {}
# pvae_lv_names = dict.fromkeys(z_scaled.columns)
# for row1 in df_B.T.items():
#     row1_name = row1[0]
#     plier_lv_vec = row1[1].values
#     corr_dict_pcorr[row1_name] = {"LV" + lv_name.split("lv")[1]: None for lv_name in pvae_lv_names}
#     corr_dict_scorr[row1_name] = {"LV" + lv_name.split("lv")[1]:None for lv_name in pvae_lv_names}
#     print("Currently on row :", row1_name)
#     for row2 in z_scaled.items():
#         row2_name = "LV" + row2[0].split("lv")[1]
#         pvae_lv_vec = row2[1].values
#         p_corr, p_value = pearsonr(plier_lv_vec , pvae_lv_vec)
#         s_corr = spearmanr(plier_lv_vec , pvae_lv_vec)#dcor.distance_correlation(plier_lv_vec , pvae_lv_vec)
#         corr_dict_pcorr[row1_name][row2_name] = p_corr
#         corr_dict_scorr[row1_name][row2_name] = s_corr

# %%
# pd.DataFrame([corr_dict_scorr["LV1"]])

# %%
Vae_gene_gene_coexp_matrix = Vae_gene_gene_coexp_common_df.to_numpy()
Plier_gene_gene_coexp_matrix = Plier_gene_gene_coexp_common_df.to_numpy()
ori_gene_gene_coexp_matrix = ori_gene_gene_coexp_common_df.to_numpy()
Plier_delta = np.abs(ori_gene_gene_coexp_matrix - Plier_gene_gene_coexp_matrix).flatten()
Vae_delta = np.abs(ori_gene_gene_coexp_matrix - Vae_gene_gene_coexp_matrix).flatten()

fig, ax = plt.subplots()
sns.histplot(Plier_delta, bins=100, ax = ax, label="Plier", color="red") #Customize bins, color
sns.histplot(Vae_delta, bins=100, ax = ax, label="Vae", color="green") #Customize bins, color
ax.legend()
plt.title('Mice data comparison') #Add a title
plt.xlabel('Delta') #Label x-axis
plt.ylabel('Frequency') #Label y-axis
plt.show()


# %%
mice_nah_data = pd.read_feather(INPUT_FILES["mice_gene_expr_norm_adult_heart_input"]).set_index("index")

# %%
set3 = set(mice_nah_data.index.to_list())
set13 = set1.intersection(set3)
set23 = set2.intersection(set3)

# %%
display(len(set1), len(set2), len(set13))

# %%
