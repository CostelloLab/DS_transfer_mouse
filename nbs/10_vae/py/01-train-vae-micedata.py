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
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from IPython.display import display
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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

# %% tags=[] scrolled=true jupyter={"source_hidden": true}
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

# %% tags=[]
folds_train_losses, folds_val_losses = train_vae(
    create_model,
    dataset,
    pathways=np.zeros((0, 0)),
    pathways_training_perc=PATHWAYS_TRAINING_PERC,
    models_output_dir=OUTPUT_FOLDER,
    vae_output_file_template=VAE_MODEL_OUTPUT_FILE,
    scaler_output_files_template=DATA_SCALER_OUTPUT_FILE,
    k_folds=K_FOLDS,
    n_folds_to_run=N_FOLDS_TO_RUN,
    epochs=N_EPOCS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    wd=WEIGHT_DECAY,
    random_state=random_state,
)

# %% [markdown]
# # Average performance across folds

# %% [markdown]
# ## Prepare

# %% tags=[]
df_perf = []

for fold in range(len(folds_train_losses)):
    fold_perf = folds_train_losses[fold]
    df_train = pd.DataFrame(dict(fold_perf.items()))
    df_train = df_train.assign(fold=fold, mode="train")
    df_train = df_train.reset_index().rename(columns={"index": "epoch"})
    df_train = df_train.set_index(["mode", "fold", "epoch"]).rename_axis(columns="perf")

    fold_perf = folds_val_losses[fold]
    df_val = pd.DataFrame(dict(fold_perf.items()))
    df_val = df_val.assign(fold=fold, mode="val")
    df_val = df_val.reset_index().rename(columns={"index": "epoch"})
    df_val = df_val.set_index(["mode", "fold", "epoch"]).rename_axis(columns="perf")

    df_all = pd.concat([df_train, df_val], axis=0)

    df_perf.append(df_all)

# %% tags=[]
df_perf = pd.concat(df_perf)

# %% tags=[]
df_perf.shape

# %% tags=[]
df_perf.head()

# %% [markdown]
# ## Stats

# %% tags=[]
df_perf.loc["train"].describe()

# %% tags=[]
df_perf.loc["val"].describe()

# %% [markdown]
# ### Save

# %%
df_perf.to_pickle(OUTPUT_FILES["performance_stats"])

# %% [markdown]
# ## Plots

# %% tags=[]
df_plot = df_perf.stack().rename("value").reset_index()

# %% tags=[]
df_plot

# %% tags=[]
df_plot = df_plot.astype(
    {
        "perf": "category",
        "mode": "category",
    }
)

# %% tags=[]
with sns.axes_style("whitegrid"):
    sns.boxplot(
        data=df_plot, x="perf", y="value", hue="mode", order=["full", "mse", "pathway"]
    )

# %% [markdown]
# # Learning curves

# %%
train_losses = folds_train_losses[0]
val_losses = folds_val_losses[0]

# %% tags=[]
train_losses.keys()

# %% tags=[]
val_losses.keys()

# %% [markdown]
# ## Combined: MSE and pathway

# %% tags=[]
with sns.axes_style("whitegrid"):  # , sns.color_palette("Paired"):
    ax = sns.lineplot(
        data=df_plot,
        x="epoch",
        y="value",
        hue="perf",
        hue_order=["mse"],
        style="mode",
    )
    ax.set_xticks(df_plot["epoch"].unique())

# %% [markdown]
# ## Full loss

# %% tags=[]
with sns.axes_style("whitegrid"):  # , sns.color_palette("Paired"):
    ax = sns.lineplot(
        data=df_plot, x="epoch", y="value", hue="perf", hue_order=["full"], style="mode"
    )
    ax.set_xticks(df_plot["epoch"].unique())

# %% [markdown]
# ## MSE loss

# %% tags=[]
with sns.axes_style("whitegrid"):  # , sns.color_palette("Paired"):
    ax = sns.lineplot(
        data=df_plot, x="epoch", y="value", hue="perf", hue_order=["mse"], style="mode"
    )
    ax.set_xticks(df_plot["epoch"].unique())

# %% [markdown]
# ## KL loss

# %% tags=[]
with sns.axes_style("whitegrid"):  # , sns.color_palette("Paired"):
    ax = sns.lineplot(
        data=df_plot, x="epoch", y="value", hue="perf", hue_order=["kl"], style="mode"
    )
    ax.set_xticks(df_plot["epoch"].unique())

# %% [markdown]
# ## Pathway prediction loss

# %% tags=[]
with sns.axes_style("whitegrid"):  # , sns.color_palette("Paired"):
    ax = sns.lineplot(
        data=df_plot,
        x="epoch",
        y="value",
        hue="perf",
        hue_order=["pathway"],
        style="mode",
    )
    ax.set_xticks(df_plot["epoch"].unique())

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
