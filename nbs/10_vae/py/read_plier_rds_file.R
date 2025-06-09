# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all,-execution,-papermill,-trusted
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %%
mice_model <- readRDS("/home/prashant/Documents/milton_lab/pvae/base/input/mice_data/orig/plier.rds")

# %%
names(mice_model)

# %%
mice_model$L2

# %%
mice_model_B <- as.data.frame(mice_model$B)
rownames(mice_model_B) <- paste0("LV", 1:nrow(mice_model_B))
head(mice_model_B)

# %%
mice_model_Z <- as.data.frame(mice_model$Z)
colnames(mice_model_Z) <- paste0("LV", 1:ncol(mice_model_Z))
head(mice_model_Z)

# %%
write.csv(mice_model_Z, "/home/prashant/Documents/milton_lab/pvae/base/input/mice_data/processed/mice_model_Z_ori.csv")
mice_model_Z <- data.frame(scale(mice_model_Z))
write.csv(mice_model_Z, "/home/prashant/Documents/milton_lab/pvae/base/input/mice_data/processed/mice_model_Z_std.csv")

# %%
write.csv(mice_model_B, "/home/prashant/Documents/milton_lab/pvae/base/input/mice_data/processed/mice_model_B_ori.csv")
mice_model_B <- data.frame(scale(mice_model_B))
write.csv(mice_model_B, "/home/prashant/Documents/milton_lab/pvae/base/input/mice_data/processed/mice_model_B_std.csv")

# %%
