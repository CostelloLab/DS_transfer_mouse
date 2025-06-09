# DS_transfer_mouse
The goal of this project is to learn the relationship between the transcriptomic profiles of Down syndrome (DS) mouse models (Ts65Dn, Dp16, TcMAC21, etc.,  see !["Moyer et al."](https://www.cell.com/trends/genetics/fulltext/S0168-9525(20)30253-5?dgcid=raven_jbs_aip_email#secst0080) for a more detailed description of DS animal models). The relationship between mouse models with typical karyotypes and those simulating trisomy 21 is complicated by the 1.5 upregulation of genes homologous to those on human chromosome 21 (HSA12). Therefore, we hypothesize that learning this relationship requires models appopriate for learning this complex function. The repository provides the code and data necessary for creating and evaluating these deep learning models.  

## Setup

### Conda environment

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda.

1. Install main Python environment (allows to run most steps):

   ```bash
   export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu126"
   conda env create --name pvae --file setup/envs/environment.yml
   ```

1. Activate the environment:

    ```bash
    conda activate pvae
    ```

1. Install R and packages:

   ```bash
   bash setup/install.sh -g r
   ```

1. Install pvae package in editable mode (needs to be done only once):

    ```bash
    pip install -e .
    ```

## Problem Description ##

1. Plier a linear model captures the biologically meaningful latent variables (lv’s) from gene expression data with prior knowledge using biological pathways. VAE a non linear model uses only gene expression data to compute latent variables (lv’s). Lv’s are representation of subset of genes and their expression values. We want to compare Plier and VAE, if they are learning similar or different latent variables. To do this we are training Plier and VAE on mouse data. 

2. To evaluate whether models capture biologically meaningful lv’s in mice data, hold out one or more mouse experiments not seen during training. Biological meaningful lv’s for instance if we see some significant lv’s for a specific tissue sample compared to random samples. Apply the same preprocessing and use metadata (tissue type) as interpretable labels.  Perform differential expression analysis for each latent dimension to identify how many are significantly associated with the labels. Then, shuffle the labels and repeat the analysis to assess how many latent dimensions remain significantly associated.  

## Data ##
1. Data used for train VAE and PLIER is recount3 mouse RNAseq samples . We got the data from  git repo https://github.com/greenelab/mousiplier. It consists of 190,110 samples with 10849 genes. 

2. Unseen data used for evaluating the models with their tissue types are GSE218883: adult mouse heart tissue (16 samples x 10851 genes); GSE218885: adult mouse brain tissue (18 samples x 10851 genes); GSE218887: embryonic mouse facial mesenchyme tissue (9 samples x 10851 genes);  GSE218890: adult mouse mesenteric lymph nodes (17 samples x 10851 genes). Available with this paper https://www.nature.com/articles/s41588-023-01399-7#data-availability  Total unseen data consists of 60 samples x 10851 genes. 

## Statistical Test ##
**Goal:** Determine whether the number of significant LVs associated with all tissues from unseen data is greater than what would be expected by chance.
**Null Hypothesis:** The number of significant lvs in randomly shuffled tissue lables is more than number of significant lvs observed in real data. 
**STEPS:**
1. Start with a trained model (e.g., PLIER, VAE) that outputs LVs per sample and real tissue labels to the output. Total number of LVs in both models are 168. 

2. Split the output samples into tissue group (e.g., heart), non-tissue group (all other samples) 

3. For a given tissue group and non-tissue group use the Wilcoxon rank-sum test to compute p-values for each lv.  

4. Repeat this step 3 for each tissue and non-tissue groups and count total number of significant lvs i.e. lvs with p-value < 0.5. Let’s call it observed_significant_lvs_count 

5. Randomly shuffle the tissue labels of the output sample data and repeat the steps 2, 3, 4. Let’s call it permutation_significant_lvs_count. 

6. Repeat the step 5, 100 times and save all the permutation_significant_lvs_count.  

7. Repeat these steps for both Plier and VAE.
 
**RESULTS:**
**Figure 1** shows null distribution of the total significant lvs count and corresponding observed total significant lvs count for true labels in case of Plier. Similarly, in **Figure 2** for VAE. Both the models, total significant lv count for true tissue label (observed) is significantly more than random shuffles (null distribution).  In both cases Plier and VAE p-values is close to zero(<< 0.05) with total significant lvs in Plier ~170 and VAE ~115 out of 168 total lvs. Indicating both models (Plier, VAE) were able to generalize to unseen data. Further, we took global significant lvs from VAE and Plier, do correlation heatmap plot in **Figure 3** showing significantly low positive correlation between lvs of Plier and VAE, indicating models are learning different lvs. Similar to **Figure 3**, **Figure 4** is showing maximum correlation for each lv in VAE across all lvs in Plier. It shows that for some lvs in VAE there exist an lv in Plier that has strong positive correlation but in most cases maximum positive correlation is low. In **Figure 5**,  we created cluster heatmap showing lack similarity between lvs of the Plier and VAE. Overall we can say Plier and VAE models are generalizing to unseen data with small similarity between the lvs of Plier and VAE.

|![Figure 1] (https://github.com/CostelloLab/DS_transfer_mouse/figures/Null_distribution_Plier_plot.PNG)|
|:--:|
| *Figure 1* |

|![Figure 2] (https://github.com/CostelloLab/DS_transfer_mouse/figures/Null_distribution_VAE_plot.PNG)|
|:--:|
| *Figure 2* |

|![Figure 3] (https://github.com/CostelloLab/DS_transfer_mouse/figures/vae_plier_significant_lvs_correlation.PNG)|
|:--:|
| *Figure 3* |

|![Figure 4] (https://github.com/CostelloLab/DS_transfer_mouse/figures/vae_plier_significant_lvs_max_correlation.PNG)|
|:--:|
| *Figure 4* |

|![Figure 5] (https://github.com/CostelloLab/DS_transfer_mouse/figures/correlation_cluster_heatmap.png)|
|:--:|
| *Figure 5* |
