#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --time=03:00:00
#SBATCH --partition=amilan
#SBATCH --output=%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=lucas.gillenwater@cuanschutz.edu
#SBATCH --job-name=download-mouse-blood

# # load necessary modules
module load R

cd /projects/lgillenwater@xsede.org/DS_transfer_mouse/data/mouse/archs4

Rscript /projects/lgillenwater@xsede.org/DS_transfer_mouse/src/download/download_archs4_mouse_blood.R
