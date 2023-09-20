#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --time=03:00:00
#SBATCH --partition=amilan
#SBATCH --output=/projects/lgillenwater@xsede.org/DS_transfer_mouse/src/shell_scripts/logs/%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=lucas.gillenwater@cuanschutz.edu
#SBATCH --job-name=metadata-mouse-blood

module load python
pip install archs4py
pip install pandas

python /projects/lgillenwater@xsede.org/DS_transfer_mouse/src/python/retrieve_metadata.py

