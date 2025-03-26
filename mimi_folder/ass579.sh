#!/bin/bash

#SBATCH -p all # partition (queue) 
#SBATCH -c 4 # number of cores 
#SBATCH --mem=100G 
#SBATCH --propagate=NONE 
#SBATCH -t 1-00:00 # time (D-HH:MM) 
#SBATCH --output=output.log 
#SBATCH --error=error.log 
#SBATCH --qos=comp579-0gpu-4cpu-72h 
#SBATCH --account=winter2025-comp579

module load miniconda/miniconda-fall2024

pip cache purge 
pip install -r requirements.txt

python assault1.py