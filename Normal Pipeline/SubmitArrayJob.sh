#!/bin/bash

#SBATCH -p chromia
#SBATCH -t 24:00:00
#SBATCH -c 40
#SBATCH --output=Slurm_Output/slurm_output_%A_%a.out
#SBATCH --array=0-100%1

python LearnGrowth.py $SLURM_ARRAY_TASK_ID
