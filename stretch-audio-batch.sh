#!/bin/bash

#SBATCH -J stretch_gsc
#SBATCH -p dl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=09:30:00
#SBATCH -o out/stretch_gsc_%a.out
#SBATCH -e out/stretch_gsc_%a.out
#SBATCH --mail-user=skarukas@iu.edu
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --array=0-7

module load deeplearning/2.6.0

Speeds=("10.00" "05.00" "02.50" "01.25" "00.80" "00.40" "00.20" "00.10")
Speed=${Speeds[$SLURM_ARRAY_TASK_ID]}

srun python3 stretch_wav.py $Speed
