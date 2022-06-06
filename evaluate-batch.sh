#!/bin/bash

#SBATCH -J Evaluate
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH -o out/Deep_LP_train/running_jobs/evaluate_%J_%a.out
#SBATCH -e out/Deep_LP_train/running_jobs/evaluate_%J_%a.err
#SBATCH --mail-user=skarukas@iu.edu
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --array=0-3

module load deeplearning/2.6.0

RELATIVE_PATHS=(resnet20_0 resnet32_1 resnet44_2 resnet56_3)
PATH_PREFIX="out/LP_ResNet_train/rotsvhn/svhn_weight_decay_533620/"

RELATIVE_PATH=${RELATIVE_PATHS[$SLURM_ARRAY_TASK_ID]}
EXPERIMENT_PATH="${PATH_PREFIX}${RELATIVE_PATH}"

echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running in ${EXPERIMENT_PATH}"

srun python3 evaluate_2d.py $EXPERIMENT_PATH