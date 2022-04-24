#!/bin/bash

#SBATCH -J Evaluate
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH -o out/Deep_LP_train/running_jobs/evaluate_%J_%a.out
#SBATCH -e out/Deep_LP_train/running_jobs/evaluate_%J_%a.err
#SBATCH --mail-user=skarukas@iu.edu
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --array=0-1

module load deeplearning/2.6.0

RELATIVE_PATHS=(med_model_large_topk_389594/30deg_rotations_tk12_lg_1 med_model_large_topk_389594/30deg_rotations_tk12_md_0)
PATH_PREFIX="out/Deep_LP_train/mnist/full_mnist_angle_invariance/"

RELATIVE_PATH=${RELATIVE_PATHS[$SLURM_ARRAY_TASK_ID]}
EXPERIMENT_PATH="${PATH_PREFIX}${RELATIVE_PATH}"

echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running in ${EXPERIMENT_PATH}"

srun python3 evaluate_2d.py $EXPERIMENT_PATH