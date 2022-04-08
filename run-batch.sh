#!/bin/bash

#SBATCH -J Deep_LP_train
#SBATCH -p dl
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:30:00
#SBATCH -o out/Deep_LP_train/running_jobs/%J_%a.out
#SBATCH -e out/Deep_LP_train/running_jobs/%J_%a.err
#SBATCH --mail-user=skarukas@iu.edu
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --array=0-3

# NOTE: make sure the output/error folders exist before running
module load deeplearning/2.6.0

ParamFiles=(local_16_8 local_8_8 local_5_5_5 local_5_5_5_fullpool)
ExperimentRelativePath="cifar10/local_filter_2_large"

ParamFile=${ParamFiles[$SLURM_ARRAY_TASK_ID]}
EXPERIMENT_DIR="out/$SLURM_JOB_NAME/${ExperimentRelativePath}_${SLURM_ARRAY_JOB_ID}/${ParamFile}_${SLURM_ARRAY_TASK_ID}"
ParamFile=param-files/Deep-LP/CIFAR10/ab_tests/${ParamFile}.yaml

mkdir -p $EXPERIMENT_DIR

echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Using params from $ParamFile"

srun python3 train.py --param ${ParamFile} --out_dir $EXPERIMENT_DIR
