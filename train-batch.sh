#!/bin/bash

#SBATCH -J Single_LP_train
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=9:00:00
#SBATCH -o out/Single_LP_train/running_jobs/%J_%a.out
#SBATCH -e out/Single_LP_train/running_jobs/%J_%a.err
#SBATCH --mail-user=skarukas@iu.edu
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --array=0-1


# NOTE: make sure the output/error folders exist before running
module load deeplearning/2.6.0

ParamFiles=(single_lp_1_bn single_lp_2_bn)
ExperimentRelativePath="mnist_r/simple_new"

ParamFile=${ParamFiles[$SLURM_ARRAY_TASK_ID]}
EXPERIMENT_DIR="out/$SLURM_JOB_NAME/${ExperimentRelativePath}_${SLURM_ARRAY_JOB_ID}/${ParamFile}_${SLURM_ARRAY_TASK_ID}"
ParamFile=param-files/Single_LP/MNIST/${ParamFile}.yaml

mkdir -p $EXPERIMENT_DIR

echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Using params from $ParamFile"

python3 train.py --param ${ParamFile} --out_dir $EXPERIMENT_DIR