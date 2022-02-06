#!/bin/bash

#SBATCH -J SITH_train
#SBATCH -p dl
#SBATCH --gpus-per-node p100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH -o out/SITH_train/.%A_%a.out
#SBATCH -e out/SITH_train/.%A_%a.err
#SBATCH --mail-user=skarukas@iu.edu
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --array=0-1

module load deeplearning/2.6.0

ParamFiles=(base-sithcon-params.yaml base-sithcon-params-dropout.yaml)
ExperimentName="sithcon_gsc_test"

ParamFile=${ParamFiles[$SLURM_ARRAY_TASK_ID]}
EXPERIMENT_DIR="out/$SLURM_JOB_NAME/${ExperimentName}_${SLURM_JOB_ID}"

mkdir -p $EXPERIMENT_DIR

echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Using params from $ParamFile"

python3 train.py --param ${ParamFile} --out_dir $EXPERIMENT_DIR  > "${EXPERIMENT_DIR}/output.log"
