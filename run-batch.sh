#!/bin/bash

#SBATCH -J Deep_LP_train
#SBATCH -p dl
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=9:00:00
#SBATCH -o out/Deep_LP_train/running_jobs/%J_%a.out
#SBATCH -e out/Deep_LP_train/running_jobs/%J_%a.err
#SBATCH --mail-user=skarukas@iu.edu
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --array=0-0

# NOTE: make sure the output/error folders exist before running
module load deeplearning/2.6.0

ParamFiles=(combo-quick-4-med-augt)
ExperimentName="lp_mnist_combo_quick_augt_med"

ParamFile=${ParamFiles[$SLURM_ARRAY_TASK_ID]}
EXPERIMENT_DIR="out/$SLURM_JOB_NAME/${ExperimentName}_${SLURM_ARRAY_JOB_ID}/${ParamFile}_${SLURM_ARRAY_TASK_ID}"
ParamFile=param-files/Deep-LP/${ParamFile}.yaml

mkdir -p $EXPERIMENT_DIR

echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Using params from $ParamFile"

srun python3 train.py --param ${ParamFile} --out_dir $EXPERIMENT_DIR
