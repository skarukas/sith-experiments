#!/bin/bash

#SBATCH -J LP_ResNet_train
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH -o out/LP_ResNet_train/running_jobs/%J_%a.out
#SBATCH -e out/LP_ResNet_train/running_jobs/%J_%a.err
#SBATCH --mail-user=skarukas@iu.edu
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --array=0-3


# NOTE: make sure the output/error folders exist before running
module load deeplearning/2.6.0

ParamFiles=(resnet20 resnet32 resnet44 resnet56)

# output dir
ExperimentRelativePath="rotvshn/svhn_std_max"

ParamFile=${ParamFiles[$SLURM_ARRAY_TASK_ID]}
EXPERIMENT_DIR="out/$SLURM_JOB_NAME/${ExperimentRelativePath}_${SLURM_ARRAY_JOB_ID}/${ParamFile}_${SLURM_ARRAY_TASK_ID}"

# path of input YAML file
ParamFile=param-files/LPResNet/RotSVHN/${ParamFile}.yaml

mkdir -p $EXPERIMENT_DIR

echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Using params from $ParamFile"

python3 train.py --param ${ParamFile} --out_dir $EXPERIMENT_DIR