#!/bin/bash

#SBATCH -J Deep_LP_train
#SBATCH -p dl-debug
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-00:02:00
#SBATCH -o out/Deep_LP_train/running_jobs/%J_%a.out
#SBATCH -e out/Deep_LP_train/running_jobs/%J_%a.err
#SBATCH --mail-user=skarukas@iu.edu
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --array=0-0

# asks SLURM to send the USR1 signal 60 seconds before end of the time limit
#SBATCH --signal=B:USR1@60
cleanup_logs()
{
    echo "Job canceled, clearing log files from running_jobs directory."
    rm -rf "out/$SLURM_JOB_NAME/running_jobs/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}".*
}
trap 'cleanup_logs' USR1

# NOTE: make sure the output/error folders exist before running
module load deeplearning/2.6.0

ParamFiles=(local_8_8_fulltheta)
ExperimentRelativePath="cifar10/big_fulltheta"

ParamFile=${ParamFiles[$SLURM_ARRAY_TASK_ID]}
EXPERIMENT_DIR="out/$SLURM_JOB_NAME/${ExperimentRelativePath}_${SLURM_ARRAY_JOB_ID}/${ParamFile}_${SLURM_ARRAY_TASK_ID}"
ParamFile=param-files/Deep-LP/CIFAR10/ab_tests/${ParamFile}.yaml

mkdir -p $EXPERIMENT_DIR

echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Using params from $ParamFile"

python3 train.py --param ${ParamFile} --out_dir $EXPERIMENT_DIR