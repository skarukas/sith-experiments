#!/bin/bash
#SBATCH -J SITH_train
#SBATCH -p dl
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH -o out/%j/%A/%a/log/%a.out
#SBATCH -e out/%j/%A/%a/log/%a.err
#SBATCH --mail-user=skarukas@iu.edu
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --array=0-0
cd "$PWD"
module load deeplearning/2.6.0

ParamFiles=(base-sithcon-params.yaml)
ExperimentName="sithcon_gsc_test"

ParamFile=${ParamFiles[$SLURM_ARRAY_TASK_ID]}

echo Task ID: $SLURM_ARRAY_TASK_ID 
echo "Using params from $ParamFile"
cat ${ParamFile}

srun python train.py --param ${ParamFile} --out_dir "out/$SLURM_JOB_NAME/$ExperimentName/$SLURM_ARRAY_TASK_ID"