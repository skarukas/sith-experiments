#!/bin/bash
#SBATCH -J SITH_train
#SBATCH -p dl
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --time=09:30:00
#SBATCH -o out/%j/%A/%a/log/%a.out
#SBATCH -e out/%j/%A/%a/log/%a.err
#SBATCH --mail-user=skarukas@iu.edu
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --array=0-5
cd "$PWD"
module load deeplearning/2.6.0

ParamFiles=(example-params.yaml)

ParamFile=${ParamFiles[$SLURM_ARRAY_TASK_ID]}

echo Task ID: $SLURM_ARRAY_TASK_ID 
echo "Using params from $ParamFile"
cat ${ParamFile}

srun python train.py --param ${ParamFile} --out_dir "out/$SLURM_JOB_NAME/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID"