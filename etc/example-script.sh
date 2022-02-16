#!/bin/bash
#SBATCH -J CP100
#SBATCH -p general
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --time=09:30:00
#SBATCH -o CP100Sith_%A_%a.out
#SBATCH -e CP100Sith_%A_%a.err
#SBATCH --mail-user=sahmaini@iu.edu
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --array=0-5
cd "$PWD"
module load deeplearning/2.6.0

temp=(1)
LearningRate=(0.001 0.01 0.1)
ModelName=("SITH" "SITH_F")
LearningRates=()
ModelNames=()

for k in ${temp[@]}
do
	for i in ${LearningRate[@]}
	do
		for j in ${ModelName[@]}
		do
			LearningRates+=($i)
			ModelNames+=($j)
		done
	done
done


LRate=${LearningRates[$SLURM_ARRAY_TASK_ID]}
MName=${ModelNames[$SLURM_ARRAY_TASK_ID]}

echo Task ID: $SLURM_ARRAY_TASK_ID 
echo LR:${LRate}
echo MN:${MName}




srun python alpha_train.py --model_type ${MName} --dataset outputs/datasets/CountPixelsdata_2000.npz --train_size 60 --valid_size 20 --test_size 20 --epochs 1000 --lr ${LRate} --batch_size 1 --n_taus 50 --tstr_min 0.005 --tstr_max 20.0 --k 8 --g 1 --dt 0.001 --num_inputs_sith 1 --num_extern_inputs_sith 16 --output_dir outputs --bptt_type "tbptt" --k_1 200
