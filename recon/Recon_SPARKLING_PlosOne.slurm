#!/bin/bash
#SBATCH --job-name=ReconFMRI
#SBATCH --partition=gpu_p2l
#SBATCH --account=hih@gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --output=fmriRecon_arraygpu_%A_%a.out # nom du fichier de sortie
#SBATCH --error=fmriRecon_arraygpu_%A_%a.out
#SBATCH --array=1-25

set -x
#NUM_GPUS=8
GROUP=5
source $HOME/mod_loading.sh
ctr=0
for ((j=0; j<25; j++))
do
	#export CUDA_VISIBLE_DEVICES=$(($(($((ctr))%NUM_GPUS))))
        ctr=$((ctr+1))
	for ((i=0; i<$GROUP; i++))
	do
		if [ $ctr -eq $SLURM_ARRAY_TASK_ID ]
		then    
			index=$(($GROUP*$j+$i))     	
			python recon_code_jean_zay.py --i=$index --obs="/gpfsstore/rech/hih/uwa98fg/InputData/V3/meas_MID290_3D_Snorekling_1mmiso_2_4s_120rep_antiClock_FID9812.dat" --out="/gpfsstore/rech/hih/uwa98fg/Review_PLOSONE/AntiClock_Adj/"  --mask="/gpfsstore/rech/hih/uwa98fg/InputData/V3/AntiClock/mask_B0.npy" --b0="/gpfsstore/rech/hih/uwa98fg/InputData/V3/AntiClock/B0.npy" --smaps="/gpfsstore/rech/hih/uwa98fg/InputData/V3/smaps_anticlock.npy"   &
			echo "Running $index"
		fi 	
	done

done  
wait
