#!/bin/bash
#SBATCH -A STF246
#SBATCH -J DQAOA
#SBATCH -p batch-gpu
#SBATCH -N 2
#SBATCH -t 23:59:00
#SBATCH -o slurm-DQAOA-%j.out

cd $SLURM_SUBMIT_DIR
date

module load PrgEnv-cray/8.6.0
source /ccsopen/home/gvk/defiant/CudaQ/CudaQ/bin/activate

srun -n 16 python DQAOA_run.py
