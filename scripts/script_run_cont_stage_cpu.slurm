#!/bin/bash
#SBATCH -C rome
#SBATCH --nodes=1
#SBATCH -N1 --ntasks-per-node=8 -o /mnt/home/pgrover/continous_cell_cycle_stage_pred/logs/log_run_cont_stage_cpu.%j.out -e /mnt/home/pgrover/continous_cell_cycle_stage_pred/logs/log_run_cont_stage_cpu.%j.err

module load gcc/10
module load python/3
cd /mnt/home/pgrover/continous_cell_cycle_stage_pred
python run_cont_stage_cpu.py
