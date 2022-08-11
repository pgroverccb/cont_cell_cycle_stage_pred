#!/bin/bash
#SBATCH -C rome
#SBATCH --nodes=1
#SBATCH -N1 --ntasks-per-node=1 -o /mnt/home/pgrover/continous_cell_cycle_stage_pred/logs/log_form_dataset.%j.out -e /mnt/home/pgrover/continous_cell_cycle_stage_pred/logs/log_form_dataset.%j.err

module load gcc/10
module load python/3
cd /mnt/home/pgrover/continous_cell_cycle_stage_pred
python form_dataset.py
