#!/bin/bash
#SBATCH -C rome
#SBATCH --nodes=1
#SBATCH -N1 --ntasks-per-node=1 -o /mnt/home/pgrover/continous_cell_cycle_stage_pred/logs/log_explore_lineage.%j.out -e /mnt/home/pgrover/continous_cell_cycle_stage_pred/logs/log_explore_lineage.%j.err

module load gcc/10
module load python/3
cd /mnt/home/pgrover/continous_cell_cycle_stage_pred
python explore_lineage.py
