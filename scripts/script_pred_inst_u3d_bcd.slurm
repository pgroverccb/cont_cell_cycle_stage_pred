#!/bin/bash
#SBATCH -p gpu
#SBATCH -C a100
#SBATCH --gpus=1
#SBATCH -N1 --ntasks-per-node=1 -o /mnt/home/pgrover/continous_cell_cycle_stage_pred/logs/log_pred_inst_u3d_bcd.%j.out -e /mnt/home/pgrover/continous_cell_cycle_stage_pred/logs/log_pred_inst_u3d_bcd.%j.err

module load gcc/10
module load python/3
module load cuda
cd /mnt/home/pgrover/continous_cell_cycle_stage_pred
python pred_inst_u3d_bcd.py
