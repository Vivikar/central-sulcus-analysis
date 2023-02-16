#!/bin/bash
#SBATCH --partition=HPC
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1

python -m training.vanialla_unet