#!/bin/bash
#SBATCH --partition=HPC
#SBATCH --reservation=vladyslavz_19
python -m src.train_sst
