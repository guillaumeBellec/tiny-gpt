#!/bin/bash
#SBATCH --job-name=speed-gpt
#SBATCH --partition=GPU-h100
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=last_dist_log.txt
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
###SBATCH --mail-user=guillaume.bellec@tuwien.ac.at
###SBATCH --mail-type=END,FAIL

# Activate conda environment
source ~/.bashrc  # or wherever conda is initialized
conda activate tiny-gpt

# Set some useful environment variables
#export MASTER_ADDR=$(hostname)
#export MASTER_PORT=12355

torchrun --standalone --nproc_per_node=4 main.py