#!/bin/bash
#SBATCH --job-name=speed-gpt
#SBATCH --partition=GPU-h100
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=last_dist_log.txt
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --mail-user=guillaume.bellec@tuwien.ac.at
#SBATCH --mail-type=END,FAIL

conda activate tiny-gpt
torchrun --standalone --nproc_per_node=4 main.py