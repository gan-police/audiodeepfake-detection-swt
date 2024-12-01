#!/bin/bash

#SBATCH --account=hai_fnetwlet
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH -e out/slurm-%j.err
#SBATCH -o out/slurm-%j.out

module load Stages/2024 GCC/12.3.0 Python/3.11.3 virtualenv/20.23.1 PyTorch/2.1.2
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python -O -u script/generate_dataset.py \
--DOWNLOADS_DIRECTORY /p/project/hai_fnetwlet/downloads \
--OUTPUT_DIRECTORY /p/scratch/hai_fnetwlet/datasets/loaded_65536 \
--DATASET_TYPE loaded \
--NUM_WORKERS 48 \
--BATCH_SIZE 256 \
--TARGET_SAMPLE_LENGTH 65536 \
--LOAD_SAMPLE_OFFSET 0.25 \
--RANDOM_SAMPLE_SLICE true 

echo "-- sbatch completed --"
