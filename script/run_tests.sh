#!/bin/bash

#SBATCH --account=hai_fnetwlet
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00
#SBATCH -e out/tests/slurm-%j.err
#SBATCH -o out/tests/slurm-%j.out

module load Stages/2024 GCC/12.3.0 Python/3.11.3 virtualenv/20.23.1 PyTorch/2.1.2
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

srun -n1 --gres="gpu:1" --gpu-bind=closest pytest tests

echo "-- sbatch completed --"
