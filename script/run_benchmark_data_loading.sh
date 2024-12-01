#!/bin/bash

#SBATCH --account=hai_fnetwlet
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH -e out/bench/slurm-%j.err
#SBATCH -o out/bench/slurm-%j.out

# all arguments are forwarded to the python script

module load Stages/2024 GCC/12.3.0 Python/3.11.3 virtualenv/20.23.1 PyTorch/2.1.2
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

srun -n4 --cpus-per-task=$SLURM_CPUS_PER_TASK --threads-per-core=$SLURM_THREADS_PER_CORE --mem=120G --gres="gpu:4" --gpu-bind=closest \
python -O -u script/benchmark_data_loading.py $@

echo "-- sbatch completed --"