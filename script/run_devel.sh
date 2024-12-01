#!/bin/bash

#SBATCH --account=hai_fnetwlet
#SBATCH --partition=develbooster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH -e out/devel/slurm-%j.err
#SBATCH -o out/devel/slurm-%j.out

# expects first argument ($1) to be a path to a python script (.py) file
# all other arguments are forwarded to the python script

module load Stages/2024 GCC/12.3.0 Python/3.11.3 virtualenv/20.23.1 PyTorch/2.1.2
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

srun -n1 --cpus-per-task=12 --threads-per-core=2 --mem="120G" --gres="gpu:4" --gpu-bind=closest \
python $@

echo "-- sbatch completed --"