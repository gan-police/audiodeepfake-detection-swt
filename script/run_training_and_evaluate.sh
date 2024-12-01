#!/bin/bash

#SBATCH --account=hai_fnetwlet
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH -e out/slurm-%j.err
#SBATCH -o out/slurm-%j.out

# expects first argument ($1) to be a config.yaml
# all other arguments are forwarded to the config_evaluate.py script

# if SLURM_NTASKS is not set, default to 1
if ! [[ $SLURM_NTASKS =~ ^[0-9]+$ ]] ; then
   SLURM_NTASKS=1
fi

# if SLURM_CPUS_PER_TASK is not set, default to 1
if ! [[ $SLURM_CPUS_PER_TASK =~ ^[0-9]+$ ]] ; then
   SLURM_CPUS_PER_TASK=1
fi

# if SLURM_THREADS_PER_CORE is not set, default to 1
if ! [[ $SLURM_THREADS_PER_CORE =~ ^[0-9]+$ ]] ; then
   SLURM_THREADS_PER_CORE=1
fi

# number of gpus
GPUS=$(($SLURM_NTASKS<4 ? $SLURM_NTASKS : 4))
# maximum memory per gpu
MEM=$((480 / $GPUS))

module load Stages/2024 GCC/12.3.0 Python/3.11.3 virtualenv/20.23.1 PyTorch/2.1.2
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# start training tasks
srun --ntasks=$SLURM_NTASKS --cpu-bind=verbose --cpus-per-task=$SLURM_CPUS_PER_TASK --threads-per-core=$SLURM_THREADS_PER_CORE --mem="${MEM}G" --gres="gpu:${GPUS}" --gpu-bind=closest --exact --exclusive --kill-on-bad-exit=0 --wait=0 \
python -O -u script/config_train.py --config $1
# gather training results
sleep 5
python -O -u script/gather_training_results.py --config $1
echo "finished training"

# start evaluating models
srun --ntasks=$SLURM_NTASKS --cpu-bind=verbose --cpus-per-task=$SLURM_CPUS_PER_TASK --threads-per-core=$SLURM_THREADS_PER_CORE --mem="${MEM}G" --gres="gpu:${GPUS}" --gpu-bind=closest --exact --exclusive --kill-on-bad-exit=0 --wait=0 \
python -O -u script/config_evaluate.py --config $1 ${@:2}
# gather evaluation results
sleep 5
python -O -u script/gather_evaluation_results.py --config $1
echo "finished evaluation"

echo "-- sbatch completed --"
