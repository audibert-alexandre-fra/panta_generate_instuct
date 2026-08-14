#!/bin/bash
#SBATCH --job-name=Trainning_TTP
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=./logs_slurm/step_2e5.out
#SBATCH --error=./logs_slurm/step_2e5.err
#SBATCH --account=oou@h100
#SBATCH --mail-type=BEGIN,FAIL,ABORT,END
#SBATCH --constraint=h100

module purge
module load arch/a100

set -x -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

srun uv run python -u -m panta_generate_data_instruct.generate --model Qwen/Qwen3-8B --n-per-cell 5 --output data/raw/generated.jsonl --raw-log data/raw/generated_raw_log.json --log-level INFO
