#!/bin/bash
#SBATCH --job-name=Trainning_TTP
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=./logs_slurm/generate_instruct.out
#SBATCH --error=./logs_slurm/generate_instruct.err
#SBATCH --account=oou@h100
#SBATCH --mail-type=BEGIN,FAIL,ABORT,END
#SBATCH --constraint=h100

module purge
module load arch/a100
module load python/3.11.5

set -x -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Les noeuds de calcul n'ont pas d'accès Internet : le modèle doit être
# pré-téléchargé (cf. download_model.py depuis un noeud frontal/prepost),
# ces variables empêchent vLLM/HF de tenter de contacter le Hub.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

srun uv run python -u -m panta_generate_data_instruct.generate --model Qwen/Qwen3-8B --n-per-cell 5 --output data/raw/generated.jsonl --raw-log data/raw/generated_raw_log.json --log-level INFO
