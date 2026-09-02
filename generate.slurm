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
module load arch/h100
module load python/3.11.5
module load cuda/12.4.1

set -x -e

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Les noeuds de calcul n'ont pas d'accès Internet : le modèle doit être
# pré-téléchargé (cf. download_model.py depuis un noeud frontal/prepost),
# ces variables empêchent vLLM/HF de tenter de contacter le Hub.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Triton compile un petit module C (cuda_utils) à la volée et a besoin
# des headers de développement du Python du venv (uv) — sans ça gcc
# retombe sur le Python système, qui n'a pas Python.h, et la compilation
# échoue (fatal error: Python.h: No such file or directory).
PY_INCLUDE=$(uv run python -c "import sysconfig; print(sysconfig.get_paths()['include'])")
export CPATH="${PY_INCLUDE}:${CPATH}"

srun uv run python -u -m panta_generate_data_instruct.generate --model Qwen/Qwen3-32B --n-per-cell 5 --output data/raw/generated.jsonl --raw-log data/raw/generated_raw_log.json --log-level INFO