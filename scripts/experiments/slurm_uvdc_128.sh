#!/usr/bin/env bash
#SBATCH --job-name=uvdc128
#SBATCH --partition=pro6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/nfs/home/stario/deepuv/logs/slurm_uvdc_128_%j.out
#SBATCH --error=/data/nfs/home/stario/deepuv/logs/slurm_uvdc_128_%j.err

set -euo pipefail

source /nfs/home/stario/anaconda3/etc/profile.d/conda.sh
conda activate asta

cd /data/nfs/home/stario/deepuv

DATA_ROOT="${DATA_ROOT:-/data/nfs/home/stario/datasets/deepuv/polarrec}"
SPLIT_FILE="${SPLIT_FILE:-${DATA_ROOT}/splits/polarrec_seed0_train70_val10_test20.json}"
OUT_DIR="${OUT_DIR:-results/polarrec/uvdc_128}"

python scripts/experiments/train_uvdc.py \
  --data-root "${DATA_ROOT}" \
  --split-file "${SPLIT_FILE}" \
  --output-dir "${OUT_DIR}" \
  --num-fourier 128 \
  --epochs "${EPOCHS:-80}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --lr "${LR:-2e-4}" \
  --workers "${WORKERS:-4}" \
  --stages "${STAGES:-5}" \
  --hidden-channels "${HIDDEN_CHANNELS:-64}" \
  --blocks-per-stage "${BLOCKS_PER_STAGE:-4}"

python scripts/experiments/summarize_results.py --results-root results/polarrec --output results/polarrec/comparison.md
