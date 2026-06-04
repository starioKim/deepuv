#!/bin/bash
#SBATCH --job-name=deepuv_base
#SBATCH --partition=pro6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/data/nfs/home/stario/deepuv/logs/%x_%j.out
#SBATCH --error=/data/nfs/home/stario/deepuv/logs/%x_%j.err

set -euo pipefail

METHOD="${METHOD:?METHOD is required}"
REPO_ROOT="/data/nfs/home/stario/deepuv"
DATA_ROOT="${DATA_ROOT:-/data/nfs/home/stario/datasets/deepuv/polarrec}"
SPLIT_FILE="${SPLIT_FILE:-${DATA_ROOT}/splits/polarrec_seed0_train70_val10_test20.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/polarrec/${METHOD}_128}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-4}"
LR="${LR:-2e-4}"

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

source /data/nfs/home/stario/anaconda3/etc/profile.d/conda.sh
conda activate asta

python scripts/experiments/train_baseline.py \
  --method "${METHOD}" \
  --data-root "${DATA_ROOT}" \
  --split-file "${SPLIT_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-fourier 128 \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --lr "${LR}"
