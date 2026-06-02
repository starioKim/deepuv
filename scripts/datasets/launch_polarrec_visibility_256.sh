#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/nfs/home/stario/deepuv"
DATA_ROOT="/datasets/deepuv/polarrec"
CONDA_BIN="/nfs/home/stario/anaconda3/bin/conda"
ENV_NAME="SMat"
LOG_DIR="${DATA_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

echo "[$(date -Is)] 256FC generation started" >> "${LOG_DIR}/generate_256_all.log"

for split in MG IRSG UTSG EGB; do
  echo "[$(date -Is)] starting ${split} 256FC" >> "${LOG_DIR}/generate_256_all.log"
  "${CONDA_BIN}" run -n "${ENV_NAME}" python scripts/datasets/generate_polarrec_visibility.py \
    --data-root "${DATA_ROOT}" \
    --split "${split}" \
    --num-fourier 256 \
    --eht-npix 200 \
    --obs-type eht \
    --sample-ttype nfft \
    > "${LOG_DIR}/generate_256_${split}.log" 2>&1 &
done

wait
echo "[$(date -Is)] 256FC generation finished" >> "${LOG_DIR}/generate_256_all.log"
