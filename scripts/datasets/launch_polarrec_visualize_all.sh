#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/nfs/home/stario/deepuv"
DATA_ROOT="/datasets/deepuv/polarrec"
LOG_DIR="${DATA_ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

echo "[$(date -Is)] full visualization generation started" >> "${LOG_DIR}/visualize_all.log"

for split in MG IRSG UTSG EGB; do
  echo "[$(date -Is)] starting ${split}" >> "${LOG_DIR}/visualize_all.log"
  python scripts/datasets/visualize_polarrec_dataset.py \
    --data-root "${DATA_ROOT}" \
    --out-root "${DATA_ROOT}/visualizations" \
    --splits "${split}" \
    --num-fourier 128 256 \
    --all-indices \
    --skip-existing \
    > "${LOG_DIR}/visualize_${split}.log" 2>&1 &
done

wait
echo "[$(date -Is)] full visualization generation finished" >> "${LOG_DIR}/visualize_all.log"
