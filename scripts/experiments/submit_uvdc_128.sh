#!/usr/bin/env bash
set -euo pipefail

cd /data/nfs/home/stario/deepuv
mkdir -p /datasets/deepuv/polarrec/logs

pending=$(squeue -u "${USER}" -h -t PD 2>/dev/null | wc -l)
if [ "${pending}" -ge 8 ]; then
  echo "Not submitting: pending Slurm jobs=${pending}, limit=8"
  exit 1
fi

split_file="/datasets/deepuv/polarrec/splits/polarrec_seed0_train70_val10_test20.json"
if [ ! -f "${split_file}" ]; then
  /nfs/home/stario/anaconda3/envs/asta/bin/python scripts/experiments/create_polarrec_splits.py \
    --data-root /datasets/deepuv/polarrec \
    --seed 0 \
    --train 0.7 \
    --val 0.1 \
    --test 0.2 \
    --output "${split_file}"
fi

sbatch --export=ALL,SPLIT_FILE="${split_file}" scripts/experiments/slurm_uvdc_128.sh
