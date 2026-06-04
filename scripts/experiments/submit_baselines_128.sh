#!/bin/bash
set -euo pipefail

METHODS=("$@")
if [ "${#METHODS[@]}" -eq 0 ]; then
  METHODS=(vis_unet leia_unet leia_gunet polish_edsr r2d2_series polarrec_nf)
fi

REPO_ROOT="/data/nfs/home/stario/deepuv"
PENDING_LIMIT="${PENDING_LIMIT:-8}"
cd "${REPO_ROOT}"
mkdir -p logs results/polarrec

pending_jobs() {
  squeue -h -u "${USER}" -t PD | wc -l
}

for method in "${METHODS[@]}"; do
  while [ "$(pending_jobs)" -ge "${PENDING_LIMIT}" ]; do
    echo "pending job limit ${PENDING_LIMIT} reached; sleeping 60s"
    sleep 60
  done
  echo "submitting ${method}"
  sbatch --job-name="deepuv_${method}" --export=ALL,METHOD="${method}" scripts/experiments/slurm_baseline_128.sh
done
