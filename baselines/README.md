# Baselines

Imported baselines live in this directory as ordinary source code tracked by
this repository. Do not keep nested `.git/` directories from upstream projects.

## PolarRec

- Imported from: https://github.com/RapidsAtHKUST/PolarRec
- Paper: https://arxiv.org/abs/2308.14610
- Dataset root: `/datasets/deepuv/polarrec`

The baseline expects Galaxy10 DECaLS image data and precomputed EHT visibility
HDF5 files. Use the project dataset scripts under `scripts/datasets/` to prepare
those files outside Git.
