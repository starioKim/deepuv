# deepuv

Interferometric image reconstruction using deep learning.

## Project layout

- Code and project files: `/data/nfs/home/stario/deepuv`
- Large datasets: `/datasets/deepuv`
- Imported baselines: `baselines/`

Keep large dataset files outside this Git repository. Store them under
`/datasets/deepuv` instead.

## Baselines

Third-party baselines are imported under `baselines/` without their original
`.git/` metadata, so this repository owns the tracked copy.

The first imported baseline is PolarRec:

- Source: https://github.com/RapidsAtHKUST/PolarRec
- Paper: https://arxiv.org/abs/2308.14610

## Datasets

PolarRec-compatible data is stored under `/datasets/deepuv/polarrec`.

Prepare the public Galaxy10 DECaLS dataset and the four morphology subsets used
by the PolarRec paper:

```bash
python scripts/datasets/prepare_polarrec_dataset.py
```

The PolarRec training scripts expect precomputed visibility files in the same
dataset directory. Generate them after installing the PolarRec environment:

```bash
python scripts/datasets/generate_polarrec_visibility.py --split all --num-fourier 128
```
