# PolarRec Results

Evaluation uses the held-out test partition from
`/datasets/deepuv/polarrec/splits/polarrec_seed0_train70_val10_test20.json`.
The test set has `1517` samples and was not used during training or validation.

| method | FC | n_test | PSNR ↑ | SSIM ↑ | MSE ↓ | MAE ↓ | LFD ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| UVDCNet | 128 | 1517 | 19.2128 ± 1.5508 | 0.4704 ± 0.0599 | 0.012785 ± 0.004937 | 0.052352 ± 0.012269 | 0.007599 ± 0.004771 |
| Zero-filled sparse UV | 128 | 1517 | 5.6989 ± 1.0820 | 0.1042 ± 0.0173 | 0.277598 ± 0.069012 | 0.504557 ± 0.064187 | 0.022771 ± 0.006543 |

## UVDCNet Run

- Slurm job: `9642`
- Partition: `pro6000`
- Status: `COMPLETED`
- Elapsed time: `02:58:35`
- Best validation epoch: `76`
- Best validation loss: `0.015530472739272822`
- Output directory: `results/polarrec/uvdc_128`
- Checkpoints: `best.pt`, `last.pt` in the output directory

The comparison table will be extended as each imported/open baseline is adapted
to the same train/test split.
