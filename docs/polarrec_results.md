# PolarRec Results

Evaluation uses the held-out test partition from
`/datasets/deepuv/polarrec/splits/polarrec_seed0_train70_val10_test20.json`.
The test set has `1517` samples and was not used during training or validation.

| method | FC | n_test | PSNR ↑ | SSIM ↑ | MSE ↓ | MAE ↓ | LFD ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| UVDCNet | 128 | 1517 | 19.2128 ± 1.5508 | 0.4704 ± 0.0599 | 0.012785 ± 0.004937 | 0.052352 ± 0.012269 | 0.007599 ± 0.004771 |
| PolarRec-paper-style | 128 | 1517 | 18.3926 ± 2.2089 | 0.4535 ± 0.0577 | 0.016568 ± 0.009427 | 0.058953 ± 0.020761 | 0.007663 ± 0.005325 |
| Zero-filled sparse UV | 128 | 1517 | 5.6989 ± 1.0820 | 0.1042 ± 0.0173 | 0.277598 ± 0.069012 | 0.504557 ± 0.064187 | 0.022771 ± 0.006543 |
| Nearest UV interpolation | 128 | 1517 | 4.5268 ± 1.5390 | 0.1182 ± 0.0296 | 0.372313 ± 0.110625 | 0.594044 ± 0.098762 | 0.030552 ± 0.008380 |
| Hogbom CLEAN | 128 | 1517 | 3.5942 ± 0.9598 | 0.0829 ± 0.0137 | 0.446224 ± 0.078551 | 0.643909 ± 0.065774 | N/A |

## UVDCNet Run

- Slurm job: `9642`
- Partition: `pro6000`
- Status: `COMPLETED`
- Elapsed time: `02:58:35`
- Best validation epoch: `76`
- Best validation loss: `0.015530472739272822`
- Output directory: `results/polarrec/uvdc_128`
- Checkpoints: `best.pt`, `last.pt` in the output directory

## No-Retraining Baselines

After the scope was changed to methods that do not require retraining, the
supervised neural baseline jobs were cancelled before final evaluation. The
completed no-retraining baselines are:

- `zero_filled_128`: sparse measured UV coefficients, unobserved UV coefficients
  set to zero.
- `nearest_uv_128`: unobserved UV coefficients filled from the nearest observed
  UV grid location.
- `hogbom_clean_128`: classical Hogbom CLEAN on the dirty image and PSF. This
  method outputs an image only, so `LFD` is not applicable.

Pretrained weights were not present in the imported PolarRec, POLISH,
VIC-DDPM, LeIA, R2D2, or QuantifAI source trees, so those learned methods cannot
be tested without either downloading compatible checkpoints or retraining.

## PolarRec Reproduction Run

PolarRec-paper-style retraining completed successfully after retraining was
approved.

- Slurm job: `9680`
- Partition: `pro6000`
- Status: `COMPLETED`
- Elapsed time: `14:52:43`
- Output directory: `results/polarrec/polarrec_paper_128`
- Epochs: `400`
- Batch size: `2`
- Learning rate: `1e-4`
- Best validation epoch: `54`
- Best validation loss: `0.0176104090046285`
- Final validation loss at epoch 400: `0.018456205964029464`

This run uses the fixed train/val/test JSON split and evaluates only the held-out
test set after training.
