# PolarRec Comparison Baselines

This file tracks radio-interferometric reconstruction methods considered for
the PolarRec dataset comparison. Baseline source trees are imported under
`baselines/` with nested Git metadata removed, so this repository owns the code
copies.

Large model/data artifacts from upstream repositories are intentionally not
tracked when they match the repository-level `.gitignore` patterns. Experiments
must use `/datasets/deepuv/polarrec` as the data root.

## Dataset Protocol

- Dataset: `/datasets/deepuv/polarrec`
- Slurm-visible staging path for 128FC training:
  `/data/nfs/home/stario/datasets/deepuv/polarrec`
- Split file: `/datasets/deepuv/polarrec/splits/polarrec_seed0_train70_val10_test20.json`
- Stratified split across `MG`, `IRSG`, `UTSG`, `EGB`
- Counts: train `5307`, val `758`, test `1517`
- Evaluation rule: train/validation may use only train/val items; final metrics
  must be computed only on the test items.

## Metrics

The comparison table will report the metrics commonly used by the PolarRec and
related learned RI reconstruction papers:

- `PSNR` on normalized reconstructed image.
- `SSIM` on normalized reconstructed image.
- `MSE` and `MAE` on normalized reconstructed image.
- `LFD`, the PolarRec-style log frequency distance on dense visibility.

## Imported Baselines

| Method | Imported Path | Upstream | Current Status |
|---|---|---|---|
| PolarRec | `baselines/PolarRec` | `https://github.com/RapidsAtHKUST/PolarRec` | Imported; no pretrained checkpoint found, so testing requires retraining or a compatible checkpoint. |
| POLISH | `baselines/POLISH` | `https://github.com/liamconnor/polish-pub` | Imported; no pretrained weights found, so testing requires retraining or a compatible checkpoint. |
| AIRI | `baselines/AIRI` | `https://github.com/basp-group/AIRI` | Imported; official MATLAB plug-and-play adapter still pending. |
| R2D2 | `baselines/R2D2-SII` | `https://github.com/basp-group/R2D2-SII` | Imported; no pretrained checkpoint found, so testing requires retraining or a compatible checkpoint. |
| QuantifAI | `baselines/QuantifAI` | `https://github.com/astro-informatics/QuantifAI` | Imported; Bayesian/UQ code likely needs separate env and RI operator adapter. |
| LeIA | `baselines/LeIA` | `https://github.com/astro-informatics/LeIA` | Imported; no pretrained checkpoint found, so testing requires retraining or a compatible checkpoint. |
| VIC-DDPM | `baselines/VIC-DDPM` | `https://github.com/RapidsAtHKUST/VIC-DDPM` | Imported; no pretrained checkpoint found, so testing requires retraining or a compatible checkpoint. |
| Deep Split-Bregman Deconvolution | `baselines/Deep-Split-Bregman-Deconvolution-Network` | `https://github.com/MoerAttempts/the-Deep-Split-Bregman-Deconvolution-Network` | Imported; example-only unsupervised per-sample adapter pending. |

## No-Retraining Evaluations

The supervised neural baseline jobs submitted on `2026-06-04` were cancelled
after the scope changed to methods that do not require retraining. No Slurm jobs
are currently active.

Completed no-retraining test-set evaluations:

| Method | Output Directory | Notes |
|---|---|---|
| Zero-filled sparse UV | `results/polarrec/zero_filled_128` | Existing sparse-UV baseline. |
| Nearest UV interpolation | `results/polarrec/nearest_uv_128` | Fills unobserved UV grid points from nearest observed UV samples. |
| Hogbom CLEAN | `results/polarrec/hogbom_clean_128` | Classical image-only CLEAN; `LFD` is not applicable. |

## Own Method

`UVDCNet` is implemented in `deepuv/uvdc_model.py`.

Algorithm structure:

1. Start from sparse gridded complex visibility.
2. Predict dense visibility with a convolutional residual block.
3. Enforce hard UV data consistency by replacing observed sparse-grid
   coefficients with the measured values.
4. Repeat the predict/data-consistency block for multiple unrolled stages.
5. Train with dense-visibility loss plus image-domain loss.

Training entrypoint:

```bash
scripts/experiments/submit_uvdc_128.sh
```

Current Slurm job status should be checked with `squeue -u "$USER"`.

## Baseline Entry Points

The supervised adapted baseline trainer is available but should only be used
when retraining is allowed:

```bash
python scripts/experiments/train_baseline.py --method vis_unet ...
```

The Slurm submitter is:

```bash
scripts/experiments/submit_baselines_128.sh
```

The no-retraining evaluator is:

```bash
python scripts/experiments/eval_no_retrain_baselines.py --method nearest_uv
```
