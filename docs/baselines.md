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
| PolarRec | `baselines/PolarRec` | `https://github.com/RapidsAtHKUST/PolarRec` | Imported; adapted neural-field evaluation job `polarrec_nf_128` submitted. |
| POLISH | `baselines/POLISH` | `https://github.com/liamconnor/polish-pub` | Imported; adapted EDSR image-domain evaluation job `polish_edsr_128` submitted. |
| AIRI | `baselines/AIRI` | `https://github.com/basp-group/AIRI` | Imported; official MATLAB plug-and-play adapter still pending. |
| R2D2 | `baselines/R2D2-SII` | `https://github.com/basp-group/R2D2-SII` | Imported; adapted residual-series evaluation job `r2d2_series_128` submitted. |
| QuantifAI | `baselines/QuantifAI` | `https://github.com/astro-informatics/QuantifAI` | Imported; Bayesian/UQ code likely needs separate env and RI operator adapter. |
| LeIA | `baselines/LeIA` | `https://github.com/astro-informatics/LeIA` | Imported; adapted U-Net/GU-Net image-domain jobs `leia_unet_128` and `leia_gunet_128` submitted. |
| VIC-DDPM | `baselines/VIC-DDPM` | `https://github.com/RapidsAtHKUST/VIC-DDPM` | Imported; diffusion training adapter needed. |
| Deep Split-Bregman Deconvolution | `baselines/Deep-Split-Bregman-Deconvolution-Network` | `https://github.com/MoerAttempts/the-Deep-Split-Bregman-Deconvolution-Network` | Imported; example-only unsupervised per-sample adapter pending. |

## Active Baseline Jobs

Submitted on `2026-06-04` through `scripts/experiments/submit_baselines_128.sh`.

| Job | Method | Output Directory | Adapter Type |
|---:|---|---|---|
| `9674` | `vis_unet` | `results/polarrec/vis_unet_128` | Visibility U-Net dense-grid baseline. |
| `9675` | `leia_unet` | `results/polarrec/leia_unet_128` | LeIA-style image U-Net from dirty image. |
| `9676` | `leia_gunet` | `results/polarrec/leia_gunet_128` | LeIA/GU-Net-style dirty image plus PSF input. |
| `9677` | `polish_edsr` | `results/polarrec/polish_edsr_128` | POLISH-style EDSR image baseline. |
| `9678` | `r2d2_series` | `results/polarrec/r2d2_series_128` | R2D2-style residual image-series baseline. |
| `9679` | `polarrec_nf` | `results/polarrec/polarrec_nf_128` | PolarRec-style transformer-conditioned neural field. |

These adapted jobs use the same train/val/test JSON split, same 128FC gridded
visibility files, and the same metric writer as `UVDCNet`. Image-domain methods
do not directly predict dense visibility, so their `LFD` field is recorded as
`NaN` unless a visibility-domain output is available.

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

The adapted baseline trainer is:

```bash
python scripts/experiments/train_baseline.py --method vis_unet ...
```

The Slurm submitter is:

```bash
scripts/experiments/submit_baselines_128.sh
```
