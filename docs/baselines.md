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
| PolarRec | `baselines/PolarRec` | `https://github.com/RapidsAtHKUST/PolarRec` | Already imported earlier; needs train/test protocol adapter. |
| POLISH | `baselines/POLISH` | `https://github.com/liamconnor/polish-pub` | Imported; dirty-image to clean-image adapter needed. |
| AIRI | `baselines/AIRI` | `https://github.com/basp-group/AIRI` | Imported; plug-and-play RI operator adapter needed. |
| R2D2 | `baselines/R2D2-SII` | `https://github.com/basp-group/R2D2-SII` | Imported; residual-series training adapter needed. |
| QuantifAI | `baselines/QuantifAI` | `https://github.com/astro-informatics/QuantifAI` | Imported; Bayesian/UQ code likely needs separate env and RI operator adapter. |
| LeIA | `baselines/LeIA` | `https://github.com/astro-informatics/LeIA` | Imported; varying-coverage learned imaging adapter needed. |
| VIC-DDPM | `baselines/VIC-DDPM` | `https://github.com/RapidsAtHKUST/VIC-DDPM` | Imported; diffusion training adapter needed. |

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

Current Slurm job:

- Job id: `9634`
- Method: `uvdc_128`
- Partition: `a6000`

