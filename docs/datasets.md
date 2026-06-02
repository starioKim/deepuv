# Datasets

Large datasets for this project belong under `/datasets/deepuv`, not inside the
Git repository.

## PolarRec

Dataset root:

```text
/datasets/deepuv/polarrec
```

The PolarRec paper evaluates public Galaxy10 DECaLS morphologies:

| Split | Galaxy10 DECaLS class id | Class name |
| --- | ---: | --- |
| `MG` | 1 | Merging Galaxies |
| `IRSG` | 3 | In-between Round Smooth Galaxies |
| `UTSG` | 6 | Unbarred Tight Spiral Galaxies |
| `EGB` | 9 | Edge-on Galaxies with Bulge |

Prepare the public image dataset and these subsets:

```bash
python scripts/datasets/prepare_polarrec_dataset.py
```

Expected source file:

```text
/datasets/deepuv/polarrec/Galaxy10_DECals.h5
```

Expected generated subset files:

```text
/datasets/deepuv/polarrec/subsets/Galaxy10_DECals_MG.h5
/datasets/deepuv/polarrec/subsets/Galaxy10_DECals_IRSG.h5
/datasets/deepuv/polarrec/subsets/Galaxy10_DECals_UTSG.h5
/datasets/deepuv/polarrec/subsets/Galaxy10_DECals_EGB.h5
```

PolarRec training also expects precomputed visibility files:

```text
/datasets/deepuv/polarrec/eht_cont_200im_Galaxy10_DECals_full.h5
/datasets/deepuv/polarrec/eht_grid_128FC_200im_Galaxy10_DECals_full.h5
```

Generate them after installing the PolarRec Python/eht-imaging environment:

```bash
python scripts/datasets/generate_polarrec_visibility.py --split all --num-fourier 128
```

The EHT metadata files referenced by PolarRec are stored under
`baselines/PolarRec/code/`. The generator still requires the PolarRec runtime
dependencies, including PyTorch, torchvision, and eht-imaging.
