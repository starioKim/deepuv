#!/usr/bin/env python3
"""Generate PolarRec-compatible EHT visibility HDF5 files.

This script uses the imported PolarRec simulator. It requires the PolarRec
Python environment, including PyTorch, torchvision, and eht-imaging.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
POLARREC_ROOT = REPO_ROOT / "baselines" / "PolarRec"
DATA_ROOT = Path("/datasets/deepuv/polarrec")
SPLIT_FILES = {
    "all": "Galaxy10_DECals.h5",
    "MG": "subsets/Galaxy10_DECals_MG.h5",
    "IRSG": "subsets/Galaxy10_DECals_IRSG.h5",
    "UTSG": "subsets/Galaxy10_DECals_UTSG.h5",
    "EGB": "subsets/Galaxy10_DECals_EGB.h5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="all")
    parser.add_argument("--num-fourier", type=int, default=128)
    parser.add_argument("--eht-npix", type=int, default=200)
    parser.add_argument("--obs-type", choices=["eht", "sparse", "dense"], default="eht")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit generation for smoke tests; default is all samples.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def check_baseline_assets() -> None:
    required = [
        POLARREC_ROOT / "code" / "EHT2017.txt",
        POLARREC_ROOT / "code" / "avery_m87_2_eofn.txt",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "PolarRec simulator metadata files are missing:\n"
            f"{joined}\n"
            "Supply these files under baselines/PolarRec/code/ before generating visibility data."
        )


def import_polarrec_helpers():
    sys.path.insert(0, str(POLARREC_ROOT))
    import torch  # noqa: PLC0415
    from data_ehtim_cont import Galaxy10_DECals_Dataset, obs_with_eht, upscale_tensor  # noqa: PLC0415

    return Galaxy10_DECals_Dataset, obs_with_eht, torch, upscale_tensor


def output_paths(data_root: Path, split: str, num_fourier: int, eht_npix: int) -> tuple[Path, Path]:
    suffix = "Galaxy10_DECals_full" if split == "all" else f"Galaxy10_DECals_{split}"
    cont = data_root / f"eht_cont_{eht_npix}im_{suffix}.h5"
    grid = data_root / f"eht_grid_{num_fourier}FC_{eht_npix}im_{suffix}.h5"
    return cont, grid


def create_dataset_from_sample(group: h5py.File, key: str, sample: np.ndarray, n_samples: int) -> h5py.Dataset:
    shape = (sample.shape[0], n_samples)
    return group.create_dataset(
        key,
        shape=shape,
        dtype=np.float32,
        chunks=(sample.shape[0], 1),
        compression="gzip",
        compression_opts=4,
    )


def simulate_visibility(
    image_dataset,
    idx: int,
    *,
    obs_with_eht,
    torch,
    upscale_tensor,
    obs_type: str,
    eht_npix: int,
    num_fourier: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    img, _ = image_dataset[idx]
    img_res_initial = int(torch.numel(img) ** 0.5)
    img = img.reshape((img_res_initial, img_res_initial))
    if img_res_initial != 200:
        img = upscale_tensor(img, final_res=200, method="cubic")
        img = torch.from_numpy(img)

    eht_obs, eht_im, _, _, _ = obs_with_eht(img, obs_type=obs_type, eht_npix=eht_npix)
    u_eht = np.asarray(eht_obs.unpack(["u"], conj=True), dtype=float).reshape(-1)
    v_eht = np.asarray(eht_obs.unpack(["v"], conj=True), dtype=float).reshape(-1)
    vis_eht = np.asarray(eht_obs.unpack(["vis"], conj=True), dtype=complex).reshape(-1)
    uv_dist_eht = np.asarray(eht_obs.unpack(["uvdist"], conj=True), dtype=float).reshape(-1)

    cont_sparse = {
        "uv": np.stack((u_eht, v_eht), axis=1),
        "vis": vis_eht,
    }

    max_base = np.max(uv_dist_eht)
    x = np.linspace(-max_base, max_base, num_fourier)
    y = np.linspace(-max_base, max_base, num_fourier)
    xv, yv = np.meshgrid(x, y)
    grid_dense_uv = np.stack((xv.ravel(), yv.ravel()), axis=1)
    grid_dense = {
        "uv": grid_dense_uv,
        "vis": eht_im.sample_uv(grid_dense_uv)[0],
    }

    x_centers = (x[1:] + x[:-1]) / 2
    y_centers = (y[1:] + y[:-1]) / 2
    u_dig = np.digitize(u_eht, x_centers)
    v_dig = np.digitize(v_eht, y_centers)
    uv_dig = np.stack((x[u_dig], y[v_dig]), axis=1)
    grid_sparse_uv = np.unique(uv_dig, axis=0)
    grid_sparse = {
        "uv": grid_sparse_uv,
        "vis": eht_im.sample_uv(grid_sparse_uv)[0],
    }

    return grid_dense, cont_sparse, grid_sparse


def main() -> int:
    args = parse_args()
    check_baseline_assets()

    image_path = args.data_root / SPLIT_FILES[args.split]
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image dataset: {image_path}")

    cont_path, grid_path = output_paths(args.data_root, args.split, args.num_fourier, args.eht_npix)
    for path in (cont_path, grid_path):
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} already exists; use --force to overwrite")

    Galaxy10DECalsDataset, obs_with_eht, torch, upscale_tensor = import_polarrec_helpers()
    cwd = Path.cwd()
    os.chdir(POLARREC_ROOT)
    try:
        dataset = Galaxy10DECalsDataset(str(image_path), None)
        n_samples = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
        if n_samples <= 0:
            raise ValueError("No samples requested")

        tmp_cont = cont_path.with_suffix(cont_path.suffix + ".tmp")
        tmp_grid = grid_path.with_suffix(grid_path.suffix + ".tmp")
        for path in (tmp_cont, tmp_grid):
            if path.exists():
                path.unlink()

        with h5py.File(tmp_cont, "w") as cont_h5, h5py.File(tmp_grid, "w") as grid_h5:
            first_grid_dense, first_cont_sparse, first_grid_sparse = simulate_visibility(
                dataset,
                0,
                obs_with_eht=obs_with_eht,
                torch=torch,
                upscale_tensor=upscale_tensor,
                obs_type=args.obs_type,
                eht_npix=args.eht_npix,
                num_fourier=args.num_fourier,
            )

            grid_h5.create_dataset("u_sparse", data=first_grid_sparse["uv"][:, 0].astype(np.float32))
            grid_h5.create_dataset("v_sparse", data=first_grid_sparse["uv"][:, 1].astype(np.float32))
            grid_h5.create_dataset("u_dense", data=first_grid_dense["uv"][:, 0].astype(np.float32))
            grid_h5.create_dataset("v_dense", data=first_grid_dense["uv"][:, 1].astype(np.float32))
            cont_h5.create_dataset("u_cont", data=first_cont_sparse["uv"][:, 0].astype(np.float32))
            cont_h5.create_dataset("v_cont", data=first_cont_sparse["uv"][:, 1].astype(np.float32))

            grid_re_sparse = create_dataset_from_sample(
                grid_h5, "vis_re_sparse", np.real(first_grid_sparse["vis"]), n_samples
            )
            grid_im_sparse = create_dataset_from_sample(
                grid_h5, "vis_im_sparse", np.imag(first_grid_sparse["vis"]), n_samples
            )
            grid_re_dense = create_dataset_from_sample(
                grid_h5, "vis_re_dense", np.real(first_grid_dense["vis"]), n_samples
            )
            grid_im_dense = create_dataset_from_sample(
                grid_h5, "vis_im_dense", np.imag(first_grid_dense["vis"]), n_samples
            )
            cont_re = create_dataset_from_sample(
                cont_h5, "vis_re_cont", np.real(first_cont_sparse["vis"]), n_samples
            )
            cont_im = create_dataset_from_sample(
                cont_h5, "vis_im_cont", np.imag(first_cont_sparse["vis"]), n_samples
            )

            for idx in range(n_samples):
                if idx == 0:
                    grid_dense, cont_sparse, grid_sparse = (
                        first_grid_dense,
                        first_cont_sparse,
                        first_grid_sparse,
                    )
                else:
                    grid_dense, cont_sparse, grid_sparse = simulate_visibility(
                        dataset,
                        idx,
                        obs_with_eht=obs_with_eht,
                        torch=torch,
                        upscale_tensor=upscale_tensor,
                        obs_type=args.obs_type,
                        eht_npix=args.eht_npix,
                        num_fourier=args.num_fourier,
                    )

                grid_re_sparse[:, idx] = np.real(grid_sparse["vis"]).astype(np.float32)
                grid_im_sparse[:, idx] = np.imag(grid_sparse["vis"]).astype(np.float32)
                grid_re_dense[:, idx] = np.real(grid_dense["vis"]).astype(np.float32)
                grid_im_dense[:, idx] = np.imag(grid_dense["vis"]).astype(np.float32)
                cont_re[:, idx] = np.real(cont_sparse["vis"]).astype(np.float32)
                cont_im[:, idx] = np.imag(cont_sparse["vis"]).astype(np.float32)

                if idx % 25 == 0:
                    print(f"Generated {idx + 1}/{n_samples} samples")

            for handle in (cont_h5, grid_h5):
                handle.attrs["image_dataset"] = str(image_path)
                handle.attrs["split"] = args.split
                handle.attrs["num_fourier"] = args.num_fourier
                handle.attrs["eht_npix"] = args.eht_npix
                handle.attrs["obs_type"] = args.obs_type

        os.replace(tmp_cont, cont_path)
        os.replace(tmp_grid, grid_path)
        print(f"Wrote {cont_path}")
        print(f"Wrote {grid_path}")
    finally:
        os.chdir(cwd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
