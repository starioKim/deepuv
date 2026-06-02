#!/usr/bin/env python3
"""Write PNG previews for PolarRec image and visibility HDF5 datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DATA_ROOT = Path("/datasets/deepuv/polarrec")
OUT_ROOT = DATA_ROOT / "visualizations"
SPLITS = ("MG", "IRSG", "UTSG", "EGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--all-indices",
        action="store_true",
        help="Visualize every sample in each selected split.",
    )
    parser.add_argument("--num-fourier", nargs="+", type=int, default=[128, 256])
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a sample directory when all expected PNG files already exist.",
    )
    return parser.parse_args()


def normalize01(x: np.ndarray, *, percentile: bool = False) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if percentile:
        lo, hi = np.percentile(x, [1, 99])
    else:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float64)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def rgb_to_luma(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64) / 255.0
    return 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]


def visibility_to_image(vis_grid: np.ndarray) -> np.ndarray:
    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(vis_grid)))
    return normalize01(np.abs(image), percentile=True)


def sparse_to_grid(
    u_dense: np.ndarray,
    v_dense: np.ndarray,
    u_sparse: np.ndarray,
    v_sparse: np.ndarray,
    vis_sparse: np.ndarray,
    num_fourier: int,
) -> np.ndarray:
    u_vals = np.unique(u_dense)
    v_vals = np.unique(v_dense)
    grid = np.zeros((num_fourier, num_fourier), dtype=np.complex64)
    u_idx = np.abs(u_sparse[:, None] - u_vals[None, :]).argmin(axis=1)
    v_idx = np.abs(v_sparse[:, None] - v_vals[None, :]).argmin(axis=1)
    grid[v_idx, u_idx] = vis_sparse
    return grid


def save_image(path: Path, image: np.ndarray, *, cmap: str = "gray") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    ax.imshow(image, cmap=cmap)
    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_visibility_map(path: Path, vis_grid: np.ndarray, *, cmap: str = "magma") -> None:
    amp = np.log1p(np.abs(vis_grid))
    save_image(path, normalize01(amp, percentile=True), cmap=cmap)


def save_sparse_scatter(path: Path, u: np.ndarray, v: np.ndarray, vis: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    color = np.log1p(np.abs(vis))
    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    ax.scatter(u, v, c=color, s=5, cmap="magma", linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title("Sparse UV samples")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_overview(
    path: Path,
    clean_rgb: np.ndarray,
    dense_image: np.ndarray,
    dirty_image: np.ndarray,
    dense_vis: np.ndarray,
    sparse_vis: np.ndarray,
    u_sparse: np.ndarray,
    v_sparse: np.ndarray,
    vis_sparse: np.ndarray,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=160)
    axes[0, 0].imshow(clean_rgb)
    axes[0, 0].set_title("Clean image")
    axes[0, 1].imshow(dense_image, cmap="gray")
    axes[0, 1].set_title("Clean image from dense visibility")
    axes[0, 2].imshow(dirty_image, cmap="gray")
    axes[0, 2].set_title("Dirty image from sparse visibility")
    axes[1, 0].imshow(normalize01(np.log1p(np.abs(dense_vis)), percentile=True), cmap="magma")
    axes[1, 0].set_title("Clean visibility amplitude")
    axes[1, 1].imshow(normalize01(np.log1p(np.abs(sparse_vis)), percentile=True), cmap="magma")
    axes[1, 1].set_title("Dirty/sparse visibility amplitude")
    axes[1, 2].scatter(u_sparse, v_sparse, c=np.log1p(np.abs(vis_sparse)), s=3, cmap="magma", linewidths=0)
    axes[1, 2].set_aspect("equal", adjustable="box")
    axes[1, 2].set_title("Sparse UV samples")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def expected_pngs(sample_dir: Path) -> list[Path]:
    return [
        sample_dir / "image_clean_original.png",
        sample_dir / "image_clean_luma.png",
        sample_dir / "image_clean_from_dense_visibility.png",
        sample_dir / "image_dirty_from_sparse_visibility.png",
        sample_dir / "visibility_clean_dense_amplitude.png",
        sample_dir / "visibility_dirty_sparse_amplitude.png",
        sample_dir / "visibility_dirty_sparse_uv_scatter.png",
        sample_dir / "overview.png",
    ]


def visualize_split(
    data_root: Path,
    out_root: Path,
    split: str,
    indices: list[int] | None,
    fcs: list[int],
    *,
    skip_existing: bool,
) -> None:
    image_path = data_root / "subsets" / f"Galaxy10_DECals_{split}.h5"
    with h5py.File(image_path, "r") as image_h5:
        labels = image_h5["ans"]
        active_indices = list(range(len(labels))) if indices is None else indices
        for fc in fcs:
            grid_path = data_root / f"eht_grid_{fc}FC_200im_Galaxy10_DECals_{split}.h5"
            with h5py.File(grid_path, "r") as grid_h5:
                u_dense = grid_h5["u_dense"][:]
                v_dense = grid_h5["v_dense"][:]
                u_sparse = grid_h5["u_sparse"][:]
                v_sparse = grid_h5["v_sparse"][:]
                for count, idx in enumerate(active_indices, start=1):
                    if idx < 0 or idx >= len(labels):
                        raise IndexError(f"{split} index {idx} out of range 0..{len(labels)-1}")
                    sample_dir = out_root / split / f"{fc}FC" / f"sample_{idx:04d}"
                    outputs = expected_pngs(sample_dir)
                    if skip_existing and all(path.exists() for path in outputs):
                        if count % 100 == 0:
                            print(f"SKIP {split} {fc}FC {count}/{len(active_indices)}", flush=True)
                        continue

                    clean_rgb = image_h5["images"][idx]
                    dense_vis = (
                        grid_h5["vis_re_dense"][:, idx] + 1j * grid_h5["vis_im_dense"][:, idx]
                    ).reshape(fc, fc)
                    sparse_vis_values = (
                        grid_h5["vis_re_sparse"][:, idx] + 1j * grid_h5["vis_im_sparse"][:, idx]
                    )
                    sparse_vis = sparse_to_grid(
                        u_dense, v_dense, u_sparse, v_sparse, sparse_vis_values, fc
                    )
                    clean_luma = rgb_to_luma(clean_rgb)
                    dense_image = visibility_to_image(dense_vis)
                    dirty_image = visibility_to_image(sparse_vis)

                    save_image(sample_dir / "image_clean_original.png", clean_rgb)
                    save_image(sample_dir / "image_clean_luma.png", clean_luma)
                    save_image(sample_dir / "image_clean_from_dense_visibility.png", dense_image)
                    save_image(sample_dir / "image_dirty_from_sparse_visibility.png", dirty_image)
                    save_visibility_map(sample_dir / "visibility_clean_dense_amplitude.png", dense_vis)
                    save_visibility_map(sample_dir / "visibility_dirty_sparse_amplitude.png", sparse_vis)
                    save_sparse_scatter(
                        sample_dir / "visibility_dirty_sparse_uv_scatter.png",
                        u_sparse,
                        v_sparse,
                        sparse_vis_values,
                    )
                    save_overview(
                        sample_dir / "overview.png",
                        clean_rgb,
                        dense_image,
                        dirty_image,
                        dense_vis,
                        sparse_vis,
                        u_sparse,
                        v_sparse,
                        sparse_vis_values,
                        f"{split} sample {idx} ({fc}FC)",
                    )
                    print(f"WROTE {split} {fc}FC {count}/{len(active_indices)} {sample_dir}", flush=True)


def main() -> int:
    args = parse_args()
    indices = None if args.all_indices else args.indices
    for split in args.splits:
        visualize_split(
            args.data_root,
            args.out_root,
            split,
            indices,
            args.num_fourier,
            skip_existing=args.skip_existing,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
