from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SPLITS = ("MG", "IRSG", "UTSG", "EGB")


@dataclass(frozen=True)
class PolarRecSample:
    split: str
    index: int


def split_path(data_root: Path, seed: int, train: float, val: float, test: float) -> Path:
    tag = f"seed{seed}_train{int(train * 100):02d}_val{int(val * 100):02d}_test{int(test * 100):02d}"
    return data_root / "splits" / f"polarrec_{tag}.json"


def write_stratified_split(
    data_root: Path,
    output_path: Path,
    *,
    seed: int = 0,
    train: float = 0.7,
    val: float = 0.1,
    test: float = 0.2,
) -> None:
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1")

    rng = np.random.default_rng(seed)
    payload: dict[str, object] = {
        "seed": seed,
        "fractions": {"train": train, "val": val, "test": test},
        "items": {"train": [], "val": [], "test": []},
        "counts": {},
    }
    counts: dict[str, dict[str, int]] = {}
    items = payload["items"]
    assert isinstance(items, dict)

    for split in SPLITS:
        image_path = data_root / "subsets" / f"Galaxy10_DECals_{split}.h5"
        with h5py.File(image_path, "r") as handle:
            n_samples = int(handle["images"].shape[0])
        order = rng.permutation(n_samples)
        n_train = int(round(n_samples * train))
        n_val = int(round(n_samples * val))
        buckets = {
            "train": order[:n_train],
            "val": order[n_train : n_train + n_val],
            "test": order[n_train + n_val :],
        }
        counts[split] = {name: int(len(indices)) for name, indices in buckets.items()}
        for name, indices in buckets.items():
            items[name].extend({"split": split, "index": int(idx)} for idx in sorted(indices.tolist()))

    payload["counts"] = counts
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_samples(path: Path, subset: str) -> list[PolarRecSample]:
    payload = json.loads(path.read_text())
    return [PolarRecSample(item["split"], int(item["index"])) for item in payload["items"][subset]]


def luma_from_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    return 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]


def normalize_image_batch(image: torch.Tensor) -> torch.Tensor:
    flat = image.flatten(1)
    lo = flat.min(dim=1).values[:, None, None, None]
    hi = flat.max(dim=1).values[:, None, None, None]
    return (image - lo) / (hi - lo).clamp_min(1e-6)


class PolarRecGridDataset(Dataset):
    """PolarRec grid visibility dataset with gridded sparse observations."""

    def __init__(
        self,
        data_root: Path,
        split_file: Path,
        subset: str,
        *,
        num_fourier: int = 128,
        max_samples: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.num_fourier = int(num_fourier)
        self.samples = load_samples(split_file, subset)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        self._image_h5: dict[str, h5py.File] = {}
        self._grid_h5: dict[str, h5py.File] = {}
        self._sparse_indices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._uv_coords_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def close(self) -> None:
        for handle in [*self._image_h5.values(), *self._grid_h5.values()]:
            handle.close()
        self._image_h5.clear()
        self._grid_h5.clear()

    def _image_handle(self, split: str) -> h5py.File:
        if split not in self._image_h5:
            path = self.data_root / "subsets" / f"Galaxy10_DECals_{split}.h5"
            self._image_h5[split] = h5py.File(path, "r")
        return self._image_h5[split]

    def _grid_handle(self, split: str) -> h5py.File:
        if split not in self._grid_h5:
            path = self.data_root / f"eht_grid_{self.num_fourier}FC_200im_Galaxy10_DECals_{split}.h5"
            self._grid_h5[split] = h5py.File(path, "r")
        return self._grid_h5[split]

    def _indices(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        if split not in self._sparse_indices:
            handle = self._grid_handle(split)
            u_dense = handle["u_dense"][:]
            v_dense = handle["v_dense"][:]
            u_sparse = handle["u_sparse"][:]
            v_sparse = handle["v_sparse"][:]
            u_vals = np.unique(u_dense)
            v_vals = np.unique(v_dense)
            x_idx = np.abs(u_sparse[:, None] - u_vals[None, :]).argmin(axis=1)
            y_idx = np.abs(v_sparse[:, None] - v_vals[None, :]).argmin(axis=1)
            self._sparse_indices[split] = (y_idx.astype(np.int64), x_idx.astype(np.int64))
        return self._sparse_indices[split]

    def _uv_coords(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        if split in self._uv_coords_cache:
            return self._uv_coords_cache[split]
        handle = self._grid_handle(split)
        fc = self.num_fourier
        u_dense = handle["u_dense"][:].astype(np.float32).reshape(fc, fc)
        v_dense = handle["v_dense"][:].astype(np.float32).reshape(fc, fc)
        max_base = float(max(np.abs(u_dense).max(), np.abs(v_dense).max(), 1.0))
        dense_uv = np.stack([u_dense.reshape(-1), v_dense.reshape(-1)], axis=-1) / (2.0 * max_base)
        y_idx, x_idx = self._indices(split)
        sparse_uv = np.stack([u_dense[y_idx, x_idx], v_dense[y_idx, x_idx]], axis=-1) / (2.0 * max_base)
        coords = (sparse_uv.astype(np.float32), dense_uv.astype(np.float32))
        self._uv_coords_cache[split] = coords
        return coords

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.samples[item]
        grid_h5 = self._grid_handle(sample.split)
        image_h5 = self._image_handle(sample.split)
        fc = self.num_fourier

        dense_re = grid_h5["vis_re_dense"][:, sample.index].reshape(fc, fc).astype(np.float32)
        dense_im = grid_h5["vis_im_dense"][:, sample.index].reshape(fc, fc).astype(np.float32)
        sparse_re = grid_h5["vis_re_sparse"][:, sample.index].astype(np.float32)
        sparse_im = grid_h5["vis_im_sparse"][:, sample.index].astype(np.float32)
        y_idx, x_idx = self._indices(sample.split)
        sparse_uv, dense_uv = self._uv_coords(sample.split)

        measured = np.zeros((2, fc, fc), dtype=np.float32)
        mask = np.zeros((1, fc, fc), dtype=np.float32)
        measured[0, y_idx, x_idx] = sparse_re
        measured[1, y_idx, x_idx] = sparse_im
        mask[0, y_idx, x_idx] = 1.0

        dense = np.stack([dense_re, dense_im], axis=0)
        scale = np.percentile(np.abs(sparse_re + 1j * sparse_im), 99).astype(np.float32)
        scale = float(max(scale, 1e-6))
        measured /= scale
        dense /= scale
        sparse_vis = np.stack([sparse_re / scale, sparse_im / scale], axis=-1)

        image = luma_from_rgb_uint8(image_h5["images"][sample.index])
        image_t = torch.from_numpy(image)[None, None]
        if image_t.shape[-1] != fc:
            image_t = torch.nn.functional.interpolate(image_t, size=(fc, fc), mode="bilinear", align_corners=False)
        image_t = image_t[0]

        return {
            "measured": torch.from_numpy(measured),
            "mask": torch.from_numpy(mask),
            "target_vis": torch.from_numpy(dense),
            "target_image": image_t.float(),
            "sparse_uv": torch.from_numpy(sparse_uv),
            "sparse_vis": torch.from_numpy(sparse_vis.astype(np.float32)),
            "dense_uv": torch.from_numpy(dense_uv),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "split": sample.split,
            "index": sample.index,
        }
