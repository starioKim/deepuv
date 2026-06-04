#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from deepuv.metrics import aggregate, image_metrics, lfd
from deepuv.polarrec_dataset import PolarRecGridDataset, normalize_image_batch
from deepuv.uvdc_model import centered_ifft_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate no-retraining baselines on the PolarRec test split.")
    parser.add_argument("--method", choices=["zero_filled", "nearest_uv", "hogbom_clean"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/data/nfs/home/stario/datasets/deepuv/polarrec"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-fourier", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--clean-iters", type=int, default=80)
    parser.add_argument("--clean-gain", type=float, default=0.1)
    parser.add_argument("--clean-threshold", type=float, default=0.02)
    return parser.parse_args()


def raw_ifft_image(vis: torch.Tensor) -> torch.Tensor:
    z = torch.complex(vis[:, 0], vis[:, 1])
    z = torch.fft.ifftshift(z, dim=(-2, -1))
    image = torch.fft.ifft2(z, norm="ortho")
    return torch.fft.fftshift(image, dim=(-2, -1)).real[:, None]


def normalize_np(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo = float(image.min())
    hi = float(image.max())
    return (image - lo) / max(hi - lo, 1e-6)


class NearestUVFill:
    def __init__(self) -> None:
        self.cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def fill(self, measured: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        measured_np = measured.numpy()
        mask_np = mask[:, 0].numpy() > 0
        out = np.empty_like(measured_np)
        for i in range(measured_np.shape[0]):
            key = mask_np[i].tobytes()
            if key not in self.cache:
                _, indices = ndimage.distance_transform_edt(~mask_np[i], return_indices=True)
                self.cache[key] = (indices[0], indices[1])
            yy, xx = self.cache[key]
            out[i, 0] = measured_np[i, 0, yy, xx]
            out[i, 1] = measured_np[i, 1, yy, xx]
            out[i, :, mask_np[i]] = measured_np[i, :, mask_np[i]]
        return torch.from_numpy(out)


class HogbomClean:
    def __init__(self, *, iters: int, gain: float, threshold: float) -> None:
        self.iters = iters
        self.gain = gain
        self.threshold = threshold
        self.psf_cache: dict[bytes, np.ndarray] = {}

    def _psf(self, mask: torch.Tensor) -> np.ndarray:
        mask_np = mask.numpy().astype(np.float32)
        key = mask_np.tobytes()
        if key not in self.psf_cache:
            vis = torch.from_numpy(np.stack([mask_np, np.zeros_like(mask_np)], axis=0))[None]
            psf = raw_ifft_image(vis)[0, 0].numpy()
            peak = float(np.max(np.abs(psf)))
            self.psf_cache[key] = psf / max(peak, 1e-6)
        return self.psf_cache[key]

    def clean_one(self, measured: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        dirty = raw_ifft_image(measured[None])[0, 0].numpy()
        psf = self._psf(mask[0])
        residual = dirty.copy()
        model = np.zeros_like(dirty)
        center = np.array(psf.shape) // 2
        initial_peak = max(float(np.max(np.abs(residual))), 1e-6)
        for _ in range(self.iters):
            pos = np.unravel_index(int(np.argmax(np.abs(residual))), residual.shape)
            peak = float(residual[pos])
            if abs(peak) < self.threshold * initial_peak:
                break
            component = self.gain * peak
            model[pos] += component
            shift = (int(pos[0] - center[0]), int(pos[1] - center[1]))
            residual -= component * np.roll(psf, shift=shift, axis=(0, 1))
        return normalize_np(model + residual)

    def clean_batch(self, measured: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        images = [self.clean_one(measured[i], mask[i]) for i in range(measured.shape[0])]
        return np.stack(images, axis=0)[:, None]


@torch.no_grad()
def main() -> int:
    args = parse_args()
    split_file = args.split_file or args.data_root / "splits" / "polarrec_seed0_train70_val10_test20.json"
    output_dir = args.output_dir or Path(f"results/polarrec/{args.method}_128")
    dataset = PolarRecGridDataset(
        args.data_root,
        split_file,
        "test",
        num_fourier=args.num_fourier,
        max_samples=args.max_test_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")

    nearest = NearestUVFill()
    cleaner = HogbomClean(iters=args.clean_iters, gain=args.clean_gain, threshold=args.clean_threshold)
    rows: list[dict[str, float]] = []

    for batch in loader:
        if args.method == "zero_filled":
            pred_vis = batch["measured"]
            pred_image = centered_ifft_image(pred_vis).numpy()
            pred_vis_np = pred_vis.numpy()
        elif args.method == "nearest_uv":
            pred_vis = nearest.fill(batch["measured"], batch["mask"])
            pred_image = centered_ifft_image(pred_vis).numpy()
            pred_vis_np = pred_vis.numpy()
        else:
            pred_image = cleaner.clean_batch(batch["measured"], batch["mask"])
            pred_vis_np = None

        target_image = normalize_image_batch(batch["target_image"]).numpy()
        target_vis_np = batch["target_vis"].numpy()
        for i in range(pred_image.shape[0]):
            row = image_metrics(pred_image[i, 0], target_image[i, 0])
            row["lfd"] = lfd(pred_vis_np[i], target_vis_np[i]) if pred_vis_np is not None else float("nan")
            row["split"] = str(batch["split"][i])
            row["index"] = int(batch["index"][i])
            rows.append(row)

    summary = aggregate(rows)
    summary["n_test"] = float(len(rows))
    with (output_dir / "test_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "test_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
