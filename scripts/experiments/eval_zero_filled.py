#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from deepuv.metrics import aggregate, image_metrics, lfd
from deepuv.polarrec_dataset import PolarRecGridDataset, normalize_image_batch
from deepuv.uvdc_model import centered_ifft_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate zero-filled sparse UV baseline on PolarRec.")
    parser.add_argument("--data-root", type=Path, default=Path("/data/nfs/home/stario/datasets/deepuv/polarrec"))
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/polarrec/zero_filled_128"))
    parser.add_argument("--num-fourier", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    split_file = args.split_file or args.data_root / "splits" / "polarrec_seed0_train70_val10_test20.json"
    dataset = PolarRecGridDataset(args.data_root, split_file, "test", num_fourier=args.num_fourier)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    for batch in loader:
        pred_vis = batch["measured"]
        pred_image = centered_ifft_image(pred_vis).numpy()
        target_image = normalize_image_batch(batch["target_image"]).numpy()
        pred_vis_np = pred_vis.numpy()
        target_vis_np = batch["target_vis"].numpy()
        for i in range(pred_vis_np.shape[0]):
            row = image_metrics(pred_image[i, 0], target_image[i, 0])
            row["lfd"] = lfd(pred_vis_np[i], target_vis_np[i])
            row["split"] = str(batch["split"][i])
            row["index"] = int(batch["index"][i])
            rows.append(row)

    summary = aggregate(rows)
    summary["n_test"] = float(len(rows))
    with (args.output_dir / "test_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "test_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

