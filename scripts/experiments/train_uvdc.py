#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deepuv.metrics import aggregate, image_metrics, lfd
from deepuv.polarrec_dataset import PolarRecGridDataset, normalize_image_batch
from deepuv.uvdc_model import UVDCNet, centered_ifft_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate an unrolled UV data-consistency model on PolarRec.")
    parser.add_argument("--data-root", type=Path, default=Path("/datasets/deepuv/polarrec"))
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/polarrec/uvdc_128"))
    parser.add_argument("--num-fourier", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stages", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--blocks-per-stage", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--eval-only", type=Path, default=None, help="Checkpoint path to evaluate without training.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_loader(args: argparse.Namespace, subset: str, max_samples: int | None, shuffle: bool) -> DataLoader:
    dataset = PolarRecGridDataset(
        args.data_root,
        args.split_file,
        subset,
        num_fourier=args.num_fourier,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.workers > 0,
    )


def loss_fn(pred_vis: torch.Tensor, batch: dict[str, torch.Tensor], image_weight: float = 0.25) -> torch.Tensor:
    target_vis = batch["target_vis"]
    vis_loss = F.smooth_l1_loss(pred_vis, target_vis)
    pred_image = centered_ifft_image(pred_vis)
    target_image = normalize_image_batch(batch["target_image"])
    image_loss = F.l1_loss(pred_image, target_image)
    return vis_loss + image_weight * image_loss


@torch.no_grad()
def validate(model: UVDCNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch["measured"], batch["mask"])
        loss = loss_fn(pred, batch)
        total += float(loss.item()) * pred.shape[0]
        count += pred.shape[0]
    return total / max(count, 1)


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def train(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    device = torch.device(args.device)
    model = UVDCNet(
        stages=args.stages,
        hidden_channels=args.hidden_channels,
        blocks_per_stage=args.blocks_per_stage,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(args, "train", args.max_train_samples, True)
    val_loader = make_loader(args, "val", args.max_val_samples, False)
    history_path = args.output_dir / "history.csv"
    best_path = args.output_dir / "best.pt"
    best_val = math.inf

    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "seconds"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            tic = time.time()
            model.train()
            running = 0.0
            seen = 0
            for batch in train_loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(batch["measured"], batch["mask"])
                loss = loss_fn(pred, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running += float(loss.item()) * pred.shape[0]
                seen += pred.shape[0]
            train_loss = running / max(seen, 1)
            val_loss = validate(model, val_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "seconds": time.time() - tic,
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row), flush=True)
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, args.output_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, best_path)
    return best_path


@torch.no_grad()
def evaluate(args: argparse.Namespace, checkpoint_path: Path) -> dict[str, float]:
    device = torch.device(args.device)
    model = UVDCNet(
        stages=args.stages,
        hidden_channels=args.hidden_channels,
        blocks_per_stage=args.blocks_per_stage,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    test_loader = make_loader(args, "test", args.max_test_samples, False)
    rows: list[dict[str, float]] = []
    pred_dir = args.output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for batch in test_loader:
        batch = move_batch(batch, device)
        pred_vis = model(batch["measured"], batch["mask"])
        pred_image = centered_ifft_image(pred_vis).cpu().numpy()
        target_image = normalize_image_batch(batch["target_image"]).cpu().numpy()
        pred_vis_np = pred_vis.cpu().numpy()
        target_vis_np = batch["target_vis"].cpu().numpy()
        for i in range(pred_vis_np.shape[0]):
            row = image_metrics(pred_image[i, 0], target_image[i, 0])
            row["lfd"] = lfd(pred_vis_np[i], target_vis_np[i])
            row["split"] = str(batch["split"][i])
            row["index"] = int(batch["index"][i])
            rows.append(row)

    metrics = aggregate(rows)
    metrics["n_test"] = float(len(rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "test_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "test_summary.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return metrics


def main() -> int:
    args = parse_args()
    checkpoint = args.eval_only or train(args)
    evaluate(args, Path(checkpoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
