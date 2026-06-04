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

from deepuv.baseline_models import EDSRImage, PolarNeuralField, R2D2ImageSeries, UNet2D
from deepuv.metrics import aggregate, image_metrics, lfd
from deepuv.polarrec_dataset import PolarRecGridDataset, normalize_image_batch
from deepuv.uvdc_model import centered_ifft_image


IMAGE_METHODS = {"leia_unet", "leia_gunet", "polish_edsr", "r2d2_series"}
VIS_METHODS = {"vis_unet", "polarrec_nf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate adapted PolarRec baselines on the fixed split.")
    parser.add_argument("--method", choices=sorted(IMAGE_METHODS | VIS_METHODS), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/data/nfs/home/stario/datasets/deepuv/polarrec"))
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-fourier", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-weight", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--eval-only", type=Path, default=None)
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


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def build_model(method: str) -> torch.nn.Module:
    if method == "vis_unet":
        return UNet2D(3, 2, base_channels=48, depth=3)
    if method == "leia_unet":
        return UNet2D(1, 1, base_channels=32, depth=3)
    if method == "leia_gunet":
        return UNet2D(2, 1, base_channels=32, depth=3)
    if method == "polish_edsr":
        return EDSRImage(in_channels=1, channels=64, blocks=8)
    if method == "r2d2_series":
        return R2D2ImageSeries(steps=6, channels=48)
    if method == "polarrec_nf":
        return PolarNeuralField(fourier_bands=8, token_dim=128, context_dim=128, transformer_layers=2)
    raise ValueError(f"unknown method {method}")


def dirty_image(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return centered_ifft_image(batch["measured"])


def psf_image(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return centered_ifft_image(torch.cat([batch["mask"], torch.zeros_like(batch["mask"])], dim=1))


def forward_model(model: torch.nn.Module, method: str, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if method == "vis_unet":
        pred_vis = model(torch.cat([batch["measured"], batch["mask"]], dim=1))
        return {"pred_vis": pred_vis, "pred_image": centered_ifft_image(pred_vis)}
    if method == "polarrec_nf":
        bsz = batch["target_vis"].shape[0]
        fc = batch["target_vis"].shape[-1]
        pred_flat = model(batch["sparse_uv"], batch["sparse_vis"], batch["dense_uv"])
        pred_vis = pred_flat.transpose(1, 2).reshape(bsz, 2, fc, fc)
        return {"pred_vis": pred_vis, "pred_image": centered_ifft_image(pred_vis)}
    if method == "leia_unet":
        pred_image = torch.sigmoid(model(dirty_image(batch)))
        return {"pred_image": pred_image}
    if method == "leia_gunet":
        pred_image = torch.sigmoid(model(torch.cat([dirty_image(batch), psf_image(batch)], dim=1)))
        return {"pred_image": pred_image}
    if method == "polish_edsr":
        return {"pred_image": model(dirty_image(batch))}
    if method == "r2d2_series":
        return {"pred_image": model(dirty_image(batch))}
    raise ValueError(f"unknown method {method}")


def loss_fn(outputs: dict[str, torch.Tensor], method: str, batch: dict[str, torch.Tensor], image_weight: float) -> torch.Tensor:
    target_image = normalize_image_batch(batch["target_image"])
    if method in VIS_METHODS:
        vis_loss = F.smooth_l1_loss(outputs["pred_vis"], batch["target_vis"])
        image_loss = F.l1_loss(outputs["pred_image"], target_image)
        return vis_loss + image_weight * image_loss
    return F.l1_loss(outputs["pred_image"], target_image) + F.mse_loss(outputs["pred_image"], target_image)


@torch.no_grad()
def validate(model: torch.nn.Module, args: argparse.Namespace, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = forward_model(model, args.method, batch)
        loss = loss_fn(outputs, args.method, batch, args.image_weight)
        total += float(loss.item()) * batch["target_image"].shape[0]
        count += batch["target_image"].shape[0]
    return total / max(count, 1)


def train(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    device = torch.device(args.device)
    model = build_model(args.method).to(device)
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
                outputs = forward_model(model, args.method, batch)
                loss = loss_fn(outputs, args.method, batch, args.image_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                n = batch["target_image"].shape[0]
                running += float(loss.item()) * n
                seen += n
            val_loss = validate(model, args, val_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": running / max(seen, 1),
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
    model = build_model(args.method).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    test_loader = make_loader(args, "test", args.max_test_samples, False)
    rows: list[dict[str, float]] = []
    for batch in test_loader:
        batch = move_batch(batch, device)
        outputs = forward_model(model, args.method, batch)
        pred_image = outputs["pred_image"].detach().cpu().numpy()
        target_image = normalize_image_batch(batch["target_image"]).detach().cpu().numpy()
        pred_vis = outputs.get("pred_vis")
        pred_vis_np = pred_vis.detach().cpu().numpy() if pred_vis is not None else None
        target_vis_np = batch["target_vis"].detach().cpu().numpy()
        for i in range(pred_image.shape[0]):
            row = image_metrics(pred_image[i, 0], target_image[i, 0])
            row["lfd"] = lfd(pred_vis_np[i], target_vis_np[i]) if pred_vis_np is not None else float("nan")
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
