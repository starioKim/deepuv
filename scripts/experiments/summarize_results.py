#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PolarRec reconstruction metrics as a Markdown table.")
    parser.add_argument("--results-root", type=Path, default=Path("results/polarrec"))
    parser.add_argument("--output", type=Path, default=Path("results/polarrec/comparison.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for path in sorted(args.results_root.glob("*/test_summary.json")):
        if "smoke" in path.parent.name:
            continue
        metrics = json.loads(path.read_text())
        rows.append((path.parent.name, metrics))
    lines = [
        "| method | n_test | PSNR ↑ | SSIM ↑ | MSE ↓ | MAE ↓ | LFD ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(mean: float, std: float, digits: int) -> str:
        if math.isnan(mean) or math.isnan(std):
            return "N/A"
        return f"{mean:.{digits}f} ± {std:.{digits}f}"

    for name, metrics in rows:
        lines.append(
            "| {name} | {n:.0f} | {psnr} | {ssim} | {mse} | {mae} | {lfd} |".format(
                name=name,
                n=metrics.get("n_test", 0.0),
                psnr=fmt(metrics.get("psnr_mean", float("nan")), metrics.get("psnr_std", float("nan")), 4),
                ssim=fmt(metrics.get("ssim_mean", float("nan")), metrics.get("ssim_std", float("nan")), 4),
                mse=fmt(metrics.get("mse_mean", float("nan")), metrics.get("mse_std", float("nan")), 6),
                mae=fmt(metrics.get("mae_mean", float("nan")), metrics.get("mae_std", float("nan")), 6),
                lfd=fmt(metrics.get("lfd_mean", float("nan")), metrics.get("lfd_std", float("nan")), 6),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
