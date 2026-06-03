#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    for name, metrics in rows:
        lines.append(
            "| {name} | {n:.0f} | {psnr:.4f} ± {psnr_std:.4f} | {ssim:.4f} ± {ssim_std:.4f} | "
            "{mse:.6f} ± {mse_std:.6f} | {mae:.6f} ± {mae_std:.6f} | {lfd:.6f} ± {lfd_std:.6f} |".format(
                name=name,
                n=metrics.get("n_test", 0.0),
                psnr=metrics.get("psnr_mean", float("nan")),
                psnr_std=metrics.get("psnr_std", float("nan")),
                ssim=metrics.get("ssim_mean", float("nan")),
                ssim_std=metrics.get("ssim_std", float("nan")),
                mse=metrics.get("mse_mean", float("nan")),
                mse_std=metrics.get("mse_std", float("nan")),
                mae=metrics.get("mae_mean", float("nan")),
                mae_std=metrics.get("mae_std", float("nan")),
                lfd=metrics.get("lfd_mean", float("nan")),
                lfd_std=metrics.get("lfd_std", float("nan")),
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
