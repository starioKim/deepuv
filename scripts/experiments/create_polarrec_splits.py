#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deepuv.polarrec_dataset import split_path, write_stratified_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits for PolarRec datasets.")
    parser.add_argument("--data-root", type=Path, default=Path("/datasets/deepuv/polarrec"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or split_path(args.data_root, args.seed, args.train, args.val, args.test)
    write_stratified_split(
        args.data_root,
        output,
        seed=args.seed,
        train=args.train,
        val=args.val,
        test=args.test,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
