#!/usr/bin/env python3
"""Prepare the public Galaxy10 DECaLS dataset for PolarRec experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import h5py
import numpy as np


DATA_ROOT = Path("/datasets/deepuv/polarrec")
SOURCE_URL = "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5"
SOURCE_SHA256 = "19aefc477c41bb7f77ff07599a6b82a038dc042f889a111b0d4d98bb755c1571"
SOURCE_FILE = "Galaxy10_DECals.h5"

MORPHOLOGY_SPLITS = {
    "MG": {
        "class_id": 1,
        "name": "Merging Galaxies",
    },
    "IRSG": {
        "class_id": 3,
        "name": "In-between Round Smooth Galaxies",
    },
    "UTSG": {
        "class_id": 6,
        "name": "Unbarred Tight Spiral Galaxies",
    },
    "EGB": {
        "class_id": 9,
        "name": "Edge-on Galaxies with Bulge",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Directory for PolarRec dataset files.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download Galaxy10_DECals.h5; require it to already exist.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload Galaxy10_DECals.h5 even if it already exists.",
    )
    parser.add_argument(
        "--force-subsets",
        action="store_true",
        help="Regenerate morphology subset HDF5 files.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")
    start_byte = part_path.stat().st_size if part_path.exists() else 0
    request = urllib.request.Request(url)
    if start_byte:
        request.add_header("Range", f"bytes={start_byte}-")
        mode = "ab"
        print(f"Resuming {dest.name} from {start_byte / (1024**2):.1f} MiB")
    else:
        mode = "wb"

    start = time.time()
    last_report = start
    downloaded = start_byte
    with urllib.request.urlopen(request) as response, part_path.open(mode) as out:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            out.write(block)
            downloaded += len(block)
            now = time.time()
            if now - last_report >= 30:
                speed = (downloaded - start_byte) / max(now - start, 1)
                print(
                    f"Downloaded {downloaded / (1024**2):.1f} MiB "
                    f"({speed / (1024**2):.2f} MiB/s)",
                    flush=True,
                )
                last_report = now

    os.replace(part_path, dest)
    elapsed = time.time() - start
    size_gb = dest.stat().st_size / (1024**3)
    print(f"Downloaded {dest} ({size_gb:.2f} GiB) in {elapsed / 60:.1f} min")


def ensure_source_file(data_root: Path, skip_download: bool, force_download: bool) -> Path:
    source_path = data_root / SOURCE_FILE
    if source_path.exists() and not force_download:
        print(f"Using existing {source_path}")
    elif skip_download:
        raise FileNotFoundError(f"{source_path} does not exist and --skip-download was set")
    else:
        print(f"Downloading {SOURCE_URL}")
        download_file(SOURCE_URL, source_path)

    actual_sha = sha256_file(source_path)
    if actual_sha.lower() != SOURCE_SHA256:
        raise ValueError(
            f"SHA256 mismatch for {source_path}: expected {SOURCE_SHA256}, got {actual_sha}"
        )
    print(f"Verified SHA256 for {source_path}")
    return source_path


def copy_selected_rows(src: h5py.File, dst: h5py.File, indices: np.ndarray) -> None:
    for key in src.keys():
        item = src[key]
        if not isinstance(item, h5py.Dataset):
            continue

        data = item[indices] if item.shape and item.shape[0] == len(src["ans"]) else item[()]
        dst.create_dataset(
            key,
            data=data,
            compression="gzip" if getattr(data, "ndim", 0) > 0 else None,
            compression_opts=4 if getattr(data, "ndim", 0) > 0 else None,
        )

        for attr_key, attr_value in item.attrs.items():
            dst[key].attrs[attr_key] = attr_value


def write_subsets(source_path: Path, data_root: Path, force: bool) -> dict[str, dict[str, object]]:
    subset_root = data_root / "subsets"
    subset_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}

    with h5py.File(source_path, "r") as src:
        labels = np.asarray(src["ans"])
        for split, meta in MORPHOLOGY_SPLITS.items():
            class_id = int(meta["class_id"])
            out_path = subset_root / f"Galaxy10_DECals_{split}.h5"
            indices = np.flatnonzero(labels == class_id)

            if out_path.exists() and not force:
                print(f"Keeping existing {out_path}")
            else:
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                if tmp_path.exists():
                    tmp_path.unlink()
                with h5py.File(tmp_path, "w") as dst:
                    copy_selected_rows(src, dst, indices)
                    dst.attrs["source_file"] = str(source_path)
                    dst.attrs["source_url"] = SOURCE_URL
                    dst.attrs["split"] = split
                    dst.attrs["class_id"] = class_id
                    dst.attrs["class_name"] = str(meta["name"])
                os.replace(tmp_path, out_path)
                print(f"Wrote {out_path} ({len(indices)} samples)")

            manifest[split] = {
                "class_id": class_id,
                "class_name": meta["name"],
                "samples": int(len(indices)),
                "path": str(out_path),
            }

    return manifest


def write_layout_files(data_root: Path, source_path: Path, subsets: dict[str, dict[str, object]]) -> None:
    manifest = {
        "dataset": "Galaxy10 DECaLS",
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "source_path": str(source_path),
        "paper": "PolarRec: Improving Radio Interferometric Data Reconstruction Using Polar Coordinates",
        "paper_url": "https://arxiv.org/abs/2308.14610",
        "visibility_files": {
            "continuous_sparse": str(data_root / "eht_cont_200im_Galaxy10_DECals_full.h5"),
            "grid_sparse_dense": str(data_root / "eht_grid_128FC_200im_Galaxy10_DECals_full.h5"),
        },
        "morphology_subsets": subsets,
    }
    manifest_path = data_root / "polarrec_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


def main() -> int:
    args = parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    source_path = ensure_source_file(args.data_root, args.skip_download, args.force_download)
    subsets = write_subsets(source_path, args.data_root, args.force_subsets)
    write_layout_files(args.data_root, source_path, subsets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
