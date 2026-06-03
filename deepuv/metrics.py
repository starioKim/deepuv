from __future__ import annotations

import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def image_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    return {
        "mse": float(mean_squared_error(target, pred)),
        "mae": float(np.mean(np.abs(target - pred))),
        "psnr": float(peak_signal_noise_ratio(target, pred, data_range=1.0)),
        "ssim": float(structural_similarity(target, pred, data_range=1.0)),
    }


def lfd(pred_vis: np.ndarray, target_vis: np.ndarray) -> float:
    diff = np.asarray(pred_vis, dtype=np.float32) - np.asarray(target_vis, dtype=np.float32)
    freq_distance = diff[0] ** 2 + diff[1] ** 2
    return float(np.mean(np.log1p(freq_distance)))


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in rows[0] if isinstance(rows[0][key], float)]
    out: dict[str, float] = {}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std())
    return out

