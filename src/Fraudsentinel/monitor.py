"""
monitor.py — Prediction Logging & Probability Shift Monitoring
================================================================
Lightweight inference-time monitoring utilities for FraudSentinel.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score

from Fraudsentinel.logger import get_logger

logger = get_logger("FraudSentinel.Monitor")

DEFAULT_LOG_PATH = Path("reports/prediction_log.csv")


def _to_numpy(arr: np.ndarray | torch.Tensor | Sequence) -> np.ndarray:
    """Convert input array or torch.Tensor to 1D numpy float array."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float64).ravel()
    return arr


def log_prediction_batch(
    probas: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor | None = None,
    batch_id: str | None = None,
    eval_threshold: float = 0.5,
    save_path: str | Path = DEFAULT_LOG_PATH,
) -> dict[str, float | str | int]:
    """Log summary statistics for a batch of predicted probabilities and append to CSV."""
    p = _to_numpy(probas)
    batch_size = len(p)
    if batch_size == 0:
        raise ValueError("Cannot log empty prediction batch.")

    mean_p = float(np.mean(p))
    std_p = float(np.std(p))
    min_p = float(np.min(p))
    max_p = float(np.max(p))

    prec: float | None = None
    rec: float | None = None

    if y_true is not None:
        y = _to_numpy(y_true)
        valid_mask = y >= 0
        if np.any(valid_mask):
            y_valid = y[valid_mask].astype(int)
            p_valid = p[valid_mask]
            preds = (p_valid >= eval_threshold).astype(int)
            prec = float(precision_score(y_valid, preds, zero_division=0))
            rec = float(recall_score(y_valid, preds, zero_division=0))

    timestamp = datetime.now(timezone.utc).isoformat()
    if batch_id is None:
        batch_id = f"batch_{timestamp[:19]}"

    record = {
        "timestamp": timestamp,
        "batch_id": batch_id,
        "batch_size": batch_size,
        "mean_proba": round(mean_p, 4),
        "std_proba": round(std_p, 4),
        "min_proba": round(min_p, 4),
        "max_proba": round(max_p, 4),
        "precision": round(prec, 4) if prec is not None else np.nan,
        "recall": round(rec, 4) if rec is not None else np.nan,
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df_row = pd.DataFrame([record])
    if save_path.exists():
        df_row.to_csv(save_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(save_path, mode="w", header=True, index=False)

    msg = (
        f"Logged batch '{batch_id}' (n={batch_size}): "
        f"mean_p={mean_p:.4f}, std_p={std_p:.4f}, range=[{min_p:.4f}, {max_p:.4f}]"
    )
    if prec is not None and rec is not None:
        msg += f" | prec={prec:.4f}, rec={rec:.4f}"
    logger.info(msg)

    return record


def check_probability_shift(
    probas: np.ndarray | torch.Tensor,
    ref_mean: float,
    threshold_delta: float = 0.10,
    batch_id: str | None = None,
) -> tuple[bool, float]:
    """Compare batch mean predicted probability against reference baseline and flag shifts."""
    p = _to_numpy(probas)
    batch_mean = float(np.mean(p))
    delta = abs(batch_mean - ref_mean)

    shifted = delta > threshold_delta
    id_str = f" [{batch_id}]" if batch_id else ""

    if shifted:
        logger.warning(
            f"PROBABILITY DRIFT WARNING{id_str}: Mean predicted probability shift "
            f"|{batch_mean:.4f} - {ref_mean:.4f}| = {delta:.4f} exceeds threshold "
            f"({threshold_delta:.4f}). Threshold recalibration recommended."
        )
    else:
        logger.info(
            f"Probability check PASSED{id_str}: Shift delta = {delta:.4f} "
            f"<= threshold {threshold_delta:.4f} (ref mean = {ref_mean:.4f}, "
            f"batch mean = {batch_mean:.4f})."
        )

    return shifted, delta
