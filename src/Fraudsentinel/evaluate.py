"""
Evaluation utilities for FraudSentinel.

evaluate_model(model, data, mask, model_type)
    Computes Precision, Recall, F1-illicit, and PR-AUC for either the
    XGBoost or GraphSAGE model on any data split mask.
    Finds the threshold that maximises F1 on the given split.

compare_models(xgb_metrics, sage_metrics) -> pd.DataFrame
    Returns a clean side-by-side comparison table of both models.

plot_pr_curves(xgb_metrics, sage_metrics, save_path)
    Saves an overlaid Precision-Recall curve for both models.

Why F1-illicit and not accuracy?
    The dataset is ~10 % illicit / 90 % licit.
    A model predicting "licit" for everything achieves ~90 % accuracy
    but detects zero fraud. F1-illicit and PR-AUC are the only metrics
    that honestly reflect fraud-detection performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")   # headless — no display needed

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch_geometric.data import Data

from Fraudsentinel.logger import get_logger
from Fraudsentinel.models import (
    GATClassifier,
    GCNClassifier,
    GraphSAGEClassifier,
    XGBoostFraudClassifier,
)

logger = get_logger("FraudSentinel.Evaluate")


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate_model(
    model: XGBoostFraudClassifier | GraphSAGEClassifier | GCNClassifier | GATClassifier,
    data: Data,
    mask: torch.Tensor,
    model_type: Literal["xgboost", "graphsage", "gcn", "gat"],
    device: str | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    """
    Evaluate a model on the nodes selected by `mask`.

    Parameters
    ----------
    model      : Fitted classifier
    data       : PyG Data object
    mask       : Boolean mask selecting which nodes to evaluate
    model_type : "xgboost" | "graphsage" | "gcn" | "gat"
    device     : torch device string; auto-detected if None
    threshold  : Classification threshold. If None, we find the threshold
                 that maximises F1 on this split (so call with val_mask
                 to find threshold, then pass it to test_mask evaluation).

    Returns
    -------
    dict with keys: f1, precision, recall, pr_auc, threshold, n_samples
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # ── Get probabilities ──────────────────────────────────────────────
    if model_type == "xgboost":
        X = data.x[mask].cpu().numpy()
        y_true = data.y[mask].cpu().numpy()
        proba  = model.predict_proba(X)

    elif model_type in ("graphsage", "gcn", "gat"):
        model.eval()
        model = model.to(dev)
        proba, y_true = _get_sage_probas_fullbatch(model, data, mask, dev)

    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # ── Find optimal threshold (maximise F1) ───────────────────────────
    if threshold is None:
        threshold = _find_best_threshold(y_true, proba)
        logger.info(f"[{model_type}] Optimal threshold (max F1): {threshold:.3f}")

    y_pred = (proba >= threshold).astype(int)

    f1        = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    pr_auc    = average_precision_score(y_true, proba)

    metrics = {
        "f1":        round(f1, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "pr_auc":   round(pr_auc, 4),
        "threshold": round(threshold, 4),
        "n_samples": int(mask.sum().item()),
        # Store raw arrays for PR curve plotting
        "_proba":  proba,
        "_y_true": y_true,
    }

    logger.info(
        f"[{model_type}] n={metrics['n_samples']} | "
        f"F1={f1:.4f} | P={precision:.4f} | R={recall:.4f} | "
        f"PR-AUC={pr_auc:.4f} | threshold={threshold:.3f}"
    )
    return metrics


# ── Comparison table ──────────────────────────────────────────────────────────

def compare_models(
    xgb_metrics: dict,
    sage_metrics: dict,
    gcn_metrics: dict,
    gat_metrics: dict,
    split_name: str = "Test",
) -> pd.DataFrame:
    """
    Build a human-readable comparison DataFrame for all four models.

    Parameters
    ----------
    xgb_metrics  : Output of evaluate_model(..., model_type="xgboost")
    sage_metrics : Output of evaluate_model(..., model_type="graphsage")
    gcn_metrics  : Output of evaluate_model(..., model_type="gcn")
    gat_metrics  : Output of evaluate_model(..., model_type="gat")
    split_name   : Label for the split (e.g. "Validation", "Test")

    Returns
    -------
    pd.DataFrame with models as columns and metrics as rows
    """
    display_keys = ["f1", "precision", "recall", "pr_auc", "threshold", "n_samples"]

    df = pd.DataFrame(
        {
            "XGBoost Baseline": {k: xgb_metrics[k] for k in display_keys},
            "GraphSAGE (Max)": {k: sage_metrics[k] for k in display_keys},
            "GCN Baseline": {k: gcn_metrics[k] for k in display_keys},
            "GAT (Attention)": {k: gat_metrics[k] for k in display_keys},
        }
    )
    df.index.name = f"Metric ({split_name})"

    logger.info(f"\n{'='*65}\nModel Comparison ({split_name} Split)\n{'='*65}")
    logger.info(f"\n{df.to_string()}\n")
    return df


# ── PR curve plot ─────────────────────────────────────────────────────────────

def plot_pr_curves(
    xgb_metrics: dict,
    sage_metrics: dict,
    gcn_metrics: dict,
    gat_metrics: dict,
    save_path: str | Path = "reports/figures/pr_curves.png",
    split_name: str = "Validation",
) -> None:
    """
    Save overlaid Precision-Recall curves for all four benchmarked models.

    Parameters
    ----------
    xgb_metrics  : Output of evaluate_model
    sage_metrics : Output of evaluate_model
    gcn_metrics  : Output of evaluate_model
    gat_metrics  : Output of evaluate_model
    save_path    : Where to write the PNG
    split_name   : Displayed in the plot title
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, metrics, color, ls in [
        ("XGBoost Baseline", xgb_metrics, "#E07B39", "-"),
        ("GraphSAGE (Max)",  sage_metrics, "#4C78A8", "--"),
        ("GCN Baseline",     gcn_metrics,  "#55A868", "-."),
        ("GAT (Attention)",  gat_metrics,  "#C44E52", ":"),
    ]:
        prec, rec, _ = precision_recall_curve(
            metrics["_y_true"], metrics["_proba"], pos_label=1
        )
        auc = metrics["pr_auc"]
        ax.plot(
            rec, prec,
            label=f"{label} (AUC={auc:.3f})",
            color=color, linestyle=ls, linewidth=2.2
        )
        # Mark the chosen operating point
        ax.scatter(
            [metrics["recall"]], [metrics["precision"]],
            color=color, s=80, zorder=5
        )

    illicit_rate = float(xgb_metrics["_y_true"].mean())
    ax.axhline(
        illicit_rate, linestyle=":", color="grey", linewidth=1,
        label=f"Random classifier (P={illicit_rate:.2f})"
    )

    ax.set_xlabel("Recall (Illicit)", fontsize=13)
    ax.set_ylabel("Precision (Illicit)", fontsize=13)
    ax.set_title(
        f"Precision-Recall Curve -- {split_name} Split\n"
        f"Elliptic Bitcoin Fraud Detection",
        fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve saved to {save_path}")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _find_best_threshold(
    y_true: np.ndarray, proba: np.ndarray
) -> float:
    """
    Sweep thresholds in [0.1, 0.9] and return the one maximising F1-illicit.
    This is done on the *same* split to avoid leakage — caller is responsible
    for using the val split to find the threshold and the test split to report.
    """
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        preds = (proba >= t).astype(int)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


@torch.no_grad()
def _get_sage_probas_fullbatch(
    model: GraphSAGEClassifier,
    data: Data,
    mask: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full-graph forward pass -> (probabilities, true_labels) for masked nodes.
    Applies scaler to topological features only (cols 165+), matching training.
    """
    import pickle
    TOPO_START = 165
    x_np = data.x.cpu().numpy().copy()

    # Apply the same scaler used during training
    scaler = getattr(model, "scaler", None)
    if scaler is None:
        from pathlib import Path
        scaler_path = Path("models/feature_scaler.pkl")
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            logger.info("Loaded feature_scaler.pkl for GraphSAGE inference")
        else:
            logger.warning(
                "No feature scaler found. Using raw topological features."
            )

    if scaler is not None:
        x_np[:, TOPO_START:] = scaler.transform(x_np[:, TOPO_START:])

    x          = torch.tensor(x_np, dtype=torch.float).to(device)
    edge_index = data.edge_index.to(device)
    all_logits = model(x, edge_index).squeeze(-1)
    proba  = torch.sigmoid(all_logits[mask]).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    return proba, labels