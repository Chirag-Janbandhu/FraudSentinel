"""
FraudSentinel Semi-supervised Labels — Label Propagation Setup
===============================================================
Generates pseudo-labels using PyG LabelPropagation on graph nodes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from sklearn.metrics import average_precision_score
from torch_geometric.nn import LabelPropagation

from Fraudsentinel.evaluate import _find_best_threshold
from Fraudsentinel.logger import get_logger

logger = get_logger("FraudSentinel.LabelPropagation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FraudSentinel — Semi-supervised Label Propagation Setup"
    )
    parser.add_argument(
        "--num-layers", type=int, default=5,
        help="Number of propagation steps/layers (default: 5)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.9,
        help="Diffusion coefficient alpha in (0, 1) (default: 0.9)"
    )
    parser.add_argument(
        "--licit-threshold", type=float, default=0.95,
        help="Probability threshold to assign licit pseudo-label (default: 0.95)"
    )
    parser.add_argument(
        "--illicit-threshold", type=float, default=0.90,
        help="Probability threshold to assign illicit pseudo-label (default: 0.90)"
    )
    parser.add_argument(
        "--data-path", default="data/processed/graph_data.pt",
        help="Path to the saved PyG Data object"
    )
    parser.add_argument(
        "--output-path", default="data/processed/graph_with_pseudo.pt",
        help="Path to save the graph with generated pseudo-labels"
    )
    return parser.parse_args()


def evaluate_intra_timestep(
    data: any,
    num_layers: int,
    alpha: float
) -> tuple[dict, torch.Tensor]:
    """Runs Label Propagation restricted strictly to intra-timestep edges."""
    logger.info(f"Setting up PyG LabelPropagation (layers={num_layers}, alpha={alpha})...")
    lp = LabelPropagation(num_layers=num_layers, alpha=alpha)

    train_labeled = data.train_mask & data.labeled_mask
    mask_indices = torch.where(train_labeled)[0]

    num_nodes = data.num_nodes
    y_train = torch.full((num_nodes,), -1, dtype=torch.long)
    y_train[mask_indices] = data.y[mask_indices]

    logger.info(f"Running LabelPropagation over graph ({num_nodes:,} nodes, {data.num_edges:,} edges)...")
    out_probs = lp(y_train, data.edge_index, mask=train_labeled)

    val_mask = data.val_mask & data.labeled_mask
    val_y = data.y[val_mask].cpu().numpy()
    val_probs = out_probs[val_mask, 1].cpu().numpy()

    pr_auc_val = average_precision_score(val_y, val_probs)
    best_thresh, f1_val, prec_val, rec_val = _find_best_threshold(val_y, val_probs)

    logger.info(f"[VAL EVALUATION] PR-AUC: {pr_auc_val:.4f} | Max F1: {f1_val:.4f} @ thresh={best_thresh:.4f}")

    metrics = {
        "val_pr_auc": pr_auc_val,
        "val_best_f1": f1_val,
        "val_best_thresh": best_thresh,
        "val_prec": prec_val,
        "val_rec": rec_val
    }
    return metrics, out_probs


def generate_pseudo_labels(
    data: any,
    out_probs: torch.Tensor,
    licit_threshold: float,
    illicit_threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates high-confidence pseudo-labels for unlabeled training nodes."""
    unlabeled_train_mask = data.train_mask & (~data.labeled_mask)
    unlabeled_indices = torch.where(unlabeled_train_mask)[0]

    prob_illicit_unlabeled = out_probs[unlabeled_indices, 1]

    licit_pseudo_mask = prob_illicit_unlabeled <= (1.0 - licit_threshold)
    illicit_pseudo_mask = prob_illicit_unlabeled >= illicit_threshold

    licit_pseudo_indices = unlabeled_indices[licit_pseudo_mask]
    illicit_pseudo_indices = unlabeled_indices[illicit_pseudo_mask]

    pseudo_indices = torch.cat([licit_pseudo_indices, illicit_pseudo_indices], dim=0)
    pseudo_labels = torch.cat([
        torch.zeros(len(licit_pseudo_indices), dtype=torch.long),
        torch.ones(len(illicit_pseudo_indices), dtype=torch.long)
    ], dim=0)

    logger.info(
        f"Generated {len(pseudo_indices):,} pseudo-labels "
        f"({len(licit_pseudo_indices):,} Licit, {len(illicit_pseudo_indices):,} Illicit)."
    )
    return pseudo_indices, pseudo_labels


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)

    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        sys.exit(1)

    logger.info(f"Loading graph data from {data_path}...")
    data = torch.load(data_path, weights_only=False)

    metrics, out_probs = evaluate_intra_timestep(data, args.num_layers, args.alpha)
    pseudo_indices, pseudo_labels = generate_pseudo_labels(
        data, out_probs, args.licit_threshold, args.illicit_threshold
    )

    data.pseudo_indices = pseudo_indices
    data.pseudo_labels = pseudo_labels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
    logger.info(f"Saved updated graph with pseudo-labels to {output_path}")


if __name__ == "__main__":
    main()
