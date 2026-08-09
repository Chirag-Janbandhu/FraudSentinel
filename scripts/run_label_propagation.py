"""
FraudSentinel Semi-supervised Labels (Part 1) — Label Propagation Setup
======================================================================
Usage:
    py scripts/run_label_propagation.py --num-layers 5 --alpha 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from torch_geometric.nn import LabelPropagation

from Fraudsentinel.logger import get_logger
from Fraudsentinel.evaluate import _find_best_threshold

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
    split_mask: torch.Tensor,
    num_layers: int,
    alpha: float,
    seed: int = 42
) -> dict[str, float]:
    """
    Evaluates Label Propagation accuracy by performing a random 80/20 split
    on labeled nodes *within each timestep* of the selected split.
    This respects the disconnected temporal subgraph structure of the dataset.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Find labeled nodes in this split
    labeled_split_mask = split_mask & data.labeled_mask
    labeled_indices = torch.where(labeled_split_mask)[0].cpu().numpy()

    # If too few nodes, return dummy metrics
    if len(labeled_indices) < 10:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "pr_auc": 0.0}

    # Group labeled indices by timestep
    timesteps = data.time_step[labeled_split_mask].cpu().numpy()
    unique_timesteps = np.unique(timesteps)

    train_idx_list = []
    eval_idx_list = []

    for t in unique_timesteps:
        t_mask = (data.time_step == t) & labeled_split_mask
        t_indices = torch.where(t_mask)[0].cpu().numpy()
        
        # Shuffle and split 80% / 20%
        np.random.shuffle(t_indices)
        split_point = int(0.8 * len(t_indices))
        train_idx_list.extend(t_indices[:split_point])
        eval_idx_list.extend(t_indices[split_point:])

    train_indices = torch.tensor(train_idx_list, dtype=torch.long)
    eval_indices = torch.tensor(eval_idx_list, dtype=torch.long)

    # Prepare labels and mask for LP
    y_lp = data.y.clone()
    y_lp[y_lp == -1] = 0

    # Propagation source is all training labels plus the 80% train split nodes
    prop_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    # If evaluating validation/test, we can also use training labels as context
    if not torch.equal(split_mask, data.train_mask):
        prop_mask[data.train_mask & data.labeled_mask] = True
    prop_mask[train_indices] = True

    lp_model = LabelPropagation(num_layers=num_layers, alpha=alpha)
    out_probs = lp_model(y_lp, data.edge_index, mask=prop_mask)

    # Evaluate on the 20% evaluation nodes
    y_true = data.y[eval_indices].cpu().numpy()
    proba = out_probs[eval_indices, 1].cpu().numpy()

    best_threshold = _find_best_threshold(y_true, proba)
    preds = (proba >= best_threshold).astype(int)

    f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
    precision = precision_score(y_true, preds, pos_label=1, zero_division=0)
    recall = recall_score(y_true, preds, pos_label=1, zero_division=0)
    pr_auc = average_precision_score(y_true, proba)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
        "threshold": best_threshold
    }


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Graph data not found at {data_path}. Run graph_construction.py first.")
        sys.exit(1)

    logger.info(f"Loading graph data from {data_path}...")
    data = torch.load(data_path, weights_only=False)

    logger.info("Setting up label propagation inputs...")
    y_lp = data.y.clone()
    # Map -1 (unlabeled/unknown) to 0 temporarily so one_hot doesn't crash on negative values.
    # The actual unlabeled values are ignored during propagation since we specify the mask.
    y_lp[y_lp == -1] = 0

    # To label the unlabeled ~77% across ALL timesteps, we clamp all known ground-truth labels.
    # Because there are zero cross-timestep edges, label propagation will run in parallel
    # within each independent timestep subgraph, diffusing labels from the labeled nodes of
    # each timestep to the unlabeled nodes of that same timestep.
    full_labeled_mask = data.labeled_mask
    
    logger.info(
        f"Running LabelPropagation(num_layers={args.num_layers}, alpha={args.alpha}) "
        f"clamping all {full_labeled_mask.sum().item()} known labels..."
    )
    lp_model = LabelPropagation(num_layers=args.num_layers, alpha=args.alpha)
    out_probs = lp_model(y_lp, data.edge_index, mask=full_labeled_mask)

    # ── Evaluate Label Propagation as a Classifier ───────────────────────────
    logger.info("Evaluating Label Propagation accuracy (intra-timestep 80/20 splits)...")
    
    # 1. Validation split evaluation
    val_metrics = evaluate_intra_timestep(data, data.val_mask, args.num_layers, args.alpha)
    logger.info(
        f"Validation | F1: {val_metrics['f1']:.4f} | Precision: {val_metrics['precision']:.4f} | "
        f"Recall: {val_metrics['recall']:.4f} | PR-AUC: {val_metrics['pr_auc']:.4f} | "
        f"Optimal Threshold: {val_metrics['threshold']:.3f}"
    )

    # 2. Test split evaluation
    test_metrics = evaluate_intra_timestep(data, data.test_mask, args.num_layers, args.alpha)
    logger.info(
        f"Test Set   | F1: {test_metrics['f1']:.4f} | Precision: {test_metrics['precision']:.4f} | "
        f"Recall: {test_metrics['recall']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f} | "
        f"Optimal Threshold: {test_metrics['threshold']:.3f}"
    )

    # ── Generate Pseudo-Labels for the Unlabeled Nodes ───────────────────────
    logger.info("Generating pseudo-labels for unlabeled nodes...")
    
    unlabeled_mask = ~data.labeled_mask
    total_unlabeled = unlabeled_mask.sum().item()

    proba_licit = out_probs[:, 0]
    proba_illicit = out_probs[:, 1]

    # Mask of high confidence predictions on unlabeled nodes
    pseudo_licit_mask = unlabeled_mask & (proba_licit >= args.licit_threshold)
    pseudo_illicit_mask = unlabeled_mask & (proba_illicit >= args.illicit_threshold)

    num_licit_pseudo = pseudo_licit_mask.sum().item()
    num_illicit_pseudo = pseudo_illicit_mask.sum().item()
    num_unlabeled_left = total_unlabeled - num_licit_pseudo - num_illicit_pseudo

    logger.info(f"Unlabeled Nodes Breakdown (Total={total_unlabeled}):")
    logger.info(f"  - Pseudo-labeled Licit (Prob >= {args.licit_threshold:.2f}): {num_licit_pseudo} ({num_licit_pseudo/total_unlabeled*100:.2f}%)")
    logger.info(f"  - Pseudo-labeled Illicit (Prob >= {args.illicit_threshold:.2f}): {num_illicit_pseudo} ({num_illicit_pseudo/total_unlabeled*100:.2f}%)")
    logger.info(f"  - Unresolved (confidence too low): {num_unlabeled_left} ({num_unlabeled_left/total_unlabeled*100:.2f}%)")

    # Construct final pseudo-labels fields
    pseudo_y = torch.full_like(data.y, -1)
    # 1. Copy ground-truth labels
    pseudo_y[data.labeled_mask] = data.y[data.labeled_mask]
    # 2. Insert high-confidence pseudo-labels
    pseudo_y[pseudo_licit_mask] = 0
    pseudo_y[pseudo_illicit_mask] = 1

    # Keep track of which nodes have labels (ground truth OR pseudo-labels)
    pseudo_mask = data.labeled_mask | pseudo_licit_mask | pseudo_illicit_mask

    # Save to the PyG Data object
    data.soft_y = out_probs
    data.pseudo_y = pseudo_y
    data.pseudo_mask = pseudo_mask

    # Add hyperparameter info to metadata
    data.lp_num_layers = args.num_layers
    data.lp_alpha = args.alpha
    data.lp_licit_threshold = args.licit_threshold
    data.lp_illicit_threshold = args.illicit_threshold

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving graph with pseudo-labels to {output_path}...")
    torch.save(data, output_path)
    logger.info("Label propagation pipeline complete!")


if __name__ == "__main__":
    main()
