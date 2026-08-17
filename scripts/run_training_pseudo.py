"""
FraudSentinel Semi-supervised Labels — Re-evaluate GraphSAGE with Pseudo-labels
========================================================================================
Retrains GraphSAGE Incorporating propagated pseudo-labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from Fraudsentinel.evaluate import evaluate_model
from Fraudsentinel.logger import get_logger
from Fraudsentinel.train import DEFAULT_CFG, train_graphsage

logger = get_logger("FraudSentinel.TrainingPseudo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FraudSentinel — Train GraphSAGE with pseudo-labels"
    )
    parser.add_argument(
        "--data-path", default="data/processed/graph_with_pseudo.pt",
        help="Path to the saved PyG Data object with pseudo-labels"
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override GraphSAGE epochs limit"
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device: 'cpu' or 'cuda'"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(
            f"Graph data with pseudo-labels not found at {data_path}. "
            "Run run_label_propagation.py first."
        )
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading graph data with pseudo-labels from {data_path}...")
    data = torch.load(data_path, weights_only=False)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    sage_cfg = DEFAULT_CFG["sage"].copy()
    sage_cfg["save_path"] = str(model_dir / "graphsage_pseudo_best.pt")
    sage_cfg["scaler_path"] = str(model_dir / "scaler.pkl")

    params_file = model_dir / "graphsage_best_params.json"
    if params_file.exists():
        with open(params_file) as f:
            best_params = json.load(f)
            sage_cfg.update(best_params)

    if args.epochs is not None:
        sage_cfg["epochs"] = args.epochs

    pseudo_labels = (data.pseudo_indices, data.pseudo_labels)

    logger.info("Training GraphSAGE incorporating pseudo-labels...")
    model_pseudo, scaler, history = train_graphsage(
        data, cfg=sage_cfg, device=device, pseudo_labels=pseudo_labels
    )

    metrics_val = evaluate_model(model_pseudo, data, data.val_mask, split_name="val")
    metrics_test = evaluate_model(model_pseudo, data, data.test_mask, split_name="test")

    logger.info(f"Retrained GraphSAGE Val PR-AUC: {metrics_val['pr_auc']:.4f} | Test PR-AUC: {metrics_test['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
