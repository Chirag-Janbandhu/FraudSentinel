"""
FraudSentinel Semi-supervised Labels (Part 2) — Re-evaluate GraphSAGE with Pseudo-labels
========================================================================================
Usage:
    py scripts/run_training_pseudo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import pandas as pd

from Fraudsentinel.logger import get_logger
from Fraudsentinel.train import train_graphsage, DEFAULT_CFG
from Fraudsentinel.evaluate import evaluate_model

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

    logger.info(f"Loading graph data with pseudo-labels from {data_path}...")
    data = torch.load(data_path, weights_only=False)

    dev = args.device
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {dev}")

    # Load best hyperparameters from Optuna sweep
    sage_cfg = DEFAULT_CFG["sage"].copy()
    best_params_path = Path(args.model_dir) / "graphsage_best_params.json"
    if best_params_path.exists():
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        logger.info(f"Loading optimized GraphSAGE parameters: {best_params}")
        sage_cfg.update(best_params)

    if args.epochs is not None:
        sage_cfg["epochs"] = args.epochs

    # ── 1. Train GraphSAGE (Max) with Pseudo-labels ──────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING GRAPHSAGE WITH PSEUDO-LABELS")
    logger.info("=" * 60)

    # Use a separate subdirectory/checkpoint to avoid overwriting standard model
    pseudo_model_dir = Path(args.model_dir) / "pseudo_trained"
    pseudo_model_dir.mkdir(parents=True, exist_ok=True)

    # Train
    model_pseudo = train_graphsage(
        data,
        cfg=sage_cfg,
        model_dir=pseudo_model_dir,
        device=dev,
        use_pseudo=True,
    )

    # Save pseudo-model checkpoint to main models folder
    best_pseudo_ckpt = pseudo_model_dir / "graphsage_best.pt"
    final_pseudo_ckpt = Path(args.model_dir) / "graphsage_pseudo_best.pt"
    if best_pseudo_ckpt.exists():
        import shutil
        shutil.copy(best_pseudo_ckpt, final_pseudo_ckpt)
        logger.info(f"Copied best pseudo-trained checkpoint to {final_pseudo_ckpt}")

    # ── 2. Evaluate on Ground Truth ──────────────────────────────────────────
    logger.info("\nEvaluating pseudo-trained GraphSAGE on validation split...")
    val_mask_labeled = data.val_mask & data.labeled_mask
    val_metrics_pseudo = evaluate_model(
        model_pseudo, data, val_mask_labeled,
        model_type="graphsage",
        device=dev,
    )

    logger.info("\nEvaluating pseudo-trained GraphSAGE on test split (drift)...")
    test_mask_labeled = data.test_mask & data.labeled_mask
    test_metrics_pseudo = evaluate_model(
        model_pseudo, data, test_mask_labeled,
        model_type="graphsage",
        device=dev,
    )

    # ── 3. Load and Evaluate Standard GraphSAGE (Max) ────────────────────────
    logger.info("\nLoading and evaluating standard GraphSAGE (ground-truth only)...")
    # Instantiate standard GraphSAGE model
    from Fraudsentinel.models import GraphSAGEClassifier
    model_std = GraphSAGEClassifier(
        in_channels=data.num_node_features,
        hidden_channels=sage_cfg["hidden_channels"],
        dropout=sage_cfg["dropout"],
        aggr=sage_cfg.get("aggr", "max"),
    ).to(dev)

    std_ckpt_path = Path(args.model_dir) / "graphsage_best.pt"
    if std_ckpt_path.exists():
        model_std.load_state_dict(torch.load(std_ckpt_path, map_location=dev, weights_only=True))
        logger.info(f"Loaded standard GraphSAGE checkpoint from {std_ckpt_path}")
        
        # Load scaler
        import pickle
        scaler_path = Path(args.model_dir) / "feature_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                model_std.scaler = pickle.load(f)
            logger.info("Loaded standard feature scaler.")
        
        val_metrics_std = evaluate_model(
            model_std, data, val_mask_labeled,
            model_type="graphsage",
            device=dev,
        )
        test_metrics_std = evaluate_model(
            model_std, data, test_mask_labeled,
            model_type="graphsage",
            device=dev,
        )
    else:
        logger.warning(f"Standard GraphSAGE checkpoint not found at {std_ckpt_path}. Skipping standard comparison.")
        val_metrics_std = None
        test_metrics_std = None

    # ── 4. Print Comparison ──────────────────────────────────────────────────
    if val_metrics_std and test_metrics_std:
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON: STANDARD VS. PSEUDO-LABEL TRAINED GRAPHSAGE")
        logger.info("=" * 60)

        # Validation Comparison
        val_comparison = pd.DataFrame({
            "Metric (Val)": ["F1-illicit", "Precision", "Recall", "PR-AUC", "Threshold"],
            "Standard SAGE": [
                val_metrics_std["f1"],
                val_metrics_std["precision"],
                val_metrics_std["recall"],
                val_metrics_std["pr_auc"],
                val_metrics_std["threshold"],
            ],
            "Pseudo-Trained SAGE": [
                val_metrics_pseudo["f1"],
                val_metrics_pseudo["precision"],
                val_metrics_pseudo["recall"],
                val_metrics_pseudo["pr_auc"],
                val_metrics_pseudo["threshold"],
            ]
        })
        print("\nVALIDATION SPLIT COMPARISON:")
        print(val_comparison.to_string(index=False))

        # Test Comparison
        test_comparison = pd.DataFrame({
            "Metric (Test)": ["F1-illicit", "Precision", "Recall", "PR-AUC", "Threshold"],
            "Standard SAGE": [
                test_metrics_std["f1"],
                test_metrics_std["precision"],
                test_metrics_std["recall"],
                test_metrics_std["pr_auc"],
                test_metrics_std["threshold"],
            ],
            "Pseudo-Trained SAGE": [
                test_metrics_pseudo["f1"],
                test_metrics_pseudo["precision"],
                test_metrics_pseudo["recall"],
                test_metrics_pseudo["pr_auc"],
                test_metrics_pseudo["threshold"],
            ]
        })
        print("\nTEST SPLIT COMPARISON (UNDER CONCEPT DRIFT):")
        print(test_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
