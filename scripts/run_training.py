"""
FraudSentinel Training Entry Point
====================================
Usage
-----
    # Train all models (XGBoost → GraphSAGE → GCN → GAT → comparison):
    py scripts/run_training.py

    # Train only GCN:
    py scripts/run_training.py --model gcn

    # Train only GAT:
    py scripts/run_training.py --model gat

Output
------
    models/xgboost_baseline.json     — serialised XGBoost model
    models/graphsage_best.pt         — best GraphSAGE checkpoint
    models/gcn_best.pt               — best GCN checkpoint
    models/gat_best.pt               — best GAT checkpoint
    reports/figures/pr_curves_val.png — overlaid PR curves (val split)
    reports/figures/pr_curves_test.png - overlaid PR curves (test split)
    logs/training_<date>.log         — full training log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from Fraudsentinel.logger import get_logger
from Fraudsentinel.train import (
    train_xgboost,
    train_graphsage,
    train_gcn,
    train_gat,
    DEFAULT_CFG,
)
from Fraudsentinel.evaluate import evaluate_model, compare_models, plot_pr_curves

logger = get_logger("FraudSentinel.RunTraining")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FraudSentinel — train XGBoost, GraphSAGE, GCN, and/or GAT"
    )
    parser.add_argument(
        "--model", choices=["xgboost", "graphsage", "gcn", "gat", "all"],
        default="all", help="Which model(s) to train (default: all)"
    )
    parser.add_argument(
        "--data-path", default="data/processed/graph_data.pt",
        help="Path to the saved PyG Data object"
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="Directory to save model artifacts"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override GraphSAGE/GCN/GAT max epochs"
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device: 'cpu' or 'cuda' (auto-detected if not set)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load graph data ──────────────────────────────────────────────────
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(
            f"Graph data not found at {data_path}. "
            "Run graph_construction.py first."
        )
        sys.exit(1)

    logger.info(f"Loading graph data from {data_path} ...")
    data = torch.load(data_path, weights_only=False)
    logger.info(
        f"Graph loaded: {data.num_nodes} nodes | "
        f"{data.num_edges} edges | "
        f"{data.num_node_features} features"
    )
    logger.info(
        f"Labeled splits — "
        f"Train: {(data.train_mask & data.labeled_mask).sum().item()} | "
        f"Val: {(data.val_mask & data.labeled_mask).sum().item()} | "
        f"Test: {(data.test_mask & data.labeled_mask).sum().item()}"
    )

    # ── Build configs ────────────────────────────────────────────────────
    epochs_override = args.epochs

    xgb_model  = None
    sage_model = None
    gcn_model  = None
    gat_model  = None

    xgb_val_metrics  = None
    sage_val_metrics = None
    gcn_val_metrics  = None
    gat_val_metrics  = None

    # ── XGBoost ──────────────────────────────────────────────────────────
    if args.model in ("xgboost", "all"):
        xgb_model = train_xgboost(
            data,
            cfg=DEFAULT_CFG["xgb"],
            model_dir=args.model_dir,
        )

        val_mask_labeled = data.val_mask & data.labeled_mask
        xgb_val_metrics = evaluate_model(
            xgb_model, data, val_mask_labeled,
            model_type="xgboost",
        )
        logger.info(f"XGBoost | Val F1={xgb_val_metrics['f1']:.4f}")

    # ── GraphSAGE ────────────────────────────────────────────────────────
    if args.model in ("graphsage", "all"):
        sage_cfg = DEFAULT_CFG["sage"].copy()
        best_params_path = Path(args.model_dir) / "graphsage_best_params.json"
        if best_params_path.exists():
            with open(best_params_path, "r") as f:
                best_params = json.load(f)
            logger.info(f"Loading optimized GraphSAGE parameters: {best_params}")
            sage_cfg.update(best_params)
        if epochs_override is not None:
            sage_cfg["epochs"] = epochs_override
        sage_model = train_graphsage(
            data,
            cfg=sage_cfg,
            model_dir=args.model_dir,
            device=args.device,
        )

        val_mask_labeled = data.val_mask & data.labeled_mask
        sage_val_metrics = evaluate_model(
            sage_model, data, val_mask_labeled,
            model_type="graphsage",
            device=args.device,
        )
        logger.info(f"GraphSAGE | Val F1={sage_val_metrics['f1']:.4f}")

    # ── GCN ──────────────────────────────────────────────────────────────
    if args.model in ("gcn", "all"):
        gcn_cfg = DEFAULT_CFG["gcn"].copy()
        best_params_path = Path(args.model_dir) / "gcn_best_params.json"
        if best_params_path.exists():
            with open(best_params_path, "r") as f:
                best_params = json.load(f)
            logger.info(f"Loading optimized GCN parameters: {best_params}")
            gcn_cfg.update(best_params)
        if epochs_override is not None:
            gcn_cfg["epochs"] = epochs_override
        gcn_model = train_gcn(
            data,
            cfg=gcn_cfg,
            model_dir=args.model_dir,
            device=args.device,
        )

        val_mask_labeled = data.val_mask & data.labeled_mask
        gcn_val_metrics = evaluate_model(
            gcn_model, data, val_mask_labeled,
            model_type="gcn",
            device=args.device,
        )
        logger.info(f"GCN | Val F1={gcn_val_metrics['f1']:.4f}")

    # ── GAT ──────────────────────────────────────────────────────────────
    if args.model in ("gat", "all"):
        gat_cfg = DEFAULT_CFG["gat"].copy()
        best_params_path = Path(args.model_dir) / "gat_best_params.json"
        if best_params_path.exists():
            with open(best_params_path, "r") as f:
                best_params = json.load(f)
            logger.info(f"Loading optimized GAT parameters: {best_params}")
            gat_cfg.update(best_params)
        if epochs_override is not None:
            gat_cfg["epochs"] = epochs_override
        gat_model = train_gat(
            data,
            cfg=gat_cfg,
            model_dir=args.model_dir,
            device=args.device,
        )

        val_mask_labeled = data.val_mask & data.labeled_mask
        gat_val_metrics = evaluate_model(
            gat_model, data, val_mask_labeled,
            model_type="gat",
            device=args.device,
        )
        logger.info(f"GAT | Val F1={gat_val_metrics['f1']:.4f}")

    # ── Comparison + PR curves (only if all four were trained) ───────────
    if xgb_val_metrics and sage_val_metrics and gcn_val_metrics and gat_val_metrics:
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SPLIT — MODEL COMPARISON")
        logger.info("=" * 60)
        comparison_df = compare_models(
            xgb_val_metrics, sage_val_metrics, gcn_val_metrics, gat_val_metrics, "Validation"
        )
        print("\n" + comparison_df.to_string() + "\n")

        plot_pr_curves(
            xgb_val_metrics,
            sage_val_metrics,
            gcn_val_metrics,
            gat_val_metrics,
            save_path="reports/figures/pr_curves_val.png",
            split_name="Validation",
        )

        # ── Test set evaluation (threshold from val, locked) ──
        logger.info("\n" + "=" * 60)
        logger.info("TEST SPLIT — FINAL NUMBERS (threshold locked from val)")
        logger.info("=" * 60)

        test_mask_labeled = data.test_mask & data.labeled_mask
        
        xgb_test_metrics = evaluate_model(
            xgb_model, data, test_mask_labeled,
            model_type="xgboost",
            threshold=xgb_val_metrics["threshold"],
        )
        sage_test_metrics = evaluate_model(
            sage_model, data, test_mask_labeled,
            model_type="graphsage",
            device=args.device,
            threshold=sage_val_metrics["threshold"],
        )
        gcn_test_metrics = evaluate_model(
            gcn_model, data, test_mask_labeled,
            model_type="gcn",
            device=args.device,
            threshold=gcn_val_metrics["threshold"],
        )
        gat_test_metrics = evaluate_model(
            gat_model, data, test_mask_labeled,
            model_type="gat",
            device=args.device,
            threshold=gat_val_metrics["threshold"],
        )

        test_comparison = compare_models(
            xgb_test_metrics, sage_test_metrics, gcn_test_metrics, gat_test_metrics, "Test"
        )
        print("\n" + "=" * 60)
        print("TEST SET RESULTS")
        print("=" * 60)
        print(test_comparison.to_string())
        print()

        plot_pr_curves(
            xgb_test_metrics,
            sage_test_metrics,
            gcn_test_metrics,
            gat_test_metrics,
            save_path="reports/figures/pr_curves_test.png",
            split_name="Test",
        )

    logger.info("run_training.py complete.")


if __name__ == "__main__":
    main()
