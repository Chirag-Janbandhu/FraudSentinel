"""
FraudSentinel Training Entry Point
====================================
Trains XGBoost, GraphSAGE, GCN, and GAT models on graph data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from Fraudsentinel.evaluate import evaluate_model, plot_pr_curves
from Fraudsentinel.logger import get_logger
from Fraudsentinel.train import (
    DEFAULT_CFG,
    train_gat,
    train_gcn,
    train_graphsage,
    train_xgboost,
)

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
        "--output-dir", default="reports/figures",
        help="Directory to save plots and reports"
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device: 'cpu' or 'cuda' (auto-detected if not set)"
    )
    return parser.parse_args()


def load_best_params(model_dir: Path, model_name: str) -> dict | None:
    params_file = model_dir / f"{model_name}_best_params.json"
    if params_file.exists():
        logger.info(f"Loading tuned hyperparameters from {params_file}...")
        with open(params_file) as f:
            return json.load(f)
    return None


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        logger.error(f"Data file not found at '{data_path}'. Run graph construction script first.")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading graph data from '{data_path}'...")
    data = torch.load(data_path, weights_only=False)

    run_all = args.model == "all"
    metrics_cache = {}

    if run_all or args.model == "xgboost":
        logger.info("Training XGBoost baseline...")
        xgb_cfg = DEFAULT_CFG["xgb"].copy()
        xgb_cfg["save_path"] = str(model_dir / "xgboost_baseline.json")
        xgb_model = train_xgboost(data, cfg=xgb_cfg)

        metrics_cache["xgb_val"] = evaluate_model(xgb_model, data, data.val_mask, split_name="val")
        metrics_cache["xgb_test"] = evaluate_model(xgb_model, data, data.test_mask, split_name="test")

    if run_all or args.model == "graphsage":
        logger.info("Training GraphSAGE...")
        sage_cfg = DEFAULT_CFG["sage"].copy()
        sage_cfg["save_path"] = str(model_dir / "graphsage_best.pt")
        sage_cfg["scaler_path"] = str(model_dir / "scaler.pkl")

        tuned = load_best_params(model_dir, "graphsage")
        if tuned:
            sage_cfg.update(tuned)

        sage_model, scaler, _ = train_graphsage(data, cfg=sage_cfg, device=device)
        metrics_cache["sage_val"] = evaluate_model(sage_model, data, data.val_mask, split_name="val")
        metrics_cache["sage_test"] = evaluate_model(sage_model, data, data.test_mask, split_name="test")

    if run_all or args.model == "gcn":
        logger.info("Training GCN...")
        gcn_cfg = DEFAULT_CFG["gcn"].copy()
        gcn_cfg["save_path"] = str(model_dir / "gcn_best.pt")
        gcn_cfg["scaler_path"] = str(model_dir / "scaler.pkl")

        tuned = load_best_params(model_dir, "gcn")
        if tuned:
            gcn_cfg.update(tuned)

        gcn_model, _, _ = train_gcn(data, cfg=gcn_cfg, device=device)
        metrics_cache["gcn_val"] = evaluate_model(gcn_model, data, data.val_mask, split_name="val")
        metrics_cache["gcn_test"] = evaluate_model(gcn_model, data, data.test_mask, split_name="test")

    if run_all or args.model == "gat":
        logger.info("Training GAT...")
        gat_cfg = DEFAULT_CFG["gat"].copy()
        gat_cfg["save_path"] = str(model_dir / "gat_best.pt")
        gat_cfg["scaler_path"] = str(model_dir / "scaler.pkl")

        tuned = load_best_params(model_dir, "gat")
        if tuned:
            gat_cfg.update(tuned)

        gat_model, _, _ = train_gat(data, cfg=gat_cfg, device=device)
        metrics_cache["gat_val"] = evaluate_model(gat_model, data, data.val_mask, split_name="val")
        metrics_cache["gat_test"] = evaluate_model(gat_model, data, data.test_mask, split_name="test")

    if "xgb_val" in metrics_cache and "sage_val" in metrics_cache:
        plot_pr_curves(metrics_cache["xgb_val"], metrics_cache["sage_val"], output_dir / "pr_curves_val.png", gnn_name="GraphSAGE")
        plot_pr_curves(metrics_cache["xgb_test"], metrics_cache["sage_test"], output_dir / "pr_curves_test.png", gnn_name="GraphSAGE")

    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
