"""
FraudSentinel GNN Hyperparameter Optimization Sweeps (Optuna)
==============================================================
Runs Optuna hyperparameter sweeps for GraphSAGE, GCN, and GAT.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import optuna
import torch

from Fraudsentinel.evaluate import evaluate_model
from Fraudsentinel.logger import get_logger
from Fraudsentinel.train import train_gat, train_gcn, train_graphsage

logger = get_logger("FraudSentinel.RunSweeps")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FraudSentinel — Hyperparameter sweeps using Optuna"
    )
    parser.add_argument(
        "--models", nargs="+", choices=["graphsage", "gcn", "gat", "all"],
        default=["all"], help="Which model(s) to optimize (default: all)"
    )
    parser.add_argument(
        "--n-trials", type=int, default=15,
        help="Number of search trials per GNN model (default: 15)"
    )
    parser.add_argument(
        "--epochs", type=int, default=40,
        help="Maximum epochs to train per trial (default: 40 for speed)"
    )
    parser.add_argument(
        "--data-path", default="data/processed/graph_data.pt",
        help="Path to the saved PyG Data object"
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="Directory to save parameter JSON files"
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device: 'cpu' or 'cuda' (auto-detected if not set)"
    )
    return parser.parse_args()


def optimize_graphsage(data: torch.Tensor, n_trials: int, max_epochs: int, device: str) -> dict:
    def objective(trial: optuna.Trial) -> float:
        cfg = {
            "hidden_channels": trial.suggest_categorical("hidden_channels", [64, 128, 256]),
            "num_layers": 2,
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "aggr": trial.suggest_categorical("aggr", ["mean", "max"]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "epochs": max_epochs,
            "patience": 10,
            "save_path": "models/temp_sage_trial.pt",
            "scaler_path": "models/scaler.pkl",
        }
        model, _, _ = train_graphsage(data, cfg=cfg, device=device)
        metrics = evaluate_model(model, data, data.val_mask, split_name="val")
        return float(metrics["f1"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def optimize_gcn(data: torch.Tensor, n_trials: int, max_epochs: int, device: str) -> dict:
    def objective(trial: optuna.Trial) -> float:
        cfg = {
            "hidden_channels": trial.suggest_categorical("hidden_channels", [64, 128, 256]),
            "num_layers": 2,
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "epochs": max_epochs,
            "patience": 10,
            "save_path": "models/temp_gcn_trial.pt",
            "scaler_path": "models/scaler.pkl",
        }
        model, _, _ = train_gcn(data, cfg=cfg, device=device)
        metrics = evaluate_model(model, data, data.val_mask, split_name="val")
        return float(metrics["f1"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def optimize_gat(data: torch.Tensor, n_trials: int, max_epochs: int, device: str) -> dict:
    def objective(trial: optuna.Trial) -> float:
        cfg = {
            "hidden_channels": trial.suggest_categorical("hidden_channels", [32, 64, 128]),
            "num_layers": 2,
            "heads": trial.suggest_categorical("heads", [4, 8]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "epochs": max_epochs,
            "patience": 10,
            "save_path": "models/temp_gat_trial.pt",
            "scaler_path": "models/scaler.pkl",
        }
        model, _, _ = train_gat(data, cfg=cfg, device=device)
        metrics = evaluate_model(model, data, data.val_mask, split_name="val")
        return float(metrics["f1"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading graph data from {data_path} ...")
    data = torch.load(data_path, weights_only=False)

    target_models = args.models
    if "all" in target_models:
        target_models = ["graphsage", "gcn", "gat"]

    for name in target_models:
        t0 = time.time()
        if name == "graphsage":
            best_params = optimize_graphsage(data, args.n_trials, args.epochs, device)
        elif name == "gcn":
            best_params = optimize_gcn(data, args.n_trials, args.epochs, device)
        elif name == "gat":
            best_params = optimize_gat(data, args.n_trials, args.epochs, device)

        elapsed = time.time() - t0
        save_file = model_dir / f"{name}_best_params.json"
        with open(save_file, "w") as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Finished {name} sweep in {elapsed:.1f}s. Best params saved to {save_file}")


if __name__ == "__main__":
    main()
