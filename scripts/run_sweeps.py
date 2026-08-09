"""
FraudSentinel GNN Hyperparameter Optimization Sweeps (Optuna)
==============================================================
This script runs Optuna hyperparameter sweeps for GraphSAGE, GCN, and GAT.
The best parameters are written to models/<model>_best_params.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
import time

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import optuna

from Fraudsentinel.logger import get_logger
from Fraudsentinel.train import train_graphsage, train_gcn, train_gat
from Fraudsentinel.evaluate import evaluate_model

# Force standard logger output to terminal
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


def objective_sage(trial: optuna.Trial, data: any, epochs: int, model_dir: Path, dev: str) -> float:
    # 1. Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_choice = trial.suggest_categorical("hidden_channels_idx", [0, 1, 2, 3])
    hidden_options = [
        [128, 64],
        [256, 128],
        [64, 64],
        [128, 128],
    ]
    hidden_channels = hidden_options[hidden_choice]
    aggr = trial.suggest_categorical("aggr", ["max", "mean"])

    trial_cfg = {
        "hidden_channels": hidden_channels,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "aggr": aggr,
        "epochs": epochs,
        "patience": 10,
        "random_state": 42,
    }

    logger.info(f"[Trial {trial.number}] SAGE parameters: {trial_cfg}")

    try:
        # Temporary model output path for the trial to avoid overwriting best overall model
        trial_dir = model_dir / f"trial_sage_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        model = train_graphsage(data, cfg=trial_cfg, model_dir=trial_dir, device=dev)
        val_mask_labeled = data.val_mask & data.labeled_mask
        metrics = evaluate_model(model, data, val_mask_labeled, model_type="graphsage", device=dev)
        val_f1 = metrics["f1"]

        logger.info(f" -> Trial {trial.number} Finished. Val F1-illicit: {val_f1:.4f}")
        return val_f1
    except Exception as e:
        logger.error(f" -> Trial {trial.number} Failed with error: {e}")
        return 0.0


def objective_gcn(trial: optuna.Trial, data: any, epochs: int, model_dir: Path, dev: str) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_choice = trial.suggest_categorical("hidden_channels_idx", [0, 1, 2, 3])
    hidden_options = [
        [128, 64],
        [256, 128],
        [64, 64],
        [128, 128],
    ]
    hidden_channels = hidden_options[hidden_choice]

    trial_cfg = {
        "hidden_channels": hidden_channels,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": 10,
        "random_state": 42,
    }

    logger.info(f"[Trial {trial.number}] GCN parameters: {trial_cfg}")

    try:
        trial_dir = model_dir / f"trial_gcn_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        model = train_gcn(data, cfg=trial_cfg, model_dir=trial_dir, device=dev)
        val_mask_labeled = data.val_mask & data.labeled_mask
        metrics = evaluate_model(model, data, val_mask_labeled, model_type="gcn", device=dev)
        val_f1 = metrics["f1"]

        logger.info(f" -> Trial {trial.number} Finished. Val F1-illicit: {val_f1:.4f}")
        return val_f1
    except Exception as e:
        logger.error(f" -> Trial {trial.number} Failed with error: {e}")
        return 0.0


def objective_gat(trial: optuna.Trial, data: any, epochs: int, model_dir: Path, dev: str) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    heads = trial.suggest_categorical("heads", [2, 4, 8])

    hidden_choice = trial.suggest_categorical("hidden_channels_idx", [0, 1, 2, 3])
    hidden_options = [
        [32, 64],
        [64, 128],
        [32, 128],
        [64, 64],
    ]
    hidden_channels = hidden_options[hidden_choice]

    trial_cfg = {
        "hidden_channels": hidden_channels,
        "dropout": dropout,
        "heads": heads,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": 10,
        "random_state": 42,
    }

    logger.info(f"[Trial {trial.number}] GAT parameters: {trial_cfg}")

    try:
        trial_dir = model_dir / f"trial_gat_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        model = train_gat(data, cfg=trial_cfg, model_dir=trial_dir, device=dev)
        val_mask_labeled = data.val_mask & data.labeled_mask
        metrics = evaluate_model(model, data, val_mask_labeled, model_type="gat", device=dev)
        val_f1 = metrics["f1"]

        logger.info(f" -> Trial {trial.number} Finished. Val F1-illicit: {val_f1:.4f}")
        return val_f1
    except Exception as e:
        logger.error(f" -> Trial {trial.number} Failed with error: {e}")
        return 0.0


def clean_trial_directories(model_dir: Path, prefix: str) -> None:
    """Removes trial directories to save space after study finishes."""
    for path in model_dir.glob(f"trial_{prefix}_*"):
        if path.is_dir():
            for f in path.iterdir():
                f.unlink()
            path.rmdir()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Graph data not found at {data_path}. Run graph_construction.py first.")
        sys.exit(1)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading graph data from {data_path}...")
    data = torch.load(data_path, weights_only=False)

    dev = args.device
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Sweeping models using device: {dev}")

    models_to_run = args.models
    if "all" in models_to_run:
        models_to_run = ["graphsage", "gcn", "gat"]

    # Storage for studies database
    db_path = model_dir / "optuna.db"
    storage_url = f"sqlite:///{db_path.resolve()}"

    for model_name in models_to_run:
        logger.info("\n" + "=" * 60)
        logger.info(f"STARTING OPTUNA STUDY: {model_name.upper()}")
        logger.info("=" * 60)

        study_name = f"study_{model_name}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="maximize",
            load_if_exists=True
        )

        t0 = time.time()
        if model_name == "graphsage":
            study.optimize(
                lambda trial: objective_sage(trial, data, args.epochs, model_dir, dev),
                n_trials=args.n_trials
            )
        elif model_name == "gcn":
            study.optimize(
                lambda trial: objective_gcn(trial, data, args.epochs, model_dir, dev),
                n_trials=args.n_trials
            )
        elif model_name == "gat":
            study.optimize(
                lambda trial: objective_gat(trial, data, args.epochs, model_dir, dev),
                n_trials=args.n_trials
            )

        elapsed = time.time() - t0
        best_trial = study.best_trial

        logger.info(f"Finished {model_name} study in {elapsed:.1f}s")
        logger.info(f"Best Trial F1: {best_trial.value:.4f}")
        logger.info(f"Best parameters: {best_trial.params}")

        # Construct parameter payload (remapping index choice back to hidden_channels)
        best_params = best_trial.params.copy()
        if "hidden_channels_idx" in best_params:
            idx = best_params.pop("hidden_channels_idx")
            if model_name == "gat":
                hidden_options = [[32, 64], [64, 128], [32, 128], [64, 64]]
            else:
                hidden_options = [[128, 64], [256, 128], [64, 64], [128, 128]]
            best_params["hidden_channels"] = hidden_options[idx]

        # Write parameters JSON
        out_path = model_dir / f"{model_name}_best_params.json"
        with open(out_path, "w") as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Saved optimized parameters to {out_path}")

        # Clean space
        clean_trial_directories(model_dir, model_name[:3])


if __name__ == "__main__":
    main()
