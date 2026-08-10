"""
demo_monitoring.py — Lightweight Inference Monitoring Demo
============================================================

Demonstrates FraudSentinel's prediction logging and probability shift monitoring:
  1. Loads processed graph data and saved XGBoost model.
  2. Logs validation set predictions as reference baseline (`log_prediction_batch`).
  3. Logs test set predictions (`log_prediction_batch`).
  4. Runs `check_probability_shift` comparing test probability distribution against
     val baseline, demonstrating that a drift warning is automatically triggered.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from Fraudsentinel.logger import get_logger
from Fraudsentinel.models import XGBoostFraudClassifier
from Fraudsentinel.monitor import check_probability_shift, log_prediction_batch

logger = get_logger("FraudSentinel.DemoMonitoring")


def main() -> None:
    data_path = Path("data/processed/graph_data.pt")
    model_path = Path("models/xgboost_baseline.json")

    if not data_path.exists():
        logger.error(f"Graph data not found at {data_path}. Run graph_construction.py first.")
        sys.exit(1)
    if not model_path.exists():
        logger.error(f"XGBoost model not found at {model_path}. Run run_training.py first.")
        sys.exit(1)

    logger.info(f"Loading graph data from {data_path} ...")
    data = torch.load(data_path, weights_only=False)

    logger.info(f"Loading trained XGBoost baseline from {model_path} ...")
    model = XGBoostFraudClassifier()
    model.load(model_path)

    # Convert node features to numpy for XGBoost
    X_val = data.x[data.val_mask].cpu().numpy()
    y_val = data.y[data.val_mask].cpu().numpy()

    X_test = data.x[data.test_mask].cpu().numpy()
    y_test = data.y[data.test_mask].cpu().numpy()

    logger.info("Computing predictions for Validation set (Reference) ...")
    val_probas = model.predict_proba(X_val)

    logger.info("Computing predictions for Test set (Current Batch) ...")
    test_probas = model.predict_proba(X_test)

    # 1. Log validation batch (reference)
    logger.info("Logging Validation predictions ...")
    log_prediction_batch(
        probas=val_probas,
        y_true=y_val,
        batch_id="val_reference_split",
        save_path="reports/prediction_log.csv",
    )

    # 2. Log test batch (production test period)
    logger.info("Logging Test predictions ...")
    log_prediction_batch(
        probas=test_probas,
        y_true=y_test,
        batch_id="test_t43_t49_split",
        save_path="reports/prediction_log.csv",
    )

    # 3. Check for output probability shift
    logger.info("Checking for output probability shift (Val vs Test) ...")
    is_drifted = check_probability_shift(
        current_probas=test_probas,
        reference_probas=val_probas,
        threshold=0.05,
    )

    if is_drifted:
        logger.info("[SUCCESS] Monitoring hook successfully caught the val-to-test probability shift!")
    else:
        logger.info("[NOTE] No probability shift detected.")


if __name__ == "__main__":
    main()
