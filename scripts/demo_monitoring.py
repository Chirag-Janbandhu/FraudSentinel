"""
demo_monitoring.py — Inference Monitoring Demo for FraudSentinel.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
        logger.error(f"Graph data not found at {data_path}.")
        sys.exit(1)
    if not model_path.exists():
        logger.error(f"XGBoost model not found at {model_path}.")
        sys.exit(1)

    logger.info(f"Loading graph data from {data_path} ...")
    data = torch.load(data_path, weights_only=False)

    logger.info(f"Loading trained XGBoost baseline from {model_path} ...")
    model = XGBoostFraudClassifier()
    model.load(model_path)

    X_val = data.x[data.val_mask].cpu().numpy()
    y_val = data.y[data.val_mask].cpu().numpy()

    X_test = data.x[data.test_mask].cpu().numpy()
    y_test = data.y[data.test_mask].cpu().numpy()

    logger.info("Computing predictions for Validation set (Reference) ...")
    val_probas = model.predict_proba(X_val)

    logger.info("Computing predictions for Test set (Current Batch) ...")
    test_probas = model.predict_proba(X_test)

    log_path = Path("reports/prediction_log.csv")

    val_record = log_prediction_batch(
        probas=val_probas,
        y_true=y_val,
        batch_id="val_split_reference",
        eval_threshold=0.5,
        save_path=log_path,
    )

    test_record = log_prediction_batch(
        probas=test_probas,
        y_true=y_test,
        batch_id="test_split_batch",
        eval_threshold=0.5,
        save_path=log_path,
    )

    val_mean_proba = float(val_record["mean_proba"])

    shifted, delta = check_probability_shift(
        probas=test_probas,
        ref_mean=val_mean_proba,
        threshold_delta=0.10,
        batch_id="test_split_batch",
    )

    logger.info(f"Monitoring demo completed successfully. Log saved to {log_path}")


if __name__ == "__main__":
    main()
