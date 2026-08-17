"""
FraudSentinel — Drift Detection Entry Point
============================================
Runs Population Stability Index (PSI) drift detection scans across temporal splits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from Fraudsentinel.drift_check import run_drift_check
from Fraudsentinel.logger import get_logger

logger = get_logger("FraudSentinel.RunDriftCheck")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PSI feature drift detection on FraudSentinel graph data."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/graph_data.pt",
        help="Path to serialized PyG Data object.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save CSV drift reports.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of quantile bins for PSI calculation.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save CSV drift reports to disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(
            f"Graph data not found at '{data_path}'. "
            "Run graph construction script first."
        )
        sys.exit(1)

    logger.info(f"Loading graph data from '{data_path}'...")
    data = torch.load(data_path, weights_only=False)

    ref_mask_control = data.train_mask & data.labeled_mask
    comp_mask_control = data.val_mask & data.labeled_mask

    result_control = run_drift_check(
        data=data,
        ref_mask=ref_mask_control,
        comp_mask=comp_mask_control,
        label="Control: Train (t=1..34) vs Val (t=35..42)",
        n_bins=args.n_bins,
    )

    ref_mask_primary = (data.train_mask | data.val_mask) & data.labeled_mask
    comp_mask_primary = data.test_mask & data.labeled_mask

    result_primary = run_drift_check(
        data=data,
        ref_mask=ref_mask_primary,
        comp_mask=comp_mask_primary,
        label="Primary: TrainVal (t=1..42) vs Test (t=43..49)",
        n_bins=args.n_bins,
    )

    output_dir = Path(args.output_dir)
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

        path_ctrl = output_dir / "drift_report_train_vs_val.csv"
        result_control.psi_table.to_csv(path_ctrl, index=False)
        logger.info(f"  Control report saved to {path_ctrl}")

        path_prim = output_dir / "drift_report_trainval_vs_test.csv"
        result_primary.psi_table.to_csv(path_prim, index=False)
        logger.info(f"  PSI report saved to {path_prim}")

    logger.info("======================================================================")
    logger.info("  DRIFT DETECTION SUMMARY")
    logger.info("======================================================================")
    logger.info("  Metric                                    CONTROL       PRIMARY")
    logger.info("  -----------------------------------  ------------  ------------")
    logger.info("  Comparison                             Train->Val  TrainVal->Test")
    logger.info(f"  Reference nodes                     {result_control.n_ref:>12,}  {result_primary.n_ref:>12,}")
    logger.info(f"  Comparison nodes                    {result_control.n_comp:>12,}  {result_primary.n_comp:>12,}")
    logger.info(f"  Mean PSI (all features)             {result_control.mean_psi:>12.4f}  {result_primary.mean_psi:>12.4f}")
    logger.info(f"  Max PSI                             {result_control.max_psi:>12.4f}  {result_primary.max_psi:>12.4f}")
    logger.info(f"  Features: MAJOR (PSI > 0.25)        {result_control.n_major:>12d}  {result_primary.n_major:>12d}")
    logger.info(f"  Features: Moderate (0.10-0.25)      {result_control.n_moderate:>12d}  {result_primary.n_moderate:>12d}")
    logger.info(f"  Features: Stable (< 0.10)           {result_control.n_stable:>12d}  {result_primary.n_stable:>12d}")
    logger.info("======================================================================")


if __name__ == "__main__":
    main()
