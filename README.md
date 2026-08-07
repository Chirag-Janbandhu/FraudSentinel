# FraudSentinel


# FraudSentinel

Graph neural network-based fraud/money-laundering detection on the
Elliptic Bitcoin transaction dataset. Compares a tabular baseline
(XGBoost) against GNN architectures (GraphSAGE, GAT, GCN) to evaluate
whether transaction network structure improves fraud detection beyond
individual transaction features alone.

## Status
EDA complete (see `research/EDA.ipynb`). Key findings:
- Severe class imbalance: ~2.2% illicit within full dataset, ~9.8%
  within labeled data — PR-AUC/F1 on the illicit class used instead
  of accuracy.
- Illicit transactions cluster in the network: neighbors of illicit
  nodes are illicit at 2.4x the base rate, supporting the use of
  graph-based modeling.
- Temporal train/val/test split: time steps 1-34 / 35-42 / 43-49.

## Dataset
[Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
— place raw CSVs in `data/raw/` (not tracked in git).

## Setup
```powershell
python -m venv Fsenv
.\Fsenv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Structure
- `research/` — EDA and exploratory notebooks
- `src/Fraudsentinel/` — reusable pipeline modules
- `data/` — raw and processed data (gitignored)