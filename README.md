# 🛡️ FraudSentinel

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-PyTorch_Geometric-ee4c2c.svg)](https://pytorch-geometric.readthedocs.io/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Graph Neural Network-based Bitcoin anti-money laundering and transaction fraud detection engine on the Elliptic dataset.** 

FraudSentinel benchmarks a tabular baseline (XGBoost) against three GNN architectures (**GraphSAGE**, **GCN**, **GAT**) on 203,769 Bitcoin transactions across 49 temporal subgraphs. It incorporates topological feature engineering (PageRank, Louvain community ratios), leakage-free `LayerNorm` message passing, semi-supervised label propagation, Population Stability Index (PSI) drift monitoring, and local GNNExplainer attribution visualizers.

---

## 🚀 Interactive Streamlit Dashboard

FraudSentinel includes a production-ready interactive web application for real-time model evaluation, 2-hop transaction network graph exploration, live fraud risk scoring, and temporal drift monitoring.

```powershell
# Activate environment and launch dashboard locally
.\Fsenv\Scripts\Activate.ps1
streamlit run app.py
```

### Dashboard Features
- **📊 Executive Overview & Leaderboard:** Compare F1-illicit, PR-AUC, and lift metrics across model architectures with an interactive threshold tuning simulator.
- **🌐 Transaction Network Explorer:** Visualize 2-hop local receptive field subgraphs around transaction nodes with node risk inspection.
- **🔍 Live Predictor & Feature Importance:** Real-time transaction fraud scoring with topological metric attributions.
- **📈 Temporal Drift & PSI Monitoring:** Automated Population Stability Index (PSI) feature scan across temporal splits.
- **📖 Technical Architecture:** Walkthrough of neighborhood clustering factors, bidirectional propagation, and max-pooling resilience.

---

## 🏆 Architectural Benchmark Summary

Models were evaluated under strict chronological splits (Train: timesteps 1–34, Val: 35–42, Test: 43–49):

| Model Architecture | Val F1-Illicit | Val PR-AUC | Test PR-AUC (Drift) | Test Lift over Base Rate | Key Architectural Insight |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **XGBoost Baseline** | 0.9090 | 0.9320 | 0.0408 | 2.33x | Local features + Louvain community size ratios |
| **GCN (Tuned)** | 0.7305 | 0.7638 | 0.0507 | 2.89x | Degree-symmetric Laplacian aggregation + LayerNorm |
| **GraphSAGE (Max) [Champion]** | **0.7239** | **0.7916** | **0.0663** | **3.78x** | Element-wise max-pooling (logical OR-gate resilience) |
| **GAT (Tuned)** | 0.6289 | 0.6363 | 0.0424 | 2.42x | Multi-head attention (8 heads) over-fits post-drift |

---

## 🔑 Key Engineering Breakthroughs

1. **Neighborhood Illicit Clustering (2.40x Base Rate):**
   Neighboring transactions of illicit nodes—in both predecessor and successor directions—exhibit an illicit rate **2.40× the base rate**, establishing the empirical rationale for graph message passing over tabular models.
2. **Edge Directionality (`to_undirected()`):**
   Converting directed transaction edges to undirected doubled graph edges to **468,710**, enabling bidirectional message passing and capturing successor signals previously missed.
3. **Leakage-Free Normalization (`LayerNorm`):**
   Replaced `BatchNorm1d` (which leaked full-batch validation/test statistics during graph passes) with node-independent `LayerNorm`.
4. **Max-Pooling OR-Gate Resilience:**
   Element-wise max aggregation (`aggr='max'`) acts as a logical OR-gate, preserving extreme anomalous signals from neighbors under post-drift temporal degradation.
5. **Semi-Supervised Label Propagation:**
   Propagating high-confidence labels over the 77% unlabeled transaction nodes provided a **+2.1% relative PR-AUC lift** and **+3.6% F1-illicit lift**.

---

## ⚙️ Quickstart & Installation

### 1. Clone Repository & Setup Virtual Environment
```powershell
git clone https://github.com/Chirag-Janbandhu/FraudSentinel.git
cd FraudSentinel

python -m venv Fsenv
.\Fsenv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
pip install -e .
```

### 3. Run Pipeline Entry Points
```powershell
# Run end-to-end model training
py scripts/run_training.py

# Run Population Stability Index (PSI) drift detection
py scripts/run_drift_check.py

# Run semi-supervised label propagation
py scripts/run_label_propagation.py
```

---

## 📂 Repository Structure

```
FraudSentinel/
├── .streamlit/                  # Custom dark mode dashboard theme config
├── app.py                       # Streamlit web application dashboard
├── data/                        # Raw and processed graph datasets (gitignored)
│   ├── raw/                     # Raw Elliptic CSV files
│   └── processed/               # Serialized PyTorch Geometric graph objects
├── models/                      # Saved checkpoints & hyperparameter configs
├── reports/                     # Benchmark analysis, PSI CSVs, and figures
│   ├── figures/                 # Precision-Recall curves & explainability plots
│   ├── drift_report_train_vs_val.csv
│   └── drift_report_trainval_vs_test.csv
├── research/                    # Exploratory analysis notebooks
├── scripts/                     # Executable entry points
│   ├── demo_monitoring.py       # Inference prediction logging demo
│   ├── run_drift_check.py       # PSI feature drift detection scan
│   ├── run_explainability.py    # GNNExplainer & Captum case study generator
│   ├── run_label_propagation.py  # Semi-supervised pseudo-label generator
│   ├── run_sweeps.py            # Optuna hyperparameter tuning
│   └── run_training.py          # End-to-end model training loop
├── src/Fraudsentinel/           # Core pipeline modules
│   ├── drift_check.py           # PSI calculation engine & severity mapping
│   ├── evaluate.py              # Precision, Recall, F1, PR-AUC evaluation
│   ├── graph_construction.py    # Topological feature engineering & PyG assembly
│   ├── logger.py                # Logging configuration
│   ├── models.py                # XGBoost, GraphSAGE, GCN, GAT definitions
│   ├── monitor.py               # Prediction batch logging & probability shift hooks
│   └── train.py                 # PyTorch & XGBoost training loops
├── tests/                       # Automated unit tests
├── requirements.txt             # Locked dependencies
└── setup.py                     # Package setup script
```

---

## 📜 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.