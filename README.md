# FraudSentinel

**Graph neural network-based fraud and money-laundering detection on the Elliptic Bitcoin dataset.** The project compares a tabular baseline (XGBoost) against three GNN architectures (GraphSAGE, GCN, GAT) to evaluate whether transaction network structure improves fraud detection beyond individual transaction features alone. The pipeline covers end-to-end development from exploratory data analysis through hyperparameter optimization, semi-supervised label propagation, and model explainability — with an honest accounting of what works, what doesn't, and exactly why.

---

## ✅ Project Status (Weeks 1–10 Complete)

| Week | Phase | Status |
| :--- | :--- | :---: |
| 1 | Exploratory Data Analysis (EDA) | ✅ Done |
| 2 | Graph Construction & Feature Engineering | ✅ Done |
| 3–4 | Baseline (XGBoost) + GNN Modeling, Bug Fixes | ✅ Done |
| 5 | GCN & GAT Architecture Additions | ✅ Done |
| 6 | Hyperparameter Sweeps (Optuna) | ✅ Done |
| 7 | Architectural Comparison & Winner Analysis | ✅ Done |
| 8 | Semi-Supervised Label Propagation | ✅ Done |
| 9 | Explainability (GNNExplainer + Captum) | ✅ Done |
| 10 | Benchmark vs. Published Literature | ✅ Done |

---

## 🔑 Key Findings by Phase

### Week 1 — EDA
- **Severe class imbalance**: ~2.23% illicit within the full dataset (203,769 nodes); ~9.76% within the labeled subset. PR-AUC and F1-illicit used as evaluation metrics throughout instead of accuracy.
- **Neighborhood clustering**: Neighbors of illicit nodes — in *both* successor and predecessor directions — are illicit at **2.40× the base rate**, empirically justifying the use of GNNs.
- **Temporal structure**: 49 timesteps of ~2 weeks each, strictly chronologically split: train=1–34, val=35–42, test=43–49. Zero cross-timestep edges exist, meaning the graph consists of 49 disconnected temporal subgraphs.

### Week 2 — Graph Construction & Feature Engineering
- Parsed raw Elliptic CSVs, remapped transaction IDs to sequential indices, and assembled a PyTorch Geometric `Data` object.
- Engineered 5 topological features per node: `in_degree`, `out_degree`, `total_degree`, `pagerank`, and Louvain `community_id` (used as a community-size ratio). Total feature dimension: 165 + 5 = **170**.

### Weeks 3–4 — Baseline + GNN Modeling & Critical Bug Fixes
Two bugs were identified and fixed during this phase:
1. **Edge directionality bug**: The initial `edge_index` was directed (234,355 edges). Because the Elliptic EDA showed that *both* predecessors and successors of illicit nodes carry anomalous signals, we converted to undirected edges via `to_undirected()`, doubling the edge count to 468,710 and enabling bidirectional message passing.
2. **Normalization leakage bug**: The initial model used `BatchNorm1d`, which computes statistics across all nodes in the full-graph forward pass. During training, this leaked val/test node statistics into the batch normalization parameters. Fixed by switching to `LayerNorm`, which normalizes each node independently and is leakage-free.

### Week 5 — GCN & GAT
Added GCN and GAT architectures to the benchmark, trained on identical splits and identical evaluation protocol. GAT uses 8 attention heads; GCN uses degree-symmetric normalization.

### Week 6 — Hyperparameter Sweeps (Optuna)
Ran Optuna sweeps (10 trials × 35 epochs per model) to optimize learning rate, weight decay, dropout, hidden dimensions, and (for GraphSAGE) aggregation function. Best configurations saved to `models/{model}_best_params.json`.

### Week 7 — Architectural Comparison & Winner
**GraphSAGE (Max)** is the clear winner for drift-resilient fraud detection. Its element-wise max aggregation acts as a logical `OR` gate — preserving the most anomalous signal from any single neighbor — unlike GCN (which dilutes signals isotropically) and GAT (whose learned attention coefficients overfit to pre-drift feature patterns and collapse post-drift).

| Model | Val F1-illicit | Test PR-AUC (drift) |
| :--- | :---: | :---: |
| XGBoost | 0.9090 | 0.0408 |
| GCN (Tuned) | 0.7305 | 0.0507 |
| **GraphSAGE (Max)** | **0.7239** | **0.0663** |
| GAT (Tuned) | 0.6289 | 0.0424 |

### Week 8 — Semi-Supervised Label Propagation
Applied PyG's `LabelPropagation` to generate pseudo-labels for the unlabeled ~77% of nodes. Used strict confidence thresholds (0.95 licit, 0.90 illicit) to produce high-quality pseudo-labels (181 licit, 55 illicit). Retraining GraphSAGE with these pseudo-labels yielded a +2.1% relative lift in test PR-AUC and +3.6% lift in test F1-illicit.

### Week 9 — Explainability (GNNExplainer + Captum)
Ran explanations on 5 test-split case studies (True Positive, True Negative, False Positive, False Negative, High-Degree TP) using:
- **GNNExplainer**: Edge mask + feature mask optimization (200 epochs per node) on local 2-hop receptive field subgraphs (2-hop subgraph extraction reduced runtime from ~55 minutes to ~13 seconds).
- **Captum (Integrated Gradients)**: Feature-only attribution as a comparison baseline.

**Key failure mode diagnosed**: A structurally isolated illicit node (degree = 1, connected only to a licit address) was completely missed by the GNN — max-pooling propagated a clean signal from the single licit neighbor, defeating the detector.

### Week 10 — Benchmark vs. Literature
Our pipeline outperforms the published Weber et al. (2019) baseline on the stable validation split (XGBoost: 0.804 → 0.909; GCN: 0.574 → 0.731), driven by topological feature engineering and the edge directionality + LayerNorm fixes.

**Honest caveat on temporal drift**: Under strict chronological evaluation with a threshold locked on validation, GraphSAGE's test F1 drops sharply due to the concept drift documented by Elliptic (Bellei, 2019) — the disappearance of a major dark market at timestep 43. The PR-AUC ranking metric (which is threshold-free) reveals a consistent ~1.6× lift over XGBoost, but any production deployment would require dynamic threshold recalibration. See `reports/elliptic_benchmark_comparison.md` for the full analysis.

---

## 📁 Dataset

[Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) — place raw CSVs in `data/raw/` (not tracked in git).

---

## ⚙️ Setup

```powershell
python -m venv Fsenv
.\Fsenv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

> **Note on `requirements.txt`**: The current `requirements.txt` lists package names **without version pins**. Before reproducing results, run `pip freeze > requirements.txt` in your activated environment to lock exact versions. See Fix #4 note in the project log.

---

## 📂 Project Structure

```
FraudSentinel/
├── data/                        # Raw and processed data (gitignored)
│   ├── raw/                     # Place Elliptic CSVs here
│   └── processed/               # Serialized PyG Data objects
├── models/                      # Saved model checkpoints & hyperparameter configs
├── reports/                     # Analysis reports and figures
│   ├── figures/
│   │   └── explainability/      # GNNExplainer + Captum case-study plots
│   ├── gnn_comparison_analysis.md
│   ├── elliptic_benchmark_comparison.md
│   ├── explainability_report.md
│   └── project_summary.md
├── research/                    # EDA and exploratory notebooks
│   └── EDA.ipynb
├── scripts/                     # Executable pipeline entry points
│   ├── run_training.py          # Train all models
│   ├── run_sweeps.py            # Optuna hyperparameter sweeps
│   ├── run_label_propagation.py # Semi-supervised label propagation
│   ├── run_training_pseudo.py   # Retrain with pseudo-labels
│   └── run_explainability.py   # GNNExplainer + Captum case studies
├── src/Fraudsentinel/           # Reusable pipeline modules
│   ├── models.py                # XGBoost, GraphSAGE, GCN, GAT definitions
│   ├── train.py                 # Training loops with pseudo-label support
│   ├── evaluate.py              # PR-AUC, F1-illicit, threshold sweep
│   └── graph_construction.py   # Data loading, feature engineering, PyG assembly
├── tests/                       # Unit tests
├── requirements.txt
└── setup.py
```