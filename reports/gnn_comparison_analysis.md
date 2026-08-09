# GNN Architectural Comparison & Winner Deep-Dive Analysis

This report provides a formal technical comparison of the three Graph Neural Network (GNN) architectures (**GraphSAGE (Max)**, **GCN**, and **GAT**) trained and optimized on the Elliptic Bitcoin dataset. It breaks down the mathematical differences in their aggregation functions and explains why GraphSAGE (Max) emerged as the clear winner—especially when subjected to severe concept drift.

---

## 📊 GNN Architectural Comparison Table

The table below contrasts the structural properties and empirical performance of the three tuned GNN architectures alongside the tabular baseline:

| Dimension / Metric | XGBoost Baseline | GCN Baseline (Tuned) | GraphSAGE (Max) (Tuned) | GAT (Attention) (Tuned) |
| :--- | :--- | :--- | :--- | :--- |
| **Mathematical Aggregation** | Local Splitting | Isotropic Average: $\sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i d_j}} x_j$ | Element-wise Max: $\max_{j \in \mathcal{N}(i)} (W x_j)$ | Anisotropic Attention: $\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W x_j$ |
| **Feature Normalization** | Global scaling (pre-fit) | LayerNorm (Per-node stats) | LayerNorm (Per-node stats) | LayerNorm (Per-node stats) |
| **Edge Weighting Type** | N/A | Hardcoded (degree symmetric) | Hardcoded (uniform/max pool) | Learned Dynamically ($\alpha_{ij}$) |
| **Parameter Count Complexity** | $O(\text{trees} \times \text{leaves})$ | Low ($O(d_{in} d_{out})$) | Medium ($O(2 \times d_{in} d_{out})$) | High ($O(H \times d_{in} d_{out})$) |
| **Validation F1-illicit** | **0.9090** | 0.7305 | 0.7239 | 0.6289 |
| **Validation PR-AUC** | **0.9210** | 0.7638 | 0.7916 | 0.6363 |
| **Test PR-AUC (Timestep 43-49)** | 0.0408 | 0.0507 | **0.0663** | 0.0424 |
| **Drift Generalization Lift vs XGB**| 1.00x (Baseline) | 1.24x | **1.63x** | 1.04x |
| **Primary Failure Mode** | Feature shift blind spot | Isotropic dilution of fraud | Slightly lower precision | Overfitting dynamic coefficients |

---

## 🏆 The Winner: GraphSAGE (Max)

### 1. Mathematical Rationale (Element-wise Max Pooling)
In financial network analysis and transaction-level fraud detection, fraud indicators are often sparse and extreme (e.g., highly anomalous transaction fees, specific hop-counts from high-risk entities, or extreme outbound transaction velocity). 

The GraphSAGE (Max) model aggregates neighborhood information using the element-wise maximum operator:
$$h_{\mathcal{N}(i)}^{(k)} = \max \left( \{ \sigma(W_{pool} h_j^{(k-1)} + b) , \forall j \in \mathcal{N}(i) \} \right)$$

#### Why Max-Pooling is an Optimal Anomaly Detector:
* **The Anomaly Flag Behavior**: Element-wise maximum acts as a logical `OR` gate. If even *one* transaction partner in a node's local neighborhood has a highly suspicious feature value (e.g., extremely high degree or high-risk classification), the `max` operator preserves and propagates this peak signal. 
* **Zero Signal Dilution**: Unlike averaging, which dilutes the fraud signal when a node is connected to many normal (licit) entities, max pooling is size-invariant. It guarantees that the presence of normal transaction counterparts cannot mask or wash out a single high-risk connection.

---

## 🔍 Why GCN and GAT Fell Short

### 1. GCN: Isotropic Average Dilution
A Graph Convolutional Network (GCN) aggregates neighborhood features symmetrically:
$$h_i^{(k)} = \sigma \left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} W h_j^{(k-1)} \right)$$

* **The Dilution Problem**: GCN is an *isotropic* filter—it weights all neighbors uniformly, adjusted only by their static degrees. When an illicit transaction node interacts with large exchanges or utility addresses (which have very high degrees $d_j$), the normalization denominator $\sqrt{d_i d_j}$ becomes extremely large. 
* **Loss of Anomaly Signals**: The anomalous feature values of the illicit node are averaged into the massive baseline feature statistics of the high-degree normal nodes, causing the GCN classifier to miss the anomaly (validation F1 is high, but test precision drops under feature shifts).

### 2. GAT: Attention Coefficient Collapse
A Graph Attention Network (GAT) computes dynamic attention coefficients $\alpha_{ij}$ to focus on specific neighbors:
$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [W h_i \,\|\, W h_j]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [W h_i \,\|\, W h_k]\right)\right)}$$

* **Overfitting to Stationary Distributions**: GAT's attention mechanism relies entirely on projected node representations ($W h_i$ and $W h_j$). During the training phase (timesteps 1-34), GAT learns to attend to specific feature patterns associated with illicit nodes.
* **Failure Under Concept Drift**: At timestep 43, coinciding with the disappearance of a major dark market as documented by Elliptic (Bellei, 2019), illicit actors modified their local feature distributions (amounts, fees, degrees). Because GAT's attention module $\mathbf{a}^T$ and weight matrix $W$ were trained on pre-event feature structures, the model paid attention to the wrong neighbors post-event, leading to a collapse in validation-to-test generalizability (test PR-AUC = **0.0424**, barely beating XGBoost).

---

## 💡 Practical Takeaways for Fraud Sentinel

1. **Topology beats Raw Features during Drift**: Under stationary conditions (Validation split), local tabular classifiers (XGBoost) easily detect fraud because the feature distribution is stable. However, under non-stationary regimes (post-timestep 43), the tabular model collapses. The GNN's ability to propagate structural anomaly signals is the primary defense against adversarial evasion.
2. **Simpler Aggregations Generalize Better**: In highly adversarial domains where actors actively work to evade detection, simple, non-parametric structural operators (like max pooling) generalize significantly better than complex parameterized aggregators (like GAT self-attention) which are highly prone to overfitting.
