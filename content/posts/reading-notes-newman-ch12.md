---
title: "Reading Notes: Newman's Networks Chapter 12 - Network Inference"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 12 of Newman's 'Networks: An Introduction' covering statistical inference, network reconstruction, and missing data imputation"
tags: ["reading-notes", "network-theory", "network-inference", "statistical-inference", "network-reconstruction"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 12 of Newman's *Networks: An Introduction* explores **network inference** - the process of reconstructing or inferring network structure from partial or noisy data. This chapter covers statistical methods, machine learning approaches, and practical techniques for network reconstruction.

## 12.1 Statistical Inference

### Maximum Likelihood Estimation

#### Likelihood Function

**Likelihood function**:
$$L(\theta) = \prod_{i,j} P(A_{ij}|\theta)$$

Where:
- $A_{ij}$: Observed adjacency matrix
- $\theta$: Model parameters
- $P(A_{ij}|\theta)$: Probability of edge $(i,j)$ given parameters

#### Log-Likelihood

**Log-likelihood**:
$$\ell(\theta) = \sum_{i,j} \log P(A_{ij}|\theta)$$

**MLE solution**:
$$\hat{\theta} = \arg\max_{\theta} \ell(\theta)$$

#### Gradient Descent

**Gradient**:
$$\frac{\partial \ell}{\partial \theta} = \sum_{i,j} \frac{1}{P(A_{ij}|\theta)} \frac{\partial P(A_{ij}|\theta)}{\partial \theta}$$

**Update rule**:
$$\theta^{(t+1)} = \theta^{(t)} + \alpha \frac{\partial \ell}{\partial \theta}$$

### Bayesian Inference

#### Prior Distribution

**Prior distribution**:
$$P(\theta) = \text{prior knowledge about parameters}$$

**Common priors**:
- **Uniform**: $P(\theta) = \text{constant}$
- **Gaussian**: $P(\theta) = \mathcal{N}(\mu, \sigma^2)$
- **Beta**: $P(\theta) = \text{Beta}(\alpha, \beta)$

#### Posterior Distribution

**Posterior distribution**:
$$P(\theta|A) = \frac{P(A|\theta)P(\theta)}{P(A)}$$

Where $P(A)$ is the marginal likelihood:
$$P(A) = \int P(A|\theta)P(\theta) \, d\theta$$

#### Markov Chain Monte Carlo

**Metropolis-Hastings algorithm**:
1. Propose new parameter $\theta'$ from proposal distribution $q(\theta'|\theta)$
2. Accept with probability:
   $$P_{\text{accept}} = \min\left(1, \frac{P(\theta'|A)q(\theta|\theta')}{P(\theta|A)q(\theta'|\theta)}\right)$$
3. Repeat until convergence

### Model Selection

#### Information Criteria

**Akaike Information Criterion (AIC)**:
$$AIC = -2\ell(\hat{\theta}) + 2k$$

**Bayesian Information Criterion (BIC)**:
$$BIC = -2\ell(\hat{\theta}) + k \log n$$

Where $k$ is the number of parameters.

#### Cross-Validation

**k-fold cross-validation**:
1. Split data into $k$ folds
2. Train on $k-1$ folds, test on remaining fold
3. Repeat for all folds
4. Average performance

## 12.2 Network Reconstruction

### Link Prediction

#### Similarity Measures

**Common neighbors**:
$$S_{ij} = |\mathcal{N}_i \cap \mathcal{N}_j|$$

**Jaccard coefficient**:
$$S_{ij} = \frac{|\mathcal{N}_i \cap \mathcal{N}_j|}{|\mathcal{N}_i \cup \mathcal{N}_j|}$$

**Adamic-Adar**:
$$S_{ij} = \sum_{k \in \mathcal{N}_i \cap \mathcal{N}_j} \frac{1}{\log k_k}$$

**Preferential attachment**:
$$S_{ij} = k_i \cdot k_j$$

**Resource allocation**:
$$S_{ij} = \sum_{k \in \mathcal{N}_i \cap \mathcal{N}_j} \frac{1}{k_k}$$

#### Prediction Accuracy

**Area Under Curve (AUC)**:
$$AUC = \frac{1}{n(n-1)} \sum_{i \neq j} \mathbb{I}(S_{ij} > S_{\text{random}})$$

**Precision**:
$$P = \frac{TP}{TP + FP}$$

**Recall**:
$$R = \frac{TP}{TP + FN}$$

**F1-score**:
$$F1 = \frac{2PR}{P + R}$$

### Matrix Completion

#### Nuclear Norm Minimization

**Problem**:
$$\min_{X} ||X - A||_F^2 + \lambda ||X||_*$$

Where:
- $||X||_*$: Nuclear norm (sum of singular values)
- $\lambda$: Regularization parameter

#### Alternating Least Squares

**Algorithm**:
1. Initialize $U$ and $V$ randomly
2. Fix $V$, solve for $U$: $U = \arg\min_U ||UV^T - A||_F^2$
3. Fix $U$, solve for $V$: $V = \arg\min_V ||UV^T - A||_F^2$
4. Repeat until convergence

**Solution**: $X = UV^T$

### Graph Neural Networks

#### Graph Convolutional Networks

**Layer update**:
$$H^{(l+1)} = \sigma(\tilde{A} H^{(l)} W^{(l)})$$

Where:
- $\tilde{A} = D^{-1/2} A D^{-1/2}$: Normalized adjacency matrix
- $H^{(l)}$: Node features at layer $l$
- $W^{(l)}$: Weight matrix at layer $l$

#### Link Prediction with GNNs

**Node embeddings**:
$$z_i = \text{GNN}(A, X)_i$$

**Link prediction**:
$$P(A_{ij} = 1) = \sigma(z_i^T z_j)$$

## 12.3 Missing Data Imputation

### Missing Data Types

#### Missing Completely at Random (MCAR)

**Definition**: Missingness is independent of observed and unobserved data

**Mathematical condition**:
$$P(M|X, Y) = P(M)$$

Where $M$ is the missingness indicator.

#### Missing at Random (MAR)

**Definition**: Missingness depends only on observed data

**Mathematical condition**:
$$P(M|X, Y) = P(M|X)$$

#### Missing Not at Random (MNAR)

**Definition**: Missingness depends on unobserved data

**Mathematical condition**:
$$P(M|X, Y) \neq P(M|X)$$

### Imputation Methods

#### Mean Imputation

**Method**: Replace missing values with mean of observed values

**Formula**:
$$\hat{x}_{ij} = \frac{1}{n_j} \sum_{i: x_{ij} \text{ observed}} x_{ij}$$

**Advantages**: Simple, fast
**Disadvantages**: Reduces variance, may bias estimates

#### Regression Imputation

**Method**: Use regression to predict missing values

**Formula**:
$$\hat{x}_{ij} = \beta_0 + \sum_{k \neq j} \beta_k x_{ik}$$

**Advantages**: Preserves relationships
**Disadvantages**: Assumes linear relationships

#### Multiple Imputation

**Method**: Generate multiple imputed datasets

**Algorithm**:
1. Generate $m$ imputed datasets
2. Analyze each dataset separately
3. Combine results using Rubin's rules

**Combined estimate**:
$$\bar{\theta} = \frac{1}{m} \sum_{i=1}^m \hat{\theta}_i$$

**Variance**:
$$\text{Var}(\bar{\theta}) = \frac{1}{m} \sum_{i=1}^m \text{Var}(\hat{\theta}_i) + \frac{1}{m-1} \sum_{i=1}^m (\hat{\theta}_i - \bar{\theta})^2$$

## 12.4 Network Tomography

### Network Discovery

#### Traceroute

**Method**: Send packets with increasing TTL values

**Information obtained**:
- Path from source to destination
- Round-trip time
- Packet loss rate

**Limitations**:
- May not discover all paths
- Load balancing can cause inconsistencies

#### Ping

**Method**: Send ICMP echo requests

**Information obtained**:
- Round-trip time
- Packet loss rate
- Network reachability

**Limitations**:
- Only end-to-end information
- May be blocked by firewalls

### Topology Inference

#### Graph Construction

**From traceroute data**:
1. Collect paths from multiple sources to multiple destinations
2. Construct graph by connecting consecutive nodes in paths
3. Merge duplicate edges

**From ping data**:
1. Test connectivity between all pairs of nodes
2. Add edge if nodes can communicate
3. May miss intermediate nodes

#### Validation

**Consistency checks**:
- **Triangle inequality**: $d_{ij} \leq d_{ik} + d_{kj}$
- **Symmetry**: $d_{ij} = d_{ji}$
- **Transitivity**: If $A_{ik} = 1$ and $A_{kj} = 1$, then $A_{ij} = 1$

**Statistical validation**:
- Compare inferred topology with known topology
- Use cross-validation techniques

## 12.5 Applications to Materials Science

### Defect Network Inference

#### Problem

**Given**: Partial observations of defect interactions
**Goal**: Infer complete defect network structure

**Mathematical formulation**:
$$\min_{A} ||A - A_{\text{observed}}||_F^2 + \lambda R(A)$$

Where $R(A)$ is a regularization term.

#### Methods

**Sparse reconstruction**:
$$R(A) = ||A||_1$$

**Low-rank reconstruction**:
$$R(A) = ||A||_*$$

**Community structure**:
$$R(A) = \sum_{c} ||A_c||_F^2$$

### Nanowire Network Reconstruction

#### Problem

**Given**: Partial electrical measurements
**Goal**: Infer complete nanowire network

**Constraints**:
- **Electrical**: Ohm's law must be satisfied
- **Topological**: Network must be connected
- **Physical**: Edge weights must be positive

#### Optimization

**Objective function**:
$$\min_{A, R} \sum_{i,j} (V_i - V_j - R_{ij} I_{ij})^2 + \lambda ||A||_1$$

**Constraints**:
- $A_{ij} \geq 0$ (positive resistance)
- $A_{ij} = 0$ if no nanowire between $i$ and $j$
- $\sum_j A_{ij} > 0$ (each node must have at least one connection)

### Phase Transition Inference

#### Problem

**Given**: Partial observations of phase transitions
**Goal**: Infer complete phase diagram

**Network representation**:
- **Nodes**: Different phases
- **Edges**: Phase transitions
- **Weights**: Transition probabilities

#### Methods

**Bayesian inference**:
$$P(\text{phase}|T, P) = \frac{P(T, P|\text{phase})P(\text{phase})}{P(T, P)}$$

**Machine learning**:
$$f: (T, P) \rightarrow \text{phase}$$

## Code Example: Network Inference

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from collections import defaultdict

def generate_partial_network(G, missing_fraction=0.3):
    """Generate partial network by removing edges"""
    
    edges = list(G.edges())
    n_remove = int(len(edges) * missing_fraction)
    edges_to_remove = np.random.choice(len(edges), n_remove, replace=False)
    
    partial_G = G.copy()
    partial_G.remove_edges_from([edges[i] for i in edges_to_remove])
    
    return partial_G, edges_to_remove

def calculate_similarity_measures(G):
    """Calculate various similarity measures for link prediction"""
    
    n = G.number_of_nodes()
    similarities = {}
    
    # Common neighbors
    cn = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            cn[i, j] = len(set(G.neighbors(i)) & set(G.neighbors(j)))
            cn[j, i] = cn[i, j]
    similarities['common_neighbors'] = cn
    
    # Jaccard coefficient
    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            neighbors_i = set(G.neighbors(i))
            neighbors_j = set(G.neighbors(j))
            union = neighbors_i | neighbors_j
            if len(union) > 0:
                jaccard[i, j] = len(neighbors_i & neighbors_j) / len(union)
                jaccard[j, i] = jaccard[i, j]
    similarities['jaccard'] = jaccard
    
    # Adamic-Adar
    aa = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            common_neighbors = set(G.neighbors(i)) & set(G.neighbors(j))
            aa[i, j] = sum(1/np.log(G.degree(k)) for k in common_neighbors if G.degree(k) > 1)
            aa[j, i] = aa[i, j]
    similarities['adamic_adar'] = aa
    
    # Preferential attachment
    pa = np.zeros((n, n))
    degrees = [G.degree(i) for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            pa[i, j] = degrees[i] * degrees[j]
            pa[j, i] = pa[i, j]
    similarities['preferential_attachment'] = pa
    
    return similarities

def predict_links(G, similarities, method='common_neighbors'):
    """Predict links using similarity measures"""
    
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G).toarray()
    
    # Get similarity scores
    scores = similarities[method]
    
    # Create training and test sets
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    labels = [A[i, j] for i, j in edges]
    features = [scores[i, j] for i, j in edges]
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42
    )
    
    # Train classifier
    clf = LogisticRegression(random_state=42)
    clf.fit(np.array(X_train).reshape(-1, 1), y_train)
    
    # Predict probabilities
    y_pred_proba = clf.predict_proba(np.array(X_test).reshape(-1, 1))[:, 1]
    y_pred = clf.predict(np.array(X_test).reshape(-1, 1))
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba
    }

def matrix_completion(A_observed, rank=5, lambda_reg=0.1):
    """Perform matrix completion using nuclear norm minimization"""
    
    def objective(X_flat):
        X = X_flat.reshape(A_observed.shape)
        # Nuclear norm (sum of singular values)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        nuclear_norm = np.sum(s)
        
        # Frobenius norm of observed entries
        mask = ~np.isnan(A_observed)
        frobenius_norm = np.sum((X[mask] - A_observed[mask])**2)
        
        return frobenius_norm + lambda_reg * nuclear_norm
    
    # Initialize with random matrix
    X_init = np.random.randn(*A_observed.shape)
    X_init = X_init.flatten()
    
    # Optimize
    result = minimize(objective, X_init, method='L-BFGS-B')
    X_completed = result.x.reshape(A_observed.shape)
    
    return X_completed

def impute_missing_data(A, method='mean'):
    """Impute missing data in adjacency matrix"""
    
    A_imputed = A.copy()
    
    if method == 'mean':
        # Mean imputation
        mean_val = np.nanmean(A)
        A_imputed[np.isnan(A)] = mean_val
    
    elif method == 'regression':
        # Regression imputation
        for j in range(A.shape[1]):
            missing_mask = np.isnan(A[:, j])
            if np.any(missing_mask) and not np.all(missing_mask):
                # Use other columns to predict missing values
                X = A[:, ~missing_mask]
                y = A[~missing_mask, j]
                
                if X.shape[1] > 0:
                    # Simple linear regression
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    A_imputed[missing_mask, j] = X[missing_mask] @ beta
    
    elif method == 'matrix_completion':
        # Matrix completion
        A_imputed = matrix_completion(A)
    
    return A_imputed

def evaluate_reconstruction(G_original, G_reconstructed):
    """Evaluate network reconstruction quality"""
    
    A_original = nx.adjacency_matrix(G_original).toarray()
    A_reconstructed = nx.adjacency_matrix(G_reconstructed).toarray()
    
    # Edge accuracy
    edge_accuracy = np.mean(A_original == A_reconstructed)
    
    # Precision and recall
    tp = np.sum((A_original == 1) & (A_reconstructed == 1))
    fp = np.sum((A_original == 0) & (A_reconstructed == 1))
    fn = np.sum((A_original == 1) & (A_reconstructed == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Structural similarity
    degree_correlation = np.corrcoef(
        [G_original.degree(i) for i in G_original.nodes()],
        [G_reconstructed.degree(i) for i in G_reconstructed.nodes()]
    )[0, 1]
    
    return {
        'edge_accuracy': edge_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'degree_correlation': degree_correlation
    }

def plot_inference_results(G_original, G_partial, G_reconstructed, 
                          link_prediction_results, title="Network Inference"):
    """Plot network inference results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original network
    pos = nx.spring_layout(G_original, k=1, iterations=50)
    nx.draw(G_original, pos, ax=ax1, node_color='lightblue', 
            edge_color='gray', alpha=0.6, node_size=50)
    ax1.set_title('Original Network')
    ax1.axis('off')
    
    # Partial network
    nx.draw(G_partial, pos, ax=ax2, node_color='lightcoral', 
            edge_color='gray', alpha=0.6, node_size=50)
    ax2.set_title('Partial Network (Missing Edges)')
    ax2.axis('off')
    
    # Reconstructed network
    nx.draw(G_reconstructed, pos, ax=ax3, node_color='lightgreen', 
            edge_color='gray', alpha=0.6, node_size=50)
    ax3.set_title('Reconstructed Network')
    ax3.axis('off')
    
    # Link prediction ROC curve
    if 'y_test' in link_prediction_results and 'y_pred_proba' in link_prediction_results:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(link_prediction_results['y_test'], 
                               link_prediction_results['y_pred_proba'])
        ax4.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {link_prediction_results["auc"]:.3f}')
        ax4.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('Link Prediction ROC Curve')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Example: Network inference
G = nx.barabasi_albert_graph(50, 3)

# Generate partial network
G_partial, removed_edges = generate_partial_network(G, missing_fraction=0.3)

# Calculate similarity measures
similarities = calculate_similarity_measures(G_partial)

# Predict links
link_prediction_results = predict_links(G_partial, similarities, method='common_neighbors')

# Reconstruct network
A_partial = nx.adjacency_matrix(G_partial).toarray()
A_reconstructed = impute_missing_data(A_partial, method='matrix_completion')

# Create reconstructed network
G_reconstructed = nx.from_numpy_array(A_reconstructed)
G_reconstructed = nx.Graph(G_reconstructed)  # Remove self-loops and parallel edges

# Evaluate reconstruction
evaluation = evaluate_reconstruction(G, G_reconstructed)

print("Network Inference Results:")
print(f"Edge accuracy: {evaluation['edge_accuracy']:.3f}")
print(f"Precision: {evaluation['precision']:.3f}")
print(f"Recall: {evaluation['recall']:.3f}")
print(f"F1-score: {evaluation['f1']:.3f}")
print(f"Degree correlation: {evaluation['degree_correlation']:.3f}")
print(f"Link prediction AUC: {link_prediction_results['auc']:.3f}")

# Plot results
plot_inference_results(G, G_partial, G_reconstructed, link_prediction_results)
```

## Key Takeaways

1. **Statistical inference**: Maximum likelihood and Bayesian methods for parameter estimation
2. **Link prediction**: Various similarity measures for predicting missing edges
3. **Matrix completion**: Nuclear norm minimization for network reconstruction
4. **Missing data**: Different types and imputation methods
5. **Network tomography**: Inferring topology from partial observations
6. **Applications**: Important for materials science and defect networks
7. **Evaluation**: Multiple metrics for assessing reconstruction quality

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Liben-Nowell, D., & Kleinberg, J. (2007). The link-prediction problem for social networks. Journal of the American Society for Information Science and Technology, 58(7), 1019-1031.
3. Candès, E. J., & Recht, B. (2009). Exact matrix completion via convex optimization. Foundations of Computational Mathematics, 9(6), 717-772.
4. Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.

---

*Network inference provides powerful tools for reconstructing network structure from partial data, with important applications in materials science and complex systems.*
