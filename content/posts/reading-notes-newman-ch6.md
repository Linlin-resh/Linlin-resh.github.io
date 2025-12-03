---
title: "Reading Notes: Newman's Networks Chapter 6 - Mathematical Foundations"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 6 of Newman's 'Networks: An Introduction' covering graph theory, matrix representations, and mathematical foundations of network analysis"
tags: ["reading-notes", "network-theory", "mathematics", "graph-theory", "linear-algebra"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 6 of Newman's *Networks: An Introduction* establishes the **mathematical foundations** of network analysis. This chapter provides the essential mathematical tools and concepts needed to understand and analyze complex networks, from basic graph theory to advanced matrix methods.

## 6.1 Graph Theory Fundamentals

### Basic Definitions

#### Graph Components

A **graph** $G = (V, E)$ consists of:

- **Vertices (nodes)**: $V = \{v_1, v_2, \ldots, v_n\}$
- **Edges (links)**: $E = \{(v_i, v_j) : v_i, v_j \in V\}$
- **Order**: $n = |V|$ (number of vertices)
- **Size**: $m = |E|$ (number of edges)

#### Graph Types

**Undirected graph**: Edges have no direction
- **Edge**: $(v_i, v_j) = (v_j, v_i)$
- **Degree**: $k_i = |\{j : (v_i, v_j) \in E\}|$

**Directed graph (digraph)**: Edges have direction
- **Edge**: $(v_i, v_j) \neq (v_j, v_i)$
- **In-degree**: $k_i^{\text{in}} = |\{j : (v_j, v_i) \in E\}|$
- **Out-degree**: $k_i^{\text{out}} = |\{j : (v_i, v_j) \in E\}|$

**Weighted graph**: Edges have weights
- **Weight**: $w_{ij} \in \mathbb{R}$ for edge $(v_i, v_j)$
- **Weighted degree**: $k_i^w = \sum_{j} w_{ij}$

### Graph Properties

#### Connectivity

**Path**: Sequence of vertices $(v_1, v_2, \ldots, v_k)$ where $(v_i, v_{i+1}) \in E$

**Walk**: Path where vertices can be repeated

**Trail**: Path where edges can be repeated

**Cycle**: Path where $v_1 = v_k$ and all other vertices are distinct

**Connected graph**: Path exists between any two vertices

**Strongly connected**: Directed path exists between any two vertices

#### Distance and Diameter

**Distance**: $d_{ij} = \min\{\text{length of path from } i \text{ to } j\}$

**Diameter**: $D = \max_{i,j} d_{ij}$

**Average path length**: $L = \frac{1}{n(n-1)} \sum_{i \neq j} d_{ij}$

#### Clustering

**Local clustering coefficient**:
$$C_i = \frac{2e_i}{k_i(k_i-1)}$$

Where $e_i$ is the number of edges between neighbors of vertex $i$.

**Global clustering coefficient**:
$$C = \frac{1}{n} \sum_{i=1}^n C_i$$

**Transitivity**:
$$T = \frac{3 \times \text{number of triangles}}{\text{number of connected triples}}$$

## 6.2 Matrix Representations

### Adjacency Matrix

#### Definition

For an undirected graph:
$$A_{ij} = \begin{cases} 
1 & \text{if } (v_i, v_j) \in E \\
0 & \text{otherwise}
\end{cases}$$

For a directed graph:
$$A_{ij} = \begin{cases} 
1 & \text{if } (v_i, v_j) \in E \\
0 & \text{otherwise}
\end{cases}$$

#### Properties

**Undirected graphs**:
- **Symmetric**: $A = A^T$
- **Trace**: $\text{tr}(A) = 0$ (no self-loops)
- **Sum of rows**: $\sum_j A_{ij} = k_i$ (degree of vertex $i$)

**Directed graphs**:
- **Asymmetric**: $A \neq A^T$ (in general)
- **Row sums**: $\sum_j A_{ij} = k_i^{\text{out}}$ (out-degree)
- **Column sums**: $\sum_i A_{ij} = k_j^{\text{in}}$ (in-degree)

#### Powers of Adjacency Matrix

**$A^2$ interpretation**:
$$(A^2)_{ij} = \sum_k A_{ik} A_{kj} = \text{number of paths of length 2 from } i \text{ to } j$$

**General power**:
$$(A^k)_{ij} = \text{number of paths of length } k \text{ from } i \text{ to } j$$

**Reachability matrix**:
$$R = \sum_{k=1}^{n-1} A^k$$

Where $R_{ij} = 1$ if there exists a path from $i$ to $j$.

### Laplacian Matrix

#### Definition

**Laplacian matrix**:
$$L = D - A$$

Where $D$ is the **degree matrix**:
$$D_{ij} = \begin{cases} 
k_i & \text{if } i = j \\
0 & \text{otherwise}
\end{cases}$$

#### Properties

**Eigenvalues**:
- **Smallest eigenvalue**: $\lambda_1 = 0$ (always)
- **Multiplicity of 0**: Number of connected components
- **Second smallest eigenvalue**: $\lambda_2 > 0$ if graph is connected

**Eigenvectors**:
- **First eigenvector**: $\mathbf{v}_1 = \frac{1}{\sqrt{n}} \mathbf{1}$ (constant vector)
- **Fiedler vector**: Second eigenvector (used for graph partitioning)

#### Normalized Laplacian

**Symmetric normalized Laplacian**:
$$L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

**Random walk normalized Laplacian**:
$$L_{\text{rw}} = D^{-1} L = I - D^{-1} A$$

### Incidence Matrix

#### Definition

**Incidence matrix** $B$:
$$B_{ie} = \begin{cases} 
1 & \text{if vertex } i \text{ is incident to edge } e \\
0 & \text{otherwise}
\end{cases}$$

#### Properties

**Dimensions**: $n \times m$ (vertices × edges)

**Rank**: $\text{rank}(B) = n - c$ where $c$ is the number of connected components

**Laplacian relationship**:
$$L = B B^T$$

## 6.3 Spectral Graph Theory

### Eigenvalue Analysis

#### Adjacency Matrix Eigenvalues

**Spectral radius**: $\rho(A) = \max_i |\lambda_i|$

**Perron-Frobenius theorem**: For connected graphs, the largest eigenvalue is:
- **Real and positive**: $\lambda_1 > 0$
- **Simple**: Multiplicity 1
- **Positive eigenvector**: All components > 0

#### Laplacian Eigenvalues

**Spectral gap**: $\lambda_2 - \lambda_1 = \lambda_2$

**Cheeger's inequality**:
$$\frac{\lambda_2}{2} \leq h(G) \leq \sqrt{2\lambda_2}$$

Where $h(G)$ is the **Cheeger constant**:
$$h(G) = \min_{S} \frac{|\partial S|}{\min(|S|, |V \setminus S|)}$$

### Graph Partitioning

#### Spectral Partitioning

**Algorithm**:
1. Compute **Fiedler vector** (second eigenvector of Laplacian)
2. Sort vertices by Fiedler vector values
3. Cut at median value

**Mathematical foundation**:
$$\min_{\mathbf{x}} \mathbf{x}^T L \mathbf{x} \quad \text{subject to } \mathbf{x}^T \mathbf{1} = 0, \mathbf{x}^T \mathbf{x} = n$$

**Solution**: Fiedler vector

#### Ratio Cut

**Ratio cut**:
$$\text{RatioCut}(S, T) = \frac{\text{cut}(S, T)}{|S|} + \frac{\text{cut}(S, T)}{|T|}$$

**Spectral relaxation**:
$$\min_{\mathbf{x}} \mathbf{x}^T L \mathbf{x} \quad \text{subject to } \mathbf{x}^T \mathbf{1} = 0, \mathbf{x}^T \mathbf{x} = n$$

## 6.4 Random Walks

### Transition Matrix

#### Definition

**Transition matrix**:
$$P_{ij} = \frac{A_{ij}}{k_i}$$

**Properties**:
- **Row stochastic**: $\sum_j P_{ij} = 1$
- **Eigenvalues**: $1 = \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n \geq -1$

#### Stationary Distribution

**Stationary distribution**:
$$\pi_i = \frac{k_i}{2m}$$

**Verification**:
$$\sum_i \pi_i P_{ij} = \sum_i \frac{k_i}{2m} \frac{A_{ij}}{k_i} = \frac{1}{2m} \sum_i A_{ij} = \frac{k_j}{2m} = \pi_j$$

### Mixing Time

#### Definition

**Mixing time**:
$$\tau_{\text{mix}} = \min\{t : \max_i ||P^t(i, \cdot) - \pi||_1 \leq \epsilon\}$$

**Spectral bound**:
$$\tau_{\text{mix}} \leq \frac{1}{1 - \lambda_2} \log\left(\frac{1}{\epsilon \pi_{\min}}\right)$$

Where $\pi_{\min} = \min_i \pi_i$.

### PageRank

#### Definition

**PageRank**:
$$\mathbf{PR} = (1-d) \frac{\mathbf{1}}{n} + d P^T \mathbf{PR}$$

**Matrix form**:
$$\mathbf{PR} = (1-d) \frac{\mathbf{1}}{n} + d \mathbf{M} \mathbf{PR}$$

Where $\mathbf{M}$ is the **stochastic matrix**:
$$M_{ij} = \frac{A_{ji}}{k_j^{\text{out}}}$$

#### Power Iteration

**Iterative solution**:
$$\mathbf{PR}^{(t+1)} = (1-d) \frac{\mathbf{1}}{n} + d \mathbf{M}^T \mathbf{PR}^{(t)}$$

**Convergence**:
$$||\mathbf{PR}^{(t+1)} - \mathbf{PR}^{(t)}||_1 < \epsilon$$

## 6.5 Centrality Measures

### Degree Centrality

**Definition**:
$$C_D(i) = \frac{k_i}{n-1}$$

**Normalized**: $0 \leq C_D(i) \leq 1$

### Betweenness Centrality

**Definition**:
$$C_B(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}$$

Where:
- $\sigma_{st}$: Number of shortest paths from $s$ to $t$
- $\sigma_{st}(i)$: Number of shortest paths from $s$ to $t$ passing through $i$

**Normalized**:
$$C_B^{\text{norm}}(i) = \frac{C_B(i)}{(n-1)(n-2)/2}$$

### Closeness Centrality

**Definition**:
$$C_C(i) = \frac{n-1}{\sum_{j \neq i} d_{ij}}$$

**Normalized**: $0 \leq C_C(i) \leq 1$

### Eigenvector Centrality

**Definition**:
$$x_i = \frac{1}{\lambda} \sum_j A_{ij} x_j$$

**Matrix form**:
$$A \mathbf{x} = \lambda \mathbf{x}$$

**Solution**: Principal eigenvector of adjacency matrix

### Katz Centrality

**Definition**:
$$x_i = \alpha \sum_j A_{ij} x_j + \beta$$

**Matrix form**:
$$\mathbf{x} = \alpha A \mathbf{x} + \beta \mathbf{1}$$

**Solution**:
$$\mathbf{x} = \beta (I - \alpha A)^{-1} \mathbf{1}$$

**Convergence**: Requires $\alpha < \frac{1}{\rho(A)}$

## 6.6 Applications to Materials Science

### Network Topology in Materials

#### Atomic Networks

**Network representation**:
- **Nodes**: Atoms
- **Edges**: Chemical bonds
- **Weights**: Bond strength, distance

**Mathematical analysis**:
- **Degree distribution**: Coordination number distribution
- **Clustering**: Local atomic environment
- **Path length**: Atomic connectivity

#### Defect Networks

**Network properties**:
- **Connectivity**: Defect percolation
- **Clustering**: Defect clustering
- **Centrality**: Critical defects

**Mathematical framework**:
$$P(\text{percolation}) = 1 - \exp\left(-\frac{\langle k^2 \rangle}{\langle k \rangle} p\right)$$

### Network-Based Materials Design

#### Structure-Property Relationships

**Network descriptors**:
- **Average degree**: $\langle k \rangle$
- **Clustering coefficient**: $C$
- **Path length**: $L$
- **Centrality measures**: $C_D, C_B, C_C$

**Property prediction**:
$$P = f(\langle k \rangle, C, L, \ldots)$$

#### Optimization

**Objective function**:
$$\min_{\text{network}} \sum_i w_i |P_i - P_i^{\text{target}}|^2$$

**Constraints**:
- **Connectivity**: Network must be connected
- **Degree bounds**: $k_{\min} \leq k_i \leq k_{\max}$
- **Clustering bounds**: $C_{\min} \leq C \leq C_{\max}$

## Code Example: Mathematical Foundations

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.sparse import csgraph

def analyze_graph_matrices(G):
    """Analyze graph using matrix representations"""
    
    # Adjacency matrix
    A = nx.adjacency_matrix(G).toarray()
    n = A.shape[0]
    
    # Degree matrix
    degrees = [d for n, d in G.degree()]
    D = np.diag(degrees)
    
    # Laplacian matrix
    L = D - A
    
    # Normalized Laplacian
    D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
    L_norm = D_sqrt_inv @ L @ D_sqrt_inv
    
    # Eigenvalue analysis
    eigenvals_A, eigenvecs_A = eig(A)
    eigenvals_L, eigenvecs_L = eig(L)
    eigenvals_L_norm, eigenvecs_L_norm = eig(L_norm)
    
    # Sort eigenvalues
    eigenvals_A = np.real(eigenvals_A)
    eigenvals_L = np.real(eigenvals_L)
    eigenvals_L_norm = np.real(eigenvals_L_norm)
    
    idx_A = np.argsort(eigenvals_A)[::-1]
    idx_L = np.argsort(eigenvals_L)
    idx_L_norm = np.argsort(eigenvals_L_norm)
    
    eigenvals_A = eigenvals_A[idx_A]
    eigenvals_L = eigenvals_L[idx_L]
    eigenvals_L_norm = eigenvals_L_norm[idx_L_norm]
    
    return {
        'adjacency_matrix': A,
        'degree_matrix': D,
        'laplacian_matrix': L,
        'normalized_laplacian': L_norm,
        'adjacency_eigenvalues': eigenvals_A,
        'laplacian_eigenvalues': eigenvals_L,
        'normalized_laplacian_eigenvalues': eigenvals_L_norm
    }

def compute_centrality_measures(G):
    """Compute various centrality measures"""
    
    # Degree centrality
    degree_centrality = nx.degree_centrality(G)
    
    # Betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Closeness centrality
    closeness_centrality = nx.closeness_centrality(G)
    
    # Eigenvector centrality
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Katz centrality
    katz_centrality = nx.katz_centrality(G, alpha=0.1)
    
    # PageRank
    pagerank = nx.pagerank(G, alpha=0.85)
    
    return {
        'degree_centrality': degree_centrality,
        'betweenness_centrality': betweenness_centrality,
        'closeness_centrality': closeness_centrality,
        'eigenvector_centrality': eigenvector_centrality,
        'katz_centrality': katz_centrality,
        'pagerank': pagerank
    }

def analyze_spectral_properties(G):
    """Analyze spectral properties of the graph"""
    
    # Matrix analysis
    matrix_results = analyze_graph_matrices(G)
    
    # Spectral gap
    laplacian_eigenvals = matrix_results['laplacian_eigenvalues']
    spectral_gap = laplacian_eigenvals[1] - laplacian_eigenvals[0]
    
    # Cheeger constant (approximation)
    cheeger_constant = spectral_gap / 2
    
    # Mixing time (approximation)
    if spectral_gap > 0:
        mixing_time = 1 / spectral_gap
    else:
        mixing_time = float('inf')
    
    # Fiedler vector
    L = matrix_results['laplacian_matrix']
    eigenvals, eigenvecs = eig(L)
    eigenvals = np.real(eigenvals)
    eigenvecs = np.real(eigenvecs)
    
    idx = np.argsort(eigenvals)
    fiedler_vector = eigenvecs[:, idx[1]]
    
    return {
        'spectral_gap': spectral_gap,
        'cheeger_constant': cheeger_constant,
        'mixing_time': mixing_time,
        'fiedler_vector': fiedler_vector
    }

def plot_spectral_analysis(G, title="Spectral Analysis"):
    """Plot spectral analysis results"""
    
    matrix_results = analyze_graph_matrices(G)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Adjacency matrix eigenvalues
    eigenvals_A = matrix_results['adjacency_eigenvalues']
    ax1.plot(eigenvals_A, 'bo-', markersize=8)
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Adjacency Matrix Eigenvalues')
    ax1.grid(True, alpha=0.3)
    
    # Laplacian matrix eigenvalues
    eigenvals_L = matrix_results['laplacian_eigenvalues']
    ax2.plot(eigenvals_L, 'ro-', markersize=8)
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Eigenvalue')
    ax2.set_title('Laplacian Matrix Eigenvalues')
    ax2.grid(True, alpha=0.3)
    
    # Normalized Laplacian eigenvalues
    eigenvals_L_norm = matrix_results['normalized_laplacian_eigenvalues']
    ax3.plot(eigenvals_L_norm, 'go-', markersize=8)
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Eigenvalue')
    ax3.set_title('Normalized Laplacian Eigenvalues')
    ax3.grid(True, alpha=0.3)
    
    # Fiedler vector
    spectral_results = analyze_spectral_properties(G)
    fiedler_vector = spectral_results['fiedler_vector']
    
    ax4.plot(fiedler_vector, 'mo-', markersize=8)
    ax4.set_xlabel('Node Index')
    ax4.set_ylabel('Fiedler Vector Value')
    ax4.set_title('Fiedler Vector (Second Eigenvector)')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Example: Analyze a scale-free network
G = nx.barabasi_albert_graph(100, 3)

# Matrix analysis
matrix_results = analyze_graph_matrices(G)
print("Matrix Analysis Results:")
print(f"Spectral gap: {matrix_results['laplacian_eigenvalues'][1] - matrix_results['laplacian_eigenvalues'][0]:.4f}")

# Centrality measures
centrality_results = compute_centrality_measures(G)
print("\nCentrality Measures (top 5 nodes):")
for measure, values in centrality_results.items():
    sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"{measure}: {sorted_nodes}")

# Spectral analysis
spectral_results = analyze_spectral_properties(G)
print(f"\nSpectral Properties:")
print(f"Cheeger constant: {spectral_results['cheeger_constant']:.4f}")
print(f"Mixing time: {spectral_results['mixing_time']:.4f}")

# Plot results
plot_spectral_analysis(G, "Scale-Free Network Spectral Analysis")
```

## Key Takeaways

1. **Matrix representations**: Adjacency, Laplacian, and incidence matrices provide different views of network structure
2. **Spectral analysis**: Eigenvalues and eigenvectors reveal important network properties
3. **Centrality measures**: Different measures capture different aspects of node importance
4. **Random walks**: Provide insights into network dynamics and mixing
5. **Graph partitioning**: Spectral methods are effective for community detection
6. **Mathematical rigor**: Solid mathematical foundation enables advanced network analysis
7. **Applications**: Matrix methods are essential for materials science applications

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Chung, F. R. K. (1997). Spectral Graph Theory. American Mathematical Society.
3. Godsil, C., & Royle, G. (2001). Algebraic Graph Theory. Springer.
4. Horn, R. A., & Johnson, C. R. (2012). Matrix Analysis. Cambridge University Press.

## Video: Mathematical Foundations Deep Dive

### Video Overview
**Duration**: 20 minutes  
**Format**: Educational video with visual demonstrations and live coding

### Video Content

#### 1. Graph Theory Fundamentals (4 minutes)
- **Visual demonstrations**: Interactive graphs showing different types and properties
- **Key concepts**: Vertices, edges, connectivity, clustering, distance measures
- **Mathematical focus**: Clustering coefficient formula and its interpretation

#### 2. Matrix Representations (3 minutes)
- **Adjacency matrix**: Construction and properties
- **Matrix powers**: Path counting and reachability
- **Laplacian matrix**: Definition and spectral properties
- **Live coding**: Matrix operations in Python

#### 3. Spectral Graph Theory (3 minutes)
- **Eigenvalue analysis**: Adjacency and Laplacian spectra
- **Fiedler vector**: Graph partitioning applications
- **Cheeger's inequality**: Connectivity and bottlenecks
- **Visualizations**: Spectral plots and partitioning results

#### 4. Random Walks and PageRank (3 minutes)
- **Transition matrix**: Random walk probabilities
- **Stationary distribution**: Long-term behavior
- **PageRank algorithm**: Web search applications
- **Mixing time**: Convergence analysis

#### 5. Centrality Measures (3 minutes)
- **Degree centrality**: Connection counting
- **Betweenness centrality**: Shortest path participation
- **Closeness centrality**: Distance averaging
- **Eigenvector centrality**: Neighbor importance
- **Visual comparisons**: Centrality heatmaps

#### 6. Materials Science Applications (2 minutes)
- **Atomic networks**: Atoms as vertices, bonds as edges
- **Defect networks**: Imperfection clustering and percolation
- **Structure-property relationships**: Network descriptors for material properties
- **Real examples**: Nanowire networks, defect clustering

#### 7. Live Coding Demonstration (2 minutes)
- **Python implementation**: NetworkX and NumPy
- **Matrix analysis**: Adjacency, Laplacian, spectral decomposition
- **Centrality computation**: All major centrality measures
- **Visualization**: Network plots with centrality coloring

### Learning Objectives

After watching this video, you will be able to:

1. **Understand graph theory basics**: Vertices, edges, connectivity, clustering
2. **Work with matrix representations**: Adjacency, Laplacian, incidence matrices
3. **Apply spectral analysis**: Eigenvalues, eigenvectors, graph partitioning
4. **Analyze random walks**: Transition matrices, stationary distributions, mixing
5. **Calculate centrality measures**: Degree, betweenness, closeness, eigenvector
6. **Connect to materials science**: Network-based material property prediction

### Key Mathematical Formulas Highlighted

**Clustering Coefficient**:
$$C_i = \frac{2e_i}{k_i(k_i-1)}$$

**Laplacian Matrix**:
$$L = D - A$$

**Cheeger's Inequality**:
$$\frac{\lambda_2}{2} \leq h(G) \leq \sqrt{2\lambda_2}$$

**PageRank**:
$$\mathbf{PR} = (1-d) \frac{\mathbf{1}}{n} + d \mathbf{M}^T \mathbf{PR}$$

**Betweenness Centrality**:
$$C_B(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}$$

### Interactive Elements

- **Pause and practice**: Key concepts with exercises
- **Code along**: Follow the Python implementations
- **Visual quizzes**: Identify graph properties and matrix elements
- **Application scenarios**: Materials science problem solving

### Additional Resources

- **Complete code examples**: Available in the accompanying materials
- **Interactive notebooks**: Jupyter notebooks for hands-on practice
- **Reference materials**: Mathematical proofs and derivations
- **Further reading**: Advanced topics in spectral graph theory

### Video Production Notes

**Visual Style**:
- Clean, modern presentation with clear mathematical notation
- Color-coded elements for different concepts
- Smooth animations for matrix operations and spectral analysis
- High-quality network visualizations

**Audio**:
- Clear, professional narration
- Appropriate pacing for mathematical content
- Emphasis on key concepts and formulas
- Subtitles available for accessibility

**Technical Quality**:
- High-resolution graphics for mathematical formulas
- Smooth screen recordings for code demonstrations
- Professional audio recording
- Multiple viewing options (desktop, mobile, tablet)

---

*The mathematical foundations of network analysis provide the essential tools for understanding complex systems, with direct applications to materials science and engineering. This video brings these concepts to life through visual demonstrations and practical implementations.*
