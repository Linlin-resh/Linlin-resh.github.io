---
title: "Math Formula Display Test"
date: 2025-08-29
draft: false
description: "Test page for math formula rendering optimization"
tags: ["test", "math", "katex", "mathjax"]
showToc: true
TocOpen: true
---

## Math Formula Display Test

This page tests the math formula rendering optimization for better web display.

### Inline Math

Here are some inline math formulas: $E = mc^2$, $P(k) = \frac{N_k}{N}$, and $\sum_{k=0}^{\infty} P(k) = 1$.

### Display Math

Here are some display math formulas:

**Degree Distribution:**
$$P(k) = \frac{\text{Number of nodes with degree } k}{n}$$

**Properties:**
$$\sum_{k=0}^{\infty} P(k) = 1$$

$$\langle k \rangle = \sum_{k=0}^{\infty} k P(k)$$

$$\langle k^2 \rangle = \sum_{k=0}^{\infty} k^2 P(k)$$

**Degree Moments:**
$$\langle k \rangle = \frac{2m}{n}$$

$$\langle k^2 \rangle = \frac{1}{n} \sum_{i=1}^n k_i^2$$

$$\sigma_k^2 = \langle k^2 \rangle - \langle k \rangle^2$$

$$CV = \frac{\sigma_k}{\langle k \rangle}$$

### Complex Formulas

**Network Efficiency:**
$$E = \frac{1}{n(n-1)} \sum_{i \neq j} \frac{1}{d_{ij}}$$

**Clustering Coefficient:**
$$C_i = \frac{2e_i}{k_i(k_i-1)}$$

**Modularity:**
$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

### Matrix Formulas

**Adjacency Matrix:**
$$A_{ij} = \begin{cases} 
1 & \text{if nodes } i \text{ and } j \text{ are connected} \\
0 & \text{otherwise}
\end{cases}$$

**Laplacian Matrix:**
$$L = D - A$$

**Eigenvalue Decomposition:**
$$A = U \Lambda U^T$$

### Long Formulas

**SIR Model:**
$$\frac{dS}{dt} = -\beta S I$$
$$\frac{dI}{dt} = \beta S I - \gamma I$$
$$\frac{dR}{dt} = \gamma I$$

**Kuramoto Model:**
$$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^N A_{ij} \sin(\theta_j - \theta_i)$$

### Mathematical Tables

| Property | Formula | Description |
|----------|---------|-------------|
| Degree | $k_i = \sum_j A_{ij}$ | Number of connections |
| Clustering | $C_i = \frac{2e_i}{k_i(k_i-1)}$ | Local clustering coefficient |
| Betweenness | $C_B(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}$ | Betweenness centrality |
| Eigenvector | $x_i = \frac{1}{\lambda} \sum_j A_{ij} x_j$ | Eigenvector centrality |

### Code with Math

Here's some Python code that uses the mathematical concepts:

```python
import networkx as nx
import numpy as np

def calculate_network_properties(G):
    """Calculate various network properties"""
    
    # Degree distribution
    degrees = [G.degree(i) for i in G.nodes()]
    n = G.number_of_nodes()
    
    # Average degree: ⟨k⟩ = 2m/n
    avg_degree = 2 * G.number_of_edges() / n
    
    # Second moment: ⟨k²⟩ = (1/n) Σ k_i²
    second_moment = np.mean([k**2 for k in degrees])
    
    # Clustering coefficient: C_i = 2e_i/(k_i(k_i-1))
    clustering = nx.average_clustering(G)
    
    # Global efficiency: E = (1/(n(n-1))) Σ (1/d_ij)
    efficiency = nx.global_efficiency(G)
    
    return {
        'avg_degree': avg_degree,
        'second_moment': second_moment,
        'clustering': clustering,
        'efficiency': efficiency
    }
```

### Conclusion

This test page demonstrates the improved math formula rendering with:

1. **Better spacing** around mathematical expressions
2. **Enhanced typography** for better readability
3. **Responsive design** that works on mobile devices
4. **Dark mode support** for better viewing experience
5. **Formula numbering** for easy reference
6. **Consistent styling** across different math environments

The math formulas should now display much better on the web compared to the original Obsidian rendering.
