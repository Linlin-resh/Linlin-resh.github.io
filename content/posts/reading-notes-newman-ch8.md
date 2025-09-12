---
title: "Reading Notes: Newman's Networks Chapter 8 - Random Graphs"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 8 of Newman's 'Networks: An Introduction' covering Erdős-Rényi model, configuration model, and random graph theory"
tags: ["reading-notes", "network-theory", "random-graphs", "erdos-renyi", "configuration-model"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 8 of Newman's *Networks: An Introduction* introduces **random graph theory** - the mathematical foundation for understanding network structure through probabilistic models. This chapter covers the classic Erdős-Rényi model, configuration models, and their applications to understanding real-world networks.

## 8.1 Erdős-Rényi Model

### Model Definition

The **Erdős-Rényi random graph** $G(n,p)$ consists of:

- **n** vertices
- **p** probability of edge existence between any two vertices
- **Expected number of edges**: $E[m] = \binom{n}{2}p = \frac{n(n-1)}{2}p$

### Mathematical Properties

#### Degree Distribution

For large networks, the degree distribution follows a **Poisson distribution**:

$$P(k) = \frac{\langle k \rangle^k e^{-\langle k \rangle}}{k!}$$

Where $\langle k \rangle = (n-1)p$ is the average degree.

#### Properties of Poisson Distribution

**Mean**: $\langle k \rangle = (n-1)p$

**Variance**: $\sigma_k^2 = \langle k \rangle$

**Standard deviation**: $\sigma_k = \sqrt{\langle k \rangle}$

**Coefficient of variation**: $CV = \frac{\sigma_k}{\langle k \rangle} = \frac{1}{\sqrt{\langle k \rangle}}$

#### Clustering Coefficient

**Expected clustering coefficient**:
$$C = p = \frac{\langle k \rangle}{n-1}$$

**For large networks**: $C \to 0$ as $n \to \infty$

#### Path Length

**Average path length**:
$$L \sim \frac{\ln n}{\ln \langle k \rangle}$$

**For large networks**: $L \sim \ln n$ (logarithmic growth)

### Phase Transitions

#### Giant Component Emergence

A **phase transition** occurs at the critical threshold:

$$p_c = \frac{1}{n-1} \approx \frac{1}{n}$$

**Below threshold** ($p < p_c$):
- Only small, isolated components exist
- Largest component size: $O(\ln n)$

**At threshold** ($p = p_c$):
- Giant component emerges
- Largest component size: $O(n^{2/3})$

**Above threshold** ($p > p_c$):
- Giant component dominates
- Largest component size: $O(n)$

#### Mathematical Analysis

**Component size distribution**:
$$P(s) \sim s^{-3/2} e^{-s/s_0}$$

Where $s_0$ is the characteristic size.

**Giant component size**:
$$S = n \left[1 - \sum_{k=0}^{\infty} P(k) u^k\right]$$

Where $u$ satisfies:
$$u = \sum_{k=0}^{\infty} \frac{k P(k) u^{k-1}}{\langle k \rangle}$$

### Connectivity

#### Probability of Connectivity

**For large networks**:
$$P(\text{connected}) \approx \exp(-n e^{-\langle k \rangle})$$

**Critical threshold for connectivity**:
$$p_c^{\text{conn}} = \frac{\ln n}{n}$$

#### Diameter

**Expected diameter**:
$$D \sim \frac{\ln n}{\ln \langle k \rangle}$$

**For large networks**: $D \sim \ln n$

## 8.2 Configuration Model

### Model Definition

The **configuration model** generates random graphs with a **prescribed degree sequence**:

- **Input**: Degree sequence $\{k_1, k_2, \ldots, k_n\}$
- **Constraint**: $\sum_{i=1}^n k_i$ must be even
- **Method**: Randomly connect stubs (half-edges)

### Mathematical Properties

#### Degree Distribution

**Given degree sequence**: $P(k) = \frac{n_k}{n}$

Where $n_k$ is the number of nodes with degree $k$.

#### Expected Number of Edges

**Total number of edges**:
$$m = \frac{1}{2} \sum_{i=1}^n k_i = \frac{n \langle k \rangle}{2}$$

#### Clustering Coefficient

**Expected clustering coefficient**:
$$C = \frac{\langle k^2 \rangle - \langle k \rangle}{n \langle k \rangle^2}$$

**For large networks**: $C \to 0$ as $n \to \infty$

#### Path Length

**Average path length**:
$$L \sim \frac{\ln n}{\ln \langle k \rangle}$$

**For scale-free networks**: $L \sim \frac{\ln n}{\ln \ln n}$

### Degree Correlation

#### Assortativity

**Degree correlation function**:
$$k_{nn}(k) = \frac{\sum_{k'} k' P(k'|k)}{\sum_{k'} P(k'|k)}$$

**For configuration model**:
$$k_{nn}(k) = \frac{\langle k^2 \rangle}{\langle k \rangle}$$

**Assortativity coefficient**:
$$r = \frac{\langle k^2 \rangle - \langle k \rangle^2}{\langle k^3 \rangle - \langle k \rangle^2}$$

### Giant Component

#### Existence Condition

**Giant component exists if**:
$$\frac{\langle k^2 \rangle}{\langle k \rangle} > 2$$

**Critical threshold**:
$$\frac{\langle k^2 \rangle}{\langle k \rangle} = 2$$

#### Size of Giant Component

**Giant component size**:
$$S = n \left[1 - \sum_{k=0}^{\infty} P(k) u^k\right]$$

Where $u$ satisfies:
$$u = \sum_{k=0}^{\infty} \frac{k P(k) u^{k-1}}{\langle k \rangle}$$

## 8.3 Random Graph Variants

### Directed Random Graphs

#### Model Definition

**Directed Erdős-Rényi model** $G(n,p)$:

- **n** vertices
- **p** probability of directed edge from any vertex to any other
- **Expected number of edges**: $E[m] = n(n-1)p$

#### Degree Distributions

**In-degree distribution**:
$$P_{\text{in}}(k) = \frac{\langle k_{\text{in}} \rangle^k e^{-\langle k_{\text{in}} \rangle}}{k!}$$

**Out-degree distribution**:
$$P_{\text{out}}(k) = \frac{\langle k_{\text{out}} \rangle^k e^{-\langle k_{\text{out}} \rangle}}{k!}$$

Where $\langle k_{\text{in}} \rangle = \langle k_{\text{out}} \rangle = (n-1)p$.

### Weighted Random Graphs

#### Model Definition

**Weighted random graph**:

- **n** vertices
- **p** probability of edge existence
- **w** weight distribution for existing edges

#### Weight Distribution

**Exponential weights**:
$$P(w) = \lambda e^{-\lambda w}$$

**Power-law weights**:
$$P(w) \sim w^{-\alpha}$$

#### Network Properties

**Weighted degree**:
$$k_i^w = \sum_{j} w_{ij}$$

**Weighted clustering**:
$$C_i^w = \frac{\sum_{j,k} w_{ij} w_{jk} w_{ki}}{\sum_{j,k} w_{ij} w_{jk}}$$

### Bipartite Random Graphs

#### Model Definition

**Bipartite random graph** $G(n_1, n_2, p)$:

- **n₁** vertices in partition 1
- **n₂** vertices in partition 2
- **p** probability of edge between partitions

#### Degree Distributions

**Partition 1 degree distribution**:
$$P_1(k) = \frac{(n_2 p)^k e^{-n_2 p}}{k!}$$

**Partition 2 degree distribution**:
$$P_2(k) = \frac{(n_1 p)^k e^{-n_1 p}}{k!}$$

## 8.4 Random Graph Algorithms

### Generation Algorithms

#### Erdős-Rényi Generation

**Algorithm**:
1. Initialize empty graph with $n$ vertices
2. For each pair $(i,j)$ where $i < j$:
   - Generate random number $r \sim U(0,1)$
   - If $r < p$, add edge $(i,j)$

**Time complexity**: $O(n^2)$

#### Configuration Model Generation

**Algorithm**:
1. Create $k_i$ stubs for each vertex $i$
2. Randomly pair stubs
3. Connect paired stubs to form edges

**Time complexity**: $O(m)$ where $m$ is the number of edges

### Sampling Algorithms

#### Random Walk Sampling

**Algorithm**:
1. Start at random vertex
2. At each step, move to random neighbor
3. Repeat for desired number of steps

**Stationary distribution**:
$$\pi_i = \frac{k_i}{2m}$$

#### Metropolis-Hastings Sampling

**Algorithm**:
1. Start at random vertex
2. Propose move to neighbor with probability $1/k_i$
3. Accept with probability $\min(1, k_i/k_j)$

**Stationary distribution**: Uniform

## 8.5 Applications to Materials Science

### Percolation Theory

#### Bond Percolation

**Model**: Random removal of edges with probability $1-p$

**Percolation probability**:
$$P(p) = 1 - \sum_{k=0}^{\infty} P(k) (1-p)^k$$

**Critical threshold**:
$$p_c = \frac{1}{\kappa - 1}$$

Where $\kappa = \frac{\langle k^2 \rangle}{\langle k \rangle}$.

#### Site Percolation

**Model**: Random removal of vertices with probability $1-p$

**Percolation probability**:
$$P(p) = 1 - \sum_{k=0}^{\infty} P(k) (1-p)^k$$

**Critical threshold**:
$$p_c = \frac{1}{\kappa - 1}$$

### Nanowire Networks

#### Network Formation

**Random geometric graph**:
- **Nodes**: Nanowire junctions
- **Edges**: Nanowire segments
- **Probability**: $p = \rho \pi r^2$ where $\rho$ is density and $r$ is interaction radius

#### Electrical Properties

**Conductivity**:
$$\sigma \sim (p - p_c)^t$$

Where $t \approx 2.0$ is the conductivity exponent.

**Critical density**:
$$\rho_c = \frac{1}{\pi r^2 (\kappa - 1)}$$

### Defect Networks

#### Defect Clustering

**Clustering coefficient**:
$$C = \frac{\langle k^2 \rangle - \langle k \rangle}{n \langle k \rangle^2}$$

**For defect networks**: $C > 0$ indicates clustering

#### Percolation Threshold

**Defect percolation**:
$$p_c = \frac{1}{\kappa - 1}$$

**For materials**: $p_c$ determines critical defect concentration

## Code Example: Random Graph Analysis

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats

def generate_erdos_renyi(n, p):
    """Generate Erdős-Rényi random graph"""
    G = nx.erdos_renyi_graph(n, p)
    return G

def generate_configuration_model(degree_sequence):
    """Generate configuration model random graph"""
    G = nx.configuration_model(degree_sequence)
    # Remove self-loops and parallel edges
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def analyze_random_graph(G, model_type="ER"):
    """Analyze random graph properties"""
    
    # Basic statistics
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = 2 * m / (n * (n - 1))
    
    # Degree analysis
    degrees = [d for n, d in G.degree()]
    degree_dist = Counter(degrees)
    
    # Poisson fitting for ER model
    if model_type == "ER":
        avg_degree = np.mean(degrees)
        poisson_dist = stats.poisson(avg_degree)
        
        # Chi-square test
        observed = list(degree_dist.values())
        expected = [poisson_dist.pmf(k) * n for k in degree_dist.keys()]
        chi2_stat, chi2_pvalue = stats.chisquare(observed, expected)
    else:
        chi2_stat = chi2_pvalue = None
    
    # Clustering analysis
    clustering = nx.average_clustering(G)
    
    # Path length analysis
    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    else:
        # Analyze largest component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        avg_path_length = nx.average_shortest_path_length(subgraph)
        diameter = nx.diameter(subgraph)
    
    # Giant component analysis
    components = list(nx.connected_components(G))
    component_sizes = [len(comp) for comp in components]
    giant_component_size = max(component_sizes) if component_sizes else 0
    giant_component_fraction = giant_component_size / n if n > 0 else 0
    
    # Phase transition analysis
    avg_degree = np.mean(degrees)
    phase_transition_threshold = 1.0  # For ER model
    
    return {
        'model_type': model_type,
        'nodes': n,
        'edges': m,
        'density': density,
        'avg_degree': avg_degree,
        'degree_variance': np.var(degrees),
        'clustering': clustering,
        'avg_path_length': avg_path_length,
        'diameter': diameter,
        'giant_component_size': giant_component_size,
        'giant_component_fraction': giant_component_fraction,
        'num_components': len(components),
        'phase_transition_threshold': phase_transition_threshold,
        'above_threshold': avg_degree > phase_transition_threshold,
        'chi2_statistic': chi2_stat,
        'chi2_pvalue': chi2_pvalue
    }

def analyze_phase_transition(n_values, p_values):
    """Analyze phase transition in random graphs"""
    
    results = []
    
    for n in n_values:
        for p in p_values:
            G = generate_erdos_renyi(n, p)
            analysis = analyze_random_graph(G, "ER")
            results.append({
                'n': n,
                'p': p,
                'giant_component_fraction': analysis['giant_component_fraction'],
                'avg_path_length': analysis['avg_path_length'],
                'clustering': analysis['clustering']
            })
    
    return results

def plot_degree_distribution(G, model_type="ER", title="Degree Distribution"):
    """Plot degree distribution with theoretical comparison"""
    
    degrees = [d for n, d in G.degree()]
    degree_dist = Counter(degrees)
    
    k_values = list(degree_dist.keys())
    counts = list(degree_dist.values())
    
    plt.figure(figsize=(10, 6))
    
    # Plot observed distribution
    plt.bar(k_values, counts, alpha=0.7, label='Observed', color='blue')
    
    # Plot theoretical distribution
    if model_type == "ER":
        avg_degree = np.mean(degrees)
        poisson_dist = stats.poisson(avg_degree)
        theoretical = [poisson_dist.pmf(k) * len(degrees) for k in k_values]
        plt.plot(k_values, theoretical, 'ro-', label='Poisson', markersize=8)
    
    plt.xlabel('Degree k')
    plt.ylabel('Count')
    plt.title(f'{title} - {model_type} Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_phase_transition(results):
    """Plot phase transition results"""
    
    # Group results by n
    n_values = sorted(set(r['n'] for r in results))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for n in n_values:
        n_results = [r for r in results if r['n'] == n]
        p_values = [r['p'] for r in n_results]
        giant_fractions = [r['giant_component_fraction'] for r in n_results]
        
        ax1.plot(p_values, giant_fractions, 'o-', label=f'n={n}', markersize=8)
    
    ax1.set_xlabel('Probability p')
    ax1.set_ylabel('Giant Component Fraction')
    ax1.set_title('Phase Transition: Giant Component')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for n in n_values:
        n_results = [r for r in results if r['n'] == n]
        p_values = [r['p'] for r in n_results]
        path_lengths = [r['avg_path_length'] for r in n_results]
        
        ax2.plot(p_values, path_lengths, 'o-', label=f'n={n}', markersize=8)
    
    ax2.set_xlabel('Probability p')
    ax2.set_ylabel('Average Path Length')
    ax2.set_title('Phase Transition: Path Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example: Analyze Erdős-Rényi random graph
n, p = 1000, 0.01
G_ER = generate_erdos_renyi(n, p)
analysis_ER = analyze_random_graph(G_ER, "ER")

print("Erdős-Rényi Random Graph Analysis:")
for key, value in analysis_ER.items():
    print(f"{key}: {value}")

# Plot degree distribution
plot_degree_distribution(G_ER, "ER", "Erdős-Rényi Random Graph")

# Example: Analyze configuration model
degree_sequence = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3] * 100  # Regular graph
G_CM = generate_configuration_model(degree_sequence)
analysis_CM = analyze_random_graph(G_CM, "CM")

print("\nConfiguration Model Analysis:")
for key, value in analysis_CM.items():
    print(f"{key}: {value}")

# Plot degree distribution
plot_degree_distribution(G_CM, "CM", "Configuration Model")

# Example: Phase transition analysis
n_values = [100, 500, 1000]
p_values = np.linspace(0.001, 0.1, 20)
phase_results = analyze_phase_transition(n_values, p_values)

# Plot phase transition
plot_phase_transition(phase_results)
```

## Key Takeaways

1. **Random graph models**: Provide null models for understanding network structure
2. **Phase transitions**: Critical thresholds determine network connectivity
3. **Degree distributions**: Poisson for ER model, prescribed for configuration model
4. **Giant component**: Emerges at critical threshold, dominates above threshold
5. **Mathematical analysis**: Rigorous theory enables prediction of network properties
6. **Applications**: Random graphs provide insights into materials science phenomena
7. **Percolation theory**: Direct application to nanowire networks and defect systems

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Erdős, P., & Rényi, A. (1959). On random graphs. Publicationes Mathematicae, 6, 290-297.
3. Molloy, M., & Reed, B. (1995). A critical point for random graphs with a given degree sequence. Random Structures & Algorithms, 6(2-3), 161-180.
4. Stauffer, D., & Aharony, A. (1994). Introduction to Percolation Theory. Taylor & Francis.

---

*Random graph theory provides the mathematical foundation for understanding network structure and behavior, with important applications in materials science and percolation phenomena.*
