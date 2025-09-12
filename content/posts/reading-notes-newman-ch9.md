---
title: "Reading Notes: Newman's Networks Chapter 9 - Network Models"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 9 of Newman's 'Networks: An Introduction' covering small-world networks, scale-free networks, and other network generation models"
tags: ["reading-notes", "network-theory", "network-models", "small-world", "scale-free"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 9 of Newman's *Networks: An Introduction* explores **network models** - mathematical frameworks for generating networks with specific structural properties. This chapter covers small-world networks, scale-free networks, and other models that capture the essential features of real-world networks.

## 9.1 Small-World Networks

### Watts-Strogatz Model

#### Model Definition

The **Watts-Strogatz model** generates networks with:

- **High clustering coefficient** (like regular lattices)
- **Short average path length** (like random graphs)
- **Small-world property**: $L \sim \ln n$ and $C \gg C_{\text{random}}$

#### Algorithm

1. **Start with regular lattice**: $n$ nodes, each connected to $k$ nearest neighbors
2. **Rewire edges**: With probability $p$, move each edge to a random new location
3. **Result**: Interpolates between regular lattice ($p=0$) and random graph ($p=1$)

#### Mathematical Properties

**Clustering coefficient**:
$$C(p) = \frac{3(k-2)}{4(k-1)}(1-p)^3$$

**Average path length**:
$$L(p) \sim \frac{n}{k} f(pkn)$$

Where $f(x)$ is a scaling function.

**Small-world coefficient**:
$$S = \frac{C/C_{\text{random}}}{L/L_{\text{random}}}$$

### Phase Transition

#### Clustering Transition

**At $p=0$**: $C = \frac{3(k-2)}{4(k-1)}$ (regular lattice)

**At $p=1$**: $C \approx \frac{k}{n}$ (random graph)

**Transition point**: $p \sim \frac{1}{n}$

#### Path Length Transition

**At $p=0$**: $L \sim \frac{n}{k}$ (regular lattice)

**At $p=1$**: $L \sim \frac{\ln n}{\ln k}$ (random graph)

**Transition point**: $p \sim \frac{1}{n}$

### Real-World Applications

#### Social Networks

**Six degrees of separation**: Milgram's experiment shows $L \sim 6$

**High clustering**: People tend to form dense local groups

**Small-world property**: $S \gg 1$ for most social networks

#### Neural Networks

**Brain connectivity**: High local clustering, short global paths

**Functional modules**: Dense local connections within modules

**Global efficiency**: Short paths between distant brain regions

## 9.2 Scale-Free Networks

### Barabási-Albert Model

#### Model Definition

The **Barabási-Albert model** generates networks with:

- **Power-law degree distribution**: $P(k) \sim k^{-\gamma}$
- **Preferential attachment**: New nodes prefer to connect to high-degree nodes
- **Growth**: Network grows by adding new nodes over time

#### Algorithm

1. **Start**: Small initial network (e.g., complete graph with $m_0$ nodes)
2. **Add node**: At each step, add one new node
3. **Connect**: New node connects to $m$ existing nodes
4. **Preferential attachment**: Probability of connecting to node $i$ is proportional to $k_i$

#### Mathematical Analysis

**Degree distribution**:
$$P(k) = \frac{2m^2}{k^3} \sim k^{-3}$$

**Exponent**: $\gamma = 3$ (universal for BA model)

**Average degree**: $\langle k \rangle = 2m$

**Second moment**: $\langle k^2 \rangle = \infty$ (diverges)

### Generalized Preferential Attachment

#### Linear Preferential Attachment

**Attachment probability**:
$$\Pi(k_i) = \frac{k_i + a}{\sum_j (k_j + a)}$$

Where $a$ is the **initial attractiveness**.

**Degree distribution**:
$$P(k) \sim k^{-\gamma}$$

Where $\gamma = 3 + \frac{a}{m}$.

#### Nonlinear Preferential Attachment

**Attachment probability**:
$$\Pi(k_i) = \frac{k_i^{\alpha}}{\sum_j k_j^{\alpha}}$$

**Degree distribution**:
- **$\alpha < 1$**: Exponential distribution
- **$\alpha = 1$**: Power-law with $\gamma = 3$
- **$\alpha > 1$**: "Winner-takes-all" (gelation)

### Real-World Examples

#### Internet

**AS-level network**: $\gamma \approx 2.2$

**Router-level network**: $\gamma \approx 2.5$

**Growth mechanism**: New ISPs prefer to connect to well-connected ISPs

#### World Wide Web

**In-degree distribution**: $\gamma \approx 2.1$

**Out-degree distribution**: $\gamma \approx 2.7$

**Growth mechanism**: New pages prefer to link to popular pages

#### Scientific Collaboration

**Degree distribution**: $\gamma \approx 2.1$

**Growth mechanism**: New researchers prefer to collaborate with established researchers

## 9.3 Other Network Models

### Configuration Model

#### Model Definition

**Configuration model** generates random graphs with:

- **Prescribed degree sequence**: $\{k_1, k_2, \ldots, k_n\}$
- **Random connections**: Edges are randomly placed between stubs
- **No degree correlation**: Degrees are uncorrelated

#### Mathematical Properties

**Degree distribution**: $P(k) = \frac{n_k}{n}$

**Clustering coefficient**:
$$C = \frac{\langle k^2 \rangle - \langle k \rangle}{n \langle k \rangle^2}$$

**Giant component condition**:
$$\frac{\langle k^2 \rangle}{\langle k \rangle} > 2$$

### Exponential Random Graph Models

#### Model Definition

**Exponential random graph models** (ERGMs) generate networks with:

- **Prescribed statistics**: Mean degree, clustering, etc.
- **Maximum entropy**: Most random network consistent with constraints
- **Exponential family**: $P(G) = \frac{1}{Z} \exp(\sum_i \theta_i S_i(G))$

#### Mathematical Formulation

**Probability distribution**:
$$P(G) = \frac{1}{Z} \exp\left(\sum_i \theta_i S_i(G)\right)$$

Where:
- $Z$: Partition function
- $\theta_i$: Parameters
- $S_i(G)$: Network statistics

**Partition function**:
$$Z = \sum_G \exp\left(\sum_i \theta_i S_i(G)\right)$$

### Geometric Random Graphs

#### Model Definition

**Geometric random graph** $G(n, r)$:

- **n** vertices placed randomly in space
- **r** connection radius
- **Edge**: $(i,j)$ exists if $d_{ij} \leq r$

#### Mathematical Properties

**Expected degree**:
$$\langle k \rangle = (n-1) \pi r^2$$

**Clustering coefficient**:
$$C = \frac{3}{4} - \frac{3}{4\pi} \approx 0.586$$

**Path length**: $L \sim \frac{1}{r}$ (for large $n$)

### Hierarchical Networks

#### Model Definition

**Hierarchical networks** have:

- **Tree structure**: Hierarchical organization
- **Local clustering**: Dense connections within levels
- **Global efficiency**: Short paths between distant nodes

#### Mathematical Properties

**Degree distribution**: $P(k) \sim k^{-\gamma}$ with $\gamma = \ln 3/\ln 2 \approx 1.585$

**Clustering coefficient**: $C(k) \sim k^{-1}$

**Path length**: $L \sim \ln n$

## 9.4 Network Evolution Models

### Growing Networks

#### Model Definition

**Growing networks** evolve by:

- **Adding nodes**: New nodes join over time
- **Adding edges**: New edges form between existing nodes
- **Removing nodes/edges**: Nodes or edges may be removed

#### Mathematical Analysis

**Master equation**:
$$\frac{\partial P(k,t)}{\partial t} = \frac{1}{t} \left[P(k-1,t) - P(k,t)\right] + \delta_{k,1}$$

**Solution**: $P(k,t) \sim k^{-\gamma}$ with $\gamma = 3$

### Aging and Preferential Attachment

#### Model Definition

**Aging model** combines:

- **Preferential attachment**: High-degree nodes attract more connections
- **Aging**: Older nodes become less attractive over time

#### Mathematical Formulation

**Attachment probability**:
$$\Pi(k_i, t_i) = \frac{k_i e^{-\alpha(t-t_i)}}{\sum_j k_j e^{-\alpha(t-t_j)}}$$

Where:
- $t_i$: Time when node $i$ was added
- $\alpha$: Aging parameter

**Degree distribution**:
$$P(k) \sim k^{-\gamma} e^{-\beta k}$$

Where $\gamma$ and $\beta$ depend on $\alpha$.

### Fitness Models

#### Model Definition

**Fitness model** assigns:

- **Fitness values**: $\eta_i$ for each node $i$
- **Attachment probability**: $\Pi(k_i, \eta_i) = \frac{\eta_i k_i}{\sum_j \eta_j k_j}$

#### Mathematical Analysis

**Degree distribution**:
$$P(k) \sim k^{-\gamma} \int \eta^{\gamma-1} \rho(\eta) \, d\eta$$

Where $\rho(\eta)$ is the fitness distribution.

## 9.5 Applications to Materials Science

### Nanowire Network Formation

#### Growth Model

**Nanowire growth** can be modeled as:

- **Nodes**: Nanowire junctions
- **Edges**: Nanowire segments
- **Growth**: New nanowires prefer to connect to existing junctions

**Mathematical model**:
$$\Pi(k_i) = \frac{k_i + a}{\sum_j (k_j + a)}$$

Where $a$ represents the intrinsic connectivity of new junctions.

#### Percolation Properties

**Critical density**:
$$\rho_c = \frac{1}{\pi r^2 (\kappa - 1)}$$

Where $\kappa = \frac{\langle k^2 \rangle}{\langle k \rangle}$.

**Conductivity**:
$$\sigma \sim (\rho - \rho_c)^t$$

Where $t \approx 2.0$ is the conductivity exponent.

### Defect Network Evolution

#### Defect Clustering

**Defect clustering** can be modeled using:

- **Preferential attachment**: Defects prefer to form near existing defects
- **Aging**: Old defects become less active over time

**Mathematical model**:
$$\Pi(k_i, t_i) = \frac{k_i e^{-\alpha(t-t_i)}}{\sum_j k_j e^{-\alpha(t-t_j)}}$$

#### Phase Transitions

**Defect percolation**:
$$P(\text{percolation}) = 1 - \sum_{k=0}^{\infty} P(k) (1-p)^k$$

**Critical threshold**:
$$p_c = \frac{1}{\kappa - 1}$$

### Self-Assembling Materials

#### Network Formation

**Self-assembly** can be modeled as:

- **Nodes**: Building blocks (molecules, particles)
- **Edges**: Interactions between building blocks
- **Growth**: New building blocks join existing structures

**Mathematical model**:
$$\Pi(k_i) = \frac{k_i^{\alpha}}{\sum_j k_j^{\alpha}}$$

Where $\alpha$ controls the strength of preferential attachment.

## Code Example: Network Models

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def generate_watts_strogatz(n, k, p):
    """Generate Watts-Strogatz small-world network"""
    G = nx.watts_strogatz_graph(n, k, p)
    return G

def generate_barabasi_albert(n, m):
    """Generate Barabási-Albert scale-free network"""
    G = nx.barabasi_albert_graph(n, m)
    return G

def generate_configuration_model(degree_sequence):
    """Generate configuration model network"""
    G = nx.configuration_model(degree_sequence)
    # Remove self-loops and parallel edges
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def generate_geometric_random_graph(n, r):
    """Generate geometric random graph"""
    # Generate random positions
    positions = np.random.uniform(0, 1, (n, 2))
    
    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Add edges based on distance
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist <= r:
                G.add_edge(i, j)
    
    return G

def analyze_network_model(G, model_type="generic"):
    """Analyze network model properties"""
    
    # Basic statistics
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = 2 * m / (n * (n - 1))
    
    # Degree analysis
    degrees = [d for n, d in G.degree()]
    degree_dist = Counter(degrees)
    
    # Power-law fitting
    if len(degree_dist) > 1:
        k_values = list(degree_dist.keys())
        counts = list(degree_dist.values())
        
        if len(k_values) > 1:
            log_k = np.log(k_values[1:])  # Skip k=0
            log_counts = np.log(counts[1:])
            
            if len(log_k) > 1:
                gamma = -np.polyfit(log_k, log_counts, 1)[0]
            else:
                gamma = None
        else:
            gamma = None
    else:
        gamma = None
    
    # Clustering analysis
    clustering = nx.average_clustering(G)
    local_clustering = nx.clustering(G)
    
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
    
    # Small-world coefficient
    if n > 1:
        # Compare to random network
        p = density
        random_G = nx.erdos_renyi_graph(n, p)
        random_clustering = nx.average_clustering(random_G)
        random_path_length = nx.average_shortest_path_length(random_G) if nx.is_connected(random_G) else float('inf')
        
        small_world_coeff = (clustering / random_clustering) / (avg_path_length / random_path_length)
    else:
        small_world_coeff = None
    
    return {
        'model_type': model_type,
        'nodes': n,
        'edges': m,
        'density': density,
        'avg_degree': np.mean(degrees),
        'gamma_estimate': gamma,
        'clustering': clustering,
        'avg_path_length': avg_path_length,
        'diameter': diameter,
        'small_world_coeff': small_world_coeff
    }

def compare_network_models(n, models):
    """Compare different network models"""
    
    results = {}
    
    for model_name, model_params in models.items():
        if model_name == "watts_strogatz":
            G = generate_watts_strogatz(n, model_params['k'], model_params['p'])
        elif model_name == "barabasi_albert":
            G = generate_barabasi_albert(n, model_params['m'])
        elif model_name == "configuration_model":
            G = generate_configuration_model(model_params['degree_sequence'])
        elif model_name == "geometric_random":
            G = generate_geometric_random_graph(n, model_params['r'])
        else:
            continue
        
        analysis = analyze_network_model(G, model_name)
        results[model_name] = analysis
    
    return results

def plot_network_models(models, title="Network Models Comparison"):
    """Plot comparison of network models"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Degree distributions
    for model_name, G in models.items():
        degrees = [d for n, d in G.degree()]
        degree_dist = Counter(degrees)
        k_values = list(degree_dist.keys())
        counts = list(degree_dist.values())
        
        ax1.loglog(k_values, counts, 'o-', label=model_name, markersize=6)
    
    ax1.set_xlabel('Degree k')
    ax1.set_ylabel('Count')
    ax1.set_title('Degree Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Clustering vs Path Length
    for model_name, G in models.items():
        clustering = nx.average_clustering(G)
        if nx.is_connected(G):
            path_length = nx.average_shortest_path_length(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            path_length = nx.average_shortest_path_length(subgraph)
        
        ax2.scatter(path_length, clustering, label=model_name, s=100)
    
    ax2.set_xlabel('Average Path Length')
    ax2.set_ylabel('Clustering Coefficient')
    ax2.set_title('Clustering vs Path Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Network visualizations
    for i, (model_name, G) in enumerate(models.items()):
        ax = ax3 if i < 2 else ax4
        if i >= 2:
            ax = ax4
        
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw(G, pos, ax=ax, node_size=50, node_color='lightblue', 
                edge_color='gray', alpha=0.6)
        ax.set_title(f'{model_name.title()} Network')
        ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Example: Compare different network models
n = 1000
models = {
    'watts_strogatz': generate_watts_strogatz(n, 6, 0.1),
    'barabasi_albert': generate_barabasi_albert(n, 3),
    'configuration_model': generate_configuration_model([3] * n),
    'geometric_random': generate_geometric_random_graph(n, 0.1)
}

# Analyze models
results = compare_network_models(n, {
    'watts_strogatz': {'k': 6, 'p': 0.1},
    'barabasi_albert': {'m': 3},
    'configuration_model': {'degree_sequence': [3] * n},
    'geometric_random': {'r': 0.1}
})

print("Network Models Comparison:")
for model_name, analysis in results.items():
    print(f"\n{model_name.upper()}:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")

# Plot comparison
plot_network_models(models, "Network Models Comparison")
```

## Key Takeaways

1. **Small-world networks**: Combine high clustering with short path lengths
2. **Scale-free networks**: Exhibit power-law degree distributions through preferential attachment
3. **Network models**: Provide frameworks for understanding real-world network structure
4. **Phase transitions**: Critical thresholds determine network properties
5. **Growth mechanisms**: Preferential attachment and aging affect network evolution
6. **Applications**: Network models help understand materials science phenomena
7. **Mathematical analysis**: Rigorous theory enables prediction of network properties

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. Nature, 393(6684), 440-442.
3. Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks. Science, 286(5439), 509-512.
4. Dorogovtsev, S. N., & Mendes, J. F. F. (2003). Evolution of Networks: From Biological Nets to the Internet and WWW. Oxford University Press.

---

*Network models provide powerful frameworks for understanding and generating networks with specific structural properties, with important applications in materials science and complex systems.*
