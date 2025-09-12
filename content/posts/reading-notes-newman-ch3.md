---
title: "Reading Notes: Newman's Networks Chapter 3 - Social Networks"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 3 of Newman's 'Networks: An Introduction' covering social network analysis, data collection methods, and structural properties"
tags: ["reading-notes", "network-theory", "social-networks", "data-collection", "small-world"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 3 of Newman's *Networks: An Introduction* explores **social networks** - the webs of relationships that connect individuals in society. Social networks exhibit unique structural properties that reflect human behavior patterns and have profound implications for information spread, influence, and social dynamics.

## 3.1 Data Collection Methods

### Survey-Based Methods

#### Interviews and Questionnaires

**Face-to-face interviews** provide high-quality data but are:

- **Expensive**: Require trained interviewers
- **Time-consuming**: Limited sample sizes
- **Subject to bias**: Social desirability effects

**Mathematical framework** for survey design:

$$n = \frac{z^2 p(1-p)}{e^2}$$

Where:
- $n$: Required sample size
- $z$: Confidence level (e.g., 1.96 for 95%)
- $p$: Expected proportion
- $e$: Margin of error

#### Online Surveys

**Advantages**:
- **Cost-effective**: Large sample sizes possible
- **Anonymity**: Reduced social desirability bias
- **Geographic reach**: Global data collection

**Challenges**:
- **Selection bias**: Digital divide effects
- **Response rates**: Often low (< 20%)
- **Data quality**: Self-reported relationships

### Observational Methods

#### Direct Observation

**Systematic observation** of social interactions:

- **Time sampling**: Record interactions at regular intervals
- **Event sampling**: Record specific interaction types
- **Focal sampling**: Follow specific individuals

**Mathematical modeling** of observation reliability:

$$R = \frac{2r_{12}}{1 + r_{12}}$$

Where $r_{12}$ is the correlation between two observers.

#### Digital Traces

**Modern data sources**:

- **Social media**: Facebook, Twitter, LinkedIn
- **Communication logs**: Email, phone calls, texts
- **Location data**: GPS, check-ins, proximity

**Privacy considerations**:
- **Anonymization**: Remove identifying information
- **Consent**: Informed consent for data use
- **Ethics**: Institutional review board approval

## 3.2 Network Sampling Strategies

### Snowball Sampling

**Method**: Start with initial contacts, then ask them to nominate others.

**Mathematical model**:

$$P(\text{reach node } i) = 1 - (1 - p)^{d_i}$$

Where:
- $p$: Probability of being nominated
- $d_i$: Degree of node $i$

**Advantages**:
- **Hidden populations**: Access hard-to-reach groups
- **Cost-effective**: Leverages existing relationships

**Disadvantages**:
- **Bias**: Favors high-degree nodes
- **Clustering**: May miss isolated components

### Contact Tracing

**Epidemiological approach**:

1. **Index case**: Start with known infected individual
2. **Contact identification**: Find all contacts
3. **Network expansion**: Repeat for each contact

**Mathematical framework**:

$$R_0 = \beta \cdot \tau \cdot c$$

Where:
- $R_0$: Basic reproduction number
- $\beta$: Transmission probability
- $\tau$: Infectious period
- $c$: Contact rate

### Random Walk Sampling

**Method**: Start at random node, follow random edges.

**Stationary distribution**:

$$\pi_i = \frac{k_i}{2m}$$

Where $k_i$ is the degree of node $i$ and $m$ is the total number of edges.

**Advantages**:
- **Unbiased**: Proportional to degree
- **Efficient**: No need for complete network

## 3.3 Structural Properties of Social Networks

### Degree Distribution

#### Power-Law Behavior

Many social networks exhibit **scale-free degree distributions**:

$$P(k) \sim k^{-\gamma}$$

Where $\gamma \approx 2.0-3.0$ for social networks.

#### Mathematical Analysis

**Cumulative distribution function**:

$$P(K \geq k) = \int_k^{\infty} P(k') \, dk' \sim k^{-(\gamma-1)}$$

**Moment analysis**:
- **First moment**: $\langle k \rangle = \int k P(k) \, dk$
- **Second moment**: $\langle k^2 \rangle = \int k^2 P(k) \, dk$

### Clustering Coefficient

#### Local Clustering

**Definition**:

$$C_i = \frac{2e_i}{k_i(k_i-1)}$$

Where $e_i$ is the number of edges between neighbors of node $i$.

#### Global Clustering

**Average clustering**:

$$C = \frac{1}{n} \sum_{i=1}^{n} C_i$$

**Transitivity**:

$$T = \frac{3 \times \text{number of triangles}}{\text{number of connected triples}}$$

#### Social Interpretation

- **High clustering**: Dense social groups, strong local connections
- **Low clustering**: Sparse connections, weak social ties

### Path Length and Small-World Effect

#### Milgram's Experiment

**Six degrees of separation**:

- **Target**: Random person in Boston
- **Method**: Mail forwarding through acquaintances
- **Result**: Average path length ≈ 6

#### Mathematical Framework

**Small-world property**:

$$L \sim \log n \quad \text{and} \quad C \gg C_{\text{random}}$$

Where:
- $L$: Average path length
- $C$: Clustering coefficient
- $C_{\text{random}}$: Clustering in random network

#### Network Efficiency

**Global efficiency**:

$$E = \frac{1}{n(n-1)} \sum_{i \neq j} \frac{1}{d_{ij}}$$

**Local efficiency**:

$$E_{\text{local}} = \frac{1}{n} \sum_{i} E_i$$

Where $E_i$ is the efficiency of the subgraph of neighbors of node $i$.

## 3.4 Community Structure

### Modularity

**Definition**:

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

Where:
- $A_{ij}$: Adjacency matrix
- $k_i, k_j$: Degrees of nodes $i, j$
- $c_i, c_j$: Community assignments
- $\delta(c_i, c_j)$: Kronecker delta

#### Interpretation

- $Q > 0$: More edges within communities than expected by chance
- $Q = 0$: Random network structure
- $Q < 0$: Fewer edges within communities than expected

### Community Detection Algorithms

#### Girvan-Newman Algorithm

**Method**:
1. Calculate **betweenness centrality** for all edges
2. Remove edge with highest betweenness
3. Recalculate betweenness for remaining edges
4. Repeat until no edges remain

**Betweenness centrality**:

$$C_B(e) = \sum_{s \neq t} \frac{\sigma_{st}(e)}{\sigma_{st}}$$

Where $\sigma_{st}(e)$ is the number of shortest paths between $s$ and $t$ that pass through edge $e$.

#### Spectral Clustering

**Laplacian matrix**:

$$L = D - A$$

Where $D$ is the degree matrix and $A$ is the adjacency matrix.

**Normalized Laplacian**:

$$L_{\text{norm}} = D^{-1/2} L D^{-1/2}$$

**Eigenvalue decomposition**:

$$L_{\text{norm}} = U \Lambda U^T$$

Use the first $k$ eigenvectors to cluster nodes.

## 3.5 Influence and Centrality

### Centrality Measures

#### Degree Centrality

**Definition**:

$$C_D(i) = \frac{k_i}{n-1}$$

**Interpretation**: Number of direct connections

#### Betweenness Centrality

**Definition**:

$$C_B(i) = \sum_{s \neq i \neq t} \frac{\sigma_{st}(i)}{\sigma_{st}}$$

**Interpretation**: How often node $i$ lies on shortest paths

#### Closeness Centrality

**Definition**:

$$C_C(i) = \frac{n-1}{\sum_{j \neq i} d_{ij}}$$

**Interpretation**: Average distance to all other nodes

#### Eigenvector Centrality

**Definition**:

$$x_i = \frac{1}{\lambda} \sum_{j} A_{ij} x_j$$

**Matrix form**:

$$A \mathbf{x} = \lambda \mathbf{x}$$

**Interpretation**: Importance based on connections to important nodes

### Influence Models

#### Linear Threshold Model

**Activation rule**:

$$\sum_{j \in \text{active neighbors}} w_{ji} \geq \theta_i$$

Where:
- $w_{ji}$: Influence weight from $j$ to $i$
- $\theta_i$: Threshold for node $i$

#### Independent Cascade Model

**Activation probability**:

$$P(\text{activate } j \text{ from } i) = p_{ij}$$

**Cascade dynamics**:

$$P(\text{node } j \text{ activated}) = 1 - \prod_{i \in \text{active}} (1 - p_{ij})$$

## 3.6 Applications to Materials Science

### Collaboration Networks

#### Scientific Collaboration

**Network structure**:
- **Nodes**: Researchers
- **Edges**: Co-authorship relationships
- **Weights**: Number of joint publications

**Mathematical properties**:
- **Degree distribution**: Power-law with $\gamma \approx 2.1$
- **Clustering**: High ($C \approx 0.3$)
- **Path length**: Short ($L \approx 6$)

#### Research Impact

**Citation networks**:
- **In-degree**: Number of citations received
- **Out-degree**: Number of papers cited
- **PageRank**: Influence based on citation network

### Knowledge Networks

#### Technology Transfer

**Network structure**:
- **Nodes**: Institutions, companies
- **Edges**: Technology transfer relationships
- **Weights**: Transfer frequency/value

**Diffusion models**:
- **Bass model**: $S(t) = S_{\max} \frac{1 - e^{-(p+q)t}}{1 + \frac{q}{p} e^{-(p+q)t}}$
- **SIR model**: Susceptible → Infected → Recovered

## Code Example: Social Network Analysis

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_social_network(G, network_name="social"):
    """Comprehensive social network analysis"""
    
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
        
        # Log-log regression
        log_k = np.log(k_values[1:])  # Skip k=0
        log_counts = np.log(counts[1:])
        
        if len(log_k) > 1:
            gamma = -np.polyfit(log_k, log_counts, 1)[0]
        else:
            gamma = None
    else:
        gamma = None
    
    # Clustering analysis
    local_clustering = nx.clustering(G)
    global_clustering = nx.average_clustering(G)
    
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
    
    # Centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Community detection
    try:
        communities = nx.community.greedy_modularity_communities(G)
        modularity = nx.community.modularity(G, communities)
    except:
        communities = None
        modularity = None
    
    # Small-world coefficient
    if n > 1:
        # Compare to random network
        p = density
        random_G = nx.erdos_renyi_graph(n, p)
        random_clustering = nx.average_clustering(random_G)
        random_path_length = nx.average_shortest_path_length(random_G) if nx.is_connected(random_G) else float('inf')
        
        small_world_coeff = (global_clustering / random_clustering) / (avg_path_length / random_path_length)
    else:
        small_world_coeff = None
    
    return {
        'network_name': network_name,
        'nodes': n,
        'edges': m,
        'density': density,
        'avg_degree': np.mean(degrees),
        'gamma_estimate': gamma,
        'global_clustering': global_clustering,
        'avg_path_length': avg_path_length,
        'diameter': diameter,
        'small_world_coeff': small_world_coeff,
        'modularity': modularity,
        'num_communities': len(communities) if communities else None,
        'degree_centrality': degree_centrality,
        'betweenness_centrality': betweenness_centrality,
        'closeness_centrality': closeness_centrality,
        'eigenvector_centrality': eigenvector_centrality
    }

def plot_degree_distribution(G, title="Degree Distribution"):
    """Plot degree distribution in log-log scale"""
    degrees = [d for n, d in G.degree()]
    degree_counts = Counter(degrees)
    
    k_values = list(degree_counts.keys())
    counts = list(degree_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.loglog(k_values, counts, 'bo', markersize=8)
    plt.xlabel('Degree k')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()

# Example: Analyze a social network
G = nx.barabasi_albert_graph(1000, 3)
results = analyze_social_network(G, "collaboration_network")

print("Social Network Analysis Results:")
for key, value in results.items():
    if key not in ['degree_centrality', 'betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality']:
        print(f"{key}: {value}")

# Plot degree distribution
plot_degree_distribution(G, "Collaboration Network Degree Distribution")
```

## Key Takeaways

1. **Data collection challenges**: Social network data is difficult to collect and often biased
2. **Scale-free structure**: Most social networks follow power-law degree distributions
3. **Small-world effect**: Short path lengths with high clustering are common
4. **Community structure**: Social networks exhibit strong modular organization
5. **Centrality matters**: Different centrality measures capture different aspects of influence
6. **Applications**: Network analysis provides insights into collaboration and knowledge transfer

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Milgram, S. (1967). The small world problem. Psychology Today, 2(1), 60-67.
3. Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. Nature, 393(6684), 440-442.
4. Girvan, M., & Newman, M. E. J. (2002). Community structure in social and biological networks. Proceedings of the National Academy of Sciences, 99(12), 7821-7826.

---

*Social networks provide a rich source of data for understanding human behavior and social dynamics, with important applications in materials science collaboration and knowledge transfer.*