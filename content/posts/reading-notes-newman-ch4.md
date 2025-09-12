---
title: "Reading Notes: Newman's Networks Chapter 4 - Information Networks"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 4 of Newman's 'Networks: An Introduction' covering the World Wide Web, citation networks, and information flow dynamics"
tags: ["reading-notes", "network-theory", "information-networks", "web", "citation-networks"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 4 of Newman's *Networks: An Introduction* examines **information networks** - the complex webs that connect information, knowledge, and ideas. These networks, including the World Wide Web and citation networks, exhibit unique structural properties that reflect how information is created, linked, and accessed in our digital age.

## 4.1 The World Wide Web

### Network Structure

The Web is a **directed network** where:

- **Nodes**: Web pages (HTML documents)
- **Edges**: Hyperlinks (directional connections)
- **Scale**: Billions of pages, trillions of links

### Topological Properties

#### Degree Distributions

The Web exhibits **power-law degree distributions** for both in-degree and out-degree:

**In-degree distribution** (pages linking to a page):
$$P_{\text{in}}(k) \sim k^{-\gamma_{\text{in}}}$$

Where $\gamma_{\text{in}} \approx 2.1$ for the Web.

**Out-degree distribution** (pages linked by a page):
$$P_{\text{out}}(k) \sim k^{-\gamma_{\text{out}}}$$

Where $\gamma_{\text{out}} \approx 2.7$ for the Web.

#### Mathematical Analysis

**Cumulative distribution functions**:

$$P_{\text{in}}(K \geq k) \sim k^{-(\gamma_{\text{in}}-1)}$$
$$P_{\text{out}}(K \geq k) \sim k^{-(\gamma_{\text{out}}-1)}$$

**Moment analysis**:
- **First moments**: $\langle k_{\text{in}} \rangle = \langle k_{\text{out}} \rangle = \frac{m}{n}$
- **Second moments**: $\langle k_{\text{in}}^2 \rangle$, $\langle k_{\text{out}}^2 \rangle$

### Web Graph Components

#### Strongly Connected Components

A **strongly connected component** (SCC) is a set of nodes where every node is reachable from every other node.

**Web structure**:
- **Giant SCC**: ~30% of all pages
- **In-component**: Pages that can reach the giant SCC
- **Out-component**: Pages reachable from the giant SCC
- **Tendrils**: Small disconnected components

#### Mathematical Framework

**Reachability matrix**:
$$R_{ij} = \begin{cases} 
1 & \text{if there exists a path from } i \text{ to } j \\
0 & \text{otherwise}
\end{cases}$$

**Strong connectivity**:
$$S_{ij} = R_{ij} \cdot R_{ji}$$

### PageRank Algorithm

#### Mathematical Definition

**PageRank** of page $i$:

$$PR(i) = \frac{1-d}{n} + d \sum_{j \in \text{in}(i)} \frac{PR(j)}{k_{\text{out}}(j)}$$

Where:
- $d$: Damping factor (typically 0.85)
- $n$: Total number of pages
- $\text{in}(i)$: Pages linking to page $i$
- $k_{\text{out}}(j)$: Out-degree of page $j$

#### Matrix Form

**PageRank vector**:
$$\mathbf{PR} = (1-d) \frac{\mathbf{1}}{n} + d \mathbf{M} \mathbf{PR}$$

Where $\mathbf{M}$ is the **stochastic matrix**:
$$M_{ij} = \frac{A_{ji}}{k_{\text{out}}(j)}$$

**Eigenvalue equation**:
$$\mathbf{PR} = \mathbf{M}^T \mathbf{PR}$$

#### Power Iteration Method

**Iterative solution**:
$$\mathbf{PR}^{(t+1)} = (1-d) \frac{\mathbf{1}}{n} + d \mathbf{M}^T \mathbf{PR}^{(t)}$$

**Convergence criterion**:
$$||\mathbf{PR}^{(t+1)} - \mathbf{PR}^{(t)}||_1 < \epsilon$$

### Web Crawling and Sampling

#### Breadth-First Search (BFS)

**Algorithm**:
1. Start with seed pages
2. Visit all neighbors of current pages
3. Add unvisited pages to queue
4. Repeat until queue is empty

**Bias**: Favors high-degree pages

#### Random Walk Sampling

**Random walk** with transition probabilities:
$$P_{ij} = \frac{A_{ij}}{k_{\text{out}}(i)}$$

**Stationary distribution**:
$$\pi_i = \frac{k_{\text{in}}(i)}{\sum_j k_{\text{in}}(j)}$$

**Metropolis-Hastings correction**:
$$P_{\text{accept}} = \min\left(1, \frac{k_{\text{out}}(j)}{k_{\text{out}}(i)}\right)$$

## 4.2 Citation Networks

### Network Structure

**Citation networks** are directed acyclic graphs (DAGs) where:

- **Nodes**: Academic papers
- **Edges**: Citations (paper A cites paper B)
- **Properties**: Acyclic (no paper cites itself directly or indirectly)

### Degree Distributions

#### In-Degree (Citations Received)

**Power-law distribution**:
$$P_{\text{in}}(k) \sim k^{-\gamma_{\text{citations}}}$$

Where $\gamma_{\text{citations}} \approx 3.0$ for citation networks.

#### Out-Degree (References Made)

**Distribution**:
$$P_{\text{out}}(k) \sim \text{Poisson}(\lambda)$$

Where $\lambda \approx 20-30$ references per paper.

### Citation Dynamics

#### Aging Effect

**Citation rate** as a function of time:
$$r(t) = r_0 e^{-\alpha t}$$

Where:
- $r_0$: Initial citation rate
- $\alpha$: Aging parameter
- $t$: Time since publication

#### Cumulative Citations

**Total citations** after time $T$:
$$C(T) = \int_0^T r(t) \, dt = \frac{r_0}{\alpha}(1 - e^{-\alpha T})$$

**Long-term behavior**:
$$C(\infty) = \frac{r_0}{\alpha}$$

### Impact Metrics

#### H-Index

**Definition**: A scientist has index $h$ if $h$ of their papers have at least $h$ citations each.

**Mathematical formulation**:
$$h = \max\{k : \text{number of papers with } \geq k \text{ citations} \geq k\}$$

#### G-Index

**Definition**: The largest number $g$ such that the top $g$ papers have at least $g^2$ total citations.

**Mathematical formulation**:
$$g = \max\{k : \sum_{i=1}^k c_i \geq k^2\}$$

Where $c_i$ are citations sorted in descending order.

#### PageRank for Citations

**Citation PageRank**:
$$CPR(i) = \frac{1-d}{n} + d \sum_{j \in \text{cites}(i)} \frac{CPR(j)}{k_{\text{out}}(j)}$$

**Advantages over simple citation count**:
- Considers **quality** of citing papers
- Reduces **self-citation** effects
- Accounts for **citation patterns**

## 4.3 Other Information Networks

### Patent Citation Networks

#### Network Properties

- **Nodes**: Patents
- **Edges**: Patent citations
- **Scale**: Millions of patents, tens of millions of citations
- **Temporal**: Citations can only go forward in time

#### Innovation Metrics

**Patent impact**:
$$I_i = \sum_{j \in \text{cited by } i} w_j \cdot CPR(j)$$

Where $w_j$ is the weight of patent $j$.

### Software Dependency Networks

#### Network Structure

- **Nodes**: Software packages
- **Edges**: Dependencies (package A depends on package B)
- **Properties**: Directed, often acyclic

#### Vulnerability Analysis

**Cascade failure** probability:
$$P_{\text{fail}}(i) = 1 - \prod_{j \in \text{dependencies}(i)} (1 - P_{\text{fail}}(j))$$

**System reliability**:
$$R = \prod_{i \in \text{critical packages}} (1 - P_{\text{fail}}(i))$$

### Knowledge Graphs

#### Semantic Networks

- **Nodes**: Concepts, entities
- **Edges**: Semantic relationships
- **Properties**: Multi-relational, weighted

#### Graph Embeddings

**Node2Vec** algorithm:
$$\max_f \sum_{u \in V} \log P(N_S(u)|f(u))$$

Where $N_S(u)$ is the neighborhood of node $u$ under sampling strategy $S$.

## 4.4 Information Spreading Models

### SIR Model for Information

#### Model Dynamics

**States**:
- **Susceptible (S)**: Haven't received information
- **Infected (I)**: Have information and can spread it
- **Recovered (R)**: Have information but won't spread it

#### Mathematical Formulation

**Differential equations**:
$$\frac{dS}{dt} = -\beta SI$$
$$\frac{dI}{dt} = \beta SI - \gamma I$$
$$\frac{dR}{dt} = \gamma I$$

**Conservation**: $S + I + R = N$ (constant)

#### Basic Reproduction Number

$$R_0 = \frac{\beta N}{\gamma}$$

**Epidemic threshold**: $R_0 > 1$

### Complex Contagion

#### Threshold Models

**Activation condition**:
$$\sum_{j \in \text{active neighbors}} w_{ji} \geq \theta_i$$

Where:
- $w_{ji}$: Influence weight
- $\theta_i$: Activation threshold

#### Mathematical Analysis

**Cascade condition**:
$$\frac{\langle k^2 \rangle}{\langle k \rangle} > \frac{1}{\theta}$$

Where $\theta$ is the average threshold.

## 4.5 Applications to Materials Science

### Scientific Collaboration Networks

#### Co-authorship Networks

**Network structure**:
- **Nodes**: Researchers
- **Edges**: Co-authorship relationships
- **Weights**: Number of joint publications

**Mathematical properties**:
- **Degree distribution**: Power-law with $\gamma \approx 2.1$
- **Clustering**: High ($C \approx 0.3$)
- **Path length**: Short ($L \approx 6$)

#### Knowledge Transfer

**Information flow** between research groups:
$$F_{ij} = \sum_{k} \frac{A_{ik} A_{jk}}{k_k}$$

Where $A_{ik}$ is the collaboration strength between groups $i$ and $k$.

### Research Impact Networks

#### Citation Networks in Materials Science

**Top journals**:
- Nature Materials
- Advanced Materials
- Materials Today
- Journal of Materials Chemistry

**Impact factors**:
$$IF = \frac{\text{Citations in year } t}{\text{Articles published in years } t-1, t-2}$$

#### Collaboration Patterns

**Geographic clustering**:
- **Local collaboration**: Same institution
- **National collaboration**: Same country
- **International collaboration**: Different countries

**Mathematical modeling**:
$$P(\text{collaboration}) = \frac{1}{1 + e^{-\alpha \cdot \text{distance}}}$$

## Code Example: Information Network Analysis

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_information_network(G, network_type="web"):
    """Analyze information network properties"""
    
    # Basic statistics
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = m / (n * (n - 1))
    
    # Degree analysis
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    # Power-law fitting for in-degree
    if len(in_degrees) > 1:
        in_degree_dist = Counter(in_degrees)
        k_values = list(in_degree_dist.keys())
        counts = list(in_degree_dist.values())
        
        if len(k_values) > 1:
            log_k = np.log(k_values[1:])  # Skip k=0
            log_counts = np.log(counts[1:])
            
            if len(log_k) > 1:
                gamma_in = -np.polyfit(log_k, log_counts, 1)[0]
            else:
                gamma_in = None
        else:
            gamma_in = None
    else:
        gamma_in = None
    
    # Strongly connected components
    sccs = list(nx.strongly_connected_components(G))
    scc_sizes = [len(scc) for scc in sccs]
    giant_scc_size = max(scc_sizes) if scc_sizes else 0
    
    # PageRank analysis
    try:
        pagerank = nx.pagerank(G, alpha=0.85)
        pagerank_values = list(pagerank.values())
        max_pagerank = max(pagerank_values)
        min_pagerank = min(pagerank_values)
    except:
        pagerank = None
        max_pagerank = min_pagerank = None
    
    # Hubs and authorities (HITS algorithm)
    try:
        hubs, authorities = nx.hits(G)
        hub_values = list(hubs.values())
        authority_values = list(authorities.values())
    except:
        hub_values = authority_values = None
    
    # Clustering (for undirected version)
    G_undirected = G.to_undirected()
    clustering = nx.average_clustering(G_undirected)
    
    # Path length analysis
    if nx.is_weakly_connected(G):
        avg_path_length = nx.average_shortest_path_length(G.to_undirected())
    else:
        # Analyze largest weakly connected component
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        subgraph = G.subgraph(largest_wcc).to_undirected()
        avg_path_length = nx.average_shortest_path_length(subgraph)
    
    return {
        'network_type': network_type,
        'nodes': n,
        'edges': m,
        'density': density,
        'avg_in_degree': np.mean(in_degrees),
        'avg_out_degree': np.mean(out_degrees),
        'gamma_in_estimate': gamma_in,
        'num_sccs': len(sccs),
        'giant_scc_size': giant_scc_size,
        'giant_scc_fraction': giant_scc_size / n if n > 0 else 0,
        'max_pagerank': max_pagerank,
        'min_pagerank': min_pagerank,
        'clustering': clustering,
        'avg_path_length': avg_path_length
    }

def plot_degree_distributions(G, title="Information Network Degree Distributions"):
    """Plot in-degree and out-degree distributions"""
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # In-degree distribution
    in_degree_dist = Counter(in_degrees)
    k_values = list(in_degree_dist.keys())
    counts = list(in_degree_dist.values())
    
    ax1.loglog(k_values, counts, 'bo', markersize=8)
    ax1.set_xlabel('In-Degree k')
    ax1.set_ylabel('Count')
    ax1.set_title('In-Degree Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Out-degree distribution
    out_degree_dist = Counter(out_degrees)
    k_values = list(out_degree_dist.keys())
    counts = list(out_degree_dist.values())
    
    ax2.loglog(k_values, counts, 'ro', markersize=8)
    ax2.set_xlabel('Out-Degree k')
    ax2.set_ylabel('Count')
    ax2.set_title('Out-Degree Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def analyze_citation_network(citations):
    """Analyze citation network properties"""
    
    # Create citation network
    G = nx.DiGraph()
    for citing, cited in citations:
        G.add_edge(citing, cited)
    
    # Basic analysis
    results = analyze_information_network(G, "citation")
    
    # Citation-specific metrics
    in_degrees = [d for n, d in G.in_degree()]
    
    # H-index calculation
    def calculate_h_index(citations):
        citations_sorted = sorted(citations, reverse=True)
        h = 0
        for i, c in enumerate(citations_sorted):
            if c >= i + 1:
                h = i + 1
            else:
                break
        return h
    
    h_index = calculate_h_index(in_degrees)
    
    # G-index calculation
    def calculate_g_index(citations):
        citations_sorted = sorted(citations, reverse=True)
        g = 0
        total_citations = 0
        for i, c in enumerate(citations_sorted):
            total_citations += c
            if total_citations >= (i + 1) ** 2:
                g = i + 1
            else:
                break
        return g
    
    g_index = calculate_g_index(in_degrees)
    
    results['h_index'] = h_index
    results['g_index'] = g_index
    
    return results

# Example: Analyze a web-like network
G = nx.scale_free_graph(1000, alpha=0.41, beta=0.54, gamma=0.05, delta_in=0.2, delta_out=0)
results = analyze_information_network(G, "web_simulation")

print("Information Network Analysis Results:")
for key, value in results.items():
    print(f"{key}: {value}")

# Plot degree distributions
plot_degree_distributions(G, "Simulated Web Network")
```

## Key Takeaways

1. **Directed networks**: Information networks are typically directed with asymmetric relationships
2. **Power-law distributions**: Both in-degree and out-degree often follow power laws
3. **Scale-free structure**: Few highly connected nodes, many sparsely connected nodes
4. **PageRank importance**: Algorithm for ranking nodes based on network structure
5. **Citation dynamics**: Citations follow predictable patterns over time
6. **Impact metrics**: H-index, G-index, and PageRank provide different measures of impact
7. **Applications**: Direct relevance to scientific collaboration and knowledge transfer

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation ranking: Bringing order to the web. Stanford InfoLab.
3. Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks. Science, 286(5439), 509-512.
4. Hirsch, J. E. (2005). An index to quantify an individual's scientific research output. Proceedings of the National Academy of Sciences, 102(46), 16569-16572.

---

*Information networks provide insights into how knowledge is created, linked, and accessed, with important applications in scientific collaboration and research impact assessment.*
