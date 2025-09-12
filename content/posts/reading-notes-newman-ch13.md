---
title: "Reading Notes: Newman's Networks Chapter 13 - Network Dynamics"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 13 of Newman's 'Networks: An Introduction' covering temporal networks, network evolution, and dynamic processes"
tags: ["reading-notes", "network-theory", "network-dynamics", "temporal-networks", "network-evolution"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 13 of Newman's *Networks: An Introduction* explores **network dynamics** - the temporal evolution of network structure and the dynamic processes that occur on networks. This chapter covers temporal networks, network evolution models, and the interplay between network structure and dynamics.

## 13.1 Temporal Networks

### Definition

**Temporal network** is a sequence of networks over time:

$$G(t) = (V, E(t)) \quad \text{for } t \in [0, T]$$

Where:
- $V$: Set of nodes (usually constant)
- $E(t)$: Set of edges at time $t$ (varies with time)
- $T$: Total time period

### Representation

#### Time-Varying Adjacency Matrix

**Adjacency matrix**:
$$A_{ij}(t) = \begin{cases} 
1 & \text{if edge } (i,j) \text{ exists at time } t \\
0 & \text{otherwise}
\end{cases}$$

**Properties**:
- **Symmetric**: $A_{ij}(t) = A_{ji}(t)$ for undirected networks
- **Time-dependent**: $A(t)$ changes over time
- **Binary**: $A_{ij}(t) \in \{0, 1\}$ for unweighted networks

#### Aggregated Network

**Aggregated adjacency matrix**:
$$A_{ij} = \int_0^T A_{ij}(t) \, dt$$

**Discrete version**:
$$A_{ij} = \sum_{t=1}^T A_{ij}(t)$$

**Properties**:
- **Weighted**: $A_{ij} \in [0, T]$ represents total connection time
- **Static**: Single network representing entire time period

### Temporal Measures

#### Temporal Degree

**Temporal degree** of node $i$:
$$k_i(t) = \sum_{j} A_{ij}(t)$$

**Average temporal degree**:
$$\langle k_i \rangle = \frac{1}{T} \int_0^T k_i(t) \, dt$$

#### Temporal Clustering

**Temporal clustering coefficient**:
$$C_i(t) = \frac{2e_i(t)}{k_i(t)(k_i(t)-1)}$$

Where $e_i(t)$ is the number of edges between neighbors of node $i$ at time $t$.

**Average temporal clustering**:
$$\langle C_i \rangle = \frac{1}{T} \int_0^T C_i(t) \, dt$$

#### Temporal Path Length

**Temporal path** from $i$ to $j$:
- Sequence of edges $(i, v_1, t_1), (v_1, v_2, t_2), \ldots, (v_k, j, t_k)$
- Where $t_1 < t_2 < \ldots < t_k$ (causality constraint)

**Temporal distance**:
$$d_{ij}^T = \min\{\text{length of temporal path from } i \text{ to } j\}$$

## 13.2 Dynamic Processes on Temporal Networks

### SIR Model on Temporal Networks

#### Model Definition

**Temporal SIR model**:
$$\frac{dS_i}{dt} = -\beta S_i \sum_{j} A_{ij}(t) I_j$$
$$\frac{dI_i}{dt} = \beta S_i \sum_{j} A_{ij}(t) I_j - \gamma I_i$$
$$\frac{dR_i}{dt} = \gamma I_i$$

Where:
- $A_{ij}(t)$: Temporal adjacency matrix
- $\beta$: Infection rate
- $\gamma$: Recovery rate

#### Basic Reproduction Number

**Temporal basic reproduction number**:
$$R_0^T = \frac{\beta}{\gamma} \frac{\langle k^2 \rangle_T}{\langle k \rangle_T}$$

Where:
- $\langle k \rangle_T = \frac{1}{T} \int_0^T \langle k(t) \rangle \, dt$
- $\langle k^2 \rangle_T = \frac{1}{T} \int_0^T \langle k^2(t) \rangle \, dt$

#### Critical Threshold

**Epidemic threshold**:
$$R_0^T > 1$$

**Critical infection rate**:
$$\beta_c^T = \frac{\gamma \langle k \rangle_T}{\langle k^2 \rangle_T}$$

### Synchronization on Temporal Networks

#### Kuramoto Model

**Temporal Kuramoto model**:
$$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j} A_{ij}(t) \sin(\theta_j - \theta_i)$$

Where:
- $\theta_i(t)$: Phase of oscillator $i$ at time $t$
- $\omega_i$: Natural frequency
- $K$: Coupling strength
- $A_{ij}(t)$: Temporal adjacency matrix

#### Order Parameter

**Temporal order parameter**:
$$r(t) = \left| \frac{1}{N} \sum_{j=1}^N e^{i\theta_j(t)} \right|$$

**Average order parameter**:
$$\langle r \rangle = \frac{1}{T} \int_0^T r(t) \, dt$$

#### Synchronization Threshold

**Critical coupling**:
$$K_c^T = \frac{2}{\pi g(0)} \frac{\langle k^2 \rangle_T}{\langle k \rangle_T}$$

Where $g(0)$ is the frequency distribution at zero.

## 13.3 Network Evolution Models

### Growing Networks

#### Preferential Attachment

**Temporal preferential attachment**:
$$\Pi(k_i, t) = \frac{k_i(t) + a}{\sum_j (k_j(t) + a)}$$

Where:
- $k_i(t)$: Degree of node $i$ at time $t$
- $a$: Initial attractiveness parameter

**Degree evolution**:
$$\frac{dk_i}{dt} = m \Pi(k_i, t)$$

Where $m$ is the number of edges added per time step.

#### Solution

**Degree distribution**:
$$P(k) \sim k^{-\gamma}$$

Where $\gamma = 3 + \frac{a}{m}$.

**Degree evolution**:
$$k_i(t) = m \left(\frac{t}{t_i}\right)^{\beta}$$

Where $\beta = \frac{1}{2}$ for $a = 0$.

### Aging and Preferential Attachment

#### Model Definition

**Aging model**:
$$\Pi(k_i, t, t_i) = \frac{k_i(t) e^{-\alpha(t-t_i)}}{\sum_j k_j(t) e^{-\alpha(t-t_j)}}$$

Where:
- $t_i$: Time when node $i$ was added
- $\alpha$: Aging parameter

#### Degree Distribution

**For $\alpha > 0$**:
$$P(k) \sim k^{-\gamma} e^{-\beta k}$$

Where:
- $\gamma = 3 + \frac{a}{m}$
- $\beta = \frac{\alpha}{m}$

**For $\alpha = 0$**: Standard preferential attachment

### Fitness Models

#### Model Definition

**Fitness model**:
$$\Pi(k_i, \eta_i) = \frac{\eta_i k_i(t)}{\sum_j \eta_j k_j(t)}$$

Where $\eta_i$ is the fitness of node $i$.

#### Degree Distribution

**Fitness distribution**:
$$P(k) \sim k^{-\gamma} \int \eta^{\gamma-1} \rho(\eta) \, d\eta$$

Where $\rho(\eta)$ is the fitness distribution.

**Power-law exponent**:
$$\gamma = 1 + \frac{1}{\langle \eta \rangle}$$

## 13.4 Network Rewiring

### Random Rewiring

#### Model Definition

**Random rewiring**:
1. Remove edge with probability $p$
2. Add new edge randomly
3. Repeat for all edges

**Rewiring probability**:
$$P(\text{rewire}) = p$$

#### Degree Distribution

**For small $p$**: Slight deviation from original distribution
**For large $p$**: Approaches random network distribution

### Preferential Rewiring

#### Model Definition

**Preferential rewiring**:
1. Remove edge with probability $p$
2. Add edge preferentially (high-degree nodes)
3. Repeat for all edges

**Rewiring probability**:
$$P(\text{rewire to node } i) = \frac{k_i}{\sum_j k_j}$$

#### Degree Distribution

**Maintains scale-free structure** for appropriate $p$

### Social Rewiring

#### Model Definition

**Social rewiring**:
- **Triadic closure**: Prefer to connect to friends of friends
- **Homophily**: Prefer to connect to similar nodes
- **Geographic proximity**: Prefer to connect to nearby nodes

**Rewiring probability**:
$$P(\text{rewire to node } j) = \frac{w_{ij}}{\sum_k w_{ik}}$$

Where $w_{ij}$ is the weight of connection to node $j$.

## 13.5 Applications to Materials Science

### Defect Network Evolution

#### Model

**Defect network evolution**:
- **Nodes**: Defect sites
- **Edges**: Defect interactions
- **Growth**: New defects appear over time
- **Rewiring**: Defects can move and reconnect

**Mathematical model**:
$$\frac{dk_i}{dt} = m \Pi(k_i, t) - \gamma k_i$$

Where:
- $m$: Rate of new defect formation
- $\gamma$: Rate of defect annihilation
- $\Pi(k_i, t)$: Preferential attachment probability

#### Phase Transitions

**Defect percolation**:
$$P(\text{percolation}) = 1 - \sum_{k=0}^{\infty} P(k) (1-p)^k$$

**Critical defect concentration**:
$$c_c = \frac{1}{\kappa - 1}$$

Where $\kappa = \frac{\langle k^2 \rangle}{\langle k \rangle}$.

### Nanowire Network Growth

#### Model

**Nanowire network growth**:
- **Nodes**: Nanowire junctions
- **Edges**: Nanowire segments
- **Growth**: New nanowires are added
- **Rewiring**: Nanowires can break and reconnect

**Mathematical model**:
$$\frac{dk_i}{dt} = m \Pi(k_i, t) - \lambda k_i$$

Where:
- $m$: Rate of new nanowire formation
- $\lambda$: Rate of nanowire breaking
- $\Pi(k_i, t)$: Preferential attachment probability

#### Electrical Properties

**Conductivity evolution**:
$$\sigma(t) = \sigma_0 \left(\frac{k(t)}{k_0}\right)^{\alpha}$$

Where:
- $\sigma_0$: Initial conductivity
- $k_0$: Initial average degree
- $\alpha$: Conductivity exponent

### Phase Transition Dynamics

#### Model

**Phase transition dynamics**:
- **Nodes**: Atomic sites
- **Edges**: Chemical bonds
- **Dynamics**: Bonds form and break over time

**Mathematical model**:
$$\frac{dA_{ij}}{dt} = \beta A_{ij} (1 - A_{ij}) - \gamma A_{ij}$$

Where:
- $\beta$: Bond formation rate
- $\gamma$: Bond breaking rate
- $A_{ij}$: Bond strength between sites $i$ and $j$

#### Order Parameter

**Order parameter evolution**:
$$\frac{d\phi}{dt} = -\frac{\partial F}{\partial \phi}$$

Where $F$ is the free energy:
$$F = \frac{1}{2} \sum_{i,j} A_{ij} (1 - A_{ij}) + \frac{1}{2} \sum_{i,j} A_{ij} A_{ji}$$

## Code Example: Network Dynamics

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from collections import defaultdict

def generate_temporal_network(n, T, p=0.1):
    """Generate temporal network with random rewiring"""
    
    # Start with random network
    G = nx.erdos_renyi_graph(n, 0.1)
    
    # Generate temporal sequence
    temporal_networks = []
    current_G = G.copy()
    
    for t in range(T):
        # Random rewiring
        edges_to_remove = []
        for edge in current_G.edges():
            if np.random.random() < p:
                edges_to_remove.append(edge)
        
        # Remove edges
        current_G.remove_edges_from(edges_to_remove)
        
        # Add new edges
        for _ in range(len(edges_to_remove)):
            i, j = np.random.choice(n, 2, replace=False)
            current_G.add_edge(i, j)
        
        temporal_networks.append(current_G.copy())
    
    return temporal_networks

def simulate_temporal_sir(G_temporal, beta=0.1, gamma=0.05, initial_infected=1):
    """Simulate SIR model on temporal network"""
    
    n = G_temporal[0].number_of_nodes()
    T = len(G_temporal)
    
    # Initial conditions
    S = np.ones(n)
    I = np.zeros(n)
    R = np.zeros(n)
    
    # Initial infected nodes
    infected_nodes = np.random.choice(n, initial_infected, replace=False)
    S[infected_nodes] = 0
    I[infected_nodes] = 1
    
    # Store results
    S_history = [S.copy()]
    I_history = [I.copy()]
    R_history = [R.copy()]
    
    # Simulate dynamics
    for t in range(T):
        G = G_temporal[t]
        
        # SIR dynamics
        dS_dt = np.zeros(n)
        dI_dt = np.zeros(n)
        dR_dt = np.zeros(n)
        
        for i in range(n):
            # Infection rate
            infection_rate = 0
            for j in G.neighbors(i):
                infection_rate += I[j]
            infection_rate *= beta * S[i]
            
            # Recovery rate
            recovery_rate = gamma * I[i]
            
            dS_dt[i] = -infection_rate
            dI_dt[i] = infection_rate - recovery_rate
            dR_dt[i] = recovery_rate
        
        # Update states
        S += dS_dt
        I += dI_dt
        R += dR_dt
        
        # Ensure non-negative values
        S = np.maximum(S, 0)
        I = np.maximum(I, 0)
        R = np.maximum(R, 0)
        
        # Store results
        S_history.append(S.copy())
        I_history.append(I.copy())
        R_history.append(R.copy())
    
    return S_history, I_history, R_history

def simulate_temporal_kuramoto(G_temporal, K=1.0, omega=None):
    """Simulate Kuramoto model on temporal network"""
    
    n = G_temporal[0].number_of_nodes()
    T = len(G_temporal)
    
    # Natural frequencies
    if omega is None:
        omega = np.random.normal(0, 1, n)
    
    # Initial phases
    theta = np.random.uniform(0, 2*np.pi, n)
    
    # Store results
    theta_history = [theta.copy()]
    r_history = []
    
    # Simulate dynamics
    for t in range(T):
        G = G_temporal[t]
        
        # Kuramoto dynamics
        dtheta_dt = np.zeros(n)
        
        for i in range(n):
            coupling = 0
            for j in G.neighbors(i):
                coupling += np.sin(theta[j] - theta[i])
            coupling *= K / n
            
            dtheta_dt[i] = omega[i] + coupling
        
        # Update phases
        theta += dtheta_dt
        
        # Calculate order parameter
        r = np.abs(np.mean(np.exp(1j * theta)))
        r_history.append(r)
        
        # Store results
        theta_history.append(theta.copy())
    
    return theta_history, r_history

def simulate_network_growth(n, T, m=1, alpha=0.1):
    """Simulate network growth with preferential attachment and aging"""
    
    # Start with small network
    G = nx.complete_graph(min(3, n))
    node_times = list(range(min(3, n)))
    
    # Store degree evolution
    degree_history = []
    
    for t in range(min(3, n), n):
        # Add new node
        G.add_node(t)
        node_times.append(t)
        
        # Add edges with preferential attachment and aging
        for _ in range(m):
            # Calculate attachment probabilities
            probs = []
            for i in G.nodes():
                if i != t:
                    k_i = G.degree(i)
                    age_factor = np.exp(-alpha * (t - node_times[i]))
                    probs.append(k_i * age_factor)
            
            # Normalize probabilities
            probs = np.array(probs)
            probs = probs / np.sum(probs)
            
            # Choose node to connect to
            if len(probs) > 0:
                target = np.random.choice(list(G.nodes())[:-1], p=probs)
                G.add_edge(t, target)
        
        # Store degree distribution
        degrees = [G.degree(i) for i in G.nodes()]
        degree_history.append(degrees.copy())
    
    return G, degree_history

def analyze_temporal_network(G_temporal):
    """Analyze temporal network properties"""
    
    T = len(G_temporal)
    n = G_temporal[0].number_of_nodes()
    
    # Calculate temporal measures
    temporal_degrees = []
    temporal_clustering = []
    temporal_path_lengths = []
    
    for t in range(T):
        G = G_temporal[t]
        
        # Degree distribution
        degrees = [G.degree(i) for i in G.nodes()]
        temporal_degrees.append(degrees)
        
        # Clustering coefficient
        clustering = nx.average_clustering(G)
        temporal_clustering.append(clustering)
        
        # Path length
        if nx.is_connected(G):
            path_length = nx.average_shortest_path_length(G)
        else:
            # Analyze largest component
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc)
            path_length = nx.average_shortest_path_length(subgraph)
        temporal_path_lengths.append(path_length)
    
    # Calculate aggregated measures
    avg_degree = np.mean([np.mean(degrees) for degrees in temporal_degrees])
    avg_clustering = np.mean(temporal_clustering)
    avg_path_length = np.mean(temporal_path_lengths)
    
    return {
        'temporal_degrees': temporal_degrees,
        'temporal_clustering': temporal_clustering,
        'temporal_path_lengths': temporal_path_lengths,
        'avg_degree': avg_degree,
        'avg_clustering': avg_clustering,
        'avg_path_length': avg_path_length
    }

def plot_temporal_analysis(G_temporal, S_history, I_history, R_history, 
                          theta_history, r_history, degree_history):
    """Plot temporal network analysis results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # SIR dynamics
    t_sir = range(len(S_history))
    S_avg = [np.mean(S) for S in S_history]
    I_avg = [np.mean(I) for I in I_history]
    R_avg = [np.mean(R) for R in R_history]
    
    ax1.plot(t_sir, S_avg, 'b-', label='Susceptible', linewidth=2)
    ax1.plot(t_sir, I_avg, 'r-', label='Infected', linewidth=2)
    ax1.plot(t_sir, R_avg, 'g-', label='Recovered', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Fraction')
    ax1.set_title('Temporal SIR Dynamics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Synchronization dynamics
    t_sync = range(len(r_history))
    ax2.plot(t_sync, r_history, 'b-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Order Parameter r')
    ax2.set_title('Temporal Synchronization')
    ax2.grid(True, alpha=0.3)
    
    # Degree evolution
    t_deg = range(len(degree_history))
    avg_degrees = [np.mean(degrees) for degrees in degree_history]
    ax3.plot(t_deg, avg_degrees, 'b-', linewidth=2)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Average Degree')
    ax3.set_title('Network Growth')
    ax3.grid(True, alpha=0.3)
    
    # Network visualization at different times
    times_to_plot = [0, len(G_temporal)//2, len(G_temporal)-1]
    for i, t in enumerate(times_to_plot):
        G = G_temporal[t]
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Plot in subplot
        ax = plt.subplot(2, 2, 4)
        nx.draw(G, pos, ax=ax, node_size=30, node_color='lightblue', 
                edge_color='gray', alpha=0.6)
        ax.set_title(f'Network at t={t}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example: Temporal network analysis
n, T = 50, 100
G_temporal = generate_temporal_network(n, T, p=0.1)

# Simulate SIR dynamics
S_history, I_history, R_history = simulate_temporal_sir(G_temporal, beta=0.1, gamma=0.05)

# Simulate synchronization
theta_history, r_history = simulate_temporal_kuramoto(G_temporal, K=1.0)

# Simulate network growth
G_grown, degree_history = simulate_network_growth(100, 50, m=2, alpha=0.1)

# Analyze temporal network
temporal_analysis = analyze_temporal_network(G_temporal)

print("Temporal Network Analysis:")
print(f"Average degree: {temporal_analysis['avg_degree']:.3f}")
print(f"Average clustering: {temporal_analysis['avg_clustering']:.3f}")
print(f"Average path length: {temporal_analysis['avg_path_length']:.3f}")

# Plot results
plot_temporal_analysis(G_temporal, S_history, I_history, R_history, 
                      theta_history, r_history, degree_history)
```

## Key Takeaways

1. **Temporal networks**: Networks that change over time
2. **Dynamic processes**: SIR, synchronization, and other processes on temporal networks
3. **Network evolution**: Growth, rewiring, and aging models
4. **Mathematical analysis**: Differential equations and probability theory
5. **Applications**: Important for materials science and complex systems
6. **Simulation**: Computational methods for studying network dynamics
7. **Emergent properties**: How network structure affects dynamic behavior

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Holme, P., & Saramäki, J. (2012). Temporal networks. Physics Reports, 519(3), 97-125.
3. Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks. Science, 286(5439), 509-512.
4. Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence. Springer.

---

*Network dynamics provides insights into how networks evolve over time and how dynamic processes unfold on them, with important applications in understanding materials behavior and system evolution.*
