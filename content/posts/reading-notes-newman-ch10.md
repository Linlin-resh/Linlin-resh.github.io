---
title: "Reading Notes: Newman's Networks Chapter 10 - Network Processes"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 10 of Newman's 'Networks: An Introduction' covering percolation, epidemic spreading, synchronization, and other dynamic processes on networks"
tags: ["reading-notes", "network-theory", "network-processes", "percolation", "epidemic-spreading"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 10 of Newman's *Networks: An Introduction* explores **network processes** - the dynamic phenomena that occur on networks. This chapter covers percolation, epidemic spreading, synchronization, and other processes that reveal how network structure affects system behavior.

## 10.1 Percolation Theory

### Bond Percolation

#### Model Definition

**Bond percolation** on a network:

- **Initial state**: All edges present
- **Process**: Remove edges with probability $1-p$
- **Question**: What is the probability of a giant connected component?

#### Mathematical Analysis

**Percolation probability**:
$$P(p) = 1 - \sum_{k=0}^{\infty} P(k) (1-p)^k$$

Where $P(k)$ is the degree distribution.

**Critical threshold**:
$$p_c = \frac{1}{\kappa - 1}$$

Where $\kappa = \frac{\langle k^2 \rangle}{\langle k \rangle}$ is the degree ratio.

#### Phase Transition

**Below threshold** ($p < p_c$):
- Only small, isolated components exist
- Largest component size: $O(\ln n)$

**At threshold** ($p = p_c$):
- Giant component emerges
- Largest component size: $O(n^{2/3})$

**Above threshold** ($p > p_c$):
- Giant component dominates
- Largest component size: $O(n)$

### Site Percolation

#### Model Definition

**Site percolation** on a network:

- **Initial state**: All nodes present
- **Process**: Remove nodes with probability $1-p$
- **Question**: What is the probability of a giant connected component?

#### Mathematical Analysis

**Percolation probability**:
$$P(p) = 1 - \sum_{k=0}^{\infty} P(k) (1-p)^k$$

**Critical threshold**:
$$p_c = \frac{1}{\kappa - 1}$$

**Same as bond percolation** for degree-uncorrelated networks.

### Percolation on Scale-Free Networks

#### Critical Threshold

**For power-law degree distribution** $P(k) \sim k^{-\gamma}$:

- **$\gamma > 3$**: $p_c > 0$ (finite threshold)
- **$2 < \gamma \leq 3$**: $p_c = 0$ (no threshold)
- **$\gamma \leq 2$**: $p_c = 0$ (no threshold)

#### Mathematical Derivation

**Second moment**:
$$\langle k^2 \rangle = \int_{k_{\min}}^{\infty} k^2 P(k) \, dk$$

**For power-law distribution**:
$$\langle k^2 \rangle \sim \int_{k_{\min}}^{\infty} k^{2-\gamma} \, dk$$

**Convergence condition**: $\gamma > 3$

**Critical threshold**:
$$p_c = \frac{1}{\kappa - 1} = \frac{\langle k \rangle}{\langle k^2 \rangle - \langle k \rangle}$$

### Applications to Materials Science

#### Nanowire Networks

**Network formation**:
- **Nodes**: Nanowire junctions
- **Edges**: Nanowire segments
- **Percolation**: Electrical connectivity

**Critical density**:
$$\rho_c = \frac{1}{\pi r^2 (\kappa - 1)}$$

Where $r$ is the interaction radius.

**Conductivity**:
$$\sigma \sim (\rho - \rho_c)^t$$

Where $t \approx 2.0$ is the conductivity exponent.

#### Defect Networks

**Defect percolation**:
- **Nodes**: Defect sites
- **Edges**: Defect interactions
- **Percolation**: Defect clustering

**Critical concentration**:
$$c_c = \frac{1}{\kappa - 1}$$

## 10.2 Epidemic Spreading

### SIR Model

#### Model Definition

**SIR model** on a network:

- **S**: Susceptible (haven't been infected)
- **I**: Infected (can spread the disease)
- **R**: Recovered (immune, cannot spread)

#### Dynamics

**Transition rates**:
- **S → I**: $\beta$ (infection rate)
- **I → R**: $\gamma$ (recovery rate)

**Differential equations**:
$$\frac{dS_i}{dt} = -\beta S_i \sum_{j} A_{ij} I_j$$
$$\frac{dI_i}{dt} = \beta S_i \sum_{j} A_{ij} I_j - \gamma I_i$$
$$\frac{dR_i}{dt} = \gamma I_i$$

#### Basic Reproduction Number

**Definition**:
$$R_0 = \frac{\beta}{\gamma} \frac{\langle k^2 \rangle}{\langle k \rangle}$$

**Epidemic threshold**:
$$R_0 > 1$$

**Critical infection rate**:
$$\beta_c = \frac{\gamma \langle k \rangle}{\langle k^2 \rangle}$$

### SIS Model

#### Model Definition

**SIS model** on a network:

- **S**: Susceptible
- **I**: Infected
- **No recovered state**: Infected nodes can become susceptible again

#### Dynamics

**Transition rates**:
- **S → I**: $\beta$ (infection rate)
- **I → S**: $\gamma$ (recovery rate)

**Differential equations**:
$$\frac{dS_i}{dt} = -\beta S_i \sum_{j} A_{ij} I_j + \gamma I_i$$
$$\frac{dI_i}{dt} = \beta S_i \sum_{j} A_{ij} I_j - \gamma I_i$$

#### Steady State

**Infection prevalence**:
$$\rho = \frac{\beta \langle k^2 \rangle}{\gamma \langle k \rangle + \beta \langle k^2 \rangle}$$

**Critical threshold**:
$$\beta_c = \frac{\gamma \langle k \rangle}{\langle k^2 \rangle}$$

### Complex Contagion

#### Threshold Models

**Activation condition**:
$$\sum_{j \in \text{active neighbors}} w_{ji} \geq \theta_i$$

Where:
- $w_{ji}$: Influence weight from $j$ to $i$
- $\theta_i$: Threshold for node $i$

#### Cascade Condition

**Global cascade condition**:
$$\frac{\langle k^2 \rangle}{\langle k \rangle} > \frac{1}{\theta}$$

Where $\theta$ is the average threshold.

#### Mathematical Analysis

**Cascade size**:
$$S = \sum_{k=0}^{\infty} P(k) \sum_{m=0}^{k} \binom{k}{m} p^m (1-p)^{k-m} \Theta(m - \theta k)$$

Where $\Theta(x)$ is the Heaviside step function.

## 10.3 Synchronization

### Kuramoto Model

#### Model Definition

**Kuramoto model** on a network:

$$\frac{d\theta_i}{dt} = \omega_i + \frac{K}{N} \sum_{j=1}^N A_{ij} \sin(\theta_j - \theta_i)$$

Where:
- $\theta_i$: Phase of oscillator $i$
- $\omega_i$: Natural frequency
- $K$: Coupling strength
- $A_{ij}$: Adjacency matrix

#### Order Parameters

**Global order parameter**:
$$r = \left| \frac{1}{N} \sum_{j=1}^N e^{i\theta_j} \right|$$

**Local order parameter**:
$$r_i = \left| \frac{1}{k_i} \sum_{j \in \mathcal{N}_i} e^{i\theta_j} \right|$$

#### Synchronization Threshold

**Critical coupling**:
$$K_c = \frac{2}{\pi g(0)} \frac{\langle k^2 \rangle}{\langle k \rangle}$$

Where $g(0)$ is the frequency distribution at zero.

### Master Stability Function

#### Linear Stability Analysis

**Perturbation equations**:
$$\frac{d\delta \theta_i}{dt} = \delta \omega_i + \frac{K}{N} \sum_{j=1}^N A_{ij} \cos(\theta_j - \theta_i) (\delta \theta_j - \delta \theta_i)$$

**Eigenvalue problem**:
$$\lambda \delta \theta = L \delta \theta$$

Where $L$ is the Laplacian matrix.

#### Stability Condition

**Synchronization is stable if**:
$$\lambda_2 > \frac{\sigma^2}{K}$$

Where $\lambda_2$ is the second smallest eigenvalue of the Laplacian.

## 10.4 Random Walks

### Simple Random Walk

#### Model Definition

**Simple random walk** on a network:

- **Start**: Random initial node
- **Move**: At each step, move to random neighbor
- **Transition matrix**: $P_{ij} = \frac{A_{ij}}{k_i}$

#### Stationary Distribution

**Stationary distribution**:
$$\pi_i = \frac{k_i}{2m}$$

**Verification**:
$$\sum_i \pi_i P_{ij} = \sum_i \frac{k_i}{2m} \frac{A_{ij}}{k_i} = \frac{1}{2m} \sum_i A_{ij} = \frac{k_j}{2m} = \pi_j$$

#### Mixing Time

**Mixing time**:
$$\tau_{\text{mix}} = \min\{t : \max_i ||P^t(i, \cdot) - \pi||_1 \leq \epsilon\}$$

**Spectral bound**:
$$\tau_{\text{mix}} \leq \frac{1}{1 - \lambda_2} \log\left(\frac{1}{\epsilon \pi_{\min}}\right)$$

Where $\lambda_2$ is the second largest eigenvalue of $P$.

### PageRank

#### Model Definition

**PageRank** as a random walk:

$$\mathbf{PR} = (1-d) \frac{\mathbf{1}}{n} + d P^T \mathbf{PR}$$

Where $P$ is the transition matrix.

#### Mathematical Properties

**Convergence**: Guaranteed for $d < 1$

**Stationary distribution**: Unique solution

**Interpretation**: Probability of being at each node in the long run

## 10.5 Applications to Materials Science

### Information Spreading in Materials

#### Defect Propagation

**Defect spreading** can be modeled as:

- **S**: Healthy sites
- **I**: Defected sites
- **R**: Repaired sites

**Mathematical model**:
$$\frac{dS_i}{dt} = -\beta S_i \sum_{j} A_{ij} I_j + \gamma R_i$$
$$\frac{dI_i}{dt} = \beta S_i \sum_{j} A_{ij} I_j - \delta I_i$$
$$\frac{dR_i}{dt} = \delta I_i - \gamma R_i$$

#### Critical Defect Rate

**Critical defect rate**:
$$\beta_c = \frac{\delta \langle k \rangle}{\langle k^2 \rangle}$$

**For scale-free networks**: $\beta_c \to 0$ as $n \to \infty$

### Phase Transitions in Materials

#### Order-Disorder Transitions

**Order parameter**:
$$\phi = \frac{1}{N} \sum_{i=1}^N \cos(\theta_i - \theta_0)$$

Where $\theta_0$ is the preferred orientation.

**Critical temperature**:
$$T_c = \frac{K \langle k^2 \rangle}{2 \langle k \rangle}$$

#### Percolation in Disordered Systems

**Defect percolation**:
$$P(\text{percolation}) = 1 - \sum_{k=0}^{\infty} P(k) (1-p)^k$$

**Critical concentration**:
$$c_c = \frac{1}{\kappa - 1}$$

### Network-Based Materials Design

#### Structure-Property Relationships

**Property prediction**:
$$P = f(\langle k \rangle, C, L, \ldots)$$

**Network optimization**:
$$\min_{\text{network}} \sum_i w_i |P_i - P_i^{\text{target}}|^2$$

**Constraints**:
- **Connectivity**: Network must be connected
- **Degree bounds**: $k_{\min} \leq k_i \leq k_{\max}$
- **Clustering bounds**: $C_{\min} \leq C \leq C_{\max}$

## Code Example: Network Processes

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.integrate import odeint

def simulate_percolation(G, p_values):
    """Simulate percolation on network"""
    
    results = []
    
    for p in p_values:
        # Remove edges with probability 1-p
        edges_to_remove = []
        for edge in G.edges():
            if np.random.random() > p:
                edges_to_remove.append(edge)
        
        # Create subgraph
        G_sub = G.copy()
        G_sub.remove_edges_from(edges_to_remove)
        
        # Analyze components
        components = list(nx.connected_components(G_sub))
        component_sizes = [len(comp) for comp in components]
        giant_component_size = max(component_sizes) if component_sizes else 0
        giant_component_fraction = giant_component_size / G.number_of_nodes()
        
        results.append({
            'p': p,
            'giant_component_fraction': giant_component_fraction,
            'num_components': len(components),
            'avg_component_size': np.mean(component_sizes)
        })
    
    return results

def simulate_sir_epidemic(G, beta, gamma, initial_infected=1):
    """Simulate SIR epidemic on network"""
    
    n = G.number_of_nodes()
    
    # Initial conditions
    S = np.ones(n)  # Susceptible
    I = np.zeros(n)  # Infected
    R = np.zeros(n)  # Recovered
    
    # Initial infected nodes
    infected_nodes = np.random.choice(n, initial_infected, replace=False)
    S[infected_nodes] = 0
    I[infected_nodes] = 1
    
    # Time points
    t = np.linspace(0, 100, 1000)
    
    # SIR dynamics
    def sir_equations(y, t):
        S, I, R = y
        
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
        
        return np.concatenate([dS_dt, dI_dt, dR_dt])
    
    # Solve ODEs
    y0 = np.concatenate([S, I, R])
    sol = odeint(sir_equations, y0, t)
    
    # Extract results
    S_t = sol[:, :n]
    I_t = sol[:, n:2*n]
    R_t = sol[:, 2*n:]
    
    return t, S_t, I_t, R_t

def simulate_kuramoto(G, K, omega, initial_phases=None):
    """Simulate Kuramoto model on network"""
    
    n = G.number_of_nodes()
    
    # Initial phases
    if initial_phases is None:
        theta = np.random.uniform(0, 2*np.pi, n)
    else:
        theta = initial_phases
    
    # Time points
    t = np.linspace(0, 100, 1000)
    
    # Kuramoto dynamics
    def kuramoto_equations(theta, t):
        dtheta_dt = np.zeros(n)
        
        for i in range(n):
            coupling = 0
            for j in G.neighbors(i):
                coupling += np.sin(theta[j] - theta[i])
            coupling *= K / n
            
            dtheta_dt[i] = omega[i] + coupling
        
        return dtheta_dt
    
    # Solve ODEs
    sol = odeint(kuramoto_equations, theta, t)
    
    return t, sol

def analyze_network_processes(G, process_type="percolation"):
    """Analyze network processes"""
    
    if process_type == "percolation":
        # Percolation analysis
        p_values = np.linspace(0.1, 1.0, 20)
        results = simulate_percolation(G, p_values)
        
        # Find critical threshold
        critical_p = None
        for i, result in enumerate(results):
            if result['giant_component_fraction'] > 0.5:
                critical_p = result['p']
                break
        
        return {
            'process_type': process_type,
            'critical_threshold': critical_p,
            'results': results
        }
    
    elif process_type == "sir":
        # SIR epidemic analysis
        beta = 0.1
        gamma = 0.05
        t, S_t, I_t, R_t = simulate_sir_epidemic(G, beta, gamma)
        
        # Calculate basic reproduction number
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees)
        degree_variance = np.var(degrees)
        R0 = (beta / gamma) * (avg_degree + degree_variance / avg_degree)
        
        return {
            'process_type': process_type,
            'R0': R0,
            'time': t,
            'S': S_t,
            'I': I_t,
            'R': R_t
        }
    
    elif process_type == "kuramoto":
        # Kuramoto synchronization analysis
        K = 1.0
        omega = np.random.normal(0, 1, G.number_of_nodes())
        t, theta_t = simulate_kuramoto(G, K, omega)
        
        # Calculate order parameter
        r_t = np.abs(np.mean(np.exp(1j * theta_t), axis=1))
        
        return {
            'process_type': process_type,
            'time': t,
            'phases': theta_t,
            'order_parameter': r_t
        }

def plot_network_processes(G, process_type="percolation"):
    """Plot network process results"""
    
    results = analyze_network_processes(G, process_type)
    
    if process_type == "percolation":
        p_values = [r['p'] for r in results['results']]
        giant_fractions = [r['giant_component_fraction'] for r in results['results']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(p_values, giant_fractions, 'bo-', markersize=8)
        plt.axvline(x=results['critical_threshold'], color='r', linestyle='--', 
                   label=f'Critical threshold: {results["critical_threshold"]:.3f}')
        plt.xlabel('Probability p')
        plt.ylabel('Giant Component Fraction')
        plt.title('Percolation Phase Transition')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    elif process_type == "sir":
        t = results['time']
        S_t = results['S']
        I_t = results['I']
        R_t = results['R']
        
        plt.figure(figsize=(12, 8))
        
        # Plot SIR curves
        plt.subplot(2, 2, 1)
        plt.plot(t, np.mean(S_t, axis=1), 'b-', label='Susceptible')
        plt.plot(t, np.mean(I_t, axis=1), 'r-', label='Infected')
        plt.plot(t, np.mean(R_t, axis=1), 'g-', label='Recovered')
        plt.xlabel('Time')
        plt.ylabel('Fraction')
        plt.title('SIR Epidemic Dynamics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot R0
        plt.subplot(2, 2, 2)
        plt.bar(['R0'], [results['R0']], color='orange')
        plt.ylabel('Basic Reproduction Number')
        plt.title('Epidemic Threshold')
        plt.grid(True, alpha=0.3)
        
        # Plot individual node dynamics
        plt.subplot(2, 2, 3)
        for i in range(min(5, S_t.shape[1])):
            plt.plot(t, S_t[:, i], 'b-', alpha=0.3)
            plt.plot(t, I_t[:, i], 'r-', alpha=0.3)
            plt.plot(t, R_t[:, i], 'g-', alpha=0.3)
        plt.xlabel('Time')
        plt.ylabel('Fraction')
        plt.title('Individual Node Dynamics')
        plt.grid(True, alpha=0.3)
        
        # Plot network visualization
        plt.subplot(2, 2, 4)
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw(G, pos, node_size=50, node_color='lightblue', 
                edge_color='gray', alpha=0.6)
        plt.title('Network Structure')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    elif process_type == "kuramoto":
        t = results['time']
        theta_t = results['phases']
        r_t = results['order_parameter']
        
        plt.figure(figsize=(12, 8))
        
        # Plot order parameter
        plt.subplot(2, 2, 1)
        plt.plot(t, r_t, 'b-', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Order Parameter r')
        plt.title('Synchronization Order Parameter')
        plt.grid(True, alpha=0.3)
        
        # Plot phase evolution
        plt.subplot(2, 2, 2)
        for i in range(min(10, theta_t.shape[1])):
            plt.plot(t, theta_t[:, i], alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Phase θ')
        plt.title('Phase Evolution')
        plt.grid(True, alpha=0.3)
        
        # Plot phase distribution
        plt.subplot(2, 2, 3)
        final_phases = theta_t[-1, :]
        plt.hist(final_phases, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Final Phase')
        plt.ylabel('Count')
        plt.title('Final Phase Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot network visualization
        plt.subplot(2, 2, 4)
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw(G, pos, node_size=50, node_color='lightblue', 
                edge_color='gray', alpha=0.6)
        plt.title('Network Structure')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Example: Analyze network processes
G = nx.barabasi_albert_graph(100, 3)

# Percolation analysis
print("Percolation Analysis:")
percolation_results = analyze_network_processes(G, "percolation")
print(f"Critical threshold: {percolation_results['critical_threshold']:.3f}")

# SIR epidemic analysis
print("\nSIR Epidemic Analysis:")
sir_results = analyze_network_processes(G, "sir")
print(f"Basic reproduction number: {sir_results['R0']:.3f}")

# Kuramoto synchronization analysis
print("\nKuramoto Synchronization Analysis:")
kuramoto_results = analyze_network_processes(G, "kuramoto")
print(f"Final order parameter: {kuramoto_results['order_parameter'][-1]:.3f}")

# Plot results
plot_network_processes(G, "percolation")
plot_network_processes(G, "sir")
plot_network_processes(G, "kuramoto")
```

## Key Takeaways

1. **Percolation theory**: Provides framework for understanding connectivity transitions
2. **Epidemic spreading**: Network structure affects disease transmission dynamics
3. **Synchronization**: Coupling strength and network topology determine synchronization
4. **Random walks**: Reveal network structure and mixing properties
5. **Phase transitions**: Critical thresholds determine system behavior
6. **Applications**: Network processes help understand materials science phenomena
7. **Mathematical analysis**: Rigorous theory enables prediction of dynamic behavior

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Stauffer, D., & Aharony, A. (1994). Introduction to Percolation Theory. Taylor & Francis.
3. Pastor-Satorras, R., & Vespignani, A. (2001). Epidemic spreading in scale-free networks. Physical Review Letters, 86(14), 3200.
4. Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence. Springer.

---

*Network processes provide insights into how dynamic phenomena unfold on complex networks, with important applications in understanding materials behavior and system dynamics.*
