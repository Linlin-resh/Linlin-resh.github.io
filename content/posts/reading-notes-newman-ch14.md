---
title: "Reading Notes: Newman's Networks Chapter 14 - Network Control"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 14 of Newman's 'Networks: An Introduction' covering network controllability, control theory, and applications to complex systems"
tags: ["reading-notes", "network-theory", "network-control", "controllability", "control-theory"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 14 of Newman's *Networks: An Introduction* explores **network control** - the theory and methods for controlling complex networks to achieve desired behaviors. This chapter covers controllability theory, control strategies, and applications to understanding and manipulating network dynamics.

## 14.1 Controllability Theory

### Linear Controllability

#### System Model

**Linear time-invariant system**:
$$\frac{dx}{dt} = Ax + Bu$$

Where:
- $x \in \mathbb{R}^n$: State vector
- $u \in \mathbb{R}^m$: Control input
- $A \in \mathbb{R}^{n \times n}$: System matrix
- $B \in \mathbb{R}^{n \times m}$: Input matrix

#### Controllability Matrix

**Controllability matrix**:
$$\mathcal{C} = [B, AB, A^2B, \ldots, A^{n-1}B]$$

**Controllability condition**: System is controllable if and only if
$$\text{rank}(\mathcal{C}) = n$$

#### Kalman's Controllability Test

**Theorem**: The system is controllable if and only if the controllability matrix has full rank.

**Proof**: Based on Cayley-Hamilton theorem and linear algebra.

### Structural Controllability

#### Definition

**Structural controllability**: System is structurally controllable if it is controllable for almost all parameter values.

**Advantage**: Depends only on network topology, not specific parameter values.

#### Minimum Number of Drivers

**Theorem**: Minimum number of driver nodes needed for structural controllability is:
$$N_D = \max_i \mu_i$$

Where $\mu_i$ is the geometric multiplicity of eigenvalue $\lambda_i$ of $A$.

#### Driver Node Selection

**Algorithm**:
1. Find maximum matching in bipartite graph
2. Unmatched nodes are driver nodes
3. Minimum number of drivers = $n - |M|$

Where $|M|$ is the size of maximum matching.

### Energy Control

#### Control Energy

**Control energy**:
$$E = \int_0^T ||u(t)||^2 dt$$

**Minimum energy control**:
$$u^*(t) = B^T e^{A^T(T-t)} W^{-1}(T) x_f$$

Where $W(T)$ is the controllability Gramian:
$$W(T) = \int_0^T e^{At} BB^T e^{A^T t} dt$$

#### Controllability Gramian

**Properties**:
- **Symmetric**: $W(T) = W^T(T)$
- **Positive definite**: If system is controllable
- **Energy bound**: $E \geq x_f^T W^{-1}(T) x_f$

### Optimal Control

#### Linear Quadratic Regulator (LQR)

**Cost function**:
$$J = \int_0^T (x^T Q x + u^T R u) dt$$

Where:
- $Q \geq 0$: State weight matrix
- $R > 0$: Control weight matrix

**Optimal control**:
$$u^*(t) = -R^{-1} B^T P(t) x(t)$$

Where $P(t)$ satisfies the Riccati equation:
$$-\frac{dP}{dt} = A^T P + PA - PBR^{-1}B^T P + Q$$

#### Steady-State Solution

**Algebraic Riccati equation**:
$$A^T P + PA - PBR^{-1}B^T P + Q = 0$$

**Optimal control**:
$$u^*(t) = -R^{-1} B^T P x(t)$$

## 14.2 Network Controllability

### Controllability of Complex Networks

#### Driver Node Analysis

**Driver nodes**: Nodes that need external control input

**Controllable nodes**: Nodes that can be controlled by driver nodes

**Mathematical formulation**:
$$\min_{B} ||B||_0 \quad \text{subject to } \text{rank}(\mathcal{C}) = n$$

Where $||B||_0$ is the number of non-zero columns in $B$.

#### Controllability Gramian

**Network controllability Gramian**:
$$W = \int_0^{\infty} e^{At} BB^T e^{A^T t} dt$$

**Properties**:
- **Trace**: $\text{tr}(W)$ measures controllability
- **Determinant**: $\det(W)$ measures controllability
- **Eigenvalues**: $\lambda_i(W)$ measure controllability in different directions

### Control Centrality

#### Definition

**Control centrality** of node $i$:
$$C_i = \text{tr}(W_i)$$

Where $W_i$ is the controllability Gramian when node $i$ is the only driver.

#### Properties

**Range**: $C_i \geq 0$

**Interpretation**: Higher $C_i$ means node $i$ is more important for control

**Calculation**: $C_i = \sum_{j=1}^n \frac{1}{\lambda_j^2}$ where $\lambda_j$ are eigenvalues of $A$.

### Controllability and Network Structure

#### Degree and Controllability

**High-degree nodes**: Often important for control

**Low-degree nodes**: May be easier to control

**Mathematical relationship**:
$$C_i \sim k_i^{\alpha}$$

Where $\alpha$ depends on network structure.

#### Clustering and Controllability

**High clustering**: May reduce controllability

**Low clustering**: May increase controllability

**Mathematical relationship**:
$$C_i \sim C_i^{-\beta}$$

Where $\beta$ depends on network structure.

## 14.3 Control Strategies

### Centralized Control

#### Single Controller

**Control law**:
$$u(t) = -K x(t)$$

Where $K$ is the feedback gain matrix.

**Design**: $K = R^{-1} B^T P$

**Advantages**: Optimal performance
**Disadvantages**: Requires global information

#### Multiple Controllers

**Control law**:
$$u_i(t) = -K_i x_i(t) + \sum_{j \in \mathcal{N}_i} K_{ij} x_j(t)$$

Where:
- $K_i$: Local feedback gain
- $K_{ij}$: Coupling gain
- $\mathcal{N}_i$: Neighbors of node $i$

### Decentralized Control

#### Local Control

**Control law**:
$$u_i(t) = -K_i x_i(t)$$

**Design**: Each controller uses only local information

**Advantages**: Scalable, robust
**Disadvantages**: May not achieve global optimum

#### Distributed Control

**Control law**:
$$u_i(t) = -K_i x_i(t) + \sum_{j \in \mathcal{N}_i} K_{ij} (x_j(t) - x_i(t))$$

**Design**: Controllers communicate with neighbors

**Advantages**: Better performance than local control
**Disadvantages**: Requires communication

### Adaptive Control

#### Model Reference Adaptive Control

**Reference model**:
$$\frac{dx_m}{dt} = A_m x_m + B_m r$$

**Control law**:
$$u(t) = K_x(t) x(t) + K_r(t) r(t)$$

**Adaptation law**:
$$\frac{dK_x}{dt} = -\Gamma_x x(t) e^T(t) P B$$
$$\frac{dK_r}{dt} = -\Gamma_r r(t) e^T(t) P B$$

Where $e(t) = x(t) - x_m(t)$ is the tracking error.

#### Self-Tuning Control

**Parameter estimation**:
$$\hat{\theta}(t) = \hat{\theta}(t-1) + \Gamma \phi(t) e(t)$$

**Control law**:
$$u(t) = \hat{\theta}^T(t) \phi(t)$$

Where $\phi(t)$ is the regressor vector.

## 14.4 Applications to Materials Science

### Defect Control

#### Problem

**Given**: Defect network with dynamics
$$\frac{dx_i}{dt} = f_i(x_i) + \sum_{j} A_{ij} g_{ij}(x_i, x_j) + u_i$$

**Goal**: Control defect density to desired level

**Control objective**:
$$\min_{u} \int_0^T ||x(t) - x_d||^2 dt$$

#### Solution

**Optimal control**:
$$u_i^*(t) = -R^{-1} B^T P (x_i(t) - x_{d,i})$$

**Feedback gain**:
$$K = R^{-1} B^T P$$

**Stability**: Guaranteed if $A$ is Hurwitz

### Phase Control

#### Problem

**Given**: Phase transition dynamics
$$\frac{d\phi}{dt} = f(\phi) + u$$

**Goal**: Control phase to desired value

**Control objective**:
$$\min_{u} \int_0^T (\phi(t) - \phi_d)^2 dt$$

#### Solution

**Optimal control**:
$$u^*(t) = -R^{-1} B^T P (\phi(t) - \phi_d)$$

**Phase transition**: Controlled by external field

### Nanowire Network Control

#### Problem

**Given**: Nanowire network with electrical dynamics
$$\frac{dV_i}{dt} = \frac{1}{C_i} \sum_{j} \frac{V_j - V_i}{R_{ij}} + u_i$$

**Goal**: Control voltage distribution

**Control objective**:
$$\min_{u} \int_0^T ||V(t) - V_d||^2 dt$$

#### Solution

**Optimal control**:
$$u_i^*(t) = -R^{-1} C_i (V_i(t) - V_{d,i})$$

**Electrical properties**: Controlled by external current

## 14.5 Robust Control

### Robust Controllability

#### Definition

**Robust controllability**: System remains controllable under parameter uncertainty

**Mathematical condition**:
$$\text{rank}(\mathcal{C}(\theta)) = n \quad \forall \theta \in \Theta$$

Where $\Theta$ is the uncertainty set.

#### Robust Control Design

**Control law**:
$$u(t) = -K(\theta) x(t)$$

**Design**: $K(\theta)$ minimizes worst-case performance

**Performance**:
$$J = \max_{\theta \in \Theta} \int_0^T (x^T Q x + u^T R u) dt$$

### H∞ Control

#### Problem

**Given**: System with disturbances
$$\frac{dx}{dt} = Ax + Bu + B_w w$$
$$z = Cx + Du$$

**Goal**: Minimize effect of disturbances

**Performance**:
$$||T_{zw}||_{\infty} < \gamma$$

Where $T_{zw}$ is the transfer function from $w$ to $z$.

#### Solution

**Control law**:
$$u(t) = -K x(t)$$

**Design**: $K$ minimizes $||T_{zw}||_{\infty}$

**Riccati equation**:
$$A^T P + PA - P(BR^{-1}B^T - \gamma^{-2}B_w B_w^T) P + C^T C = 0$$

## Code Example: Network Control

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are, eig
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

def check_controllability(A, B):
    """Check if system is controllable"""
    
    n = A.shape[0]
    
    # Build controllability matrix
    C = B
    A_power = np.eye(n)
    
    for i in range(1, n):
        A_power = A_power @ A
        C = np.hstack([C, A_power @ B])
    
    # Check rank
    rank = np.linalg.matrix_rank(C)
    is_controllable = rank == n
    
    return is_controllable, rank, C

def find_driver_nodes(A):
    """Find minimum number of driver nodes for structural controllability"""
    
    n = A.shape[0]
    
    # Create bipartite graph
    G_bipartite = nx.Graph()
    
    # Add nodes
    for i in range(n):
        G_bipartite.add_node(f'x_{i}', bipartite=0)
        G_bipartite.add_node(f'y_{i}', bipartite=1)
    
    # Add edges
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0:
                G_bipartite.add_edge(f'x_{i}', f'y_{j}')
    
    # Find maximum matching
    matching = nx.bipartite.maximum_matching(G_bipartite)
    
    # Find unmatched nodes
    matched_x = set()
    for x, y in matching.items():
        if x.startswith('x_'):
            matched_x.add(x)
    
    # Driver nodes are unmatched x nodes
    driver_nodes = []
    for i in range(n):
        if f'x_{i}' not in matched_x:
            driver_nodes.append(i)
    
    return driver_nodes, len(driver_nodes)

def design_lqr_controller(A, B, Q, R):
    """Design LQR controller"""
    
    # Solve algebraic Riccati equation
    P = solve_continuous_are(A, B, Q, R)
    
    # Calculate feedback gain
    K = np.linalg.inv(R) @ B.T @ P
    
    # Calculate closed-loop matrix
    A_cl = A - B @ K
    
    # Check stability
    eigenvals = eig(A_cl)[0]
    is_stable = np.all(np.real(eigenvals) < 0)
    
    return K, P, A_cl, is_stable

def simulate_controlled_system(A, B, K, x0, T, dt=0.01):
    """Simulate controlled system"""
    
    n = A.shape[0]
    steps = int(T / dt)
    
    # Initialize
    x = x0.copy()
    x_history = [x.copy()]
    u_history = []
    
    # Simulate
    for t in range(steps):
        # Calculate control input
        u = -K @ x
        u_history.append(u.copy())
        
        # Update state
        x = x + dt * (A @ x + B @ u)
        x_history.append(x.copy())
    
    return np.array(x_history), np.array(u_history)

def calculate_control_energy(u_history, dt):
    """Calculate control energy"""
    
    energy = np.sum(np.sum(u_history**2, axis=1)) * dt
    return energy

def design_robust_controller(A, B, Q, R, uncertainty_set):
    """Design robust controller for uncertain system"""
    
    def objective(K_flat):
        K = K_flat.reshape(B.shape[1], A.shape[0])
        
        # Check stability for all parameter values
        max_eigenval = -np.inf
        for A_uncertain in uncertainty_set:
            A_cl = A_uncertain - B @ K
            eigenvals = eig(A_cl)[0]
            max_eigenval = max(max_eigenval, np.max(np.real(eigenvals)))
        
        # Return negative of maximum real part (we want it to be negative)
        return -max_eigenval
    
    # Initial guess
    K0 = np.random.randn(B.shape[1], A.shape[0])
    
    # Optimize
    result = minimize(objective, K0.flatten(), method='BFGS')
    K_robust = result.x.reshape(B.shape[1], A.shape[0])
    
    return K_robust

def analyze_control_centrality(A, B):
    """Analyze control centrality of nodes"""
    
    n = A.shape[0]
    control_centrality = np.zeros(n)
    
    for i in range(n):
        # Create B matrix with only node i as driver
        B_i = np.zeros((n, 1))
        B_i[i, 0] = 1
        
        # Check controllability
        is_controllable, rank, C = check_controllability(A, B_i)
        
        if is_controllable:
            # Calculate controllability Gramian
            try:
                P = solve_continuous_are(A, B_i, np.eye(n), np.eye(1))
                control_centrality[i] = np.trace(P)
            except:
                control_centrality[i] = 0
        else:
            control_centrality[i] = 0
    
    return control_centrality

def plot_control_analysis(A, B, K, x_history, u_history, control_centrality):
    """Plot control analysis results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # State evolution
    t = np.linspace(0, len(x_history)-1, len(x_history))
    for i in range(min(5, x_history.shape[1])):
        ax1.plot(t, x_history[:, i], label=f'State {i}', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('State Value')
    ax1.set_title('State Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Control input
    t_u = np.linspace(0, len(u_history)-1, len(u_history))
    for i in range(min(5, u_history.shape[1])):
        ax2.plot(t_u, u_history[:, i], label=f'Control {i}', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Control Input')
    ax2.set_title('Control Input Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Control centrality
    nodes = range(len(control_centrality))
    bars = ax3.bar(nodes, control_centrality, alpha=0.7, color='skyblue')
    ax3.set_xlabel('Node Index')
    ax3.set_ylabel('Control Centrality')
    ax3.set_title('Control Centrality of Nodes')
    ax3.grid(True, alpha=0.3)
    
    # Network visualization with control centrality
    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Color nodes by control centrality
    node_colors = control_centrality
    node_sizes = 100 + 50 * control_centrality / np.max(control_centrality)
    
    nx.draw(G, pos, ax=ax4, node_color=node_colors, node_size=node_sizes,
            edge_color='gray', alpha=0.6, cmap='viridis')
    ax4.set_title('Network with Control Centrality')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example: Network control analysis
n = 20
G = nx.barabasi_albert_graph(n, 3)
A = nx.adjacency_matrix(G).toarray()

# Add self-loops for stability
A = A - np.diag(np.sum(A, axis=1))

# Create control input matrix (all nodes can be controlled)
B = np.eye(n)

# Check controllability
is_controllable, rank, C = check_controllability(A, B)
print(f"System is controllable: {is_controllable}")
print(f"Controllability rank: {rank}/{n}")

# Find driver nodes
driver_nodes, n_drivers = find_driver_nodes(A)
print(f"Driver nodes: {driver_nodes}")
print(f"Number of drivers: {n_drivers}")

# Design LQR controller
Q = np.eye(n)
R = 0.1 * np.eye(n)
K, P, A_cl, is_stable = design_lqr_controller(A, B, Q, R)
print(f"Controller is stable: {is_stable}")

# Simulate controlled system
x0 = np.random.randn(n)
T = 10
x_history, u_history = simulate_controlled_system(A, B, K, x0, T)

# Calculate control energy
energy = calculate_control_energy(u_history, 0.01)
print(f"Control energy: {energy:.3f}")

# Analyze control centrality
control_centrality = analyze_control_centrality(A, B)
print(f"Control centrality range: {np.min(control_centrality):.3f} - {np.max(control_centrality):.3f}")

# Plot results
plot_control_analysis(A, B, K, x_history, u_history, control_centrality)
```

## Key Takeaways

1. **Controllability theory**: Mathematical foundation for network control
2. **Driver nodes**: Minimum set of nodes needed for control
3. **Control strategies**: Centralized, decentralized, and adaptive control
4. **Energy optimization**: LQR and other optimal control methods
5. **Robust control**: Control under uncertainty
6. **Applications**: Important for materials science and complex systems
7. **Control centrality**: Measures importance of nodes for control

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Liu, Y. Y., Slotine, J. J., & Barabási, A. L. (2011). Controllability of complex networks. Nature, 473(7346), 167-173.
3. Kalman, R. E. (1960). On the general theory of control systems. IRE Transactions on Automatic Control, 5(4), 110-110.
4. Lewis, F. L., Vrabie, D., & Syrmos, V. L. (2012). Optimal Control. John Wiley & Sons.

---

*Network control provides powerful tools for understanding and manipulating complex systems, with important applications in materials science and engineering.*
