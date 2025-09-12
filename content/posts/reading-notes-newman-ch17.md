---
title: "Reading Notes: Newman's Networks Chapter 17 - Network Optimization"
date: 2025-08-29
draft: false
description: "Study notes for Chapter 17 of Newman's 'Networks: An Introduction' covering network optimization, design principles, and optimization algorithms"
tags: ["reading-notes", "network-theory", "network-optimization", "design-principles", "optimization-algorithms"]
showToc: true
TocOpen: true
---

## Introduction

Chapter 17 of Newman's *Networks: An Introduction* explores **network optimization** - the process of designing and modifying networks to achieve desired properties or performance. This chapter covers optimization problems, algorithms, and applications to real-world network design.

## 17.1 Network Design Problems

### Basic Optimization Problems

#### Minimum Spanning Tree

**Problem**: Find minimum weight spanning tree

**Mathematical formulation**:
$$\min_{T} \sum_{(i,j) \in T} w_{ij}$$

**Constraints**:
- $T$ is a spanning tree
- $w_{ij}$ is the weight of edge $(i,j)$

**Algorithms**:
- **Kruskal's algorithm**: $O(m \log m)$
- **Prim's algorithm**: $O(m + n \log n)$

#### Shortest Path Problem

**Problem**: Find shortest path between two nodes

**Mathematical formulation**:
$$\min_{P} \sum_{(i,j) \in P} w_{ij}$$

**Constraints**:
- $P$ is a path from source to destination
- $w_{ij}$ is the weight of edge $(i,j)$

**Algorithms**:
- **Dijkstra's algorithm**: $O(m + n \log n)$
- **Bellman-Ford algorithm**: $O(mn)$
- **Floyd-Warshall algorithm**: $O(n^3)$

#### Maximum Flow Problem

**Problem**: Find maximum flow from source to sink

**Mathematical formulation**:
$$\max_{f} \sum_{j} f_{sj}$$

**Constraints**:
- **Flow conservation**: $\sum_{j} f_{ij} - \sum_{j} f_{ji} = 0$ for $i \neq s,t$
- **Capacity constraints**: $0 \leq f_{ij} \leq c_{ij}$
- **Non-negativity**: $f_{ij} \geq 0$

**Algorithms**:
- **Ford-Fulkerson algorithm**: $O(m \cdot f^*)$
- **Edmonds-Karp algorithm**: $O(m^2 n)$
- **Push-relabel algorithm**: $O(n^2 m)$

### Network Design Problems

#### Steiner Tree Problem

**Problem**: Find minimum weight tree connecting all terminals

**Mathematical formulation**:
$$\min_{T} \sum_{(i,j) \in T} w_{ij}$$

**Constraints**:
- $T$ contains all terminal nodes
- $T$ is a tree

**Complexity**: NP-hard

**Approximation algorithms**:
- **2-approximation**: $O(n^2)$
- **1.55-approximation**: $O(n^2 \log n)$

#### Traveling Salesman Problem

**Problem**: Find minimum weight Hamiltonian cycle

**Mathematical formulation**:
$$\min_{C} \sum_{(i,j) \in C} w_{ij}$$

**Constraints**:
- $C$ is a Hamiltonian cycle
- $w_{ij}$ is the weight of edge $(i,j)$

**Complexity**: NP-hard

**Approximation algorithms**:
- **2-approximation**: $O(n^2)$
- **1.5-approximation**: $O(n^3)$

#### Facility Location Problem

**Problem**: Find optimal locations for facilities

**Mathematical formulation**:
$$\min_{x,y} \sum_{i} c_i x_i + \sum_{i,j} d_{ij} y_{ij}$$

**Constraints**:
- **Demand satisfaction**: $\sum_{i} y_{ij} = d_j$ for all $j$
- **Capacity constraints**: $\sum_{j} y_{ij} \leq c_i x_i$ for all $i$
- **Binary variables**: $x_i \in \{0,1\}$

**Complexity**: NP-hard

**Approximation algorithms**:
- **3-approximation**: $O(n^2)$
- **1.61-approximation**: $O(n^2 \log n)$

## 17.2 Optimization Algorithms

### Exact Algorithms

#### Branch and Bound

**Algorithm**:
1. **Branch**: Split problem into subproblems
2. **Bound**: Calculate lower bound for each subproblem
3. **Prune**: Eliminate subproblems with bounds worse than current best
4. **Repeat**: Until all subproblems are solved or pruned

**Complexity**: Exponential in worst case

**Advantages**: Guarantees optimal solution
**Disadvantages**: May be slow for large problems

#### Dynamic Programming

**Algorithm**:
1. **Subproblems**: Break problem into smaller subproblems
2. **Memoization**: Store solutions to subproblems
3. **Recursion**: Solve subproblems recursively
4. **Reconstruction**: Build optimal solution from subproblem solutions

**Complexity**: $O(n^2)$ to $O(n^3)$ for many problems

**Advantages**: Efficient for problems with optimal substructure
**Disadvantages**: May require exponential space

### Approximation Algorithms

#### Greedy Algorithms

**Algorithm**:
1. **Initialize**: Start with empty solution
2. **Iterate**: At each step, make locally optimal choice
3. **Terminate**: When solution is complete

**Examples**:
- **Kruskal's algorithm**: For minimum spanning tree
- **Prim's algorithm**: For minimum spanning tree
- **Dijkstra's algorithm**: For shortest path

**Advantages**: Simple, fast
**Disadvantages**: May not find optimal solution

#### Local Search

**Algorithm**:
1. **Initialize**: Start with feasible solution
2. **Neighborhood**: Define neighborhood of current solution
3. **Improve**: Move to better solution in neighborhood
4. **Terminate**: When no improvement possible

**Examples**:
- **2-opt**: For traveling salesman problem
- **3-opt**: For traveling salesman problem
- **Simulated annealing**: For general optimization

**Advantages**: Can escape local optima
**Disadvantages**: May not find global optimum

### Metaheuristic Algorithms

#### Genetic Algorithm

**Algorithm**:
1. **Initialize**: Create population of solutions
2. **Evaluate**: Calculate fitness of each solution
3. **Select**: Select parents for reproduction
4. **Crossover**: Create offspring from parents
5. **Mutate**: Apply random changes to offspring
6. **Replace**: Replace population with offspring
7. **Repeat**: Until convergence

**Parameters**:
- **Population size**: Number of solutions
- **Crossover rate**: Probability of crossover
- **Mutation rate**: Probability of mutation
- **Selection pressure**: Strength of selection

**Advantages**: Can handle complex problems
**Disadvantages**: May be slow to converge

#### Simulated Annealing

**Algorithm**:
1. **Initialize**: Start with random solution
2. **Neighborhood**: Define neighborhood of current solution
3. **Accept**: Accept better solutions, probabilistically accept worse solutions
4. **Cool**: Reduce temperature over time
5. **Terminate**: When temperature is low enough

**Parameters**:
- **Initial temperature**: $T_0$
- **Cooling rate**: $\alpha$
- **Acceptance probability**: $P = e^{-\Delta E/T}$

**Advantages**: Can escape local optima
**Disadvantages**: May be slow to converge

#### Particle Swarm Optimization

**Algorithm**:
1. **Initialize**: Create swarm of particles
2. **Evaluate**: Calculate fitness of each particle
3. **Update**: Update velocity and position of each particle
4. **Repeat**: Until convergence

**Parameters**:
- **Swarm size**: Number of particles
- **Inertia weight**: $w$
- **Acceleration coefficients**: $c_1, c_2$

**Advantages**: Simple, effective
**Disadvantages**: May get stuck in local optima

## 17.3 Network Design Principles

### Connectivity

#### Minimum Connectivity

**Principle**: Ensure network remains connected

**Mathematical formulation**:
$$\min_{G} \sum_{(i,j) \in E} w_{ij}$$

**Constraints**:
- $G$ is connected
- $w_{ij}$ is the weight of edge $(i,j)$

**Algorithms**:
- **Minimum spanning tree**: $O(m \log m)$
- **Steiner tree**: NP-hard

#### Redundancy

**Principle**: Provide multiple paths between nodes

**Mathematical formulation**:
$$\max_{G} \sum_{i \neq j} \text{paths}(i,j)$$

**Constraints**:
- $G$ is connected
- $\text{paths}(i,j)$ is the number of paths between $i$ and $j$

**Algorithms**:
- **Maximum flow**: $O(m^2 n)$
- **Minimum cut**: $O(m^2 n)$

### Efficiency

#### Shortest Paths

**Principle**: Minimize average path length

**Mathematical formulation**:
$$\min_{G} \frac{1}{n(n-1)} \sum_{i \neq j} d_{ij}$$

**Constraints**:
- $G$ is connected
- $d_{ij}$ is the shortest path length between $i$ and $j$

**Algorithms**:
- **All-pairs shortest paths**: $O(n^3)$
- **Single-source shortest paths**: $O(m + n \log n)$

#### Load Balancing

**Principle**: Distribute load evenly across network

**Mathematical formulation**:
$$\min_{G} \max_{i} \sum_{j} f_{ij}$$

**Constraints**:
- **Flow conservation**: $\sum_{j} f_{ij} - \sum_{j} f_{ji} = 0$ for all $i$
- **Capacity constraints**: $0 \leq f_{ij} \leq c_{ij}$

**Algorithms**:
- **Maximum flow**: $O(m^2 n)$
- **Minimum cost flow**: $O(m^2 n \log n)$

### Robustness

#### Fault Tolerance

**Principle**: Network remains functional under failures

**Mathematical formulation**:
$$\max_{G} \min_{F} \text{connectivity}(G \setminus F)$$

**Constraints**:
- $F$ is a set of failed edges/nodes
- $\text{connectivity}(G \setminus F)$ is the connectivity of remaining network

**Algorithms**:
- **Minimum cut**: $O(m^2 n)$
- **Maximum flow**: $O(m^2 n)$

#### Resilience

**Principle**: Network recovers quickly from failures

**Mathematical formulation**:
$$\min_{G} \max_{F} \text{recovery\_time}(G, F)$$

**Constraints**:
- $F$ is a set of failed edges/nodes
- $\text{recovery\_time}(G, F)$ is the time to recover from failure

**Algorithms**:
- **Dynamic programming**: $O(n^2)$
- **Simulation**: $O(n^3)$

## 17.4 Applications to Materials Science

### Nanowire Network Design

#### Problem

**Given**: Set of nanowire junctions and constraints
**Goal**: Design optimal nanowire network

**Mathematical formulation**:
$$\min_{G} \sum_{(i,j) \in E} w_{ij}$$

**Constraints**:
- **Electrical**: Ohm's law must be satisfied
- **Topological**: Network must be connected
- **Physical**: Edge weights must be positive

#### Solution

**Algorithm**:
1. **Initialize**: Start with empty network
2. **Add edges**: Add edges that improve objective
3. **Check constraints**: Ensure all constraints are satisfied
4. **Optimize**: Use local search to improve solution

**Complexity**: NP-hard

**Approximation**: 2-approximation algorithm

### Defect Network Optimization

#### Problem

**Given**: Defect network with dynamics
**Goal**: Optimize defect distribution

**Mathematical formulation**:
$$\min_{G} \sum_{i} \sigma_i k_i$$

**Constraints**:
- **Defect concentration**: $c \leq c_{\max}$
- **Clustering**: $C \geq C_{\min}$

#### Solution

**Algorithm**:
1. **Initialize**: Start with random defect distribution
2. **Evaluate**: Calculate objective function
3. **Optimize**: Use genetic algorithm to improve solution
4. **Validate**: Check constraints

**Complexity**: NP-hard

**Approximation**: Genetic algorithm

### Phase Transition Optimization

#### Problem

**Given**: Phase transition dynamics
**Goal**: Optimize phase transition properties

**Mathematical formulation**:
$$\min_{G} \sum_{i} |T_i - T_{\text{target}}|$$

**Constraints**:
- **Temperature**: $T_{\min} \leq T_i \leq T_{\max}$
- **Pressure**: $P_{\min} \leq P_i \leq P_{\max}$

#### Solution

**Algorithm**:
1. **Initialize**: Start with random phase distribution
2. **Simulate**: Simulate phase transition dynamics
3. **Optimize**: Use simulated annealing to improve solution
4. **Validate**: Check constraints

**Complexity**: NP-hard

**Approximation**: Simulated annealing

## 17.5 Multi-Objective Optimization

### Pareto Optimality

#### Definition

**Pareto optimal solution**: Solution that cannot be improved in one objective without worsening another

**Mathematical formulation**:
$$\min_{x} f(x) = (f_1(x), f_2(x), \ldots, f_k(x))$$

**Constraints**:
- $g_i(x) \leq 0$ for all $i$
- $h_j(x) = 0$ for all $j$

#### Pareto Front

**Pareto front**: Set of all Pareto optimal solutions

**Properties**:
- **Non-dominated**: No solution dominates another
- **Complete**: Contains all Pareto optimal solutions
- **Minimal**: No redundant solutions

### Multi-Objective Algorithms

#### Weighted Sum Method

**Algorithm**:
1. **Weights**: Assign weights to objectives
2. **Combine**: Combine objectives into single objective
3. **Optimize**: Solve single-objective problem
4. **Repeat**: For different weight combinations

**Mathematical formulation**:
$$\min_{x} \sum_{i=1}^k w_i f_i(x)$$

**Advantages**: Simple, efficient
**Disadvantages**: May miss some Pareto optimal solutions

#### ε-Constraint Method

**Algorithm**:
1. **Select**: Select one objective as primary
2. **Constrain**: Convert other objectives to constraints
3. **Optimize**: Solve constrained optimization problem
4. **Repeat**: For different constraint values

**Mathematical formulation**:
$$\min_{x} f_1(x)$$

**Constraints**:
- $f_i(x) \leq \epsilon_i$ for $i = 2, \ldots, k$
- $g_j(x) \leq 0$ for all $j$
- $h_l(x) = 0$ for all $l$

**Advantages**: Can find all Pareto optimal solutions
**Disadvantages**: May be computationally expensive

#### NSGA-II

**Algorithm**:
1. **Initialize**: Create population of solutions
2. **Evaluate**: Calculate fitness of each solution
3. **Select**: Select parents using non-dominated sorting
4. **Crossover**: Create offspring from parents
5. **Mutate**: Apply random changes to offspring
6. **Replace**: Replace population with offspring
7. **Repeat**: Until convergence

**Advantages**: Can handle complex problems
**Disadvantages**: May be slow to converge

## Code Example: Network Optimization

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import random

class NetworkOptimizer:
    """Network optimization class"""
    
    def __init__(self, G):
        self.G = G.copy()
        self.n = G.number_of_nodes()
        self.m = G.number_of_edges()
    
    def optimize_connectivity(self, budget):
        """Optimize network connectivity with budget constraint"""
        
        def objective(edges):
            # Create network with selected edges
            G_new = nx.Graph()
            G_new.add_nodes_from(self.G.nodes())
            
            for i, (u, v) in enumerate(self.G.edges()):
                if edges[i] > 0.5:  # Binary decision
                    G_new.add_edge(u, v)
            
            # Calculate connectivity (inverse of average path length)
            if nx.is_connected(G_new):
                avg_path_length = nx.average_shortest_path_length(G_new)
                return -1 / avg_path_length  # Maximize connectivity
            else:
                return -1000  # Penalty for disconnected network
        
        def constraint(edges):
            # Budget constraint
            cost = np.sum(edges)
            return budget - cost
        
        # Initial guess
        x0 = np.random.random(self.m)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', 
                         constraints={'type': 'ineq', 'fun': constraint},
                         bounds=[(0, 1)] * self.m)
        
        # Create optimized network
        G_optimized = nx.Graph()
        G_optimized.add_nodes_from(self.G.nodes())
        
        for i, (u, v) in enumerate(self.G.edges()):
            if result.x[i] > 0.5:
                G_optimized.add_edge(u, v)
        
        return G_optimized, result.fun
    
    def optimize_efficiency(self, budget):
        """Optimize network efficiency with budget constraint"""
        
        def objective(edges):
            # Create network with selected edges
            G_new = nx.Graph()
            G_new.add_nodes_from(self.G.nodes())
            
            for i, (u, v) in enumerate(self.G.edges()):
                if edges[i] > 0.5:
                    G_new.add_edge(u, v)
            
            # Calculate efficiency
            if nx.is_connected(G_new):
                efficiency = nx.global_efficiency(G_new)
                return -efficiency  # Maximize efficiency
            else:
                return -1000  # Penalty for disconnected network
        
        def constraint(edges):
            # Budget constraint
            cost = np.sum(edges)
            return budget - cost
        
        # Initial guess
        x0 = np.random.random(self.m)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP',
                         constraints={'type': 'ineq', 'fun': constraint},
                         bounds=[(0, 1)] * self.m)
        
        # Create optimized network
        G_optimized = nx.Graph()
        G_optimized.add_nodes_from(self.G.nodes())
        
        for i, (u, v) in enumerate(self.G.edges()):
            if result.x[i] > 0.5:
                G_optimized.add_edge(u, v)
        
        return G_optimized, result.fun
    
    def optimize_robustness(self, budget):
        """Optimize network robustness with budget constraint"""
        
        def objective(edges):
            # Create network with selected edges
            G_new = nx.Graph()
            G_new.add_nodes_from(self.G.nodes())
            
            for i, (u, v) in enumerate(self.G.edges()):
                if edges[i] > 0.5:
                    G_new.add_edge(u, v)
            
            # Calculate robustness (algebraic connectivity)
            if nx.is_connected(G_new):
                L = nx.laplacian_matrix(G_new).toarray()
                eigenvals = np.linalg.eigvals(L)
                eigenvals = np.real(eigenvals)
                eigenvals = np.sort(eigenvals)
                algebraic_connectivity = eigenvals[1] if len(eigenvals) > 1 else 0
                return -algebraic_connectivity  # Maximize robustness
            else:
                return -1000  # Penalty for disconnected network
        
        def constraint(edges):
            # Budget constraint
            cost = np.sum(edges)
            return budget - cost
        
        # Initial guess
        x0 = np.random.random(self.m)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP',
                         constraints={'type': 'ineq', 'fun': constraint},
                         bounds=[(0, 1)] * self.m)
        
        # Create optimized network
        G_optimized = nx.Graph()
        G_optimized.add_nodes_from(self.G.nodes())
        
        for i, (u, v) in enumerate(self.G.edges()):
            if result.x[i] > 0.5:
                G_optimized.add_edge(u, v)
        
        return G_optimized, result.fun
    
    def multi_objective_optimization(self, budget, weights):
        """Multi-objective optimization"""
        
        def objective(edges):
            # Create network with selected edges
            G_new = nx.Graph()
            G_new.add_nodes_from(self.G.nodes())
            
            for i, (u, v) in enumerate(self.G.edges()):
                if edges[i] > 0.5:
                    G_new.add_edge(u, v)
            
            if not nx.is_connected(G_new):
                return 1000  # Penalty for disconnected network
            
            # Calculate multiple objectives
            avg_path_length = nx.average_shortest_path_length(G_new)
            efficiency = nx.global_efficiency(G_new)
            
            L = nx.laplacian_matrix(G_new).toarray()
            eigenvals = np.linalg.eigvals(L)
            eigenvals = np.real(eigenvals)
            eigenvals = np.sort(eigenvals)
            algebraic_connectivity = eigenvals[1] if len(eigenvals) > 1 else 0
            
            # Weighted sum
            total_objective = (weights[0] * avg_path_length + 
                             weights[1] * (1 - efficiency) + 
                             weights[2] * (1 - algebraic_connectivity))
            
            return total_objective
        
        def constraint(edges):
            # Budget constraint
            cost = np.sum(edges)
            return budget - cost
        
        # Initial guess
        x0 = np.random.random(self.m)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP',
                         constraints={'type': 'ineq', 'fun': constraint},
                         bounds=[(0, 1)] * self.m)
        
        # Create optimized network
        G_optimized = nx.Graph()
        G_optimized.add_nodes_from(self.G.nodes())
        
        for i, (u, v) in enumerate(self.G.edges()):
            if result.x[i] > 0.5:
                G_optimized.add_edge(u, v)
        
        return G_optimized, result.fun

def genetic_algorithm_optimization(G, budget, generations=100, population_size=50):
    """Genetic algorithm for network optimization"""
    
    def fitness(individual):
        # Create network with selected edges
        G_new = nx.Graph()
        G_new.add_nodes_from(G.nodes())
        
        for i, (u, v) in enumerate(G.edges()):
            if individual[i] > 0.5:
                G_new.add_edge(u, v)
        
        if not nx.is_connected(G_new):
            return 0  # Penalty for disconnected network
        
        # Calculate fitness (efficiency)
        efficiency = nx.global_efficiency(G_new)
        return efficiency
    
    def crossover(parent1, parent2):
        # Single-point crossover
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def mutate(individual, mutation_rate=0.1):
        # Random mutation
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = 1 - individual[i]  # Flip bit
        return individual
    
    # Initialize population
    population = []
    for _ in range(population_size):
        individual = np.random.random(G.number_of_edges())
        # Ensure budget constraint
        while np.sum(individual) > budget:
            individual[np.argmax(individual)] = 0
        population.append(individual)
    
    # Evolution
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [fitness(individual) for individual in population]
        
        # Selection (tournament selection)
        new_population = []
        for _ in range(population_size):
            # Tournament selection
            tournament_size = 5
            tournament = random.sample(range(population_size), tournament_size)
            winner = max(tournament, key=lambda i: fitness_scores[i])
            new_population.append(population[winner].copy())
        
        # Crossover
        for i in range(0, population_size, 2):
            if i + 1 < population_size:
                child1, child2 = crossover(new_population[i], new_population[i + 1])
                new_population[i] = child1
                new_population[i + 1] = child2
        
        # Mutation
        for individual in new_population:
            mutate(individual)
            # Ensure budget constraint
            while np.sum(individual) > budget:
                individual[np.argmax(individual)] = 0
        
        population = new_population
    
    # Find best individual
    fitness_scores = [fitness(individual) for individual in population]
    best_individual = population[np.argmax(fitness_scores)]
    
    # Create optimized network
    G_optimized = nx.Graph()
    G_optimized.add_nodes_from(G.nodes())
    
    for i, (u, v) in enumerate(G.edges()):
        if best_individual[i] > 0.5:
            G_optimized.add_edge(u, v)
    
    return G_optimized, max(fitness_scores)

def plot_optimization_results(G_original, G_optimized, title="Network Optimization"):
    """Plot network optimization results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original network
    pos = nx.spring_layout(G_original, k=1, iterations=50)
    nx.draw(G_original, pos, ax=ax1, node_color='lightblue', 
            edge_color='gray', alpha=0.6, node_size=50)
    ax1.set_title('Original Network')
    ax1.axis('off')
    
    # Optimized network
    nx.draw(G_optimized, pos, ax=ax2, node_color='lightgreen', 
            edge_color='gray', alpha=0.6, node_size=50)
    ax2.set_title('Optimized Network')
    ax2.axis('off')
    
    # Network properties comparison
    properties = {
        'Nodes': [G_original.number_of_nodes(), G_optimized.number_of_nodes()],
        'Edges': [G_original.number_of_edges(), G_optimized.number_of_edges()],
        'Density': [nx.density(G_original), nx.density(G_optimized)],
        'Clustering': [nx.average_clustering(G_original), nx.average_clustering(G_optimized)],
        'Efficiency': [nx.global_efficiency(G_original), nx.global_efficiency(G_optimized)]
    }
    
    x = np.arange(len(properties))
    width = 0.35
    
    for i, (prop, values) in enumerate(properties.items()):
        ax3.bar(x[i] - width/2, values[0], width, label='Original', alpha=0.7)
        ax3.bar(x[i] + width/2, values[1], width, label='Optimized', alpha=0.7)
    
    ax3.set_xlabel('Properties')
    ax3.set_ylabel('Values')
    ax3.set_title('Network Properties Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(properties.keys())
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Degree distribution comparison
    degrees_original = [G_original.degree(i) for i in G_original.nodes()]
    degrees_optimized = [G_optimized.degree(i) for i in G_optimized.nodes()]
    
    ax4.hist(degrees_original, bins=20, alpha=0.7, label='Original', edgecolor='black')
    ax4.hist(degrees_optimized, bins=20, alpha=0.7, label='Optimized', edgecolor='black')
    ax4.set_xlabel('Degree')
    ax4.set_ylabel('Count')
    ax4.set_title('Degree Distribution Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Example: Network optimization
G = nx.barabasi_albert_graph(50, 3)

# Initialize optimizer
optimizer = NetworkOptimizer(G)

# Single-objective optimization
budget = 0.5  # 50% of edges
G_connectivity, connectivity_score = optimizer.optimize_connectivity(budget)
G_efficiency, efficiency_score = optimizer.optimize_efficiency(budget)
G_robustness, robustness_score = optimizer.optimize_robustness(budget)

# Multi-objective optimization
weights = [0.4, 0.3, 0.3]  # Weights for path length, efficiency, robustness
G_multi, multi_score = optimizer.multi_objective_optimization(budget, weights)

# Genetic algorithm optimization
G_genetic, genetic_score = genetic_algorithm_optimization(G, budget, generations=50)

print("Network Optimization Results:")
print(f"Original network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Connectivity optimization: {connectivity_score:.3f}")
print(f"Efficiency optimization: {efficiency_score:.3f}")
print(f"Robustness optimization: {robustness_score:.3f}")
print(f"Multi-objective optimization: {multi_score:.3f}")
print(f"Genetic algorithm optimization: {genetic_score:.3f}")

# Plot results
plot_optimization_results(G, G_efficiency, "Efficiency Optimization")
```

## Key Takeaways

1. **Optimization problems**: Various types of network optimization problems
2. **Algorithms**: Exact, approximation, and metaheuristic algorithms
3. **Design principles**: Connectivity, efficiency, and robustness
4. **Multi-objective optimization**: Pareto optimality and algorithms
5. **Applications**: Important for materials science and engineering
6. **Mathematical formulation**: Rigorous mathematical framework
7. **Practical implementation**: Code examples and real-world applications

## References

1. Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
2. Ahuja, R. K., Magnanti, T. L., & Orlin, J. B. (1993). Network Flows: Theory, Algorithms, and Applications. Prentice Hall.
3. Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning. Addison-Wesley.
4. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. Science, 220(4598), 671-680.

---

*Network optimization provides powerful tools for designing and improving complex networks, with important applications in materials science and engineering.*
