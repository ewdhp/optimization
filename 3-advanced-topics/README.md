# Layer 3: Advanced Topics

Welcome to specialized optimization methods! This layer covers advanced algorithms tailored to specific problem structures and applications.

## 🎯 Learning Objectives

By completing this layer, you will:
- Master simplex and interior point methods for linear programming
- Implement branch & bound for integer programming
- Understand dynamic programming and Bellman optimality
- Learn stochastic optimization (SGD, variance reduction)
- Apply convex programming tools (CVXPY, SDP)
- Solve multi-objective optimization problems

## 📚 Module Structure

```
3-advanced-topics/
├── linear-programming/          # LP: simplex, interior point
├── nonlinear-programming/       # NLP: SQP, global optimization
├── integer-programming/         # IP: branch & bound, cuts
├── dynamic-programming/         # DP: Bellman, value/policy iteration
├── stochastic-optimization/     # Stochastic methods: SGD, SVRG
├── convex-programming/          # Convex: SDP, SOCP, CVX tools
└── multi-objective/             # MOO: Pareto, scalarization
```

## 🗺️ Detailed Guide

### Linear Programming (Weeks 17-18)

**Problem Form:**
```
minimize    c^T x
subject to  Ax = b, x ≥ 0
```

#### Modules (`linear-programming/`)

**Core Concepts:**
- Simplex method: vertex-to-vertex movement
- Dual simplex: working in dual space
- Interior point: barrier methods for LP
- Applications: diet, transportation, production

**Why It Matters:**
- LP is most widely used optimization in practice
- Polynomial-time algorithms (interior point)
- Foundation for integer programming

**Files:**
- 🔄 `simplex_method.py` - Simplex algorithm, pivoting
- 🔄 `dual_simplex.py` - Dual simplex method
- 🔄 `interior_point_lp.py` - Primal-dual interior point
- 🔄 `applications.py` - Real-world LP problems

**Key Algorithms:**
```python
# Simplex: Move along edges of feasible polytope
while not optimal:
    select entering variable  # most negative reduced cost
    select leaving variable   # minimum ratio test
    pivot                     # Gaussian elimination

# Interior Point: Follow central path
while gap > tolerance:
    solve Newton system      # KKT with barrier
    update primal & dual     # predictor-corrector
    decrease barrier parameter
```

**Complexity:**
- Simplex: Exponential worst-case, polynomial average
- Interior point: O(n³L) iterations where L = input size

### Nonlinear Programming (Weeks 19-20)

**Problem Form:**
```
minimize    f(x)
subject to  g(x) ≤ 0, h(x) = 0
```

#### Modules (`nonlinear-programming/`)

**Core Concepts:**
- Sequential QP: solve sequence of QP subproblems
- Interior point for NLP: barrier + Newton
- Global optimization: escape local minima
- Applications: engineering design, control

**Why It Matters:**
- Most real problems are nonlinear
- SQP is state-of-the-art for smooth NLP
- Global methods handle non-convexity

**Files:**
- 🔄 `sqp_methods.py` - Sequential quadratic programming
- 🔄 `interior_point_nlp.py` - IPOPT-style algorithm
- 🔄 `global_optimization.py` - Basin hopping, multistart
- 🔄 `applications.py` - Engineering examples

**SQP Algorithm:**
```python
# Solve sequence of QP approximations
while not converged:
    # QP subproblem
    minimize    ∇f^T p + ½p^T ∇²L p
    subject to  ∇g^T p + g ≤ 0
    
    # Line search on merit function
    α = line_search(merit_function)
    x = x + α*p
    
    # Update multipliers (estimate)
    λ = least_squares_multiplier_estimate()
```

### Integer Programming (Weeks 21-22)

**Problem Form:**
```
minimize    c^T x
subject to  Ax ≤ b, x ∈ ℤⁿ or {0,1}ⁿ
```

#### Modules (`integer-programming/`)

**Core Concepts:**
- Branch & bound: divide & conquer
- Cutting planes: strengthen LP relaxation
- Mixed-integer: some variables integer
- Applications: scheduling, routing, assignment

**Why It Matters:**
- Models discrete decisions
- NP-hard but solvable in practice (CPLEX, Gurobi)
- Combinatorial optimization backbone

**Files:**
- 🔄 `branch_and_bound.py` - B&B algorithm
- 🔄 `cutting_planes.py` - Gomory cuts, valid inequalities
- 🔄 `mixed_integer.py` - MILP formulations
- 🔄 `applications.py` - TSP, scheduling, assignment

**Branch & Bound:**
```python
def branch_and_bound(problem):
    # Initialization
    best_solution = None
    best_value = +∞
    queue = [problem]  # active nodes
    
    while queue:
        node = queue.pop()
        
        # Solve LP relaxation (lower bound)
        relaxation_value, relaxation_solution = solve_lp_relaxation(node)
        
        # Prune if worse than incumbent
        if relaxation_value >= best_value:
            continue
        
        # Check if integer solution
        if is_integer(relaxation_solution):
            best_solution = relaxation_solution
            best_value = relaxation_value
        else:
            # Branch: create subproblems
            var = select_branching_variable(relaxation_solution)
            left = add_constraint(node, var ≤ floor(value))
            right = add_constraint(node, var ≥ ceil(value))
            queue.extend([left, right])
    
    return best_solution, best_value
```

### Dynamic Programming (Week 23)

**Bellman Principle:**
```
V(s) = min_a [cost(s,a) + γ V(next_state(s,a))]
```

#### Modules (`dynamic-programming/`)

**Core Concepts:**
- Bellman optimality principle
- Value iteration: iteratively improve value function
- Policy iteration: improve policy directly
- Applications: shortest path, inventory, control

**Why It Matters:**
- Solves sequential decision problems
- Foundation for reinforcement learning
- Efficient for problems with structure

**Files:**
- 🔄 `bellman_principle.py` - Principle of optimality
- 🔄 `value_iteration.py` - Value iteration algorithm
- 🔄 `policy_iteration.py` - Policy improvement
- 🔄 `applications.py` - Shortest path, optimal control

**Value Iteration:**
```python
def value_iteration(states, actions, transition, reward, γ, ε):
    V = {s: 0 for s in states}  # Initialize value function
    
    while True:
        Δ = 0
        for s in states:
            v_old = V[s]
            
            # Bellman backup
            V[s] = min(
                reward(s,a) + γ * sum(
                    transition(s,a,s_next) * V[s_next]
                    for s_next in states
                )
                for a in actions(s)
            )
            
            Δ = max(Δ, abs(V[s] - v_old))
        
        if Δ < ε:
            break
    
    # Extract optimal policy
    π = {s: argmin_a [Q(s,a)] for s in states}
    
    return V, π
```

### Stochastic Optimization (Week 24)

**SGD Update:**
```
x_{k+1} = x_k - α_k ∇f_i(x_k)  # Single sample gradient
```

#### Modules (`stochastic-optimization/`)

**Core Concepts:**
- Stochastic gradient descent: use sample gradients
- Variance reduction: SVRG, SAGA, SAG
- Online learning: regret minimization
- Applications: large-scale machine learning

**Why It Matters:**
- Only method for large-scale problems (millions of parameters)
- Used to train all modern deep learning models
- Theoretically interesting (stochastic approximation)

**Files:**
- 🔄 `sgd_methods.py` - SGD, mini-batch, momentum
- 🔄 `variance_reduction.py` - SVRG, SAGA
- 🔄 `online_learning.py` - Online convex optimization
- 🔄 `applications.py` - Training neural networks

**Algorithm Comparison:**
```python
# Full Gradient Descent
gradient = compute_full_gradient(x, data)  # Expensive!
x = x - α * gradient

# Stochastic Gradient Descent  
sample = random_sample(data)
gradient = compute_gradient(x, sample)     # Cheap!
x = x - α * gradient

# SVRG (Variance Reduction)
if k % m == 0:
    full_gradient = compute_full_gradient(x_snapshot, data)
sample_gradient = compute_gradient(x, sample)
variance_reduced = sample_gradient - compute_gradient(x_snapshot, sample) + full_gradient
x = x - α * variance_reduced
```

**Convergence:**
- SGD: O(1/√k) for convex (slower than GD!)
- SVRG: O(1/k) like GD, but cheaper per iteration

### Convex Programming (Weeks 25-26)

**Conic Form:**
```
minimize    c^T x
subject to  Ax + b ∈ K
```
where K is a convex cone (e.g., PSD cone, second-order cone)

#### Modules (`convex-programming/`)

**Core Concepts:**
- CVXPY: domain-specific language for convex problems
- Semidefinite programming: X ⪰ 0 constraints
- Second-order cone: ||Ax + b|| ≤ c^T x + d
- Applications: control, signal processing, statistics

**Why It Matters:**
- Many non-convex problems have convex relaxations
- SDP is powerful (includes LP, QP, SOCP)
- Practical tools (CVXPY, CVX, YALMIP)

**Files:**
- 🔄 `cvxpy_introduction.py` - Using CVXPY
- 🔄 `semidefinite_programming.py` - SDP formulations
- 🔄 `conic_optimization.py` - SOCP, exponential cones
- 🔄 `applications.py` - Control, ML, statistics

**CVXPY Example:**
```python
import cvxpy as cp

# Variables
x = cp.Variable(n)

# Objective
objective = cp.Minimize(cp.quad_form(x, Q) + c @ x)

# Constraints
constraints = [
    A @ x == b,       # Equality
    x >= 0,           # Non-negativity
    cp.norm(x) <= 1   # Norm constraint
]

# Solve
problem = cp.Problem(objective, constraints)
problem.solve()

print("Optimal value:", problem.value)
print("Optimal x:", x.value)
```

**SDP Example:**
```python
# Matrix variable
X = cp.Variable((n, n), symmetric=True)

# Objective
objective = cp.Minimize(cp.trace(C @ X))

# Constraints
constraints = [
    X >> 0,           # Positive semidefinite
    cp.trace(X) == 1  # Trace constraint
]

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCS)
```

### Multi-Objective Optimization (Weeks 27-28)

**Pareto Optimality:**
```
x* is Pareto optimal ⟺ 
  ∄x: f_i(x) ≤ f_i(x*) ∀i and f_j(x) < f_j(x*) for some j
```

#### Modules (`multi-objective/`)

**Core Concepts:**
- Pareto frontier: tradeoff surface
- Scalarization: combine objectives (weighted sum, ε-constraint)
- Evolutionary algorithms: NSGA-II, MOEA/D
- Applications: engineering design, portfolio

**Why It Matters:**
- Real problems have multiple conflicting objectives
- No single "best" solution (Pareto set)
- Decision maker chooses from Pareto optimal solutions

**Files:**
- 🔄 `pareto_optimality.py` - Pareto concepts
- 🔄 `scalarization.py` - Weighted sum, ε-constraint
- 🔄 `evolutionary_methods.py` - NSGA-II
- 🔄 `applications.py` - Multi-criteria design

**Scalarization Methods:**
```python
# Weighted Sum
def weighted_sum(weights):
    def objective(x):
        return sum(w * f(x) for w, f in zip(weights, objectives))
    return minimize(objective)

# ε-Constraint
def epsilon_constraint(primary_obj, other_objs, epsilons):
    def objective(x):
        return primary_obj(x)
    
    constraints = [
        other_obj(x) <= eps
        for other_obj, eps in zip(other_objs, epsilons)
    ]
    
    return minimize(objective, constraints)

# Tchebycheff
def tchebycheff(weights, reference):
    def objective(x):
        return max(w * abs(f(x) - r) 
                   for w, f, r in zip(weights, objectives, reference))
    return minimize(objective)
```

## 📖 Suggested Study Order

### Weeks 17-18: Linear Programming
1. Study simplex algorithm theory
2. Implement simplex on small problems
3. Learn interior point methods
4. Solve diet/transportation problems

### Weeks 19-20: Nonlinear Programming
1. Understand SQP framework
2. Implement simplified SQP
3. Study IPOPT/SNOPT algorithms
4. Apply to engineering design

### Weeks 21-22: Integer Programming
1. Learn branch & bound
2. Implement B&B on knapsack
3. Study cutting planes
4. Formulate scheduling problems

### Week 23: Dynamic Programming
1. Understand Bellman principle
2. Implement value iteration
3. Solve shortest path
4. Study policy iteration

### Week 24: Stochastic Optimization
1. Implement SGD vs full GD
2. Code SVRG algorithm
3. Compare convergence
4. Train neural network

### Weeks 25-26: Convex Programming
1. Learn CVXPY syntax
2. Solve LP/QP/SOCP with CVXPY
3. Formulate SDP problems
4. Apply to control/signal processing

### Weeks 27-28: Multi-Objective
1. Compute Pareto frontier
2. Implement scalarization
3. Study NSGA-II
4. Multi-criteria design project

## 🎓 Capstone Project Ideas

### Project 1: Optimization Solver Library
Build comprehensive solver with:
- LP: Simplex + interior point
- QP: Active set method
- NLP: SQP implementation
- Compare on test problems

### Project 2: Scheduling System
Develop job shop scheduler:
- Formulate as MILP
- Implement branch & bound
- Add cutting planes
- Visualize Gantt charts

### Project 3: ML Training Framework
Build training system with:
- SGD, momentum, Adam
- SVRG for variance reduction
- Learning rate schedules
- Compare on MNIST/CIFAR

### Project 4: Portfolio Optimizer
Multi-objective portfolio:
- Return vs risk
- Compute Pareto frontier
- Interactive selection tool
- Robust optimization

## 📚 Key References

### Linear Programming
- Dantzig: "Linear Programming and Extensions"
- Nocedal & Wright: Chapter 14

### Integer Programming
- Wolsey: "Integer Programming"
- Schrijver: "Theory of Linear and Integer Programming"

### Dynamic Programming
- Bertsekas: "Dynamic Programming and Optimal Control"
- Sutton & Barto: "Reinforcement Learning"

### Stochastic Optimization
- Bottou et al.: "Optimization Methods for Large-Scale ML" (2018)
- Robbins & Monro: "Stochastic Approximation" (1951)

### Convex Programming
- Boyd & Vandenberghe: Chapters 4-6, 11
- Ben-Tal & Nemirovski: "Lectures on Modern Convex Optimization"

### Multi-Objective
- Deb: "Multi-Objective Optimization using Evolutionary Algorithms"
- Miettinen: "Nonlinear Multiobjective Optimization"

## 🛠️ Software Tools

### Commercial
- **CPLEX**: LP/MILP solver (free academic)
- **Gurobi**: LP/MILP solver (free academic)
- **SNOPT**: NLP solver
- **MATLAB**: Optimization Toolbox

### Open Source
- **CVXPY**: Python convex optimization
- **SciPy**: Basic optimizers
- **OR-Tools**: Google's optimization suite
- **SCIP**: Mixed-integer solver
- **Pyomo**: Modeling language

## 💡 Practical Tips

1. **Use Existing Solvers**: Don't reinvent wheel for production
2. **Formulation Matters**: Good formulation = fast solution
3. **Start Simple**: Test on small problems first
4. **Leverage Structure**: Exploit problem structure (sparsity, separability)
5. **Warm Starting**: Use previous solution as initial guess

## 📊 Implementation Status

- **Linear Programming**: ░░░░░░░░░░ 0% (0/4 files)
- **Nonlinear Programming**: ░░░░░░░░░░ 0% (0/4 files)
- **Integer Programming**: ░░░░░░░░░░ 0% (0/4 files)
- **Dynamic Programming**: ░░░░░░░░░░ 0% (0/4 files)
- **Stochastic Optimization**: ░░░░░░░░░░ 0% (0/4 files)
- **Convex Programming**: ░░░░░░░░░░ 0% (0/4 files)
- **Multi-Objective**: ░░░░░░░░░░ 0% (0/4 files)
- **Overall**: ░░░░░░░░░░ 0% (0/28 files)

## ⏭️ Next Steps

After completing Layer 3:
- **Applications**: Apply to real-world domains (ML, OR, engineering)
- **Research**: Explore cutting-edge topics
- **Production**: Build robust optimization systems
- **Teaching**: Share knowledge with others

---

**Status**: 0/28 modules complete (0%)
**Time Estimate**: 12 weeks of focused study
**Prerequisites**: Layers 1 & 2 complete
