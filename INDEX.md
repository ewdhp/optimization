# Optimization Framework - Complete File Index

## 📍 Quick Navigation Guide

This index provides a complete mapping of all modules, organized by the three-layer learning architecture.

---

## 🏗️ LAYER 1: FOUNDATIONS (Weeks 1-4)

Mathematical prerequisites for optimization theory.

### 1.1 Calculus
**Path:** `1-foundations/calculus/`

| File | Theorems/Concepts | Status | Dependencies |
|------|------------------|--------|--------------|
| `taylor_theorem.py` | Taylor expansion, approximation error bounds | ✅ Complete | Calculus |
| `mean_value_theorem.py` | MVT, Rolle's theorem, Cauchy MVT | 🔄 Planned | Calculus |
| `implicit_function.py` | Implicit function theorem, constraint manifolds | 🔄 Planned | Advanced Calculus |
| `multivariable_calculus.py` | Gradients, Hessians, Jacobians, chain rule | 🔄 Planned | Multivariable Calculus |

**Key Learning:** Taylor's theorem is foundational for Newton's method and convergence analysis.

### 1.2 Linear Algebra
**Path:** `1-foundations/linear-algebra/`

| File | Theorems/Concepts | Status | Dependencies |
|------|------------------|--------|--------------|
| `eigenvalues.py` | Eigendecomposition, spectral theorem | 🔄 Planned | Linear Algebra |
| `matrix_decomposition.py` | QR, SVD, Cholesky decomposition | 🔄 Planned | Linear Algebra |
| `positive_definiteness.py` | PD/PSD testing, Sylvester's criterion | 🔄 Planned | Matrix Theory |
| `vector_spaces.py` | Norms, inner products, projections | 🔄 Planned | Linear Algebra |

**Key Learning:** Matrix definiteness determines critical point type (min/max/saddle).

### 1.3 Real Analysis
**Path:** `1-foundations/real-analysis/`

| File | Theorems/Concepts | Status | Dependencies |
|------|------------------|--------|--------------|
| `weierstrass_theorem.py` | Extreme value theorem, existence of optima | ✅ Complete | Real Analysis |
| `continuity_lipschitz.py` | Continuity, Lipschitz constants | 🔄 Planned | Real Analysis |
| `sequences_limits.py` | Convergence, Cauchy sequences | 🔄 Planned | Real Analysis |
| `bolzano_weierstrass.py` | Compactness, subsequence convergence | 🔄 Planned | Real Analysis |

**Key Learning:** Weierstrass guarantees continuous functions on compact sets have minima.

### 1.4 Convexity
**Path:** `1-foundations/convexity/`

| File | Theorems/Concepts | Status | Dependencies |
|------|------------------|--------|--------------|
| `convex_sets.py` | Convex sets, operations, properties | 🔄 Planned | Linear Algebra |
| `convex_functions.py` | Convex functions, properties, characterizations | 🔄 Planned | Calculus |
| `jensen_inequality.py` | Jensen's inequality, applications | ✅ Complete | Convexity |
| `separation_theorems.py` | Separating hyperplanes, supporting hyperplanes | 🔄 Planned | Linear Algebra |

**Key Learning:** Jensen's inequality is fundamental for proving convexity properties.

---

## 🎯 LAYER 2: CORE METHODS (Weeks 5-16)

Fundamental optimization algorithms and theory.

### 2.1 Unconstrained Optimization

#### 2.1.1 Optimality Conditions
**Path:** `2-core-methods/unconstrained/optimality-conditions/`

| File | Theorems/Concepts | Status | Dependencies |
|------|------------------|--------|--------------|
| `first_order.py` | FONC (∇f=0), critical points, descent directions | ✅ Complete | Calculus |
| `second_order.py` | SONC, SOSC, Hessian analysis | ✅ Complete | Linear Algebra, Calculus |
| `examples.py` | Worked examples, KKT for constraints | ✅ Complete | Above |

**Key Learning:**
- FONC: ∇f(x*) = 0 (necessary for minimum)
- SOSC: ∇f(x*)=0 AND ∇²f(x*)≻0 (sufficient for strict minimum)

#### 2.1.2 Gradient Methods
**Path:** `2-core-methods/unconstrained/gradient-methods/`

| File | Algorithms | Status | Dependencies |
|------|-----------|--------|--------------|
| `steepest_descent.py` | GD with constant/backtracking/adaptive step | ✅ Complete | FONC |
| `line_search.py` | Armijo, Wolfe, strong Wolfe conditions | ✅ Complete | FONC |
| `convergence_analysis.py` | Lipschitz, strong convexity, rates | ✅ Complete | Real Analysis |
| `conjugate_gradient.py` | CG, Polak-Ribière, Fletcher-Reeves | 🔄 Planned | GD |

**Key Learning:**
- Gradient descent: x_{k+1} = x_k - α∇f(x_k)
- Convergence: O(1/k) for convex, linear for strongly convex

#### 2.1.3 Newton Methods
**Path:** `2-core-methods/unconstrained/newton-methods/`

| File | Algorithms | Status | Dependencies |
|------|-----------|--------|--------------|
| `newton_method.py` | Pure/damped Newton, Hessian computation | ✅ Complete | SOSC, Taylor |
| `quasi_newton.py` | BFGS, L-BFGS, DFP, SR1 | ✅ Complete | Newton |
| `trust_region.py` | TR with dogleg, Cauchy point | ✅ Complete | Newton |
| `convergence_theory.py` | Quadratic/superlinear convergence proofs | ✅ Complete | Real Analysis |

**Key Learning:**
- Newton: p_k = -[∇²f(x_k)]^{-1}∇f(x_k), quadratic convergence
- BFGS: approximates Hessian, superlinear convergence

#### 2.1.4 Momentum Methods
**Path:** `2-core-methods/unconstrained/momentum-methods/`

| File | Algorithms | Status | Dependencies |
|------|-----------|--------|--------------|
| `heavy_ball.py` | Classical momentum method | 🔄 Planned | GD |
| `nesterov_momentum.py` | Accelerated gradient descent | 🔄 Planned | GD |
| `adaptive_methods.py` | AdaGrad, RMSprop, Adam | 🔄 Planned | GD |

**Key Learning:**
- Nesterov: O(1/k²) convergence for smooth convex functions

### 2.2 Constrained Optimization

#### 2.2.1 Lagrange Multipliers
**Path:** `2-core-methods/constrained/lagrange/`

| File | Theorems/Concepts | Status | Dependencies |
|------|------------------|--------|--------------|
| `lagrange_multipliers.py` | Lagrange theorem, equality constraints | ✅ Complete | FONC |
| `equality_constraints.py` | h(x)=0 problems, examples | 🔄 Planned | Lagrange |
| `examples.py` | Worked constrained problems | 🔄 Planned | Lagrange |
| `geometric_interpretation.py` | Visual understanding, tangent spaces | 🔄 Planned | Calculus |

**Key Learning:**
- Lagrange: ∇f(x*) + Σλᵢ∇hᵢ(x*) = 0 at optimum

#### 2.2.2 KKT Conditions
**Path:** `2-core-methods/constrained/kkt-conditions/`

| File | Theorems/Concepts | Status | Dependencies |
|------|------------------|--------|--------------|
| `kkt_theory.py` | KKT necessary/sufficient conditions | ✅ Complete | Lagrange |
| `constraint_qualification.py` | LICQ, MFCQ, Slater condition | 🔄 Planned | Real Analysis |
| `complementarity.py` | Complementary slackness, λg=0 | 🔄 Planned | KKT |
| `examples.py` | KKT worked examples | 🔄 Planned | KKT |

**Key Learning:**
- KKT: Stationarity, feasibility, dual feasibility, complementarity

#### 2.2.3 Penalty Methods
**Path:** `2-core-methods/constrained/penalty-methods/`

| File | Algorithms | Status | Dependencies |
|------|-----------|--------|--------------|
| `penalty_functions.py` | Exterior penalty methods | 🔄 Planned | Unconstrained methods |
| `barrier_methods.py` | Interior point/log barrier | 🔄 Planned | Unconstrained methods |
| `augmented_lagrangian.py` | Method of multipliers | 🔄 Planned | Lagrange |

**Key Learning:**
- Convert constrained → unconstrained with penalty term

#### 2.2.4 Active Set Methods
**Path:** `2-core-methods/constrained/active-set/`

| File | Algorithms | Status | Dependencies |
|------|-----------|--------|--------------|
| `active_set_method.py` | Active set strategy for QP | 🔄 Planned | KKT |
| `qp_solver.py` | Quadratic programming solver | 🔄 Planned | Linear Algebra |
| `sequential_qp.py` | SQP for nonlinear problems | 🔄 Planned | QP |

**Key Learning:**
- Identify which constraints are active at optimum

### 2.3 Duality Theory
**Path:** `2-core-methods/duality/`

| File | Theorems/Concepts | Status | Dependencies |
|------|------------------|--------|--------------|
| `duality_theory.py` | Weak/strong duality theorems | ✅ Complete | Lagrange, Convexity |
| `dual_problems.py` | Constructing Lagrangian dual | 🔄 Planned | Duality |
| `saddle_points.py` | Saddle point interpretation | 🔄 Planned | Duality |
| `minimax_theorem.py` | Von Neumann minimax theorem | 🔄 Planned | Convexity |

**Key Learning:**
- Strong duality: p* = d* under Slater condition
- Dual provides lower bound on primal objective

---

## 🚀 LAYER 3: ADVANCED TOPICS (Weeks 17-24)

Specialized methods for specific problem classes.

### 3.1 Linear Programming
**Path:** `3-advanced-topics/linear-programming/`

| File | Algorithms | Status | Dependencies |
|------|-----------|--------|--------------|
| `simplex_method.py` | Simplex algorithm, pivoting | 🔄 Planned | Linear Algebra |
| `dual_simplex.py` | Dual simplex method | 🔄 Planned | Simplex, Duality |
| `interior_point_lp.py` | Primal-dual interior point for LP | 🔄 Planned | Barrier methods |
| `applications.py` | Diet problem, transportation | 🔄 Planned | LP |

### 3.2 Nonlinear Programming
**Path:** `3-advanced-topics/nonlinear-programming/`

| File | Algorithms | Status | Dependencies |
|------|-----------|--------|--------------|
| `sqp_methods.py` | Sequential quadratic programming | 🔄 Planned | SQP, Newton |
| `interior_point_nlp.py` | Interior point for general NLP | 🔄 Planned | Barrier |
| `global_optimization.py` | Basin hopping, multistart | 🔄 Planned | Local methods |
| `applications.py` | Engineering design optimization | 🔄 Planned | NLP |

### 3.3 Integer Programming
**Path:** `3-advanced-topics/integer-programming/`

| File | Algorithms | Status | Dependencies |
|------|-----------|--------|--------------|
| `branch_and_bound.py` | Branch & bound algorithm | 🔄 Planned | LP |
| `cutting_planes.py` | Gomory cuts, valid inequalities | 🔄 Planned | LP |
| `mixed_integer.py` | MILP formulations, modeling | 🔄 Planned | IP |
| `applications.py` | Scheduling, TSP, assignment | 🔄 Planned | IP |

### 3.4 Dynamic Programming
**Path:** `3-advanced-topics/dynamic-programming/`

| File | Algorithms | Status | Dependencies |
|------|-----------|--------|--------------|
| `bellman_principle.py` | Principle of optimality | 🔄 Planned | Recursion |
| `value_iteration.py` | Value iteration algorithm | 🔄 Planned | Bellman |
| `policy_iteration.py` | Policy iteration method | 🔄 Planned | Bellman |
| `applications.py` | Shortest path, inventory control | 🔄 Planned | DP |

### 3.5 Stochastic Optimization
**Path:** `3-advanced-topics/stochastic-optimization/`

| File | Algorithms | Status | Dependencies |
|------|-----------|--------|--------------|
| `sgd_methods.py` | Stochastic gradient descent, mini-batch | 🔄 Planned | GD |
| `variance_reduction.py` | SVRG, SAGA, SAG | 🔄 Planned | SGD |
| `online_learning.py` | Online convex optimization | 🔄 Planned | Convexity |
| `applications.py` | Large-scale ML training | 🔄 Planned | SGD |

### 3.6 Convex Programming
**Path:** `3-advanced-topics/convex-programming/`

| File | Algorithms | Status | Dependencies |
|------|-----------|--------|--------------|
| `cvxpy_introduction.py` | Using CVXPY library | 🔄 Planned | Convexity |
| `semidefinite_programming.py` | SDP formulations, relaxations | 🔄 Planned | Linear Algebra |
| `conic_optimization.py` | Second-order cone programming | 🔄 Planned | Convexity |
| `applications.py` | Control, signal processing | 🔄 Planned | Convex |

### 3.7 Multi-Objective Optimization
**Path:** `3-advanced-topics/multi-objective/`

| File | Algorithms | Status | Dependencies |
|------|-----------|--------|--------------|
| `pareto_optimality.py` | Pareto frontier, dominance | 🔄 Planned | Basic optimization |
| `scalarization.py` | Weighted sum, ε-constraint | 🔄 Planned | MOO |
| `evolutionary_methods.py` | NSGA-II, MOEA/D | 🔄 Planned | MOO |
| `applications.py` | Multi-criteria design | 🔄 Planned | MOO |

---

## 🎨 Supporting Modules

### Visualization
**Path:** `visualization/`

| File | Purpose | Status |
|------|---------|--------|
| `knowledge_visualizer.py` | Dependency graph visualization | ✅ Complete |
| `convergence_plots.py` | Convergence analysis plots | 🔄 Planned |
| `contour_animations.py` | Animated optimization paths | 🔄 Planned |
| `3d_surfaces.py` | 3D function landscapes | 🔄 Planned |
| `constraint_visualization.py` | Feasible regions, constraints | 🔄 Planned |

### Applications
**Path:** `applications/`

Real-world implementations across domains:
- **Machine Learning**: Regression, SVM, neural networks
- **Operations Research**: Resource allocation, scheduling
- **Engineering**: Structural design, control systems
- **Finance**: Portfolio optimization, risk management
- **Physics**: Energy systems, trajectory optimization

### Benchmarks
**Path:** `benchmarks/`

Standard test problems and performance comparisons:
- `test_functions.py` - Rosenbrock, Rastrigin, Ackley, etc.
- `performance_profiles.py` - Algorithm comparisons
- `scaling_tests.py` - Computational complexity analysis

---

## 📊 Implementation Statistics

### Current Status
- **Total Modules Planned**: ~80 files
- **Completed**: 16 files (20%)
- **Layer 1 Complete**: 3/16 (19%)
- **Layer 2 Complete**: 13/30 (43%)
- **Layer 3 Complete**: 0/25 (0%)

### Completion by Category
```
Foundations:        ████░░░░░░  19% (3/16)
Unconstrained:      ████████░░  75% (12/16)
Constrained:        ██░░░░░░░░  14% (2/14)
Advanced Topics:    ░░░░░░░░░░   0% (0/25)
Applications:       ░░░░░░░░░░   0% (0/20)
Visualization:      ██░░░░░░░░  20% (1/5)
```

### Key Milestones
- ✅ **Milestone 1**: Foundation theorems (Taylor, Weierstrass, Jensen)
- ✅ **Milestone 2**: Unconstrained optimization complete
- ✅ **Milestone 3**: Basic constrained (Lagrange, KKT, Duality)
- 🔄 **Milestone 4**: Penalty/barrier methods (in progress)
- 🔄 **Milestone 5**: Linear programming (next)
- 🔄 **Milestone 6**: Applications suite (future)

---

## 🎓 Learning Pathways

### Beginner Path (Weeks 1-12)
1. **Layer 1**: All foundations
2. **Layer 2**: Unconstrained → Basic constrained

### Intermediate Path (Weeks 13-20)
1. **Layer 2**: Complete constrained + duality
2. **Layer 3**: Linear + nonlinear programming

### Advanced Path (Weeks 21-28)
1. **Layer 3**: Integer, dynamic, stochastic, convex programming
2. **Applications**: Domain-specific projects

---

## 🔍 Quick Reference

### Find by Theorem
- **Taylor**: `1-foundations/calculus/taylor_theorem.py`
- **Weierstrass**: `1-foundations/real-analysis/weierstrass_theorem.py`
- **Jensen**: `1-foundations/convexity/jensen_inequality.py`
- **Lagrange**: `2-core-methods/constrained/lagrange/lagrange_multipliers.py`
- **KKT**: `2-core-methods/constrained/kkt-conditions/kkt_theory.py`
- **Duality**: `2-core-methods/duality/duality_theory.py`
- **Bellman**: `3-advanced-topics/dynamic-programming/bellman_principle.py` (planned)

### Find by Algorithm
- **Gradient Descent**: `2-core-methods/unconstrained/gradient-methods/steepest_descent.py`
- **Newton**: `2-core-methods/unconstrained/newton-methods/newton_method.py`
- **BFGS**: `2-core-methods/unconstrained/newton-methods/quasi_newton.py`
- **Trust Region**: `2-core-methods/unconstrained/newton-methods/trust_region.py`
- **Simplex**: `3-advanced-topics/linear-programming/simplex_method.py` (planned)
- **Branch & Bound**: `3-advanced-topics/integer-programming/branch_and_bound.py` (planned)

### Find by Application
- **ML Training**: `applications/machine-learning/` + `3-advanced-topics/stochastic-optimization/`
- **Engineering Design**: `applications/engineering/` + `3-advanced-topics/nonlinear-programming/`
- **Scheduling**: `applications/operations-research/` + `3-advanced-topics/integer-programming/`

---

## 📝 Notes

- **✅ Complete**: Fully implemented with examples and visualizations
- **🔄 Planned**: Designed but not yet implemented
- **Status Updates**: This index is updated as new modules are added

**Last Updated**: October 16, 2025
