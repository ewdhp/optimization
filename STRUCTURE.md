# Optimization Theory: Complete Directory Structure

A comprehensive, pedagogically-organized optimization framework with three progressive layers: **Foundations** → **Core Methods** → **Advanced Topics**.

## 📁 Directory Architecture

```
optimization/
│
├── README.md                           # Main overview and learning roadmap
├── STRUCTURE.md                        # This file - complete structure guide
├── requirements.txt                    # Python dependencies
│
├── 1-foundations/                      # LAYER 1: Mathematical Prerequisites
│   ├── README.md                       # Foundation concepts overview
│   │
│   ├── calculus/
│   │   ├── taylor_theorem.py          # Taylor series expansion, approximation
│   │   ├── mean_value_theorem.py      # MVT, Rolle's theorem
│   │   ├── implicit_function.py       # Implicit function theorem
│   │   └── multivariable_calculus.py  # Gradients, Hessians, Jacobians
│   │
│   ├── linear-algebra/
│   │   ├── eigenvalues.py             # Eigendecomposition, spectral theory
│   │   ├── matrix_decomposition.py    # QR, SVD, Cholesky
│   │   ├── positive_definiteness.py   # PD/PSD testing, Sylvester criterion
│   │   └── vector_spaces.py           # Norms, inner products, projections
│   │
│   ├── real-analysis/
│   │   ├── weierstrass_theorem.py     # Existence of extrema, compactness
│   │   ├── continuity_lipschitz.py    # Continuity, Lipschitz conditions
│   │   ├── sequences_limits.py        # Convergence, Cauchy sequences
│   │   └── bolzano_weierstrass.py     # Compactness, subsequences
│   │
│   └── convexity/
│       ├── convex_sets.py             # Convex sets, operations
│       ├── convex_functions.py        # Convex function properties
│       ├── jensen_inequality.py       # Jensen's inequality, applications
│       └── separation_theorems.py     # Separating hyperplanes
│
├── 2-core-methods/                     # LAYER 2: Fundamental Optimization
│   ├── README.md                       # Core optimization overview
│   │
│   ├── unconstrained/
│   │   ├── README.md                   # Unconstrained optimization guide
│   │   │
│   │   ├── optimality-conditions/
│   │   │   ├── first_order.py         # FONC, critical points, descent
│   │   │   ├── second_order.py        # SONC, SOSC, Hessian analysis
│   │   │   └── examples.py            # Worked examples, classifications
│   │   │
│   │   ├── gradient-methods/
│   │   │   ├── steepest_descent.py    # Gradient descent variants
│   │   │   ├── line_search.py         # Armijo, Wolfe, strong Wolfe
│   │   │   ├── conjugate_gradient.py  # CG method, Polak-Ribière
│   │   │   └── convergence_analysis.py # Rates, Lipschitz, strong convexity
│   │   │
│   │   ├── newton-methods/
│   │   │   ├── newton_method.py       # Pure/damped Newton, Hessian
│   │   │   ├── quasi_newton.py        # BFGS, L-BFGS, DFP, SR1
│   │   │   ├── trust_region.py        # TR with dogleg, Cauchy point
│   │   │   └── convergence_theory.py  # Quadratic/superlinear convergence
│   │   │
│   │   └── momentum-methods/
│   │       ├── heavy_ball.py          # Classical momentum
│   │       ├── nesterov_momentum.py   # Accelerated gradient descent
│   │       └── adaptive_methods.py    # AdaGrad, RMSprop, Adam
│   │
│   ├── constrained/
│   │   ├── README.md                   # Constrained optimization guide
│   │   │
│   │   ├── lagrange/
│   │   │   ├── lagrange_multipliers.py # Classical Lagrange theory
│   │   │   ├── equality_constraints.py # h(x) = 0 constraints
│   │   │   ├── examples.py            # Worked constrained problems
│   │   │   └── geometric_interpretation.py # Visual understanding
│   │   │
│   │   ├── kkt-conditions/
│   │   │   ├── kkt_theory.py          # KKT necessary/sufficient conditions
│   │   │   ├── constraint_qualification.py # LICQ, MFCQ, Slater
│   │   │   ├── complementarity.py     # Complementary slackness
│   │   │   └── examples.py            # KKT worked examples
│   │   │
│   │   ├── penalty-methods/
│   │   │   ├── penalty_functions.py   # Exterior penalty
│   │   │   ├── barrier_methods.py     # Interior point/barrier
│   │   │   └── augmented_lagrangian.py # Method of multipliers
│   │   │
│   │   └── active-set/
│   │       ├── active_set_method.py   # Active set strategy
│   │       ├── qp_solver.py           # Quadratic programming
│   │       └── sequential_qp.py       # SQP methods
│   │
│   └── duality/
│       ├── README.md                   # Duality theory overview
│       ├── duality_theory.py          # Weak/strong duality
│       ├── dual_problems.py           # Constructing dual problems
│       ├── saddle_points.py           # Saddle point characterization
│       └── minimax_theorem.py         # Von Neumann minimax
│
├── 3-advanced-topics/                  # LAYER 3: Specialized Methods
│   ├── README.md                       # Advanced topics overview
│   │
│   ├── linear-programming/
│   │   ├── simplex_method.py          # Simplex algorithm
│   │   ├── dual_simplex.py            # Dual simplex
│   │   ├── interior_point_lp.py       # Primal-dual interior point
│   │   └── applications.py            # LP applications, diet problem
│   │
│   ├── nonlinear-programming/
│   │   ├── sqp_methods.py             # Sequential quadratic programming
│   │   ├── interior_point_nlp.py      # Interior point for NLP
│   │   ├── global_optimization.py     # Global methods, basin hopping
│   │   └── applications.py            # Engineering design examples
│   │
│   ├── integer-programming/
│   │   ├── branch_and_bound.py        # B&B algorithm
│   │   ├── cutting_planes.py          # Gomory cuts
│   │   ├── mixed_integer.py           # MILP formulations
│   │   └── applications.py            # Scheduling, TSP examples
│   │
│   ├── dynamic-programming/
│   │   ├── bellman_principle.py       # Principle of optimality
│   │   ├── value_iteration.py         # Value iteration algorithm
│   │   ├── policy_iteration.py        # Policy iteration
│   │   └── applications.py            # Shortest path, control
│   │
│   ├── stochastic-optimization/
│   │   ├── sgd_methods.py             # Stochastic gradient descent
│   │   ├── variance_reduction.py      # SVRG, SAGA
│   │   ├── online_learning.py         # Online convex optimization
│   │   └── applications.py            # Large-scale ML examples
│   │
│   ├── convex-programming/
│   │   ├── cvxpy_introduction.py      # Using CVXPY
│   │   ├── semidefinite_programming.py # SDP formulations
│   │   ├── conic_optimization.py      # Second-order cones
│   │   └── applications.py            # Control, signal processing
│   │
│   └── multi-objective/
│       ├── pareto_optimality.py       # Pareto frontier
│       ├── scalarization.py           # Weighted sum, ε-constraint
│       ├── evolutionary_methods.py    # NSGA-II, MOEA
│       └── applications.py            # Design tradeoffs
│
├── applications/                       # Real-World Applications
│   ├── machine-learning/
│   │   ├── linear_regression.py       # Least squares, ridge, lasso
│   │   ├── logistic_regression.py     # Classification optimization
│   │   ├── neural_networks.py         # Backpropagation, training
│   │   ├── svm.py                     # Support vector machines
│   │   └── hyperparameter_tuning.py   # Cross-validation, grid search
│   │
│   ├── operations-research/
│   │   ├── resource_allocation.py     # Production planning
│   │   ├── transportation.py          # Transportation problem
│   │   ├── assignment.py              # Assignment problem
│   │   └── scheduling.py              # Job shop scheduling
│   │
│   ├── engineering/
│   │   ├── structural_design.py       # Truss optimization
│   │   ├── control_systems.py         # Optimal control, LQR
│   │   ├── signal_processing.py       # Filter design
│   │   └── circuit_design.py          # Component optimization
│   │
│   ├── finance/
│   │   ├── portfolio_optimization.py  # Markowitz model
│   │   ├── risk_management.py         # CVaR optimization
│   │   └── option_pricing.py          # Calibration problems
│   │
│   └── physics/
│       ├── energy_systems.py          # Power system optimization
│       ├── trajectory_optimization.py # Optimal paths
│       └── quantum_optimization.py    # Variational quantum
│
├── visualization/                      # Interactive Visualizations
│   ├── convergence_plots.py           # Convergence visualization
│   ├── contour_animations.py          # Animated optimization paths
│   ├── 3d_surfaces.py                 # 3D function landscapes
│   ├── constraint_visualization.py    # Feasible regions
│   └── knowledge_visualizer.py        # Dependency graphs
│
├── benchmarks/                         # Standard Test Problems
│   ├── test_functions.py              # Rosenbrock, Rastrigin, etc.
│   ├── performance_profiles.py        # Algorithm comparisons
│   └── scaling_tests.py               # Computational complexity
│
├── utilities/                          # Helper Functions
│   ├── numerical_differentiation.py   # Finite differences
│   ├── matrix_utilities.py            # Matrix operations
│   ├── plotting_helpers.py            # Visualization utilities
│   └── test_problem_generator.py     # Random problem generation
│
└── roadmap/                            # Learning Path
    ├── week_01_foundations.md         # Weekly study guides
    ├── week_02_calculus.md
    ├── ...
    ├── week_28_capstone.md
    ├── exercises/                      # Practice problems
    │   ├── set_01_foundations.py
    │   ├── set_02_gradient_methods.py
    │   └── ...
    └── knowledge_visualizer.py         # Dependency visualization
```

## 🎯 Learning Path (3-Layer Progression)

### **Layer 1: Foundations** (Weeks 1-4)
Build mathematical prerequisites for optimization theory.

**Modules:**
- `1-foundations/calculus/` - Taylor expansion, gradients, Hessians
- `1-foundations/linear-algebra/` - Eigenvalues, definiteness, decompositions
- `1-foundations/real-analysis/` - Weierstrass, continuity, convergence
- `1-foundations/convexity/` - Convex sets/functions, Jensen inequality

**Key Theorems:** Taylor, Weierstrass, Jensen

### **Layer 2: Core Methods** (Weeks 5-16)
Master fundamental optimization algorithms and theory.

**Phase 2A: Unconstrained** (Weeks 5-8)
- `2-core-methods/unconstrained/optimality-conditions/` - FONC, SOSC
- `2-core-methods/unconstrained/gradient-methods/` - GD, line search, CG
- `2-core-methods/unconstrained/newton-methods/` - Newton, BFGS, trust region

**Phase 2B: Constrained** (Weeks 9-12)
- `2-core-methods/constrained/lagrange/` - Lagrange multipliers
- `2-core-methods/constrained/kkt-conditions/` - KKT theory
- `2-core-methods/constrained/penalty-methods/` - Penalty, barrier, augmented

**Phase 2C: Duality** (Weeks 13-16)
- `2-core-methods/duality/` - Weak/strong duality, minimax

**Key Theorems:** Lagrange, KKT, Duality, Convergence

### **Layer 3: Advanced Topics** (Weeks 17-24)
Specialized methods for specific problem classes.

- `3-advanced-topics/linear-programming/` - Simplex, interior point
- `3-advanced-topics/nonlinear-programming/` - SQP, global optimization
- `3-advanced-topics/integer-programming/` - Branch & bound
- `3-advanced-topics/dynamic-programming/` - Bellman, value iteration
- `3-advanced-topics/stochastic-optimization/` - SGD, variance reduction
- `3-advanced-topics/convex-programming/` - SDP, conic optimization

**Key Theorems:** Bellman Optimality, Simplex Fundamental, Strong Duality

### **Applications** (Weeks 25-28)
Apply optimization to real-world domains.

## 📊 Theorem Coverage Map

| Theorem | Module | Dependencies |
|---------|--------|--------------|
| Taylor's Theorem | `1-foundations/calculus/taylor_theorem.py` | Calculus |
| Mean Value Theorem | `1-foundations/calculus/mean_value_theorem.py` | Calculus |
| Weierstrass Theorem | `1-foundations/real-analysis/weierstrass_theorem.py` | Real Analysis |
| Jensen's Inequality | `1-foundations/convexity/jensen_inequality.py` | Convexity |
| Separation Theorems | `1-foundations/convexity/separation_theorems.py` | Convex Sets |
| First-Order Necessary (FONC) | `2-core-methods/unconstrained/optimality-conditions/first_order.py` | Calculus |
| Second-Order Sufficient (SOSC) | `2-core-methods/unconstrained/optimality-conditions/second_order.py` | Linear Algebra |
| Convergence Theory | `2-core-methods/unconstrained/newton-methods/convergence_theory.py` | Real Analysis |
| Lagrange Multiplier Theorem | `2-core-methods/constrained/lagrange/lagrange_multipliers.py` | FONC |
| KKT Conditions | `2-core-methods/constrained/kkt-conditions/kkt_theory.py` | Lagrange |
| Constraint Qualification | `2-core-methods/constrained/kkt-conditions/constraint_qualification.py` | Real Analysis |
| Weak Duality | `2-core-methods/duality/duality_theory.py` | Lagrange |
| Strong Duality | `2-core-methods/duality/duality_theory.py` | Convexity + Slater |
| Bellman Optimality | `3-advanced-topics/dynamic-programming/bellman_principle.py` | Dynamic Systems |
| Simplex Fundamental | `3-advanced-topics/linear-programming/simplex_method.py` | Linear Algebra |
| Complementary Slackness | `2-core-methods/constrained/kkt-conditions/complementarity.py` | KKT |

## 🔧 Algorithm Coverage Map

| Algorithm | Module | Type |
|-----------|--------|------|
| Gradient Descent | `2-core-methods/unconstrained/gradient-methods/steepest_descent.py` | First-order |
| Line Search (Armijo/Wolfe) | `2-core-methods/unconstrained/gradient-methods/line_search.py` | Step size |
| Conjugate Gradient | `2-core-methods/unconstrained/gradient-methods/conjugate_gradient.py` | First-order |
| Newton's Method | `2-core-methods/unconstrained/newton-methods/newton_method.py` | Second-order |
| BFGS | `2-core-methods/unconstrained/newton-methods/quasi_newton.py` | Quasi-Newton |
| L-BFGS | `2-core-methods/unconstrained/newton-methods/quasi_newton.py` | Limited memory |
| Trust Region | `2-core-methods/unconstrained/newton-methods/trust_region.py` | Globalization |
| Nesterov Momentum | `2-core-methods/unconstrained/momentum-methods/nesterov_momentum.py` | Accelerated |
| Adam | `2-core-methods/unconstrained/momentum-methods/adaptive_methods.py` | Adaptive |
| Penalty Method | `2-core-methods/constrained/penalty-methods/penalty_functions.py` | Constraint handling |
| Barrier Method | `2-core-methods/constrained/penalty-methods/barrier_methods.py` | Interior point |
| Augmented Lagrangian | `2-core-methods/constrained/penalty-methods/augmented_lagrangian.py` | Multiplier method |
| Active Set | `2-core-methods/constrained/active-set/active_set_method.py` | Constraint active |
| SQP | `2-core-methods/constrained/active-set/sequential_qp.py` | Sequential QP |
| Simplex | `3-advanced-topics/linear-programming/simplex_method.py` | LP solver |
| Interior Point (LP) | `3-advanced-topics/linear-programming/interior_point_lp.py` | Barrier for LP |
| Branch & Bound | `3-advanced-topics/integer-programming/branch_and_bound.py` | IP solver |
| Value Iteration | `3-advanced-topics/dynamic-programming/value_iteration.py` | DP solver |
| SGD | `3-advanced-topics/stochastic-optimization/sgd_methods.py` | Stochastic |

## 🎓 Implementation Status

### ✅ Completed (Current)
- [x] `1-foundations/convexity/convex_sets.py`
- [x] `1-foundations/convexity/convex_functions.py`
- [x] `1-foundations/calculus/taylor_theorem.py`
- [x] `1-foundations/real-analysis/weierstrass_theorem.py`
- [x] `1-foundations/convexity/jensen_inequality.py`
- [x] `2-core-methods/constrained/lagrange/lagrange_multipliers.py`
- [x] `2-core-methods/constrained/kkt-conditions/kkt_theory.py`
- [x] `2-core-methods/duality/duality_theory.py`
- [x] `2-core-methods/unconstrained/newton-methods/convergence_theory.py`
- [x] `2-core-methods/unconstrained/optimality-conditions/first_order.py`
- [x] `2-core-methods/unconstrained/optimality-conditions/second_order.py`
- [x] `2-core-methods/unconstrained/optimality-conditions/examples.py`
- [x] `2-core-methods/unconstrained/gradient-methods/steepest_descent.py`
- [x] `2-core-methods/unconstrained/gradient-methods/line_search.py`
- [x] `2-core-methods/unconstrained/gradient-methods/convergence_analysis.py`
- [x] `2-core-methods/unconstrained/newton-methods/newton_method.py`
- [x] `2-core-methods/unconstrained/newton-methods/quasi_newton.py`
- [x] `2-core-methods/unconstrained/newton-methods/trust_region.py`

### 🔄 Next Priority
- [ ] `1-foundations/linear-algebra/` (eigenvalues, definiteness)
- [ ] `2-core-methods/unconstrained/gradient-methods/conjugate_gradient.py`
- [ ] `2-core-methods/unconstrained/momentum-methods/` (Nesterov, Adam)
- [ ] `2-core-methods/constrained/penalty-methods/` (penalty, barrier, augmented)
- [ ] `3-advanced-topics/linear-programming/simplex_method.py`

## 📖 Usage Guide

```python
# Example: Following the learning path

# Week 1-4: Foundations
from optimization.foundations.calculus import taylor_theorem
from optimization.foundations.convexity import jensen_inequality

# Week 5-8: Unconstrained
from optimization.core_methods.unconstrained.gradient_methods import steepest_descent
from optimization.core_methods.unconstrained.newton_methods import quasi_newton

# Week 9-12: Constrained
from optimization.core_methods.constrained.lagrange import lagrange_multipliers
from optimization.core_methods.constrained.kkt_conditions import kkt_theory

# Week 17-24: Advanced
from optimization.advanced_topics.linear_programming import simplex_method
from optimization.advanced_topics.dynamic_programming import bellman_principle

# Applications
from optimization.applications.machine_learning import linear_regression
```

## 🔗 Cross-References

Each module includes:
- **Prerequisites**: What to study first
- **Theorem dependencies**: Which theorems are needed
- **Algorithm variants**: Related implementations
- **Applications**: Where it's used in practice
- **Next topics**: What to learn next

## 📝 Notes

- All files include comprehensive docstrings with mathematical formulations
- Each algorithm has convergence analysis and visualizations
- Examples range from toy problems to real applications
- Code is educational first, performance second
- Each layer builds on the previous layer's concepts
