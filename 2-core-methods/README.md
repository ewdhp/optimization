# Layer 2: Core Optimization Methods

Welcome to the core of optimization theory! This layer covers fundamental optimization algorithms and theoretical results that form the basis of practical optimization.

## 🎯 Learning Objectives

By completing this layer, you will:
- Master optimality conditions (FONC, SONC, SOSC)
- Implement gradient descent and variants
- Understand Newton's method and quasi-Newton approximations
- Learn trust region and line search strategies
- Master Lagrange multipliers and KKT conditions
- Understand duality theory and weak/strong duality
- Implement penalty and barrier methods

## 📚 Module Structure

```
2-core-methods/
├── unconstrained/           # Optimization without constraints
│   ├── optimality-conditions/   # Necessary and sufficient conditions
│   ├── gradient-methods/        # First-order methods
│   ├── newton-methods/          # Second-order methods
│   └── momentum-methods/        # Accelerated methods
├── constrained/             # Optimization with constraints
│   ├── lagrange/               # Lagrange multiplier theory
│   ├── kkt-conditions/         # KKT theory and applications
│   ├── penalty-methods/        # Penalty and barrier approaches
│   └── active-set/             # Active set methods
└── duality/                 # Duality theory
```

## 🗺️ Module Guide

### PART A: Unconstrained Optimization (Weeks 5-8)

#### Optimality Conditions (`unconstrained/optimality-conditions/`)

**Core Concepts:**
- **FONC** (First-Order Necessary): ∇f(x*) = 0
- **SONC** (Second-Order Necessary): ∇²f(x*) ⪰ 0
- **SOSC** (Second-Order Sufficient): ∇²f(x*) ≻ 0
- Critical point classification: min/max/saddle

**Why It Matters:**
- FONC tells us where to look for optima (stationary points)
- SOSC guarantees a point is actually a minimum
- Understanding these is essential for algorithm design

**Files:**
- ✅ `first_order.py` - FONC, critical points, descent directions
- ✅ `second_order.py` - SONC, SOSC, Hessian analysis
- ✅ `examples.py` - Comprehensive worked examples

**Key Results:**
```
Minimum ⟹ ∇f = 0 AND ∇²f ⪰ 0  (necessary)
∇f = 0 AND ∇²f ≻ 0 ⟹ Strict local minimum (sufficient)
```

#### Gradient Methods (`unconstrained/gradient-methods/`)

**Core Concepts:**
- Steepest descent: x_{k+1} = x_k - α_k ∇f(x_k)
- Line search strategies: Armijo, Wolfe conditions
- Convergence analysis: rates, Lipschitz constants
- Conjugate gradient: improved search directions

**Why It Matters:**
- Simplest and most widely used optimization method
- Scales to large problems (only needs gradients)
- Foundation for advanced methods (Adam, RMSprop, etc.)

**Files:**
- ✅ `steepest_descent.py` - GD with various step sizes
- ✅ `line_search.py` - Armijo, Wolfe, strong Wolfe
- ✅ `convergence_analysis.py` - Lipschitz, strong convexity
- 🔄 `conjugate_gradient.py` - CG method

**Convergence Rates:**
```
Convex smooth: O(1/k)
Strongly convex: O(ρᵏ) where ρ < 1 (linear convergence)
```

#### Newton Methods (`unconstrained/newton-methods/`)

**Core Concepts:**
- Newton's method: p_k = -[∇²f]^{-1}∇f
- Quasi-Newton: BFGS approximates Hessian
- Trust region: global convergence strategy
- Convergence theory: quadratic vs superlinear

**Why It Matters:**
- Fastest local convergence (quadratic)
- BFGS is workhorse of practical optimization
- Trust region ensures global convergence

**Files:**
- ✅ `newton_method.py` - Pure and damped Newton
- ✅ `quasi_newton.py` - BFGS, L-BFGS, DFP, SR1
- ✅ `trust_region.py` - TR with dogleg
- ✅ `convergence_theory.py` - Convergence proofs

**Convergence Rates:**
```
Newton: ||x_{k+1} - x*|| ≤ C||x_k - x*||² (quadratic)
BFGS: lim (||x_{k+1} - x*|| / ||x_k - x*||) = 0 (superlinear)
```

#### Momentum Methods (`unconstrained/momentum-methods/`)

**Core Concepts:**
- Heavy ball method: adds momentum term
- Nesterov acceleration: O(1/k²) convergence
- Adaptive methods: AdaGrad, RMSprop, Adam

**Why It Matters:**
- Acceleration improves convergence significantly
- Adam is default for deep learning training
- Practical importance in large-scale problems

**Files:**
- 🔄 `heavy_ball.py` - Classical momentum
- 🔄 `nesterov_momentum.py` - Accelerated GD
- 🔄 `adaptive_methods.py` - Adam, RMSprop

### PART B: Constrained Optimization (Weeks 9-12)

#### Lagrange Multipliers (`constrained/lagrange/`)

**Core Concepts:**
- Lagrangian: L(x, λ) = f(x) + Σ λᵢhᵢ(x)
- Lagrange multiplier theorem
- Geometric interpretation: gradient alignment

**Why It Matters:**
- Fundamental theory for constrained optimization
- Converts constrained → unconstrained stationarity
- Basis for KKT conditions

**Files:**
- ✅ `lagrange_multipliers.py` - Theory and examples
- 🔄 `equality_constraints.py` - h(x) = 0 problems
- 🔄 `examples.py` - Worked problems
- 🔄 `geometric_interpretation.py` - Visual understanding

**Optimality Condition:**
```
∇f(x*) + Σ λᵢ* ∇hᵢ(x*) = 0
hᵢ(x*) = 0  ∀i
```

#### KKT Conditions (`constrained/kkt-conditions/`)

**Core Concepts:**
- KKT conditions: stationarity, feasibility, complementarity
- Constraint qualification: LICQ, MFCQ, Slater
- Complementary slackness: λᵢgᵢ(x) = 0

**Why It Matters:**
- Necessary conditions for constrained optimality
- Sufficient under convexity
- Basis for all constrained algorithms

**Files:**
- ✅ `kkt_theory.py` - Complete KKT theory
- 🔄 `constraint_qualification.py` - CQ conditions
- 🔄 `complementarity.py` - Complementary slackness
- 🔄 `examples.py` - KKT examples

**KKT Conditions:**
```
1. Stationarity: ∇f + Σλᵢ∇gᵢ + Σμⱼ∇hⱼ = 0
2. Primal feasibility: gᵢ(x) ��� 0, hⱼ(x) = 0
3. Dual feasibility: λᵢ ≥ 0
4. Complementarity: λᵢgᵢ(x) = 0
```

#### Penalty Methods (`constrained/penalty-methods/`)

**Core Concepts:**
- Exterior penalty: P(x, ρ) = f(x) + ρΣ[gᵢ(x)₊]²
- Interior penalty/barrier: B(x, μ) = f(x) - μΣlog(-gᵢ(x))
- Augmented Lagrangian: combines penalty + multipliers

**Why It Matters:**
- Convert constrained → sequence of unconstrained
- Practical algorithms (IPOPT uses barrier)
- Foundation for interior point methods

**Files:**
- 🔄 `penalty_functions.py` - Exterior penalty
- 🔄 `barrier_methods.py` - Log barrier, interior point
- 🔄 `augmented_lagrangian.py` - Method of multipliers

#### Active Set Methods (`constrained/active-set/`)

**Core Concepts:**
- Active constraints: those that are tight at solution
- Active set strategy for QP
- Sequential QP for nonlinear problems

**Why It Matters:**
- Identifies which constraints matter
- Efficient for problems with few active constraints
- SQP is powerful for nonlinear problems

**Files:**
- 🔄 `active_set_method.py` - Active set for QP
- 🔄 `qp_solver.py` - Quadratic programming solver
- 🔄 `sequential_qp.py` - SQP algorithm

### PART C: Duality Theory (Weeks 13-16)

#### Duality (`duality/`)

**Core Concepts:**
- Lagrangian dual: d* = sup_λ inf_x L(x, λ)
- Weak duality: d* ≤ p* (always)
- Strong duality: d* = p* (under Slater)
- Saddle point interpretation

**Why It Matters:**
- Dual provides lower bound on optimal value
- Strong duality enables powerful solution methods
- Connects primal and dual problems

**Files:**
- ✅ `duality_theory.py` - Complete duality theory
- 🔄 `dual_problems.py` - Constructing duals
- 🔄 `saddle_points.py` - Saddle point characterization
- 🔄 `minimax_theorem.py` - Von Neumann minimax

**Duality Gap:**
```
Gap = p* - d* ≥ 0
Gap = 0 ⟺ strong duality holds
```

## 📖 Suggested Study Order

### Weeks 5-6: Unconstrained Basics
1. **Optimality Conditions** (all 3 files)
2. **Gradient Methods** (steepest_descent, line_search)
3. Implement GD on test functions

### Weeks 7-8: Advanced Unconstrained
1. **Newton Methods** (newton, quasi_newton, trust_region)
2. **Convergence Theory**
3. Compare GD vs Newton vs BFGS on Rosenbrock

### Weeks 9-10: Basic Constrained
1. **Lagrange Multipliers**
2. **KKT Conditions**
3. Solve constrained problems analytically

### Weeks 11-12: Constrained Algorithms
1. **Penalty Methods**
2. **Active Set Methods**
3. Implement penalty method

### Weeks 13-16: Duality
1. **Duality Theory**
2. **Dual Problems**
3. Verify strong duality on examples

## 🔗 Connections to Layer 3

Layer 2 provides the foundation for specialized methods in Layer 3:

- **Gradient methods → Stochastic optimization**: SGD extends GD
- **Newton → Interior point**: Barrier methods for LP/SDP
- **KKT → Linear programming**: Simplex uses complementarity
- **Duality → Convex programming**: Strong duality in CVX
- **Active set → Integer programming**: Branch & bound

## 🎓 Assessment

Before moving to Layer 3, you should be able to:

### Unconstrained
1. ✅ Verify FONC, SONC, SOSC at a point
2. ✅ Implement gradient descent from scratch
3. ✅ Implement BFGS algorithm
4. ✅ Explain convergence rates (linear, quadratic, superlinear)
5. ✅ Choose appropriate method for a problem

### Constrained
1. ✅ Write down KKT conditions for a problem
2. ✅ Verify KKT conditions at a candidate point
3. ✅ Construct Lagrangian and dual problem
4. ✅ Check constraint qualification
5. ✅ Implement penalty method

### Duality
1. ✅ Derive dual problem
2. ✅ Verify weak/strong duality
3. ✅ Interpret duality gap
4. ✅ Use dual to verify optimality

## 🛠️ Practical Projects

### Project 1: Optimization Library
Build a Python library with:
- Gradient descent (multiple step sizes)
- Newton's method
- BFGS
- Penalty method

### Project 2: Rosenbrock Challenge
Minimize Rosenbrock function using:
- Steepest descent
- Newton
- BFGS
- Trust region
Compare iterations and function evaluations.

### Project 3: Constrained Portfolio
Solve portfolio optimization:
```
minimize    -r^T x + γx^T Σ x
subject to  1^T x = 1, x ≥ 0
```
Using:
- KKT conditions (analytical)
- Penalty method (numerical)
- Compare solutions

## 📚 Key References

### Textbooks
- **Nocedal & Wright**: "Numerical Optimization" (Chapters 2-18)
- **Boyd & Vandenberghe**: "Convex Optimization" (Chapters 4-5, 9-11)
- **Bertsekas**: "Nonlinear Programming" (Chapters 1-5)

### Papers
- **BFGS**: Broyden, Fletcher, Goldfarb, Shanno (1970)
- **Trust Region**: Conn, Gould, Toint (2000)
- **Interior Point**: Karmarkar (1984), Mehrotra (1992)

## 💡 Implementation Tips

1. **Start Simple**: Implement basic GD before BFGS
2. **Test on Quadratics**: Quadratics have known solutions
3. **Visualize**: Plot convergence, contours, paths
4. **Use Test Functions**: Rosenbrock, Himmelblau, Beale
5. **Compare Methods**: See which works best for what

## 📊 Implementation Status

- **Unconstrained**: ████████░░ 80% (12/15 files)
- **Constrained**: ██░░░░░░░░ 14% (2/14 files)
- **Duality**: ██░░░░░░░░ 25% (1/4 files)
- **Overall**: ███████░░░ 46% (15/33 files)

## ⏭️ Next Steps

Once you complete Layer 2, proceed to:
- **Layer 3**: Advanced topics (LP, NLP, IP, DP)
- Start with: `3-advanced-topics/linear-programming/`

---

**Status**: 15/33 modules complete (46%)
**Time Estimate**: 12 weeks of focused study
**Prerequisites**: Layer 1 (Foundations) complete
