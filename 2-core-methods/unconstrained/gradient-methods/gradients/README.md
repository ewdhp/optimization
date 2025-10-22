# Gradient-Based Optimization Methods

This directory contains comprehensive implementations of various gradient-based optimization algorithms, from basic steepest descent to advanced adaptive methods.

## Overview

Gradient-based methods are the foundation of continuous optimization. They use gradient information (first-order derivatives) to iteratively improve solution estimates. The methods here range from simple to sophisticated, each with specific strengths and use cases.

## Contents

### 1. **steepest_descent.py** - Steepest Descent Method
The most fundamental gradient-based method. Moves in the direction of negative gradient.

**Key Features:**
- Simple and intuitive
- Multiple line search strategies (constant, backtracking, exact)
- Guaranteed convergence for convex functions
- Can be slow on ill-conditioned problems

**When to use:**
- Simple convex problems
- When you need a baseline for comparison
- Educational purposes

**Example usage:**
```python
from steepest_descent import SteepestDescent

optimizer = SteepestDescent(step_size=0.1, line_search='backtracking')
x_opt, history = optimizer.optimize(f, grad_f, x0)
```

---

### 2. **conjugate_gradient.py** - Conjugate Gradient Methods
Improved gradient method using conjugate directions instead of steepest descent.

**Key Features:**
- Converges in n steps for n-dimensional quadratic functions
- Much faster than steepest descent on ill-conditioned problems
- Multiple β formulas: Fletcher-Reeves (FR), Polak-Ribière (PR), Hestenes-Stiefel (HS), Dai-Yuan (DY)
- Memory efficient: O(n) storage

**When to use:**
- Large-scale quadratic problems
- Ill-conditioned optimization
- When Hessian computation is too expensive

**Example usage:**
```python
from conjugate_gradient import ConjugateGradient, CGConfig

config = CGConfig(beta_method='PR', max_iter=1000)
optimizer = ConjugateGradient(config)
x_opt, history = optimizer.optimize(f, grad_f, x0)
```

**Performance Comparison:**
- FR: Most robust, works well generally
- PR: Often fastest, can have issues on non-convex problems
- HS and DY: Good middle ground

---

### 3. **quasi_newton.py** - Quasi-Newton Methods (BFGS and L-BFGS)
Approximate Newton's method without computing the Hessian matrix.

**Key Features:**
- Superlinear convergence rate
- BFGS: Full inverse Hessian approximation (O(n²) memory)
- L-BFGS: Limited-memory variant (O(mn) memory, m typically 5-20)
- Most popular general-purpose optimization methods

**When to use:**
- General non-linear optimization
- When you need fast convergence
- L-BFGS for large-scale problems (n > 1000)

**Example usage:**
```python
from quasi_newton import BFGS, LBFGS, QuasiNewtonConfig

# For moderate dimensions (n < 1000)
bfgs = BFGS(QuasiNewtonConfig(max_iter=200))
x_opt, history = bfgs.optimize(f, grad_f, x0)

# For large dimensions
lbfgs = LBFGS(QuasiNewtonConfig(memory_size=10))
x_opt, history = lbfgs.optimize(f, grad_f, x0)
```

---

### 4. **momentum_methods.py** - Gradient Descent with Momentum
Accelerates gradient descent by accumulating velocity in persistent gradient directions.

**Key Features:**
- Classical (Heavy Ball) momentum
- Nesterov Accelerated Gradient (NAG)
- Reduces oscillations, speeds up convergence
- Especially effective in ravines/valleys

**When to use:**
- Problems with ravines or narrow valleys
- When steepest descent oscillates
- As a middle ground between GD and more complex methods

**Example usage:**
```python
from momentum_methods import MomentumGD, MomentumConfig

# Classical momentum
config = MomentumConfig(momentum_type='classical', beta=0.9, learning_rate=0.01)
optimizer = MomentumGD(config)
x_opt, history = optimizer.optimize(f, grad_f, x0)

# Nesterov momentum (often faster)
config = MomentumConfig(momentum_type='nesterov', beta=0.9, learning_rate=0.01)
optimizer = MomentumGD(config)
x_opt, history = optimizer.optimize(f, grad_f, x0)
```

**Hyperparameter Guidelines:**
- β = 0.9 is a good default
- β > 0.95 may cause overshooting
- Nesterov often outperforms classical momentum

---

### 5. **adam_optimizer.py** - Adam and Variants
Adaptive Moment Estimation: combines momentum and adaptive learning rates.

**Key Features:**
- Adapts learning rate for each parameter
- Very robust to hyperparameter choices
- Multiple variants: Adam, AdaMax, AMSGrad, Nadam
- Most popular optimizer in deep learning

**When to use:**
- When you want good performance with minimal tuning
- Noisy or sparse gradient problems
- Deep learning applications
- When other methods are hard to tune

**Example usage:**
```python
from adam_optimizer import Adam, AdamConfig

# Default Adam (works well in most cases)
config = AdamConfig(learning_rate=0.001)
optimizer = Adam(config)
x_opt, history = optimizer.optimize(f, grad_f, x0)

# Try different variants
for variant in ['adam', 'adamax', 'amsgrad', 'nadam']:
    config = AdamConfig(variant=variant)
    optimizer = Adam(config)
    x_opt, history = optimizer.optimize(f, grad_f, x0)
```

**Default Hyperparameters (work well):**
- learning_rate: 0.001
- beta1: 0.9
- beta2: 0.999
- epsilon: 1e-8

---

## Method Selection Guide

### Decision Tree

```
Is your problem quadratic?
├─ Yes → Use Conjugate Gradient (FR or PR)
└─ No
   ├─ Is dimension n < 1000?
   │  └─ Use BFGS
   └─ Is dimension n ≥ 1000?
      ├─ Need fast convergence? → Use L-BFGS
      └─ Need robustness/simplicity? → Use Adam
```

### Performance Comparison

On a typical 2D Rosenbrock problem starting from (-1, 1):

| Method | Iterations | Speed | Memory | Tuning Difficulty |
|--------|-----------|-------|--------|-------------------|
| Steepest Descent | ~1000 | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Momentum (NAG) | ~500 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Conjugate Gradient | ~100 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| BFGS | ~50 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| L-BFGS | ~60 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| Adam | ~200 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

⭐ = Poor, ⭐⭐⭐⭐⭐ = Excellent

### Convergence Rates

- **Steepest Descent:** Linear (for strongly convex)
- **Momentum:** Linear (faster constant)
- **Conjugate Gradient:** Superlinear (quadratic: finite steps)
- **BFGS/L-BFGS:** Superlinear
- **Adam:** Sublinear to linear (problem-dependent)

---

## Common Test Functions

All scripts include examples using these classic test functions:

### Rosenbrock Function
```
f(x, y) = (1-x)² + 100(y-x²)²
Minimum: (1, 1), f* = 0
```
- Narrow valley makes it challenging
- Classic test for optimization algorithms

### Beale Function
```
f(x, y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
Minimum: (3, 0.5), f* = 0
```
- Ravines and valleys
- Tests momentum advantages

### Rastrigin Function
```
f(x) = 10n + Σ(x² - 10cos(2πx))
Minimum: (0, ..., 0), f* = 0
```
- Highly multimodal
- Tests ability to escape local minima

---

## Running the Examples

Each file can be run independently:

```bash
# Activate your virtual environment
source ~/github/ewdhp/python/venv/bin/activate

# Run individual examples
python steepest_descent.py
python conjugate_gradient.py
python quasi_newton.py
python momentum_methods.py
python adam_optimizer.py
```

Each script includes:
1. Multiple examples demonstrating key features
2. Comparison between variants
3. Visualizations of trajectories and convergence
4. Performance analysis

---

## Visualization Features

All methods include visualization functions that show:
- **Contour plots** with optimization trajectories
- **Convergence curves** (function value vs iterations)
- **Gradient norm evolution**
- **Method-specific metrics** (velocity, step sizes, etc.)

---

## Dependencies

Required packages:
```bash
pip install numpy matplotlib scipy
```

Optional for enhanced visualizations:
```bash
pip install seaborn
```

---

## Tips and Best Practices

### 1. Always Start Simple
Begin with steepest descent or momentum to understand the problem landscape.

### 2. Check Gradients
Verify gradient implementation using finite differences:
```python
def numerical_gradient(f, x, epsilon=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += epsilon
        x_minus = x.copy()
        x_minus[i] -= epsilon
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    return grad
```

### 3. Monitor Convergence
Always check:
- Function value decreasing
- Gradient norm approaching zero
- No NaN or Inf values

### 4. Hyperparameter Tuning
- Start with default values
- If not converging: reduce learning rate
- If too slow: increase learning rate or try different method

### 5. Restart Strategies
For non-convex problems, try multiple starting points.

---

## Mathematical Background

### Gradient Descent Framework
All methods follow the general update:
```
x_{k+1} = x_k + α_k d_k
```

Where:
- `x_k`: current point
- `α_k`: step size (from line search)
- `d_k`: search direction (method-dependent)

### Search Directions
- **Steepest Descent:** `d_k = -∇f(x_k)`
- **Momentum:** `d_k = -∇f(x_k) + β d_{k-1}`
- **Conjugate Gradient:** `d_k = -∇f(x_k) + β_k d_{k-1}` (with special β)
- **Quasi-Newton:** `d_k = -B_k^{-1} ∇f(x_k)` (B_k ≈ Hessian)
- **Adam:** `d_k = -m̂_k / (√v̂_k + ε)` (adaptive)

---

## References

### Books
1. Nocedal & Wright - "Numerical Optimization" (2006)
2. Boyd & Vandenberghe - "Convex Optimization" (2004)

### Papers
1. Kingma & Ba - "Adam: A Method for Stochastic Optimization" (2014)
2. Polyak - "Some methods of speeding up the convergence of iteration methods" (1964)
3. Fletcher & Reeves - "Function minimization by conjugate gradients" (1964)
4. Broyden-Fletcher-Goldfarb-Shanno - BFGS papers (1970)

---

## Contributing

Feel free to:
- Add new test functions
- Implement additional variants
- Improve visualizations
- Add more examples

---

**Repository:** [ewdhp/optimization](https://github.com/ewdhp/optimization)  
**Path:** `2-core-methods/unconstrained/gradient-methods/gradients/`

**Last Updated:** October 2025
