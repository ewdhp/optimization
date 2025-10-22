# Convergence Theory for Gradient Methods

This directory contains theoretical analysis tools for understanding and proving convergence properties of gradient-based optimization methods.

## Files

### convergence_analysis.py
**Comprehensive Convergence Analysis Toolkit**

A powerful tool for analyzing convergence properties of optimization algorithms.

#### Features

**1. Lipschitz Constant Estimation**
- Estimates the Lipschitz constant L of ∇f
- Uses gradient sampling at random points
- Essential for determining step sizes

**2. Strong Convexity Parameter Estimation**
- Estimates the strong convexity parameter μ
- Computes condition number κ = L/μ
- Predicts convergence rate

**3. Descent Lemma Verification**
- Verifies: f(y) ≤ f(x) + ∇f(x)ᵀ(y-x) + (L/2)||y-x||²
- Essential for proving convergence
- Visualizes the quadratic upper bound

**4. Theoretical Convergence Rate Analysis**
- Computes theoretical convergence rates
- For strongly convex: linear rate ρ = (κ-1)/(κ+1)
- For convex: sublinear rate O(1/k)
- Predicts iterations needed for ε-accuracy

**5. Optimization Run Analysis**
- Analyzes actual optimization trajectories
- Computes empirical convergence rates
- Compares theory vs practice
- Generates comprehensive convergence plots

#### Class: `ConvergenceAnalysis`

```python
from convergence_analysis import ConvergenceAnalysis

# Create analyzer
analyzer = ConvergenceAnalysis(
    objective=f,
    gradient=grad_f,
    hessian=hess_f,  # optional
    name="My Function"
)

# Estimate Lipschitz constant
L = analyzer.estimate_lipschitz_constant(x0, n_samples=1000)

# Estimate strong convexity parameter
mu = analyzer.estimate_strong_convexity(x0, n_samples=1000)

# Verify descent lemma
is_valid, violations = analyzer.verify_descent_lemma(x0, n_tests=100)

# Theoretical convergence rate
theory = analyzer.theoretical_convergence_rate(mu, L, method='gradient_descent')
print(f"Condition number: {theory['condition_number']}")
print(f"Linear rate: {theory['linear_rate']}")
print(f"Iterations to 1e-6: {theory['iterations_to_1e-6']}")

# Analyze optimization run
history = [...]  # from your optimizer
analysis = analyzer.analyze_optimization_run(history, x_optimal)
```

#### Examples Included

1. **Well-conditioned Quadratic** (κ = 2)
   - Fast convergence
   - Theoretical predictions match practice

2. **Ill-conditioned Quadratic** (κ = 100)
   - Slow convergence
   - Demonstrates effect of conditioning

3. **Non-quadratic Functions**
   - Rosenbrock function
   - Shows limitations of quadratic theory

#### Visualizations

The tool generates comprehensive plots:
- **Convergence curves** (log scale)
- **Gradient norm evolution**
- **Step size adaptation**
- **Descent lemma verification**
- **Quadratic approximation quality**
- **Theoretical vs empirical rates**

#### Key Theorems Demonstrated

**Descent Lemma (L-smoothness):**
```
f(y) ≤ f(x) + ∇f(x)ᵀ(y-x) + (L/2)||y-x||²
```

**Convergence Rate (Strongly Convex + L-smooth):**
```
||x_k - x*||² ≤ ρᵏ ||x_0 - x*||²
where ρ = (κ-1)/(κ+1), κ = L/μ
```

**Iterations for ε-accuracy:**
```
k ≥ log(||x_0 - x*||/ε) / log(1/ρ)
```

#### Running the Examples

```bash
python convergence_analysis.py
```

This will:
1. Analyze well-conditioned quadratic
2. Analyze ill-conditioned quadratic
3. Display convergence theory summary
4. Generate comparison plots

#### Output

The script produces:
- **Console output** with numerical results
- **Plots** showing convergence behavior
- **Theory summary** with key formulas

Example output:
```
=================================================================
Convergence Analysis: Well-Conditioned Quadratic
=================================================================

Lipschitz Constant (L): 2.00
Strong Convexity (μ): 1.00
Condition Number (κ): 2.00

Theoretical Convergence Rate:
  Linear rate (ρ): 0.333
  Iterations to 1e-6 accuracy: 13
  Iterations to 1e-9 accuracy: 19

Empirical Analysis:
  Actual iterations: 14
  Empirical linear rate: 0.341
  Theory vs Practice: 93% match
```

---

## Mathematical Background

### Definitions

**L-smooth function:**
```
||∇f(x) - ∇f(y)|| ≤ L||x - y|| for all x, y
```

**μ-strongly convex function:**
```
f(y) ≥ f(x) + ∇f(x)ᵀ(y-x) + (μ/2)||y-x||² for all x, y
```

**Condition number:**
```
κ = L/μ
```
- κ = 1: perfectly conditioned (sphere)
- κ > 1: ill-conditioned (ellipsoid)
- κ → ∞: poorly conditioned

### Convergence Rates

| Function Class | Rate | Notation |
|----------------|------|----------|
| Strongly Convex + Smooth | Linear | O(ρᵏ), ρ < 1 |
| Convex + Smooth | Sublinear | O(1/k) |
| Non-convex | None guaranteed | - |

---

## Use Cases

### 1. Algorithm Design
- Choose appropriate step sizes
- Predict performance before running
- Compare different methods theoretically

### 2. Debugging
- Verify implementation correctness
- Check if algorithm matches theory
- Identify numerical issues

### 3. Performance Prediction
- Estimate runtime for desired accuracy
- Compare methods on your problem
- Justify algorithm choice

### 4. Education
- Understand convergence proofs
- Visualize theoretical concepts
- Connect theory to practice

---

## Prerequisites

```bash
pip install numpy matplotlib scipy
```

---

## Related

- For **implementations**, see `../gradients/`
- For **simple examples**, see `../examples/`
- For **line search theory**, see `../line-search/`

---

## References

1. **Nesterov, Y.** - "Introductory Lectures on Convex Optimization" (2004)
2. **Nocedal & Wright** - "Numerical Optimization" (2006)
3. **Boyd & Vandenberghe** - "Convex Optimization" (2004)
4. **Polyak, B.** - "Introduction to Optimization" (1987)

---

## Tips

1. **Always estimate L and μ** before optimizing
2. **Use κ to predict** convergence speed
3. **Verify descent lemma** to check L-smoothness
4. **Compare theory vs practice** to validate implementation
5. **Plot convergence curves** on log scale to see linear rates

---

**Last Updated:** October 2025
