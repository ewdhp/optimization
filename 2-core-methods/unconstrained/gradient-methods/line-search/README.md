# Line Search Methods: Theory and Implementation

## Table of Contents
- [Introduction](#introduction)
- [The Big Picture](#the-big-picture)
- [Quick Reference Guide](#quick-reference-guide)
- [Theoretical Foundations](#theoretical-foundations)
- [Line Search Framework](#line-search-framework)
- [Exact Line Search](#exact-line-search)
- [Inexact Line Search Methods](#inexact-line-search-methods)
- [Wolfe Conditions](#wolfe-conditions)
- [Goldstein Conditions](#goldstein-conditions)
- [Backtracking Line Search](#backtracking-line-search)
- [Convergence Theory](#convergence-theory)
- [Practical Considerations](#practical-considerations)

---

## Introduction

Line search is a fundamental component of iterative optimization algorithms for unconstrained optimization problems. Given a current point **x**ₖ and a search direction **d**ₖ, line search determines an appropriate step size αₖ such that:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{d}_k
$$

The quality of the step size significantly impacts the convergence rate and computational efficiency of optimization algorithms.

### The Fundamental Question

**How much should we move along the search direction?**

This seemingly simple question has profound implications:
- Too small: slow convergence, wasted iterations
- Too large: overshooting, instability, divergence
- Just right: fast convergence, efficient optimization

---

## The Big Picture

### Why Do We Need Line Search?

Imagine you're hiking down a mountain in fog. You have:
1. **Your current position** (xₖ)
2. **A compass direction** (dₖ - the descent direction)
3. **The question**: How far should you walk in that direction?

**Without line search**: You might take huge steps and fall off a cliff, or tiny steps and never reach the bottom.

**With line search**: You intelligently determine each step size to efficiently reach the valley.

### The Core Problem

In optimization, we iterate:
1. **Choose a direction** dₖ (e.g., negative gradient)
2. **Choose a step size** αₖ ← **LINE SEARCH SOLVES THIS**
3. **Update position** xₖ₊₁ = xₖ + αₖdₖ
4. **Repeat** until convergence

### Main Objectives of Line Search

```
┌─────────────────────────────────────────────────────────────┐
│  PRIMARY GOAL: Find αₖ that sufficiently decreases f        │
│                                                              │
│  SECONDARY GOALS:                                           │
│  1. Fast computation (few function evaluations)             │
│  2. Guarantee global convergence                            │
│  3. Maintain good convergence rate                          │
│  4. Be numerically stable                                   │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: We don't need the *perfect* step size, just a *good enough* one!

---

## Quick Reference Guide

### When to Use Which Method?

```
┌──────────────────────┬─────────────────┬──────────────────┬─────────────┐
│ Method               │ Best For        │ Cost             │ Reliability │
├──────────────────────┼─────────────────┼──────────────────┼─────────────┤
│ Backtracking         │ General purpose │ Low (1-3 evals)  │ High        │
│ (Armijo)             │ First choice    │ Function only    │ ⭐⭐⭐⭐⭐      │
├──────────────────────┼─────────────────┼──────────────────┼─────────────┤
│ Wolfe Conditions     │ Quasi-Newton    │ Medium (2-6)     │ Very High   │
│                      │ methods (BFGS)  │ Func + gradient  │ ⭐⭐⭐⭐⭐      │
├──────────────────────┼─────────────────┼──────────────────┼─────────────┤
│ Strong Wolfe         │ Conjugate       │ Medium (2-6)     │ Very High   │
│                      │ gradient        │ Func + gradient  │ ⭐⭐⭐⭐⭐      │
├──────────────────────┼─────────────────┼──────────────────┼─────────────┤
│ Exact Line Search    │ Quadratics      │ High (many)      │ Medium      │
│                      │ Special cases   │ Func + gradient  │ ⭐⭐⭐        │
├──────────────────────┼─────────────────┼──────────────────┼─────────────┤
│ Goldstein            │ Rarely used     │ Medium           │ Medium      │
│                      │ (historical)    │ Function only    │ ⭐⭐         │
└──────────────────────┴─────────────────┴──────────────────┴─────────────┘
```

### Decision Flowchart

```
Start with direction dₖ
         ↓
    Need simple & fast? ──YES──> Backtracking (Armijo)
         ↓ NO
         ↓
    Using BFGS/L-BFGS? ──YES──> Wolfe Conditions (c₂=0.9)
         ↓ NO
         ↓
    Using CG method? ────YES──> Strong Wolfe (c₂=0.1)
         ↓ NO
         ↓
    Quadratic function? ─YES──> Exact Line Search
         ↓ NO
         ↓
    Default: Backtracking
```

### Algorithm Comparison at a Glance

| Aspect | Backtracking | Wolfe | Exact |
|--------|-------------|-------|-------|
| **Main idea** | Shrink α until f decreases enough | f decreases AND gradient flattens | Find exact minimum along line |
| **Guarantees** | Sufficient decrease | Decrease + curvature | Optimal α on line |
| **Evaluations** | 1-3 per iteration | 2-6 per iteration | Many (solve 1D problem) |
| **Convergence** | Good for GD | Excellent for BFGS | Excellent but expensive |
| **Implementation** | Very simple | Moderate | Complex |
| **When to use** | Starting point, simple cases | Production code | Special structure |

---

## Theoretical Foundations

### The Descent Direction

A direction **d**ₖ is a **descent direction** at **x**ₖ if:

$$
\nabla f(\mathbf{x}_k)^T \mathbf{d}_k < 0
$$

This ensures that for sufficiently small positive α, we have:

$$
f(\mathbf{x}_k + \alpha \mathbf{d}_k) < f(\mathbf{x}_k)
$$

**Proof intuition**: By Taylor's theorem,

$$
f(\mathbf{x}_k + \alpha \mathbf{d}_k) = f(\mathbf{x}_k) + \alpha \nabla f(\mathbf{x}_k)^T \mathbf{d}_k + O(\alpha^2)
$$

For small α > 0 and ∇f(**x**ₖ)ᵀ**d**ₖ < 0, the linear term dominates, ensuring decrease.

### The One-Dimensional Problem

Line search converts the n-dimensional optimization problem into a sequence of one-dimensional problems:

$$
\min_{\alpha > 0} \phi(\alpha) = f(\mathbf{x}_k + \alpha \mathbf{d}_k)
$$

This function φ(α) is called the **merit function** or **line search function**.

#### Properties of φ(α)

1. **Differentiability**: If f is continuously differentiable,
   $$
   \phi'(\alpha) = \nabla f(\mathbf{x}_k + \alpha \mathbf{d}_k)^T \mathbf{d}_k
   $$

2. **Initial Slope**: 
   $$
   \phi'(0) = \nabla f(\mathbf{x}_k)^T \mathbf{d}_k < 0
   $$
   (negative for descent directions)

3. **Curvature**: 
   $$
   \phi''(\alpha) = \mathbf{d}_k^T \nabla^2 f(\mathbf{x}_k + \alpha \mathbf{d}_k) \mathbf{d}_k
   $$

---

## Line Search Framework

### General Algorithm Structure

```
Input: x_k (current point), d_k (search direction), f (objective function)
Output: α_k (step size)

1. Initialize: α = α_init (e.g., α = 1)
2. Evaluate: φ(α) = f(x_k + α·d_k)
3. Check acceptance criteria
4. If criteria satisfied: return α
5. Else: Update α (reduce or increase) and goto step 2
```

### Visual Understanding: The φ(α) Function

Think of φ(α) as a **1D slice** of your n-dimensional function along direction dₖ:

```
φ(α) = f(xₖ + α·dₖ)

     f
     ↑
     │     Current point
     │        ↓
     │        ●
     │       ╱ ╲              ← This is φ(α)
     │      ╱   ╲             We search along this curve
     │     ╱     ╲___
     │    ╱          ╲___
     │___╱               ╲___
     └─────────────────────────→ α
     0                         
     
     Goal: Find α where φ(α) is small enough
```

**Key Insight**: Instead of searching in n dimensions, we reduce it to searching along a line (1D)!

#### Interactive Plot: Line Search Visualization

```python {cmd=true matplotlib=true}
import numpy as np
import matplotlib.pyplot as plt

# Define φ(α) for our example: φ(α) = 8 - 80α + 272α²
alpha = np.linspace(0, 0.3, 1000)
phi = 8 - 80*alpha + 272*alpha**2

# Exact minimum
alpha_exact = 80/544
phi_exact = 8 - 80*alpha_exact + 272*alpha_exact**2

# Armijo condition line (c=0.1)
phi_0 = 8
phi_prime_0 = -80
armijo_line = phi_0 + 0.1 * alpha * phi_prime_0

# Backtracking accepted point
alpha_backtrack = 0.125
phi_backtrack = 8 - 80*alpha_backtrack + 272*alpha_backtrack**2

plt.figure(figsize=(10, 6))
plt.plot(alpha, phi, 'b-', linewidth=2, label='φ(α) = f(xₖ + α·dₖ)')
plt.plot(alpha, armijo_line, 'r--', linewidth=1.5, label='Armijo: φ(0) + c·α·φ\'(0)')
plt.plot(alpha_exact, phi_exact, 'go', markersize=10, label=f'Exact min (α={alpha_exact:.3f})')
plt.plot(alpha_backtrack, phi_backtrack, 'rs', markersize=10, label=f'Backtracking (α={alpha_backtrack})')
plt.axhline(y=phi_0, color='gray', linestyle=':', alpha=0.5, label='φ(0)')

plt.xlabel('Step size α', fontsize=12)
plt.ylabel('φ(α)', fontsize=12)
plt.title('Line Search: Finding the Right Step Size', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(0, 0.3)
plt.ylim(0, 10)

# Annotate regions
plt.fill_between(alpha[alpha < 0.05], 0, 10, alpha=0.1, color='red', label='Too small α')
plt.text(0.025, 9, 'Too small\n(slow progress)', ha='center', fontsize=9, color='red')
plt.text(0.2, 8, 'Acceptable region\n(below Armijo line)', ha='center', fontsize=9, color='green')

plt.tight_layout()
plt.show()
```

> **Note**: This plot will render interactively if you have Markdown Preview Enhanced installed and Python configured!

### Classification of Line Search Methods

#### Philosophy Comparison

```
┌────────────────────────────────────────────────────────────┐
│                                                             │
│  EXACT LINE SEARCH: "Find the absolute best α"             │
│  • Perfectionist approach                                  │
│  • Solves: min φ(α) exactly                                │
│  • Cost: High (many evaluations)                           │
│  • Benefit: Minimal (usually not worth it)                 │
│                                                             │
│  ─────────────────────────────────────────────────────     │
│                                                             │
│  INEXACT LINE SEARCH: "Find a good enough α"               │
│  • Pragmatic approach                                      │
│  • Uses: Acceptance criteria                               │
│  • Cost: Low (few evaluations)                             │
│  • Benefit: High (fast overall convergence)                │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

1. **Exact Line Search**
   - Find global minimum: α* = argmin_α φ(α)
   - Computationally expensive
   - Rarely used in practice
   - **Analogy**: Measuring ingredients to 0.001g precision for a casual meal

2. **Inexact Line Search**
   - Find "good enough" step size
   - Uses acceptance conditions
   - Computationally efficient
   - Most common in practice
   - **Analogy**: Measuring "approximately 1 cup" - good enough for great results!

---

## Exact Line Search

### Definition

Find αₖ that minimizes φ(α):

$$
\alpha_k = \arg\min_{\alpha > 0} f(\mathbf{x}_k + \alpha \mathbf{d}_k)
$$

### Optimality Condition

At the minimum, the first-order optimality condition requires:

$$
\phi'(\alpha_k) = \nabla f(\mathbf{x}_k + \alpha_k \mathbf{d}_k)^T \mathbf{d}_k = 0
$$

This means the gradient at the new point is **orthogonal** to the search direction.

### Geometric Interpretation

Exact line search finds the point where the gradient becomes perpendicular to the search direction. This creates a "zigzag" pattern in steepest descent methods.

### Computational Cost

- Requires solving a one-dimensional optimization problem per iteration
- May need multiple function and gradient evaluations
- Often more expensive than the benefit gained
- **Key insight**: Exact minimization is usually unnecessary for global convergence

### When to Use

Exact line search may be beneficial when:
- φ(α) has a simple closed form
- Gradient evaluations are very expensive
- High precision is required
- The dimension n is small

---

## Inexact Line Search Methods

Inexact methods find a step size that provides **sufficient decrease** in the objective function without requiring exact minimization.

### Key Principle

We don't need the best step size, just a good one that:
1. Reduces the function value sufficiently
2. Doesn't require excessive computation
3. Ensures global convergence

---

## Wolfe Conditions

The **Wolfe conditions** are the most widely used criteria for inexact line search. They consist of two conditions that ensure both sufficient decrease and adequate progress.

### Intuitive Understanding

**The Problem with Just Checking Decrease:**
If we only check "did f decrease?", we might accept α = 0.000001 (tiny step) because technically f decreased. This wastes iterations!

**Wolfe's Brilliant Solution:**
Check TWO things:
1. **Armijo (Sufficient Decrease)**: Did we decrease enough? (not tiny steps)
2. **Curvature**: Did we go far enough? (not stopping too early)

**Analogy**: 
- Armijo alone: "Did you save money?" (Yes, I saved $0.01)
- Wolfe adds: "Did you save a meaningful amount?" (Now we're talking!)

### Visual Intuition

```
φ(α)
  ↑
  │   ╱────╲        Armijo line: φ(0) + c₁·α·φ'(0)
  │  ╱      ╲       (must stay below this)
  │ ╱ ✗  ✓   ╲      
  │╱  ×  ●    ╲     ● = acceptable α
  ●───────────────→ α
  │   ✗: Armijo fails (didn't decrease enough)
  │   ✓: Both conditions satisfied
  │   
  Curvature condition ensures slope has flattened
```

#### Interactive Visualization (Mermaid Flowchart)

```mermaid
graph TD
    A[Start: α = α_init] --> B{φ(α) ≤ φ(0) + c₁·α·φ'(0)?}
    B -->|No: Armijo fails| C[Reduce: α = ρ·α]
    C --> B
    B -->|Yes: Armijo OK| D{Using Wolfe?}
    D -->|No| E[Accept α]
    D -->|Yes| F{φ'(α) ≥ c₂·φ'(0)?}
    F -->|No: Curvature fails| G[Increase α or try new α]
    G --> B
    F -->|Yes: Both OK| E[Accept α]
    E --> H[Return α]
```

### Sufficient Decrease Condition (Armijo Condition)

$$
f(\mathbf{x}_k + \alpha \mathbf{d}_k) \leq f(\mathbf{x}_k) + c_1 \alpha \nabla f(\mathbf{x}_k)^T \mathbf{d}_k
$$

where 0 < c₁ < 1 (typically c₁ = 10⁻⁴).

**Interpretation**: The function value should decrease by at least a fraction c₁ of the predicted decrease from the linear model.

**Geometric view**: The graph of φ(α) must lie below the line:
$$
\ell(\alpha) = \phi(0) + c_1 \alpha \phi'(0)
$$

### Curvature Condition

$$
\nabla f(\mathbf{x}_k + \alpha \mathbf{d}_k)^T \mathbf{d}_k \geq c_2 \nabla f(\mathbf{x}_k)^T \mathbf{d}_k
$$

where c₁ < c₂ < 1 (typically c₂ = 0.9 for Newton methods, c₂ = 0.1 for conjugate gradient).

**Interpretation**: The slope at α must be less steep (in magnitude) than the initial slope scaled by c₂. This prevents accepting tiny steps.

**Rewritten form**:
$$
\phi'(\alpha) \geq c_2 \phi'(0)
$$

Since φ'(0) < 0, this says |φ'(α)| ≤ c₂|φ'(0)|, ensuring the slope has flattened sufficiently.

### Strong Wolfe Conditions

A stronger variant replaces the curvature condition with:

$$
\left| \nabla f(\mathbf{x}_k + \alpha \mathbf{d}_k)^T \mathbf{d}_k \right| \leq c_2 \left| \nabla f(\mathbf{x}_k)^T \mathbf{d}_k \right|
$$

or equivalently:
$$
|\phi'(\alpha)| \leq c_2 |\phi'(0)|
$$

This prevents acceptance of step sizes in regions where φ(α) is increasing too rapidly.

### Why Wolfe Conditions Work

1. **Sufficient Decrease**: Ensures we make progress (not just tiny steps)
2. **Curvature**: Ensures we don't stop too early (not just accepting the first decrease)
3. **Together**: Balance between decrease and step size
4. **Theoretical**: Guarantee convergence for many algorithms

### Existence of Wolfe Points

**Theorem**: For a smooth function f that is bounded below along **x**_k + α**d**_k for all α ≥ 0, there exist step sizes satisfying the Wolfe conditions.

**Proof sketch**:
- By smoothness and boundedness, φ(α) → constant as α → ∞
- By mean value theorem, ∃α where φ'(α) ≈ 0
- The sufficient decrease condition is satisfied for small α
- Continuity guarantees existence of α satisfying both conditions

---

## Goldstein Conditions

An alternative to Wolfe conditions, the **Goldstein conditions** require:

$$
f(\mathbf{x}_k) + (1-c)\alpha \nabla f(\mathbf{x}_k)^T \mathbf{d}_k \leq f(\mathbf{x}_k + \alpha \mathbf{d}_k) \leq f(\mathbf{x}_k) + c\alpha \nabla f(\mathbf{x}_k)^T \mathbf{d}_k
$$

where 0 < c < 1/2 (typically c = 0.25).

### Interpretation

The Goldstein conditions create a **band** around the linear approximation:
- Upper bound: sufficient decrease (like Armijo)
- Lower bound: prevents too much decrease (too small steps)

### Comparison with Wolfe

**Advantages**:
- Symmetric conditions
- Easier to understand geometrically
- May reject step sizes where φ' is large and negative

**Disadvantages**:
- Lower bound can exclude the exact minimum
- May not be suitable for all algorithms (e.g., quasi-Newton methods)
- Less commonly used than Wolfe conditions

---

## Backtracking Line Search

The **backtracking** (or **Armijo backtracking**) method is one of the simplest and most practical line search algorithms.

### Algorithm

```python
def backtracking_line_search(f, grad_f, x_k, d_k, alpha_init=1.0, rho=0.5, c=1e-4):
    """
    Backtracking line search with Armijo condition.
    
    Parameters:
    - f: objective function
    - grad_f: gradient function
    - x_k: current point
    - d_k: search direction
    - alpha_init: initial step size
    - rho: contraction factor (0 < rho < 1)
    - c: Armijo parameter (0 < c < 1)
    
    Returns:
    - alpha: step size satisfying Armijo condition
    """
    alpha = alpha_init
    f_k = f(x_k)
    grad_k = grad_f(x_k)
    
    # Sufficient decrease threshold
    threshold = c * np.dot(grad_k, d_k)
    
    while f(x_k + alpha * d_k) > f_k + alpha * threshold:
        alpha *= rho  # Reduce step size
    
    return alpha
```

### Key Parameters

1. **α_init** (initial step size)
   - Often α_init = 1 for Newton-like methods
   - May scale based on previous iterations

2. **ρ** (contraction factor)
   - Typical value: ρ = 0.5 (halving)
   - Range: 0.1 ≤ ρ ≤ 0.8
   - Smaller ρ: faster reduction but more iterations
   - Larger ρ: slower reduction but fewer iterations

3. **c** (Armijo parameter)
   - Typical value: c = 10⁻⁴
   - Range: 10⁻⁵ ≤ c ≤ 0.5
   - Controls how much decrease is required

### Advantages

- **Simple**: Easy to implement and understand
- **Fast**: Typically converges in few iterations
- **Robust**: Works well in practice
- **No derivatives of φ**: Only requires function evaluations

### Disadvantages

- Only checks sufficient decrease (not curvature)
- May accept very small steps
- No guarantee of finding Wolfe points
- May be inefficient if α_init is far from optimal

### Convergence Properties

**Theorem**: For a continuously differentiable function f with bounded level sets and using a descent direction, backtracking line search terminates after a finite number of iterations.

**Proof outline**:
- By descent property, φ'(0) < 0
- By continuity, φ(α) < φ(0) + cα·φ'(0) for sufficiently small α
- Bounded level sets prevent φ from decreasing indefinitely
- Therefore, the Armijo condition is eventually satisfied

---

## Convergence Theory

### Global Convergence

A line search algorithm has **global convergence** if:

$$
\lim_{k \to \infty} \|\nabla f(\mathbf{x}_k)\| = 0
$$

starting from any initial point **x**₀.

### Zoutendijk's Theorem

**Theorem** (Zoutendijk, 1970): Suppose f is bounded below and continuously differentiable in ℝⁿ. Consider any iteration **x**ₖ₊₁ = **x**ₖ + αₖ**d**ₖ where **d**ₖ is a descent direction and αₖ satisfies the Wolfe conditions. Then:

$$
\sum_{k=0}^{\infty} \cos^2\theta_k \|\nabla f(\mathbf{x}_k)\|^2 < \infty
$$

where cos θₖ = -∇f(**x**ₖ)^T**d**ₖ / (‖∇f(**x**ₖ)‖ · ‖**d**ₖ‖).

**Implications**:
1. If cos²θₖ is bounded away from zero (bounded angle condition), then ‖∇f(**x**ₖ)‖ → 0
2. Steepest descent satisfies this with cos θₖ = 1
3. Guarantees convergence for many gradient-based methods

### Rate of Convergence

#### Linear Convergence

An algorithm has **linear convergence** if:

$$
\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \leq \mu \|\mathbf{x}_k - \mathbf{x}^*\|
$$

for some 0 < μ < 1 (convergence factor).

**Steepest descent with exact line search**:
- Linear convergence rate
- Convergence factor depends on condition number κ:
  $$
  \mu \leq \frac{\kappa - 1}{\kappa + 1}
  $$
  where κ = λ_max/λ_min (ratio of largest to smallest eigenvalue)

#### Superlinear and Quadratic Convergence

- **Superlinear**: μ_k → 0 as k → ∞
- **Quadratic**: ‖**x**_{k+1} - **x**\*‖ ≤ C‖**x**_k - **x**\*‖²

Newton's method with appropriate line search achieves quadratic convergence near the solution.

---

## Practical Considerations

### Choosing Initial Step Size

1. **Newton-like methods**: α_init = 1
   - Trust the second-order model
   - Often accepted immediately near solution

2. **Steepest descent**: 
   - Scale based on previous step: α_init = α_{k-1} · ‖**d**_{k-1}‖ / ‖**d**_k‖
   - Or use previous step size with adjustment

3. **First iteration**: 
   - Estimate based on function scale
   - α_init ≈ 1/‖∇f(**x**₀)‖ for steepest descent

### Parameter Tuning

**Wolfe parameters**:
- c₁ = 10⁻⁴ (sufficient decrease)
- c₂ = 0.9 (for Newton/quasi-Newton)
- c₂ = 0.1 (for conjugate gradient)

**Backtracking parameters**:
- ρ = 0.5 (contraction factor)
- c = 10⁻⁴ (Armijo parameter)

### Computational Costs

**Function evaluations**:
- Each line search requires multiple f evaluations
- Typical: 1-5 evaluations per iteration
- Trade-off: accuracy vs. computational cost

**Gradient evaluations**:
- Wolfe conditions require gradient evaluations
- More expensive than function-only methods
- But often worth the cost for better convergence

### Safeguards

1. **Maximum iterations**: Prevent infinite loops
2. **Minimum step size**: Detect stagnation (α_min ≈ 10⁻¹⁰)
3. **Maximum step size**: Prevent overflow (α_max ≈ 10¹⁰)
4. **Gradient checks**: Ensure descent direction

### Common Pitfalls

1. **Too strict conditions**: May fail to find acceptable α
2. **Too loose conditions**: May accept poor step sizes
3. **Poor initial guess**: Wasted evaluations
4. **Numerical errors**: Check for overflow/underflow
5. **Non-smooth functions**: Line search may fail

---

## Complete Example Walkthrough

### Simple 2D Example: Minimizing f(x,y) = x² + 4y²

Let's see how line search works in practice!

**Setup:**
- Current point: xₖ = [2, 1]
- Objective: f(x,y) = x² + 4y²
- Gradient: ∇f = [2x, 8y]
- At xₖ: ∇f(xₖ) = [4, 8]
- Direction: dₖ = -∇f(xₖ) = [-4, -8] (steepest descent)

#### Step 1: Define φ(α)

$$
\phi(\alpha) = f(x_k + \alpha d_k) = f([2,1] + \alpha[-4,-8])
$$
$$
= f([2-4\alpha, 1-8\alpha])
$$
$$
= (2-4\alpha)^2 + 4(1-8\alpha)^2
$$
$$
= 4 - 16\alpha + 16\alpha^2 + 4 - 64\alpha + 256\alpha^2
$$
$$
= 8 - 80\alpha + 272\alpha^2
$$

#### Step 2: Different Methods Apply

**Exact Line Search:**
```
Find α* = argmin φ(α)
φ'(α) = -80 + 544α = 0
α* = 80/544 ≈ 0.147

Result: x₁ = [2,1] + 0.147[-4,-8] = [1.412, -0.176]
Function value: f(x₁) ≈ 2.12 (vs f(x₀) = 8)
```

**Backtracking (α_init=1, ρ=0.5, c=0.1):**
```
α = 1.0: φ(1) = 200 > φ(0) + 0.1·1·φ'(0) = 8 - 8 = 0 ✗ Reject
α = 0.5: φ(0.5) = 50 > 4 ✗ Reject  
α = 0.25: φ(0.25) = 15 > 6 ✗ Reject
α = 0.125: φ(0.125) ≈ 4.45 < 7 ✓ Accept!

Result: x₁ = [1.5, 0] 
```

**Key Observation**: Backtracking accepted α=0.125, exact found α≈0.147. Close enough! Backtracking took 4 function evaluations vs. solving an optimization problem.

### Understanding Through Iteration

Here's what happens over multiple iterations with gradient descent + backtracking:

```
Iteration 0: x₀ = [2.0, 1.0],   f = 8.00
             ↓ (α=0.125, direction=[-4,-8])
Iteration 1: x₁ = [1.5, 0.0],   f = 2.25
             ↓ (α=0.25, direction=[-3,0])  
Iteration 2: x₂ = [0.75, 0.0],  f = 0.56
             ↓ (α=0.25, direction=[-1.5,0])
Iteration 3: x₃ = [0.375, 0.0], f = 0.14
             ↓
    ...converges to [0,0]
```

**What We Learn:**
1. Line search adapts α each iteration
2. Direction changes (gradient updates)
3. Both direction AND step size matter
4. Converges to minimum [0,0] where f=0

#### Convergence Visualization

```python {cmd=true matplotlib=true}
import numpy as np
import matplotlib.pyplot as plt

# Function: f(x,y) = x² + 4y²
def f(x, y):
    return x**2 + 4*y**2

# Gradient descent with backtracking line search
def gradient_descent_line_search(x0, max_iter=20):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    alphas = []
    
    for i in range(max_iter):
        # Gradient
        grad = np.array([2*x[0], 8*x[1]])
        
        # Direction (steepest descent)
        d = -grad
        
        # Backtracking line search
        alpha = 1.0
        rho = 0.5
        c = 0.1
        f_x = f(x[0], x[1])
        grad_d = np.dot(grad, d)
        
        while f(x[0] + alpha*d[0], x[1] + alpha*d[1]) > f_x + c*alpha*grad_d:
            alpha *= rho
        
        alphas.append(alpha)
        x = x + alpha * d
        path.append(x.copy())
        
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return np.array(path), alphas

# Run optimization
path, alphas = gradient_descent_line_search([2.0, 1.0])

# Create contour plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Convergence path on contour
x = np.linspace(-2.5, 2.5, 100)
y = np.linspace(-1.5, 1.5, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + 4*Y**2

ax1.contour(X, Y, Z, levels=20, alpha=0.6, cmap='viridis')
ax1.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=8, label='Optimization path')
ax1.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='Start')
ax1.plot(path[-1, 0], path[-1, 1], 'r*', markersize=15, label='End')
ax1.plot(0, 0, 'b*', markersize=15, label='True minimum')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Gradient Descent with Line Search', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Right plot: Step sizes over iterations
ax2.semilogy([f(p[0], p[1]) for p in path], 'b-o', linewidth=2, markersize=6)
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('f(x) [log scale]', fontsize=12)
ax2.set_title('Function Value Convergence', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Converged in {len(path)-1} iterations")
print(f"Final point: [{path[-1, 0]:.6f}, {path[-1, 1]:.6f}]")
print(f"Final value: {f(path[-1, 0], path[-1, 1]):.6e}")
```

> **Interactive plots**: These will render when you open the preview in Markdown Preview Enhanced!

---

## Understanding Each Method: A Practical Guide

### 🎯 Backtracking: The Workhorse

**Main Idea**: Start big, shrink until acceptable.

**When it "clicks"**: Think of adjusting your car's speed:
- Try 60 mph
- Too fast for this curve? Try 30 mph
- Still too fast? Try 15 mph
- Good! Proceed at 15 mph

**Code intuition**:
```python
alpha = 1.0  # Start optimistic
while f(x + alpha*d) > f(x) + c*alpha*grad.dot(d):
    alpha *= 0.5  # Too big, shrink it
return alpha  # Found it!
```

**Pro tip**: Almost always your starting point. Works 90% of the time.

### 🎯 Wolfe Conditions: The Professional

**Main Idea**: Decrease enough AND go far enough.

**When it "clicks"**: Like tuning a guitar string:
- Too loose: doesn't make sound (tiny steps)
- Too tight: might break (overshooting)
- Just right: perfect note (Wolfe!)

**Why two conditions?**
- Armijo alone: Might accept α=0.0001 (wastes time)
- Curvature adds: "Keep going until slope flattens out"

**Pro tip**: Use for BFGS/L-BFGS. The curvature condition ensures good Hessian approximation updates.

### 🎯 Exact Line Search: The Perfectionist

**Main Idea**: Find THE best α on the line.

**When it "clicks"**: Like a sniper vs. a shotgun:
- Exact: One perfect shot (expensive to aim)
- Inexact: Good enough shots (fast and effective)

**Reality check**: 
- Cost: Solving a whole optimization problem just for α
- Benefit: Slightly better α
- Verdict: Usually not worth it!

**Pro tip**: Only use when φ(α) has a closed form (e.g., quadratic functions).

---

## Summary

### Key Takeaways

1. **Line search** converts n-dimensional optimization into 1D problems
2. **Exact line search** is theoretically elegant but practically expensive
3. **Wolfe conditions** balance decrease and progress effectively
4. **Backtracking** is simple, robust, and widely used
5. **Convergence** depends on both line search and search direction
6. **Parameter choice** significantly impacts practical performance

### Recommended Approach

For most applications:
- Start with **backtracking line search** with Armijo condition
- Use **Wolfe conditions** for quasi-Newton methods
- Set α_init = 1 for Newton-like directions
- Use standard parameters (c₁ = 10⁻⁴, c₂ = 0.9, ρ = 0.5)
- Add safeguards for numerical stability

### Further Reading

- Nocedal & Wright, "Numerical Optimization" (2006), Chapter 3
- Boyd & Vandenberghe, "Convex Optimization" (2004), Chapter 9
- Fletcher, "Practical Methods of Optimization" (2013)
- Dennis & Schnabel, "Numerical Methods for Unconstrained Optimization" (1996)

---

## Implementation Notes

See the accompanying Python files in this directory:
- `line_search.py`: Core line search implementations
- `steepest_descent.py`: Steepest descent with line search
- `convergence_analysis.py`: Empirical convergence studies
- `gradient_example_1.py`, `gradient_example_2.py`: Practical examples

Each implementation includes:
- Complete documentation
- Numerical examples
- Visualization tools
- Convergence analysis
