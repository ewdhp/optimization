"""
Penalty Function Methods for Constrained Optimization
===================================================

Penalty methods convert constrained problems to unconstrained by adding
penalty terms to the objective that penalize constraint violations.

Problem Formulation:
    minimize f(x)
    subject to: h_i(x) = 0  (equality constraints)
                g_j(x) ≤ 0  (inequality constraints)

Penalty Approach:
    minimize φ(x, μ) = f(x) + μ·P(x)
    
where P(x) measures constraint violation and μ → ∞

Types:
- Quadratic Penalty: P(x) = Σ h_i(x)² + Σ max(0, g_j(x))²
- Exact Penalty (L1): P(x) = Σ |h_i(x)| + Σ max(0, g_j(x))

References:
- Fiacco & McCormick (1968). "Nonlinear Programming"
- Nocedal & Wright (2006). "Numerical Optimization"

Author: Optimization Framework
Date: October 16, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Callable, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class QuadraticPenalty:
    """
    Quadratic Penalty Method
    
    Penalty function:
        φ(x, μ) = f(x) + (μ/2)·[Σ h_i(x)² + Σ max(0, g_j(x))²]
    
    Algorithm:
        1. Start with small μ_0
        2. Minimize φ(x, μ_k) → x_k
        3. Increase μ_{k+1} = β·μ_k
        4. Repeat until convergence
    """
    
    def __init__(self, mu_init: float = 1.0, mu_increase: float = 10.0,
                 max_outer: int = 20, tol: float = 1e-6):
        """
        Initialize quadratic penalty method.
        
        Parameters:
        -----------
        mu_init : float
            Initial penalty parameter
        mu_increase : float
            Factor to increase penalty (β)
        max_outer : int
            Maximum outer iterations
        tol : float
            Convergence tolerance
        """
        self.mu_init = mu_init
        self.mu_increase = mu_increase
        self.max_outer = max_outer
        self.tol = tol
        
    def penalty_term(self, x: np.ndarray, 
                     h_funcs: List[Callable] = None,
                     g_funcs: List[Callable] = None) -> float:
        """Compute penalty term P(x)."""
        penalty = 0.0
        
        # Equality constraints: h_i(x) = 0
        if h_funcs:
            for h in h_funcs:
                penalty += h(x) ** 2
        
        # Inequality constraints: g_j(x) ≤ 0
        if g_funcs:
            for g in g_funcs:
                penalty += max(0, g(x)) ** 2
        
        return penalty
    
    def penalty_function(self, x: np.ndarray, f: Callable, mu: float,
                        h_funcs: List[Callable] = None,
                        g_funcs: List[Callable] = None) -> float:
        """Compute penalized objective φ(x, μ)."""
        penalty = self.penalty_term(x, h_funcs, g_funcs)
        return f(x) + 0.5 * mu * penalty
    
    def penalty_gradient(self, x: np.ndarray, grad_f: Callable, mu: float,
                        h_funcs: List[Callable] = None,
                        grad_h_funcs: List[Callable] = None,
                        g_funcs: List[Callable] = None,
                        grad_g_funcs: List[Callable] = None) -> np.ndarray:
        """Compute gradient of penalized objective."""
        grad = grad_f(x).copy()
        
        # Equality constraint gradients
        if h_funcs and grad_h_funcs:
            for h, grad_h in zip(h_funcs, grad_h_funcs):
                grad += mu * h(x) * grad_h(x)
        
        # Inequality constraint gradients
        if g_funcs and grad_g_funcs:
            for g, grad_g in zip(g_funcs, grad_g_funcs):
                if g(x) > 0:
                    grad += mu * g(x) * grad_g(x)
        
        return grad
    
    def optimize(self, f: Callable, grad_f: Callable, x0: np.ndarray,
                 h_funcs: List[Callable] = None,
                 grad_h_funcs: List[Callable] = None,
                 g_funcs: List[Callable] = None,
                 grad_g_funcs: List[Callable] = None) -> Tuple[np.ndarray, List]:
        """
        Solve constrained optimization using quadratic penalty.
        
        Returns:
        --------
        x_opt : ndarray
            Optimal solution
        history : list
            Optimization history
        """
        x = x0.copy()
        mu = self.mu_init
        history = []
        
        for k in range(self.max_outer):
            # Define penalized problem
            def phi(x_):
                return self.penalty_function(x_, f, mu, h_funcs, g_funcs)
            
            def grad_phi(x_):
                return self.penalty_gradient(x_, grad_f, mu, h_funcs,
                                            grad_h_funcs, g_funcs, grad_g_funcs)
            
            # Minimize penalized objective
            result = minimize(phi, x, jac=grad_phi, method='BFGS',
                            options={'maxiter': 100, 'gtol': 1e-6})
            x = result.x
            
            # Store history
            penalty = self.penalty_term(x, h_funcs, g_funcs)
            history.append({
                'iteration': k,
                'x': x.copy(),
                'f': f(x),
                'penalty': penalty,
                'phi': phi(x),
                'mu': mu,
                'constraint_violation': np.sqrt(penalty)
            })
            
            print(f"Outer iter {k}: f(x) = {f(x):.6f}, penalty = {penalty:.6e}, μ = {mu:.2e}")
            
            # Check convergence
            if penalty < self.tol:
                print(f"\nConverged! Constraint violation: {np.sqrt(penalty):.2e}")
                break
            
            # Increase penalty parameter
            mu *= self.mu_increase
        
        return x, history


class ExactPenalty:
    """
    Exact Penalty Method (L1 penalty)
    
    Penalty function:
        φ(x, μ) = f(x) + μ·[Σ |h_i(x)| + Σ max(0, g_j(x))]
    
    Advantage: Can find exact solution with finite μ
    Disadvantage: Non-smooth (requires subgradient methods)
    """
    
    def __init__(self, mu_init: float = 10.0, mu_increase: float = 2.0,
                 max_outer: int = 20, tol: float = 1e-6):
        self.mu_init = mu_init
        self.mu_increase = mu_increase
        self.max_outer = max_outer
        self.tol = tol
        
    def penalty_term(self, x: np.ndarray,
                     h_funcs: List[Callable] = None,
                     g_funcs: List[Callable] = None) -> float:
        """Compute L1 penalty term."""
        penalty = 0.0
        
        if h_funcs:
            for h in h_funcs:
                penalty += abs(h(x))
        
        if g_funcs:
            for g in g_funcs:
                penalty += max(0, g(x))
        
        return penalty
    
    def optimize(self, f: Callable, x0: np.ndarray,
                 h_funcs: List[Callable] = None,
                 g_funcs: List[Callable] = None) -> Tuple[np.ndarray, List]:
        """Solve using exact penalty (L1)."""
        x = x0.copy()
        mu = self.mu_init
        history = []
        
        for k in range(self.max_outer):
            # Define penalized problem
            def phi(x_):
                return f(x_) + mu * self.penalty_term(x_, h_funcs, g_funcs)
            
            # Minimize using Nelder-Mead (derivative-free for non-smooth)
            result = minimize(phi, x, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-8})
            x = result.x
            
            penalty = self.penalty_term(x, h_funcs, g_funcs)
            history.append({
                'iteration': k,
                'x': x.copy(),
                'f': f(x),
                'penalty': penalty,
                'mu': mu
            })
            
            print(f"Outer iter {k}: f(x) = {f(x):.6f}, L1 penalty = {penalty:.6e}, μ = {mu:.2e}")
            
            if penalty < self.tol:
                print(f"\nConverged! L1 penalty: {penalty:.2e}")
                break
            
            mu *= self.mu_increase
        
        return x, history


# ============================================================================
# TEST PROBLEMS
# ============================================================================

def test_problem_1():
    """
    Simple constrained problem:
        minimize (x - 2)² + (y - 1)²
        subject to x + y = 3
    
    Solution: x = 2.5, y = 0.5, f* = 0.5
    """
    def f(x):
        return (x[0] - 2)**2 + (x[1] - 1)**2
    
    def grad_f(x):
        return np.array([2*(x[0] - 2), 2*(x[1] - 1)])
    
    # Equality constraint: h(x) = x + y - 3 = 0
    def h(x):
        return x[0] + x[1] - 3
    
    def grad_h(x):
        return np.array([1.0, 1.0])
    
    return f, grad_f, [h], [grad_h], None, None


def test_problem_2():
    """
    Problem with inequality:
        minimize x² + y²
        subject to x + y ≥ 1  (i.e., -x - y + 1 ≤ 0)
    
    Solution: x = 0.5, y = 0.5, f* = 0.5
    """
    def f(x):
        return x[0]**2 + x[1]**2
    
    def grad_f(x):
        return np.array([2*x[0], 2*x[1]])
    
    # Inequality: g(x) = -x - y + 1 ≤ 0
    def g(x):
        return -x[0] - x[1] + 1
    
    def grad_g(x):
        return np.array([-1.0, -1.0])
    
    return f, grad_f, None, None, [g], [grad_g]


def test_problem_3():
    """
    Rosenbrock with circle constraint:
        minimize (1-x)² + 100(y-x²)²
        subject to x² + y² ≤ 2
    """
    def f(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def grad_f(x):
        dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        dy = 200*(x[1] - x[0]**2)
        return np.array([dx, dy])
    
    # Inequality: g(x) = x² + y² - 2 ≤ 0
    def g(x):
        return x[0]**2 + x[1]**2 - 2
    
    def grad_g(x):
        return np.array([2*x[0], 2*x[1]])
    
    return f, grad_f, None, None, [g], [grad_g]


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_penalty_progression():
    """Visualize how penalty method progresses."""
    print("=" * 70)
    print("PENALTY METHOD PROGRESSION VISUALIZATION")
    print("=" * 70)
    
    # Use test problem 1
    f, grad_f, h_funcs, grad_h_funcs, _, _ = test_problem_1()
    x0 = np.array([0.0, 0.0])
    
    penalty_method = QuadraticPenalty(mu_init=1.0, mu_increase=10.0, max_outer=8)
    x_opt, history = penalty_method.optimize(f, grad_f, x0, h_funcs, grad_h_funcs)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Contour with trajectory
    ax = axes[0, 0]
    x_range = np.linspace(-0.5, 3.5, 100)
    y_range = np.linspace(-0.5, 3.5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[f(np.array([x, y])) for x in x_range] for y in y_range])
    
    contour = ax.contour(X, Y, Z, levels=20, alpha=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Draw constraint line: x + y = 3
    x_line = np.linspace(0, 3.5, 100)
    y_line = 3 - x_line
    ax.plot(x_line, y_line, 'r-', linewidth=3, label='Constraint: x+y=3')
    
    # Plot trajectory
    traj = np.array([h['x'] for h in history])
    ax.plot(traj[:, 0], traj[:, 1], 'bo-', markersize=8, linewidth=2,
            label='Penalty trajectory')
    
    # Mark iterations with penalty values
    for i, h in enumerate(history):
        ax.annotate(f"μ={h['mu']:.0e}", h['x'], fontsize=9,
                   xytext=(5, 5), textcoords='offset points')
    
    ax.plot(2.5, 0.5, 'g*', markersize=20, label='True optimum')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Penalty Method Trajectory', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Objective convergence
    ax = axes[0, 1]
    f_vals = [h['f'] for h in history]
    iterations = range(len(history))
    
    ax.plot(iterations, f_vals, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='r', linestyle='--', label='f* = 0.5')
    ax.set_xlabel('Outer Iteration', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('Objective Value Convergence', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Constraint violation
    ax = axes[1, 0]
    violations = [h['constraint_violation'] for h in history]
    ax.semilogy(iterations, violations, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Outer Iteration', fontsize=12)
    ax.set_ylabel('Constraint Violation [log]', fontsize=12)
    ax.set_title('Constraint Satisfaction', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Penalty parameter
    ax = axes[1, 1]
    mu_vals = [h['mu'] for h in history]
    ax.semilogy(iterations, mu_vals, 'go-', linewidth=2, markersize=8)
    ax.set_xlabel('Outer Iteration', fontsize=12)
    ax.set_ylabel('Penalty Parameter μ [log]', fontsize=12)
    ax.set_title('Penalty Parameter Growth', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('penalty_progression.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: penalty_progression.png")
    plt.show()
    
    print(f"\nFinal solution: x = {x_opt}")
    print(f"True solution:  x = [2.5, 0.5]")
    print(f"Final f(x) = {f(x_opt):.6f}, optimal f* = 0.5")


def compare_quadratic_exact():
    """Compare quadratic vs exact penalty."""
    print("\n" + "=" * 70)
    print("COMPARISON: Quadratic vs Exact Penalty")
    print("=" * 70)
    
    f, grad_f, h_funcs, grad_h_funcs, _, _ = test_problem_1()
    x0 = np.array([0.0, 0.0])
    
    # Quadratic penalty
    print("\n1. Running Quadratic Penalty...")
    quad_penalty = QuadraticPenalty(mu_init=1.0, mu_increase=5.0, max_outer=10)
    x_quad, hist_quad = quad_penalty.optimize(f, grad_f, x0, h_funcs, grad_h_funcs)
    
    # Exact penalty
    print("\n2. Running Exact (L1) Penalty...")
    exact_penalty = ExactPenalty(mu_init=10.0, mu_increase=2.0, max_outer=10)
    x_exact, hist_exact = exact_penalty.optimize(f, x0, h_funcs)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Trajectory comparison
    ax = axes[0]
    x_range = np.linspace(-0.5, 3.5, 100)
    y_range = np.linspace(-0.5, 3.5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[f(np.array([x, y])) for x in x_range] for y in y_range])
    
    contour = ax.contour(X, Y, Z, levels=20, alpha=0.3)
    
    # Constraint line
    x_line = np.linspace(0, 3.5, 100)
    y_line = 3 - x_line
    ax.plot(x_line, y_line, 'k-', linewidth=3, label='Constraint')
    
    # Trajectories
    quad_traj = np.array([h['x'] for h in hist_quad])
    exact_traj = np.array([h['x'] for h in hist_exact])
    
    ax.plot(quad_traj[:, 0], quad_traj[:, 1], 'bo-', markersize=8,
            linewidth=2, label='Quadratic Penalty')
    ax.plot(exact_traj[:, 0], exact_traj[:, 1], 'ro-', markersize=8,
            linewidth=2, label='Exact (L1) Penalty')
    ax.plot(2.5, 0.5, 'g*', markersize=20, label='Optimum')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Trajectory Comparison', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Convergence comparison
    ax = axes[1]
    quad_f = [h['f'] for h in hist_quad]
    exact_f = [h['f'] for h in hist_exact]
    
    ax.plot(range(len(quad_f)), quad_f, 'bo-', linewidth=2,
            markersize=8, label='Quadratic')
    ax.plot(range(len(exact_f)), exact_f, 'ro-', linewidth=2,
            markersize=8, label='Exact (L1)')
    ax.axhline(y=0.5, color='g', linestyle='--', linewidth=2, label='f* = 0.5')
    
    ax.set_xlabel('Outer Iteration', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('Objective Convergence', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('penalty_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: penalty_comparison.png")
    plt.show()
    
    print(f"\nQuadratic Penalty: {len(hist_quad)} iterations")
    print(f"  Final x = {x_quad}")
    print(f"  Final f = {f(x_quad):.6f}")
    
    print(f"\nExact Penalty: {len(hist_exact)} iterations")
    print(f"  Final x = {x_exact}")
    print(f"  Final f = {f(x_exact):.6f}")


def inequality_constraint_demo():
    """Demonstrate penalty on inequality constraints."""
    print("\n" + "=" * 70)
    print("INEQUALITY CONSTRAINT DEMONSTRATION")
    print("=" * 70)
    
    f, grad_f, _, _, g_funcs, grad_g_funcs = test_problem_2()
    x0 = np.array([2.0, 2.0])
    
    penalty_method = QuadraticPenalty(mu_init=1.0, mu_increase=10.0, max_outer=10)
    x_opt, history = penalty_method.optimize(f, grad_f, x0,
                                             g_funcs=g_funcs,
                                             grad_g_funcs=grad_g_funcs)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Contour plot
    x_range = np.linspace(-0.5, 2.5, 100)
    y_range = np.linspace(-0.5, 2.5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[f(np.array([x, y])) for x in x_range] for y in y_range])
    
    contour = ax.contour(X, Y, Z, levels=20, alpha=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Feasible region: x + y ≥ 1
    x_line = np.linspace(0, 2.5, 100)
    y_line = 1 - x_line
    ax.fill_between(x_line, y_line, 2.5, alpha=0.2, color='green',
                    label='Feasible region')
    ax.plot(x_line, y_line, 'r-', linewidth=3, label='Boundary: x+y=1')
    
    # Trajectory
    traj = np.array([h['x'] for h in history])
    ax.plot(traj[:, 0], traj[:, 1], 'bo-', markersize=8, linewidth=2,
            label='Optimization path')
    
    ax.plot(0.5, 0.5, 'g*', markersize=20, label='Optimum')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Penalty Method with Inequality Constraint',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([-0.5, 2.5])
    
    plt.tight_layout()
    plt.savefig('inequality_penalty.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: inequality_penalty.png")
    plt.show()
    
    print(f"\nFinal solution: x = {x_opt}")
    print(f"True solution:  x = [0.5, 0.5]")
    print(f"Final f(x) = {f(x_opt):.6f}, optimal f* = 0.5")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PENALTY FUNCTION METHODS")
    print("=" * 70)
    
    # Run demonstrations
    visualize_penalty_progression()
    compare_quadratic_exact()
    inequality_constraint_demo()
    
    print("\n" + "=" * 70)
    print("✓ ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("\nGenerated visualizations:")
    print("  1. penalty_progression.png - Penalty method progression")
    print("  2. penalty_comparison.png - Quadratic vs Exact penalty")
    print("  3. inequality_penalty.png - Inequality constraints")
