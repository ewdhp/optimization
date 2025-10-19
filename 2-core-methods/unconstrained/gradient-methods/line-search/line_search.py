"""
Line Search Methods

Line search is a fundamental technique in optimization for determining
appropriate step sizes that guarantee convergence. Instead of using
a fixed step size, line search adaptively chooses the step size at
each iteration.

General Framework:
    x_{k+1} = x_k + α_k d_k

where:
- d_k: search direction (e.g., -∇f(x_k) for gradient descent)
- α_k: step size determined by line search

Key Line Search Methods:
1. Exact Line Search: α_k = argmin_α f(x_k + α d_k)
2. Backtracking (Armijo): Ensures sufficient decrease
3. Wolfe Conditions: Ensures sufficient decrease + curvature
4. Strong Wolfe: Stronger curvature condition

Armijo Condition (Sufficient Decrease):
    f(x_k + α d_k) ≤ f(x_k) + c₁ α ∇f(x_k)^T d_k

Wolfe Conditions:
    f(x_k + α d_k) ≤ f(x_k) + c₁ α ∇f(x_k)^T d_k  (Armijo)
    ∇f(x_k + α d_k)^T d_k ≥ c₂ ∇f(x_k)^T d_k      (Curvature)

where 0 < c₁ < c₂ < 1, typically c₁ = 1e-4, c₂ = 0.9
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')

class LineSearch:
    """
    Implementation of various line search methods.
    """
    
    def __init__(self,
                 objective: Callable[[np.ndarray], float],
                 gradient: Callable[[np.ndarray], np.ndarray],
                 name: str = "Function"):
        """
        Initialize line search.
        
        Args:
            objective: Objective function f(x)
            gradient: Gradient function ∇f(x)
            name: Function name
        """
        self.objective = objective
        self.gradient = gradient
        self.name = name
    
    def exact_line_search(self,
                         x: np.ndarray,
                         direction: np.ndarray,
                         alpha_min: float = 0,
                         alpha_max: float = 10,
                         n_samples: int = 100) -> float:
        """
        Approximate exact line search via grid search.
        
        Finds α ≈ argmin_α f(x + α·d)
        
        Args:
            x: Current point
            direction: Search direction d
            alpha_min: Minimum step size
            alpha_max: Maximum step size
            n_samples: Number of sample points
            
        Returns:
            Optimal step size α*
        """
        x = np.array(x)
        direction = np.array(direction)
        
        # Grid search
        alphas = np.linspace(alpha_min, alpha_max, n_samples)
        f_values = []
        
        for alpha in alphas:
            x_new = x + alpha * direction
            f_values.append(self.objective(x_new))
        
        # Find minimum
        min_idx = np.argmin(f_values)
        alpha_optimal = alphas[min_idx]
        
        # Refine with golden section search
        # (optional, for better accuracy)
        
        return alpha_optimal
    
    def backtracking_line_search(self,
                                x: np.ndarray,
                                direction: np.ndarray,
                                alpha_init: float = 1.0,
                                beta: float = 0.5,
                                c: float = 1e-4,
                                max_backtracks: int = 50) -> Tuple[float, int]:
        """
        Backtracking line search with Armijo condition.
        
        Ensures sufficient decrease:
            f(x + αd) ≤ f(x) + c·α·∇f^T·d
        
        Args:
            x: Current point
            direction: Search direction
            alpha_init: Initial step size
            beta: Backtracking factor (0 < β < 1)
            c: Armijo constant (0 < c < 1)
            max_backtracks: Maximum backtracking steps
            
        Returns:
            (alpha, num_backtracks)
        """
        x = np.array(x)
        direction = np.array(direction)
        
        f_x = self.objective(x)
        grad_x = self.gradient(x)
        
        # Directional derivative
        directional_deriv = np.dot(grad_x, direction)
        
        # If not a descent direction, return small step
        if directional_deriv >= 0:
            return 1e-8, 0
        
        alpha = alpha_init
        
        for i in range(max_backtracks):
            x_new = x + alpha * direction
            f_new = self.objective(x_new)
            
            # Armijo condition
            if f_new <= f_x + c * alpha * directional_deriv:
                return alpha, i
            
            # Reduce step size
            alpha *= beta
        
        # If max backtracks reached, return small step
        return alpha, max_backtracks
    
    def wolfe_line_search(self,
                         x: np.ndarray,
                         direction: np.ndarray,
                         alpha_init: float = 1.0,
                         c1: float = 1e-4,
                         c2: float = 0.9,
                         max_iters: int = 50) -> Tuple[float, int]:
        """
        Line search satisfying Wolfe conditions.
        
        Wolfe conditions:
        1. Sufficient decrease: f(x+αd) ≤ f(x) + c₁α∇f^T·d
        2. Curvature: ∇f(x+αd)^T·d ≥ c₂∇f^T·d
        
        Args:
            x: Current point
            direction: Search direction
            alpha_init: Initial step size
            c1: Armijo constant
            c2: Curvature constant
            max_iters: Maximum iterations
            
        Returns:
            (alpha, num_iterations)
        """
        x = np.array(x)
        direction = np.array(direction)
        
        f_x = self.objective(x)
        grad_x = self.gradient(x)
        
        directional_deriv = np.dot(grad_x, direction)
        
        # If not descent direction
        if directional_deriv >= 0:
            return 1e-8, 0
        
        alpha = alpha_init
        alpha_low = 0
        alpha_high = np.inf
        
        for i in range(max_iters):
            x_new = x + alpha * direction
            f_new = self.objective(x_new)
            grad_new = self.gradient(x_new)
            
            # Check Armijo condition
            armijo_satisfied = (f_new <= f_x + c1 * alpha * directional_deriv)
            
            # Check curvature condition
            directional_deriv_new = np.dot(grad_new, direction)
            curvature_satisfied = (directional_deriv_new >= c2 * directional_deriv)
            
            if armijo_satisfied and curvature_satisfied:
                # Both Wolfe conditions satisfied
                return alpha, i + 1
            
            if not armijo_satisfied:
                # α too large - reduce upper bound
                alpha_high = alpha
            elif not curvature_satisfied:
                # α too small - increase lower bound
                alpha_low = alpha
            
            # Update α using bisection
            if alpha_high < np.inf:
                alpha = 0.5 * (alpha_low + alpha_high)
            else:
                alpha = 2 * alpha
            
            # Safety check
            if alpha < 1e-16:
                return alpha, i + 1
        
        return alpha, max_iters
    
    def strong_wolfe_line_search(self,
                                x: np.ndarray,
                                direction: np.ndarray,
                                alpha_init: float = 1.0,
                                c1: float = 1e-4,
                                c2: float = 0.9,
                                max_iters: int = 50) -> Tuple[float, int]:
        """
        Strong Wolfe conditions (stricter curvature condition).
        
        Strong Wolfe:
        1. Sufficient decrease: f(x+αd) ≤ f(x) + c₁α∇f^T·d
        2. Strong curvature: |∇f(x+αd)^T·d| ≤ c₂|∇f^T·d|
        
        Args:
            x: Current point
            direction: Search direction
            alpha_init: Initial step size
            c1: Armijo constant
            c2: Strong curvature constant
            max_iters: Maximum iterations
            
        Returns:
            (alpha, num_iterations)
        """
        x = np.array(x)
        direction = np.array(direction)
        
        f_x = self.objective(x)
        grad_x = self.gradient(x)
        
        directional_deriv = np.dot(grad_x, direction)
        
        if directional_deriv >= 0:
            return 1e-8, 0
        
        alpha = alpha_init
        alpha_low = 0
        alpha_high = np.inf
        
        for i in range(max_iters):
            x_new = x + alpha * direction
            f_new = self.objective(x_new)
            grad_new = self.gradient(x_new)
            
            # Check Armijo condition
            armijo_satisfied = (f_new <= f_x + c1 * alpha * directional_deriv)
            
            # Check strong curvature condition
            directional_deriv_new = np.dot(grad_new, direction)
            strong_curvature_satisfied = (
                abs(directional_deriv_new) <= c2 * abs(directional_deriv)
            )
            
            if armijo_satisfied and strong_curvature_satisfied:
                return alpha, i + 1
            
            # Update bounds
            if not armijo_satisfied or directional_deriv_new > 0:
                alpha_high = alpha
            else:
                alpha_low = alpha
            
            # Bisection update
            if alpha_high < np.inf:
                alpha = 0.5 * (alpha_low + alpha_high)
            else:
                alpha = 2 * alpha
            
            if alpha < 1e-16:
                return alpha, i + 1
        
        return alpha, max_iters


def demonstrate_line_search():
    """
    Comprehensive demonstration of line search methods.
    """
    print("🔍 LINE SEARCH METHODS")
    print("=" * 60)
    
    # Test problem: 2D quadratic
    print("\n🎯 TEST PROBLEM: 2D Quadratic Function")
    print("-" * 50)
    
    Q = np.array([[3, 1], [1, 2]])
    b = np.array([1, -1])
    
    def objective(x):
        x = np.array(x)
        return 0.5 * np.dot(x, np.dot(Q, x)) - np.dot(b, x)
    
    def gradient(x):
        x = np.array(x)
        return np.dot(Q, x) - b
    
    line_search = LineSearch(objective, gradient, "Quadratic Function")
    
    # Test point and direction
    x_test = np.array([2, 3])
    direction = -gradient(x_test)  # Steepest descent direction
    
    print(f"Test point: x = {x_test}")
    print(f"Function value: f(x) = {objective(x_test):.6f}")
    print(f"Gradient: ∇f(x) = {gradient(x_test)}")
    print(f"Search direction: d = {direction}")
    
    # 1. Exact line search
    print("\n📊 EXACT LINE SEARCH")
    alpha_exact = line_search.exact_line_search(x_test, direction, 
                                               alpha_min=0, alpha_max=2)
    x_exact = x_test + alpha_exact * direction
    f_exact = objective(x_exact)
    
    print(f"Optimal step size: α* = {alpha_exact:.6f}")
    print(f"New point: x_new = {x_exact}")
    print(f"New function value: f(x_new) = {f_exact:.6f}")
    
    # 2. Backtracking line search
    print("\n📊 BACKTRACKING LINE SEARCH (Armijo)")
    alpha_backtrack, n_backtracks = line_search.backtracking_line_search(
        x_test, direction, alpha_init=1.0)
    x_backtrack = x_test + alpha_backtrack * direction
    f_backtrack = objective(x_backtrack)
    
    print(f"Step size: α = {alpha_backtrack:.6f}")
    print(f"Number of backtracks: {n_backtracks}")
    print(f"New function value: f(x_new) = {f_backtrack:.6f}")
    
    # 3. Wolfe line search
    print("\n📊 WOLFE LINE SEARCH")
    alpha_wolfe, n_wolfe = line_search.wolfe_line_search(x_test, direction)
    x_wolfe = x_test + alpha_wolfe * direction
    f_wolfe = objective(x_wolfe)
    
    print(f"Step size: α = {alpha_wolfe:.6f}")
    print(f"Number of iterations: {n_wolfe}")
    print(f"New function value: f(x_new) = {f_wolfe:.6f}")
    
    # 4. Strong Wolfe line search
    print("\n📊 STRONG WOLFE LINE SEARCH")
    alpha_strong, n_strong = line_search.strong_wolfe_line_search(x_test, direction)
    x_strong = x_test + alpha_strong * direction
    f_strong = objective(x_strong)
    
    print(f"Step size: α = {alpha_strong:.6f}")
    print(f"Number of iterations: {n_strong}")
    print(f"New function value: f(x_new) = {f_strong:.6f}")
    
    # Visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Function along search direction
    ax1 = plt.subplot(2, 3, 1)
    
    alphas = np.linspace(0, 1.5, 200)
    f_values = [objective(x_test + alpha * direction) for alpha in alphas]
    
    ax1.plot(alphas, f_values, 'b-', linewidth=3, label='f(x + αd)')
    
    # Mark different line search results
    ax1.plot(alpha_exact, f_exact, 'ro', markersize=12, label=f'Exact (α={alpha_exact:.3f})')
    ax1.plot(alpha_backtrack, f_backtrack, 'gs', markersize=12, 
            label=f'Backtrack (α={alpha_backtrack:.3f})')
    ax1.plot(alpha_wolfe, f_wolfe, 'c^', markersize=12, 
            label=f'Wolfe (α={alpha_wolfe:.3f})')
    ax1.plot(alpha_strong, f_strong, 'md', markersize=12,
            label=f'Strong Wolfe (α={alpha_strong:.3f})')
    
    ax1.set_xlabel('Step Size α')
    ax1.set_ylabel('f(x + αd)')
    ax1.set_title('Function Value Along Search Direction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Armijo condition visualization
    ax2 = plt.subplot(2, 3, 2)
    
    f_x = objective(x_test)
    grad_x = gradient(x_test)
    directional_deriv = np.dot(grad_x, direction)
    
    # Armijo line: f(x) + c₁·α·∇f^T·d
    c1 = 1e-4
    armijo_line = [f_x + c1 * alpha * directional_deriv for alpha in alphas]
    
    ax2.plot(alphas, f_values, 'b-', linewidth=3, label='f(x + αd)')
    ax2.plot(alphas, armijo_line, 'r--', linewidth=2, label='Armijo line')
    
    # Shade acceptable region
    acceptable_mask = np.array(f_values) <= np.array(armijo_line)
    ax2.fill_between(alphas, f_values, armijo_line, 
                    where=acceptable_mask, alpha=0.3, color='green',
                    label='Armijo satisfied')
    
    ax2.plot(alpha_backtrack, f_backtrack, 'go', markersize=12,
            label='Backtracking result')
    
    ax2.set_xlabel('Step Size α')
    ax2.set_ylabel('Function Value')
    ax2.set_title('Armijo Condition (Sufficient Decrease)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Curvature condition
    ax3 = plt.subplot(2, 3, 3)
    
    # Compute directional derivatives along line
    directional_derivs = []
    for alpha in alphas:
        x_alpha = x_test + alpha * direction
        grad_alpha = gradient(x_alpha)
        directional_derivs.append(np.dot(grad_alpha, direction))
    
    c2 = 0.9
    curvature_line = c2 * directional_deriv * np.ones_like(alphas)
    
    ax3.plot(alphas, directional_derivs, 'b-', linewidth=3, 
            label="∇f(x+αd)^T·d")
    ax3.plot(alphas, curvature_line, 'r--', linewidth=2,
            label=f'c₂·∇f(x)^T·d (c₂={c2})')
    ax3.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    
    # Mark Wolfe result
    x_wolfe_point = x_test + alpha_wolfe * direction
    grad_wolfe = gradient(x_wolfe_point)
    deriv_wolfe = np.dot(grad_wolfe, direction)
    ax3.plot(alpha_wolfe, deriv_wolfe, 'co', markersize=12,
            label='Wolfe result')
    
    ax3.set_xlabel('Step Size α')
    ax3.set_ylabel('Directional Derivative')
    ax3.set_title('Curvature Condition')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: 2D contour with line search paths
    ax4 = plt.subplot(2, 3, 4)
    
    # Create contour plot
    x_range = np.linspace(-1, 4, 100)
    y_range = np.linspace(-1, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = objective(np.array([X[i, j], Y[i, j]]))
    
    contour = ax4.contour(X, Y, Z, levels=20, alpha=0.6, colors='gray')
    ax4.clabel(contour, inline=True, fontsize=8)
    
    # Plot search line and results
    alpha_plot = np.linspace(0, 1.5, 100)
    x_line = [x_test + a * direction for a in alpha_plot]
    x_coords = [x[0] for x in x_line]
    y_coords = [x[1] for x in x_line]
    
    ax4.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='Search line')
    ax4.plot(x_test[0], x_test[1], 'ko', markersize=10, label='Start')
    ax4.plot(x_exact[0], x_exact[1], 'ro', markersize=10, label='Exact')
    ax4.plot(x_backtrack[0], x_backtrack[1], 'gs', markersize=10, label='Backtrack')
    ax4.plot(x_wolfe[0], x_wolfe[1], 'c^', markersize=10, label='Wolfe')
    
    ax4.set_xlabel('x₁')
    ax4.set_ylabel('x₂')
    ax4.set_title('Line Search in 2D Space')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Comparison of methods
    ax5 = plt.subplot(2, 3, 5)
    
    methods = ['Exact', 'Backtrack', 'Wolfe', 'Strong Wolfe']
    step_sizes = [alpha_exact, alpha_backtrack, alpha_wolfe, alpha_strong]
    f_values_comparison = [f_exact, f_backtrack, f_wolfe, f_strong]
    colors = ['red', 'green', 'cyan', 'magenta']
    
    x_pos = range(len(methods))
    bars = ax5.bar(x_pos, step_sizes, color=colors, alpha=0.7)
    
    ax5.set_ylabel('Step Size α')
    ax5.set_title('Step Size Comparison')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(methods)
    
    # Add value labels
    for bar, alpha_val in zip(bars, step_sizes):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{alpha_val:.3f}', ha='center', va='bottom')
    
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Iteration counts and efficiency
    ax6 = plt.subplot(2, 3, 6)
    
    iterations = [1, n_backtracks, n_wolfe, n_strong]
    
    bars2 = ax6.bar(x_pos, iterations, color=colors, alpha=0.7)
    
    ax6.set_ylabel('Iterations / Backtracks')
    ax6.set_title('Computational Cost')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(methods)
    
    for bar, n_iter in zip(bars2, iterations):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{n_iter}', ha='center', va='bottom')
    
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def line_search_theory():
    """
    Theoretical background on line search methods.
    """
    print("\n📚 LINE SEARCH THEORY")
    print("=" * 50)
    
    print("🔑 ARMIJO CONDITION (Sufficient Decrease):")
    print("  f(x + αd) ≤ f(x) + c₁·α·∇f(x)^T·d")
    print("  where c₁ ∈ (0, 1), typically c₁ = 10⁻⁴")
    print("  Ensures function decreases sufficiently")
    
    print("\n🔑 CURVATURE CONDITION:")
    print("  ∇f(x + αd)^T·d ≥ c₂·∇f(x)^T·d")
    print("  where c₂ ∈ (c₁, 1), typically c₂ = 0.9")
    print("  Prevents step sizes from being too small")
    
    print("\n🔑 STRONG CURVATURE CONDITION:")
    print("  |∇f(x + αd)^T·d| ≤ c₂·|∇f(x)^T·d|")
    print("  Stricter than regular curvature condition")
    print("  Ensures gradient becomes more perpendicular to direction")
    
    print("\n💡 CONVERGENCE GUARANTEES:")
    print("1. BACKTRACKING (Armijo only):")
    print("   ✓ Always finds acceptable step size (if descent direction)")
    print("   ✓ Guarantees global convergence for gradient descent")
    print("   ✓ Simple and robust")
    
    print("\n2. WOLFE CONDITIONS:")
    print("   ✓ Prevents steps that are too small")
    print("   ✓ Essential for BFGS and quasi-Newton methods")
    print("   ✓ Guarantees superlinear convergence of BFGS")
    
    print("\n3. STRONG WOLFE:")
    print("   ✓ Excludes regions where curvature is wrong sign")
    print("   ✓ Better for Newton-type methods")
    print("   ✓ More restrictive but better theoretical properties")
    
    print("\n🎯 PRACTICAL GUIDELINES:")
    print("- Use c₁ = 10⁻⁴ for Armijo condition")
    print("- Use c₂ = 0.9 for Newton methods, c₂ = 0.1 for conjugate gradient")
    print("- Start with α_init = 1 for Newton methods")
    print("- Use backtracking factor β ≈ 0.5")
    print("- Exact line search rarely needed in practice")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_line_search()
    line_search_theory()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Line search adaptively chooses step sizes for guaranteed convergence")
    print("- Armijo condition ensures sufficient decrease in function value")
    print("- Wolfe conditions add curvature requirement for quasi-Newton methods")
    print("- Backtracking is simple and robust for gradient descent")
    print("- Strong Wolfe conditions provide better theoretical guarantees")
    print("- Choice of c₁, c₂ depends on optimization method used")
    print("\nLine search is essential for robust optimization algorithms! 🚀")
