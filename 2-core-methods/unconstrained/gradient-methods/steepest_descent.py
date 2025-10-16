"""
Steepest Descent (Gradient Descent) Method

The steepest descent method is the most fundamental first-order optimization
algorithm. It iteratively moves in the direction of the negative gradient
to minimize a differentiable function.

Algorithm:
    x_{k+1} = x_k - α_k ∇f(x_k)

where:
- x_k: current iterate
- α_k: step size (learning rate)
- ∇f(x_k): gradient at x_k

Convergence Properties:
- For convex functions: Converges to global minimum
- For strongly convex functions: Linear convergence rate
- Rate depends on condition number κ = L/μ

Key Theorem (Strongly Convex Case):
If f is μ-strongly convex and L-smooth, with step size α = 1/L:
    ||x_{k+1} - x*||² ≤ (1 - μ/L)||x_k - x*||²

Convergence rate: ρ = (κ - 1)/(κ + 1) where κ = L/μ
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class SteepestDescent:
    """
    Implementation of steepest descent algorithm with various step size strategies.
    """
    
    def __init__(self, 
                 objective: Callable[[np.ndarray], float],
                 gradient: Callable[[np.ndarray], np.ndarray],
                 hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 name: str = "Function"):
        """
        Initialize steepest descent optimizer.
        
        Args:
            objective: Objective function f(x)
            gradient: Gradient function ∇f(x)
            hessian: Optional Hessian function ∇²f(x)
            name: Function name for display
        """
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.name = name
        
        # History tracking
        self.reset_history()
    
    def reset_history(self):
        """Reset optimization history."""
        self.history = {
            'x': [],
            'f': [],
            'grad_norm': [],
            'step_size': []
        }
    
    def optimize(self, 
                x0: np.ndarray,
                step_size: float = 0.01,
                max_iters: int = 1000,
                tolerance: float = 1e-6,
                verbose: bool = False) -> Dict:
        """
        Run steepest descent with constant step size.
        
        Args:
            x0: Initial point
            step_size: Constant step size α
            max_iters: Maximum iterations
            tolerance: Convergence tolerance on gradient norm
            verbose: Print progress
            
        Returns:
            Dictionary with optimization results
        """
        self.reset_history()
        
        x = np.array(x0, dtype=float)
        
        for k in range(max_iters):
            # Evaluate function and gradient
            f_val = self.objective(x)
            grad = self.gradient(x)
            grad_norm = np.linalg.norm(grad)
            
            # Store history
            self.history['x'].append(x.copy())
            self.history['f'].append(f_val)
            self.history['grad_norm'].append(grad_norm)
            self.history['step_size'].append(step_size)
            
            # Check convergence
            if grad_norm < tolerance:
                if verbose:
                    print(f"Converged in {k} iterations")
                break
            
            # Gradient descent update
            x = x - step_size * grad
            
            if verbose and k % 100 == 0:
                print(f"Iter {k}: f = {f_val:.6f}, ||∇f|| = {grad_norm:.6f}")
        
        return {
            'x_optimal': x,
            'f_optimal': self.objective(x),
            'iterations': k + 1,
            'gradient_norm': np.linalg.norm(self.gradient(x)),
            'converged': grad_norm < tolerance,
            'history': self.history
        }
    
    def optimize_backtracking(self,
                             x0: np.ndarray,
                             alpha_init: float = 1.0,
                             beta: float = 0.5,
                             c: float = 1e-4,
                             max_iters: int = 1000,
                             tolerance: float = 1e-6,
                             verbose: bool = False) -> Dict:
        """
        Steepest descent with backtracking line search (Armijo rule).
        
        Backtracking ensures sufficient decrease:
            f(x - αd) ≤ f(x) - c·α·||∇f||²
        
        Args:
            x0: Initial point
            alpha_init: Initial step size
            beta: Backtracking factor (0 < β < 1)
            c: Armijo constant (0 < c < 1)
            max_iters: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress
            
        Returns:
            Optimization results dictionary
        """
        self.reset_history()
        
        x = np.array(x0, dtype=float)
        
        for k in range(max_iters):
            f_val = self.objective(x)
            grad = self.gradient(x)
            grad_norm = np.linalg.norm(grad)
            
            # Check convergence
            if grad_norm < tolerance:
                if verbose:
                    print(f"Converged in {k} iterations")
                break
            
            # Backtracking line search
            alpha = alpha_init
            search_direction = -grad
            
            # Armijo condition: f(x + αd) ≤ f(x) + c·α·∇f^T·d
            while True:
                x_new = x + alpha * search_direction
                f_new = self.objective(x_new)
                
                # Check Armijo condition
                if f_new <= f_val + c * alpha * np.dot(grad, search_direction):
                    break
                
                # Reduce step size
                alpha *= beta
                
                if alpha < 1e-16:
                    if verbose:
                        print(f"Step size too small at iteration {k}")
                    break
            
            # Store history
            self.history['x'].append(x.copy())
            self.history['f'].append(f_val)
            self.history['grad_norm'].append(grad_norm)
            self.history['step_size'].append(alpha)
            
            # Update
            x = x_new
            
            if verbose and k % 100 == 0:
                print(f"Iter {k}: f = {f_val:.6f}, ||∇f|| = {grad_norm:.6f}, α = {alpha:.6f}")
        
        return {
            'x_optimal': x,
            'f_optimal': self.objective(x),
            'iterations': k + 1,
            'gradient_norm': np.linalg.norm(self.gradient(x)),
            'converged': grad_norm < tolerance,
            'history': self.history
        }
    
    def optimize_adaptive(self,
                         x0: np.ndarray,
                         alpha_init: float = 0.1,
                         increase_factor: float = 1.1,
                         decrease_factor: float = 0.5,
                         max_iters: int = 1000,
                         tolerance: float = 1e-6,
                         verbose: bool = False) -> Dict:
        """
        Steepest descent with adaptive step size.
        
        Increases step size when progress is good, decreases when overshooting.
        
        Args:
            x0: Initial point
            alpha_init: Initial step size
            increase_factor: Factor to increase step size
            decrease_factor: Factor to decrease step size
            max_iters: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress
            
        Returns:
            Optimization results dictionary
        """
        self.reset_history()
        
        x = np.array(x0, dtype=float)
        alpha = alpha_init
        f_prev = self.objective(x)
        
        for k in range(max_iters):
            grad = self.gradient(x)
            grad_norm = np.linalg.norm(grad)
            
            # Check convergence
            if grad_norm < tolerance:
                if verbose:
                    print(f"Converged in {k} iterations")
                break
            
            # Try step
            x_new = x - alpha * grad
            f_new = self.objective(x_new)
            
            # Adaptive step size adjustment
            if f_new < f_prev:
                # Good step - accept and possibly increase step size
                self.history['x'].append(x.copy())
                self.history['f'].append(f_prev)
                self.history['grad_norm'].append(grad_norm)
                self.history['step_size'].append(alpha)
                
                x = x_new
                f_prev = f_new
                alpha *= increase_factor  # Increase step size for next iteration
                
            else:
                # Bad step - decrease step size and retry
                alpha *= decrease_factor
                
                if alpha < 1e-16:
                    if verbose:
                        print(f"Step size too small at iteration {k}")
                    break
            
            if verbose and k % 100 == 0:
                print(f"Iter {k}: f = {f_prev:.6f}, ||∇f|| = {grad_norm:.6f}, α = {alpha:.6f}")
        
        return {
            'x_optimal': x,
            'f_optimal': self.objective(x),
            'iterations': k + 1,
            'gradient_norm': np.linalg.norm(self.gradient(x)),
            'converged': grad_norm < tolerance,
            'history': self.history
        }


def demonstrate_steepest_descent():
    """
    Comprehensive demonstration of steepest descent method.
    """
    print("📉 STEEPEST DESCENT METHOD")
    print("=" * 60)
    
    # Create test problems
    
    # 1. Well-conditioned quadratic
    print("\n🎯 EXAMPLE 1: Well-Conditioned Quadratic")
    print("-" * 50)
    
    Q1 = np.array([[2, 0.5], [0.5, 2]])  # Condition number ≈ 1.3
    b1 = np.array([1, -1])
    
    def quad1_objective(x):
        x = np.array(x)
        return 0.5 * np.dot(x, np.dot(Q1, x)) - np.dot(b1, x)
    
    def quad1_gradient(x):
        x = np.array(x)
        return np.dot(Q1, x) - b1
    
    optimizer1 = SteepestDescent(quad1_objective, quad1_gradient, 
                                name="Well-Conditioned Quadratic")
    
    x0 = np.array([5, 5])
    
    # Test different step sizes
    step_sizes = [0.01, 0.1, 0.3, 0.5]
    results_const = []
    
    print(f"Starting point: {x0}")
    print(f"Condition number: {np.linalg.cond(Q1):.2f}")
    print("\nConstant Step Size Results:")
    
    for alpha in step_sizes:
        result = optimizer1.optimize(x0, step_size=alpha, max_iters=1000, tolerance=1e-8)
        results_const.append(result)
        print(f"  α = {alpha:.2f}: {result['iterations']:4d} iterations, "
              f"f* = {result['f_optimal']:.6f}")
    
    # Backtracking line search
    result_backtrack = optimizer1.optimize_backtracking(x0, max_iters=1000, tolerance=1e-8)
    print(f"\nBacktracking: {result_backtrack['iterations']:4d} iterations, "
          f"f* = {result_backtrack['f_optimal']:.6f}")
    
    # Adaptive step size
    result_adaptive = optimizer1.optimize_adaptive(x0, max_iters=1000, tolerance=1e-8)
    print(f"Adaptive:     {result_adaptive['iterations']:4d} iterations, "
          f"f* = {result_adaptive['f_optimal']:.6f}")
    
    # 2. Ill-conditioned quadratic
    print("\n🎯 EXAMPLE 2: Ill-Conditioned Quadratic")
    print("-" * 50)
    
    Q2 = np.array([[10, 0], [0, 0.1]])  # Condition number = 100
    b2 = np.array([1, 1])
    
    def quad2_objective(x):
        x = np.array(x)
        return 0.5 * np.dot(x, np.dot(Q2, x)) - np.dot(b2, x)
    
    def quad2_gradient(x):
        x = np.array(x)
        return np.dot(Q2, x) - b2
    
    optimizer2 = SteepestDescent(quad2_objective, quad2_gradient,
                                name="Ill-Conditioned Quadratic")
    
    print(f"Condition number: {np.linalg.cond(Q2):.2f}")
    
    # Constant step size (small to ensure stability)
    result_ill_const = optimizer2.optimize(x0, step_size=0.01, max_iters=2000, tolerance=1e-6)
    print(f"Constant (α=0.01): {result_ill_const['iterations']:4d} iterations, "
          f"f* = {result_ill_const['f_optimal']:.6f}")
    
    # Backtracking
    result_ill_backtrack = optimizer2.optimize_backtracking(x0, max_iters=2000, tolerance=1e-6)
    print(f"Backtracking:      {result_ill_backtrack['iterations']:4d} iterations, "
          f"f* = {result_ill_backtrack['f_optimal']:.6f}")
    
    # Adaptive
    result_ill_adaptive = optimizer2.optimize_adaptive(x0, alpha_init=0.05, max_iters=2000, 
                                                      tolerance=1e-6)
    print(f"Adaptive:          {result_ill_adaptive['iterations']:4d} iterations, "
          f"f* = {result_ill_adaptive['f_optimal']:.6f}")
    
    # 3. Rosenbrock function
    print("\n🎯 EXAMPLE 3: Rosenbrock Function")
    print("-" * 50)
    
    def rosenbrock_objective(x):
        x = np.array(x)
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_gradient(x):
        x = np.array(x)
        grad_x = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        grad_y = 200*(x[1] - x[0]**2)
        return np.array([grad_x, grad_y])
    
    optimizer3 = SteepestDescent(rosenbrock_objective, rosenbrock_gradient,
                                name="Rosenbrock Function")
    
    x0_rosen = np.array([-1, 2])
    
    print(f"Starting point: {x0_rosen}")
    print(f"True minimum: [1, 1]")
    
    # Only backtracking and adaptive work well for Rosenbrock
    result_rosen_backtrack = optimizer3.optimize_backtracking(x0_rosen, max_iters=5000, 
                                                             tolerance=1e-4)
    print(f"Backtracking: {result_rosen_backtrack['iterations']:4d} iterations")
    print(f"  Final x: {result_rosen_backtrack['x_optimal']}")
    print(f"  Final f: {result_rosen_backtrack['f_optimal']:.6f}")
    
    result_rosen_adaptive = optimizer3.optimize_adaptive(x0_rosen, alpha_init=0.001, 
                                                        max_iters=5000, tolerance=1e-4)
    print(f"Adaptive:     {result_rosen_adaptive['iterations']:4d} iterations")
    print(f"  Final x: {result_rosen_adaptive['x_optimal']}")
    print(f"  Final f: {result_rosen_adaptive['f_optimal']:.6f}")
    
    # Visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Well-conditioned convergence
    ax1 = plt.subplot(2, 3, 1)
    
    for i, (alpha, result) in enumerate(zip(step_sizes, results_const)):
        iterations = range(len(result['history']['f']))
        ax1.semilogy(iterations, result['history']['f'], 
                    label=f'α = {alpha}', linewidth=2)
    
    ax1.semilogy(range(len(result_backtrack['history']['f'])),
                result_backtrack['history']['f'],
                label='Backtracking', linewidth=2, linestyle='--')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('f(x) (log scale)')
    ax1.set_title('Well-Conditioned: Different Step Sizes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ill-conditioned convergence
    ax2 = plt.subplot(2, 3, 2)
    
    ax2.semilogy(range(len(result_ill_const['history']['f'])),
                result_ill_const['history']['f'],
                label='Constant α=0.01', linewidth=2)
    ax2.semilogy(range(len(result_ill_backtrack['history']['f'])),
                result_ill_backtrack['history']['f'],
                label='Backtracking', linewidth=2)
    ax2.semilogy(range(len(result_ill_adaptive['history']['f'])),
                result_ill_adaptive['history']['f'],
                label='Adaptive', linewidth=2)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('f(x) (log scale)')
    ax2.set_title('Ill-Conditioned: Step Size Strategies')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gradient norm convergence
    ax3 = plt.subplot(2, 3, 3)
    
    ax3.semilogy(range(len(result_backtrack['history']['grad_norm'])),
                result_backtrack['history']['grad_norm'],
                label='Well-conditioned', linewidth=2)
    ax3.semilogy(range(len(result_ill_backtrack['history']['grad_norm'])),
                result_ill_backtrack['history']['grad_norm'],
                label='Ill-conditioned', linewidth=2)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('||∇f(x)|| (log scale)')
    ax3.set_title('Gradient Norm Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Optimization path (well-conditioned)
    ax4 = plt.subplot(2, 3, 4)
    
    # Create contour plot
    x_range = np.linspace(-2, 6, 100)
    y_range = np.linspace(-2, 6, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = quad1_objective(np.array([X[i, j], Y[i, j]]))
    
    contour = ax4.contour(X, Y, Z, levels=20, alpha=0.6, colors='gray')
    ax4.clabel(contour, inline=True, fontsize=8)
    
    # Plot optimization path
    path_x = [x[0] for x in result_backtrack['history']['x']]
    path_y = [x[1] for x in result_backtrack['history']['x']]
    
    ax4.plot(path_x, path_y, 'ro-', linewidth=2, markersize=4, alpha=0.7,
            label='Optimization path')
    ax4.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
    ax4.plot(path_x[-1], path_y[-1], 'r*', markersize=15, label='End')
    
    ax4.set_xlabel('x₁')
    ax4.set_ylabel('x₂')
    ax4.set_title('Well-Conditioned: Optimization Path')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Optimization path (ill-conditioned)
    ax5 = plt.subplot(2, 3, 5)
    
    # Create contour for ill-conditioned problem
    Z2 = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z2[i, j] = quad2_objective(np.array([X[i, j], Y[i, j]]))
    
    contour2 = ax5.contour(X, Y, Z2, levels=20, alpha=0.6, colors='gray')
    ax5.clabel(contour2, inline=True, fontsize=8)
    
    # Plot path (subsample for clarity)
    path_x2 = [x[0] for x in result_ill_backtrack['history']['x'][::10]]
    path_y2 = [x[1] for x in result_ill_backtrack['history']['x'][::10]]
    
    ax5.plot(path_x2, path_y2, 'bo-', linewidth=2, markersize=4, alpha=0.7,
            label='Optimization path')
    ax5.plot(result_ill_backtrack['history']['x'][0][0], 
            result_ill_backtrack['history']['x'][0][1], 
            'go', markersize=10, label='Start')
    ax5.plot(path_x2[-1], path_y2[-1], 'b*', markersize=15, label='End')
    
    ax5.set_xlabel('x₁')
    ax5.set_ylabel('x₂')
    ax5.set_title('Ill-Conditioned: Zig-Zag Behavior')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Step size adaptation
    ax6 = plt.subplot(2, 3, 6)
    
    # Show how step size changes in adaptive method
    iterations_adaptive = range(len(result_adaptive['history']['step_size']))
    ax6.plot(iterations_adaptive, result_adaptive['history']['step_size'],
            'g-', linewidth=2, label='Adaptive (well-cond.)')
    
    iterations_backtrack = range(len(result_backtrack['history']['step_size']))
    ax6.plot(iterations_backtrack, result_backtrack['history']['step_size'],
            'r--', linewidth=2, label='Backtracking (well-cond.)')
    
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Step Size α')
    ax6.set_title('Step Size Evolution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def theoretical_analysis():
    """
    Theoretical analysis and key insights.
    """
    print("\n📚 THEORETICAL ANALYSIS")
    print("=" * 50)
    
    print("🔑 KEY THEOREM: Convergence Rate")
    print("For μ-strongly convex and L-smooth function f:")
    print("  With optimal step size α* = 2/(μ + L):")
    print("  ||x_k - x*||² ≤ ((κ-1)/(κ+1))^k ||x_0 - x*||²")
    print("  where κ = L/μ is the condition number")
    
    print("\n📊 CONVERGENCE RATE vs CONDITION NUMBER:")
    condition_numbers = [1, 2, 5, 10, 100, 1000]
    
    for kappa in condition_numbers:
        rho = (kappa - 1) / (kappa + 1)
        iters_to_reduce_half = int(np.ceil(np.log(0.5) / np.log(rho)))
        print(f"  κ = {kappa:5d}: ρ = {rho:.6f}, "
              f"~{iters_to_reduce_half:4d} iters to reduce error by 50%")
    
    print("\n💡 KEY INSIGHTS:")
    print("1. STEP SIZE SELECTION:")
    print("   - Too large: Divergence or oscillation")
    print("   - Too small: Slow convergence")
    print("   - Optimal: α* = 2/(L + μ) for strongly convex functions")
    
    print("\n2. CONDITIONING:")
    print("   - Well-conditioned (κ ≈ 1): Fast convergence")
    print("   - Ill-conditioned (κ >> 1): Slow, zig-zag behavior")
    print("   - Preconditioning can improve conditioning")
    
    print("\n3. LINE SEARCH STRATEGIES:")
    print("   - Constant: Simple but may fail")
    print("   - Backtracking: Robust, guarantees descent")
    print("   - Adaptive: Can be faster, less robust")
    
    print("\n4. WHEN TO USE STEEPEST DESCENT:")
    print("   ✓ Large-scale problems (low memory)")
    print("   ✓ Simple implementation needed")
    print("   ✓ Function evaluation cheap")
    print("   ✗ Ill-conditioned problems")
    print("   ✗ Need fast convergence")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_steepest_descent()
    theoretical_analysis()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Steepest descent moves in direction of negative gradient")
    print("- Convergence rate depends heavily on condition number")
    print("- Step size selection is critical for success")
    print("- Backtracking line search provides robustness")
    print("- Ill-conditioned problems exhibit zig-zag behavior")
    print("- Trade-off: Simplicity vs convergence speed")
    print("\nSteepest descent is the foundation of first-order optimization! 🚀")
