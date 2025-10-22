"""
Steepest Descent Method
=======================

The steepest descent (gradient descent) method is the most fundamental gradient-based
optimization algorithm. It moves in the direction of the negative gradient at each iteration.

Key Features:
- Simple and intuitive
- Guaranteed convergence for convex functions with appropriate step size
- Can be slow near the minimum (zig-zagging behavior)
- Works well for strongly convex functions

Algorithm:
x_{k+1} = x_k - α_k ∇f(x_k)

where α_k is the step size (learning rate)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional


class SteepestDescent:
    """
    Steepest Descent optimizer with various line search strategies.
    """
    
    def __init__(self, 
                 step_size: float = 0.1,
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 line_search: str = 'constant'):
        """
        Initialize the steepest descent optimizer.
        
        Args:
            step_size: Initial step size (learning rate)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            line_search: Line search strategy ('constant', 'backtracking', 'exact')
        """
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol
        self.line_search = line_search
        self.history = []
        
    def optimize(self, 
                 f: Callable[[np.ndarray], float],
                 grad_f: Callable[[np.ndarray], np.ndarray],
                 x0: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Minimize function f starting from x0.
        
        Args:
            f: Objective function
            grad_f: Gradient of objective function
            x0: Initial point
            
        Returns:
            Optimal point and optimization history
        """
        x = x0.copy()
        self.history = [{'x': x.copy(), 'f': f(x), 'grad_norm': np.linalg.norm(grad_f(x))}]
        
        for k in range(self.max_iter):
            grad = grad_f(x)
            grad_norm = np.linalg.norm(grad)
            
            # Check convergence
            if grad_norm < self.tol:
                print(f"Converged in {k} iterations")
                break
                
            # Determine step size
            if self.line_search == 'constant':
                alpha = self.step_size
            elif self.line_search == 'backtracking':
                alpha = self._backtracking_line_search(f, grad_f, x, grad)
            elif self.line_search == 'exact':
                alpha = self._exact_line_search(f, grad_f, x, grad)
            else:
                alpha = self.step_size
                
            # Update
            x = x - alpha * grad
            
            # Store history
            self.history.append({
                'x': x.copy(),
                'f': f(x),
                'grad_norm': grad_norm,
                'step_size': alpha
            })
            
        return x, self.history
    
    def _backtracking_line_search(self, 
                                  f: Callable, 
                                  grad_f: Callable,
                                  x: np.ndarray, 
                                  grad: np.ndarray,
                                  c: float = 0.5,
                                  rho: float = 0.8) -> float:
        """
        Backtracking line search with Armijo condition.
        
        Args:
            f: Objective function
            grad_f: Gradient function
            x: Current point
            grad: Current gradient
            c: Armijo constant (typically 0.1 to 0.5)
            rho: Backtracking factor (typically 0.5 to 0.9)
            
        Returns:
            Step size
        """
        alpha = 1.0
        fx = f(x)
        grad_norm_sq = np.dot(grad, grad)
        
        while f(x - alpha * grad) > fx - c * alpha * grad_norm_sq:
            alpha *= rho
            
        return alpha
    
    def _exact_line_search(self,
                          f: Callable,
                          grad_f: Callable,
                          x: np.ndarray,
                          grad: np.ndarray) -> float:
        """
        Exact line search for quadratic functions.
        For general functions, uses a simple grid search approximation.
        
        Args:
            f: Objective function
            grad_f: Gradient function
            x: Current point
            grad: Current gradient
            
        Returns:
            Optimal step size
        """
        # For quadratic: α* = ||∇f||² / (∇f^T H ∇f)
        # For general functions, use grid search
        alphas = np.logspace(-3, 1, 50)
        f_vals = [f(x - alpha * grad) for alpha in alphas]
        return alphas[np.argmin(f_vals)]


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])


def quadratic_bowl(x: np.ndarray) -> float:
    """Simple quadratic bowl: f(x,y) = x² + 4y²"""
    return x[0]**2 + 4 * x[1]**2


def quadratic_bowl_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of quadratic bowl"""
    return np.array([2 * x[0], 8 * x[1]])


def visualize_optimization(history: List[dict], 
                          f: Callable,
                          xlim: Tuple[float, float] = (-2, 2),
                          ylim: Tuple[float, float] = (-2, 2),
                          title: str = "Steepest Descent Optimization"):
    """
    Visualize the optimization trajectory.
    
    Args:
        history: Optimization history
        f: Objective function
        xlim: X-axis limits
        ylim: Y-axis limits
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Contour plot with trajectory
    ax1 = axes[0]
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    ax1.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    
    # Plot trajectory
    traj = np.array([h['x'] for h in history])
    ax1.plot(traj[:, 0], traj[:, 1], 'r.-', linewidth=2, markersize=8, label='Trajectory')
    ax1.plot(traj[0, 0], traj[0, 1], 'go', markersize=12, label='Start')
    ax1.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=15, label='End')
    
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title(f'{title}\nTrajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Convergence
    ax2 = axes[1]
    iterations = range(len(history))
    f_vals = [h['f'] for h in history]
    grad_norms = [h['grad_norm'] for h in history]
    
    ax2_twin = ax2.twinx()
    
    line1 = ax2.semilogy(iterations, f_vals, 'b-', linewidth=2, label='f(x)')
    line2 = ax2_twin.semilogy(iterations, grad_norms, 'r--', linewidth=2, label='||∇f(x)||')
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Function Value', color='b')
    ax2_twin.set_ylabel('Gradient Norm', color='r')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    ax2.set_title('Convergence')
    ax2.grid(True, alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.show()


def example_quadratic():
    """Example: Optimize quadratic bowl function"""
    print("=" * 60)
    print("Example 1: Quadratic Bowl Function")
    print("=" * 60)
    print("f(x,y) = x² + 4y²")
    print("Minimum: (0, 0), f* = 0")
    print()
    
    x0 = np.array([1.5, 1.5])
    
    # Try different line search strategies
    strategies = ['constant', 'backtracking', 'exact']
    
    for strategy in strategies:
        print(f"\nLine Search: {strategy}")
        print("-" * 40)
        
        optimizer = SteepestDescent(
            step_size=0.1,
            max_iter=100,
            tol=1e-6,
            line_search=strategy
        )
        
        x_opt, history = optimizer.optimize(quadratic_bowl, quadratic_bowl_grad, x0)
        
        print(f"Starting point: {x0}")
        print(f"Optimal point: {x_opt}")
        print(f"Optimal value: {quadratic_bowl(x_opt):.2e}")
        print(f"Iterations: {len(history) - 1}")
        print(f"Final gradient norm: {history[-1]['grad_norm']:.2e}")
        
        if strategy == 'backtracking':
            visualize_optimization(history, quadratic_bowl, 
                                 xlim=(-2, 2), ylim=(-2, 2),
                                 title=f"Steepest Descent ({strategy})")


def example_rosenbrock():
    """Example: Optimize Rosenbrock function"""
    print("\n" + "=" * 60)
    print("Example 2: Rosenbrock Function")
    print("=" * 60)
    print("f(x,y) = (1-x)² + 100(y-x²)²")
    print("Minimum: (1, 1), f* = 0")
    print()
    
    x0 = np.array([-1.0, 1.0])
    
    print("Line Search: backtracking")
    print("-" * 40)
    
    optimizer = SteepestDescent(
        step_size=0.01,
        max_iter=5000,
        tol=1e-6,
        line_search='backtracking'
    )
    
    x_opt, history = optimizer.optimize(rosenbrock, rosenbrock_grad, x0)
    
    print(f"Starting point: {x0}")
    print(f"Optimal point: {x_opt}")
    print(f"Optimal value: {rosenbrock(x_opt):.2e}")
    print(f"Iterations: {len(history) - 1}")
    print(f"Final gradient norm: {history[-1]['grad_norm']:.2e}")
    
    visualize_optimization(history, rosenbrock,
                         xlim=(-1.5, 1.5), ylim=(-0.5, 1.5),
                         title="Steepest Descent on Rosenbrock Function")


if __name__ == "__main__":
    # Run examples
    example_quadratic()
    example_rosenbrock()
    
    print("\n" + "=" * 60)
    print("Key Observations:")
    print("=" * 60)
    print("1. Constant step size: Simple but may not converge or be inefficient")
    print("2. Backtracking: Adaptive step size, better convergence")
    print("3. Exact line search: Best step size per iteration (when feasible)")
    print("4. Steepest descent can be slow for ill-conditioned problems")
    print("5. Zig-zagging behavior common in narrow valleys (Rosenbrock)")
