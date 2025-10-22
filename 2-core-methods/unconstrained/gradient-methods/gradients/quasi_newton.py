"""
Quasi-Newton Methods (BFGS and L-BFGS)
======================================

Quasi-Newton methods approximate Newton's method without computing the Hessian matrix.
They build an approximation of the Hessian (or its inverse) using gradient information.

Key Features:
- Superlinear convergence (faster than CG, slower than Newton)
- No need to compute or store Hessian
- BFGS: stores full n×n matrix (O(n²) memory)
- L-BFGS: stores only m recent updates (O(mn) memory)
- Most popular general-purpose optimization methods

BFGS Algorithm:
1. Start with initial Hessian approximation B_0 (usually identity)
2. For k = 0, 1, 2, ...
   - Compute search direction: d_k = -B_k^{-1} ∇f(x_k)
   - Line search: α_k = argmin f(x_k + α d_k)
   - Update: x_{k+1} = x_k + α_k d_k
   - Compute: s_k = x_{k+1} - x_k, y_k = ∇f(x_{k+1}) - ∇f(x_k)
   - Update Hessian approximation using BFGS formula

L-BFGS (Limited-memory BFGS):
- Stores only m most recent {s_k, y_k} pairs
- Computes B_k^{-1} ∇f(x_k) implicitly using two-loop recursion
- Memory: O(mn) instead of O(n²)
- Typical m: 5-20
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Deque
from collections import deque
from dataclasses import dataclass


@dataclass
class QuasiNewtonConfig:
    """Configuration for Quasi-Newton methods"""
    method: str = 'BFGS'  # 'BFGS' or 'L-BFGS'
    memory_size: int = 10  # For L-BFGS only
    max_iter: int = 1000
    tol: float = 1e-6
    line_search: str = 'backtracking'


class BFGS:
    """
    BFGS (Broyden-Fletcher-Goldfarb-Shanno) optimizer.
    Stores full inverse Hessian approximation.
    """
    
    def __init__(self, config: Optional[QuasiNewtonConfig] = None):
        """Initialize BFGS optimizer."""
        self.config = config or QuasiNewtonConfig(method='BFGS')
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
        n = len(x0)
        x = x0.copy()
        grad = grad_f(x)
        H = np.eye(n)  # Initial inverse Hessian approximation
        
        self.history = [{
            'x': x.copy(),
            'f': f(x),
            'grad_norm': np.linalg.norm(grad)
        }]
        
        for k in range(self.config.max_iter):
            grad_norm = np.linalg.norm(grad)
            
            # Check convergence
            if grad_norm < self.config.tol:
                print(f"Converged in {k} iterations")
                break
            
            # Compute search direction
            d = -H @ grad
            
            # Line search
            alpha = self._backtracking_line_search(f, grad_f, x, d, grad)
            
            # Update
            s = alpha * d  # Step
            x_new = x + s
            grad_new = grad_f(x_new)
            y = grad_new - grad  # Gradient difference
            
            # BFGS update of inverse Hessian
            rho = 1.0 / (y @ s)
            if rho > 0:  # Only update if curvature condition satisfied
                I = np.eye(n)
                H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            
            # Store history
            self.history.append({
                'x': x_new.copy(),
                'f': f(x_new),
                'grad_norm': np.linalg.norm(grad_new),
                'step_size': alpha,
                'curvature': y @ s
            })
            
            # Update for next iteration
            x = x_new
            grad = grad_new
            
        return x, self.history
    
    def _backtracking_line_search(self,
                                  f: Callable,
                                  grad_f: Callable,
                                  x: np.ndarray,
                                  d: np.ndarray,
                                  grad: np.ndarray,
                                  c: float = 1e-4,
                                  rho: float = 0.9) -> float:
        """Backtracking line search with Wolfe conditions."""
        alpha = 1.0  # Start with full Newton step
        fx = f(x)
        grad_d = np.dot(grad, d)
        
        # Armijo condition
        while f(x + alpha * d) > fx + c * alpha * grad_d:
            alpha *= rho
            if alpha < 1e-10:
                break
                
        return alpha


class LBFGS:
    """
    L-BFGS (Limited-memory BFGS) optimizer.
    Stores only m recent {s, y} pairs for memory efficiency.
    """
    
    def __init__(self, config: Optional[QuasiNewtonConfig] = None):
        """Initialize L-BFGS optimizer."""
        self.config = config or QuasiNewtonConfig(method='L-BFGS', memory_size=10)
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
        grad = grad_f(x)
        
        # Storage for {s, y} pairs
        s_list: Deque = deque(maxlen=self.config.memory_size)
        y_list: Deque = deque(maxlen=self.config.memory_size)
        rho_list: Deque = deque(maxlen=self.config.memory_size)
        
        self.history = [{
            'x': x.copy(),
            'f': f(x),
            'grad_norm': np.linalg.norm(grad)
        }]
        
        for k in range(self.config.max_iter):
            grad_norm = np.linalg.norm(grad)
            
            # Check convergence
            if grad_norm < self.config.tol:
                print(f"Converged in {k} iterations")
                break
            
            # Compute search direction using two-loop recursion
            d = self._two_loop_recursion(grad, s_list, y_list, rho_list)
            
            # Line search
            alpha = self._backtracking_line_search(f, grad_f, x, d, grad)
            
            # Update
            s = alpha * d
            x_new = x + s
            grad_new = grad_f(x_new)
            y = grad_new - grad
            
            # Store {s, y} pair if curvature condition satisfied
            y_s = y @ s
            if y_s > 1e-10:
                s_list.append(s)
                y_list.append(y)
                rho_list.append(1.0 / y_s)
            
            # Store history
            self.history.append({
                'x': x_new.copy(),
                'f': f(x_new),
                'grad_norm': np.linalg.norm(grad_new),
                'step_size': alpha,
                'memory_used': len(s_list),
                'curvature': y_s
            })
            
            # Update for next iteration
            x = x_new
            grad = grad_new
            
        return x, self.history
    
    def _two_loop_recursion(self,
                           grad: np.ndarray,
                           s_list: Deque,
                           y_list: Deque,
                           rho_list: Deque) -> np.ndarray:
        """
        Two-loop recursion to compute search direction.
        Efficiently computes H_k * grad without forming H_k explicitly.
        """
        q = grad.copy()
        m = len(s_list)
        alpha = np.zeros(m)
        
        # First loop (backward)
        for i in range(m - 1, -1, -1):
            alpha[i] = rho_list[i] * (s_list[i] @ q)
            q = q - alpha[i] * y_list[i]
        
        # Initial Hessian approximation (scaling)
        if m > 0:
            gamma = (s_list[-1] @ y_list[-1]) / (y_list[-1] @ y_list[-1])
            r = gamma * q
        else:
            r = q
        
        # Second loop (forward)
        for i in range(m):
            beta = rho_list[i] * (y_list[i] @ r)
            r = r + s_list[i] * (alpha[i] - beta)
        
        return -r  # Search direction
    
    def _backtracking_line_search(self,
                                  f: Callable,
                                  grad_f: Callable,
                                  x: np.ndarray,
                                  d: np.ndarray,
                                  grad: np.ndarray,
                                  c: float = 1e-4,
                                  rho: float = 0.9) -> float:
        """Backtracking line search."""
        alpha = 1.0
        fx = f(x)
        grad_d = np.dot(grad, d)
        
        while f(x + alpha * d) > fx + c * alpha * grad_d:
            alpha *= rho
            if alpha < 1e-10:
                break
                
        return alpha


# Test functions
def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])


def extended_rosenbrock(x: np.ndarray) -> float:
    """Extended Rosenbrock function for high dimensions"""
    n = len(x)
    return sum((1 - x[i])**2 + 100 * (x[i+1] - x[i]**2)**2 for i in range(n-1))


def extended_rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of extended Rosenbrock"""
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n-1):
        grad[i] += -2 * (1 - x[i]) - 400 * x[i] * (x[i+1] - x[i]**2)
        grad[i+1] += 200 * (x[i+1] - x[i]**2)
    
    return grad


def visualize_comparison(histories: dict,
                        f: Callable,
                        xlim: Tuple[float, float] = (-2, 2),
                        ylim: Tuple[float, float] = (-2, 2),
                        title: str = "Quasi-Newton Methods"):
    """Compare different optimization methods."""
    fig = plt.figure(figsize=(16, 5))
    
    # Plot 1: Contour with trajectories
    ax1 = plt.subplot(131)
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    ax1.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    
    colors = ['r', 'b', 'g', 'orange']
    for idx, (method, history) in enumerate(histories.items()):
        traj = np.array([h['x'] for h in history])
        ax1.plot(traj[:, 0], traj[:, 1],
                f'{colors[idx]}.-', linewidth=2, markersize=6,
                label=f'{method} ({len(history)} iter)', alpha=0.7)
    
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title(f'{title}\nTrajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Function value convergence
    ax2 = plt.subplot(132)
    for idx, (method, history) in enumerate(histories.items()):
        f_vals = [h['f'] for h in history]
        ax2.semilogy(range(len(f_vals)), f_vals,
                    f'{colors[idx]}-', linewidth=2, label=method)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Function Value (log scale)')
    ax2.set_title('Convergence Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gradient norm
    ax3 = plt.subplot(133)
    for idx, (method, history) in enumerate(histories.items()):
        grad_norms = [h['grad_norm'] for h in history]
        ax3.semilogy(range(len(grad_norms)), grad_norms,
                    f'{colors[idx]}-', linewidth=2, label=method)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient Norm (log scale)')
    ax3.set_title('Gradient Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def example_2d_rosenbrock():
    """Example: Compare BFGS and L-BFGS on 2D Rosenbrock"""
    print("=" * 70)
    print("Example 1: 2D Rosenbrock Function")
    print("=" * 70)
    print("f(x,y) = (1-x)² + 100(y-x²)²")
    print("Comparing BFGS and L-BFGS")
    print()
    
    x0 = np.array([-1.0, 1.0])
    histories = {}
    
    # BFGS
    bfgs_config = QuasiNewtonConfig(method='BFGS', max_iter=200)
    bfgs = BFGS(bfgs_config)
    x_bfgs, history_bfgs = bfgs.optimize(rosenbrock, rosenbrock_grad, x0)
    histories['BFGS'] = history_bfgs
    
    print("BFGS Results:")
    print("-" * 40)
    print(f"Optimal point: {x_bfgs}")
    print(f"Optimal value: {rosenbrock(x_bfgs):.2e}")
    print(f"Iterations: {len(history_bfgs) - 1}")
    print(f"Final gradient norm: {history_bfgs[-1]['grad_norm']:.2e}")
    
    # L-BFGS
    lbfgs_config = QuasiNewtonConfig(method='L-BFGS', memory_size=5, max_iter=200)
    lbfgs = LBFGS(lbfgs_config)
    x_lbfgs, history_lbfgs = lbfgs.optimize(rosenbrock, rosenbrock_grad, x0)
    histories['L-BFGS (m=5)'] = history_lbfgs
    
    print("\nL-BFGS Results:")
    print("-" * 40)
    print(f"Optimal point: {x_lbfgs}")
    print(f"Optimal value: {rosenbrock(x_lbfgs):.2e}")
    print(f"Iterations: {len(history_lbfgs) - 1}")
    print(f"Final gradient norm: {history_lbfgs[-1]['grad_norm']:.2e}")
    
    visualize_comparison(histories, rosenbrock,
                        xlim=(-1.5, 1.5), ylim=(-0.5, 1.5),
                        title="BFGS vs L-BFGS on 2D Rosenbrock")


def example_high_dimensional():
    """Example: L-BFGS on high-dimensional problem"""
    print("\n" + "=" * 70)
    print("Example 2: High-Dimensional Extended Rosenbrock")
    print("=" * 70)
    print("Demonstrating L-BFGS efficiency in high dimensions")
    print()
    
    dimensions = [10, 50, 100]
    
    for n in dimensions:
        print(f"\nDimension n = {n}:")
        print("-" * 40)
        
        x0 = np.zeros(n)
        x0[::2] = -1.2  # Alternate initialization
        x0[1::2] = 1.0
        
        # L-BFGS with different memory sizes
        for m in [5, 10, 20]:
            if m > n // 2:  # Skip if memory size too large
                continue
                
            config = QuasiNewtonConfig(method='L-BFGS', memory_size=m, max_iter=1000)
            lbfgs = LBFGS(config)
            x_opt, history = lbfgs.optimize(extended_rosenbrock, extended_rosenbrock_grad, x0)
            
            print(f"  L-BFGS (m={m}): {len(history)-1} iterations, "
                  f"f* = {extended_rosenbrock(x_opt):.2e}")


def compare_all_methods():
    """Compare gradient descent, CG, and quasi-Newton"""
    print("\n" + "=" * 70)
    print("Example 3: Method Comparison on 2D Rosenbrock")
    print("=" * 70)
    print("Comparing convergence speed of different methods")
    print()
    
    x0 = np.array([-1.0, 1.0])
    histories = {}
    
    # Import other methods
    try:
        from steepest_descent import SteepestDescent
        from conjugate_gradient import ConjugateGradient, CGConfig
        
        # Steepest Descent
        sd = SteepestDescent(step_size=0.001, max_iter=1000, line_search='backtracking')
        _, history_sd = sd.optimize(rosenbrock, rosenbrock_grad, x0)
        histories['Steepest Descent'] = history_sd
        
        # Conjugate Gradient
        cg = ConjugateGradient(CGConfig(beta_method='PR', max_iter=200))
        _, history_cg = cg.optimize(rosenbrock, rosenbrock_grad, x0)
        histories['CG (PR)'] = history_cg
        
    except ImportError:
        print("Note: Could not import steepest_descent or conjugate_gradient")
    
    # BFGS
    bfgs = BFGS(QuasiNewtonConfig(max_iter=200))
    _, history_bfgs = bfgs.optimize(rosenbrock, rosenbrock_grad, x0)
    histories['BFGS'] = history_bfgs
    
    # Print comparison
    print("\nIterations to convergence:")
    print("-" * 40)
    for method, history in histories.items():
        print(f"{method:20s}: {len(history)-1:4d} iterations")
    
    if len(histories) > 1:
        visualize_comparison(histories, rosenbrock,
                            xlim=(-1.5, 1.5), ylim=(-0.5, 1.5),
                            title="Method Comparison")


if __name__ == "__main__":
    # Run examples
    example_2d_rosenbrock()
    example_high_dimensional()
    
    # Uncomment to compare with other methods
    # compare_all_methods()
    
    print("\n" + "=" * 70)
    print("Key Observations:")
    print("=" * 70)
    print("1. BFGS has superlinear convergence (faster than CG)")
    print("2. L-BFGS performs nearly as well as BFGS with much less memory")
    print("3. Memory size m=5-20 usually sufficient for L-BFGS")
    print("4. Quasi-Newton methods are the workhorses of optimization")
    print("5. L-BFGS is preferred for large-scale problems (n > 1000)")
    print("6. Both methods benefit from good line search")
