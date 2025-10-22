"""
Conjugate Gradient Method
==========================

The conjugate gradient (CG) method is an improved gradient method that uses
conjugate directions instead of the steepest descent direction. This leads to
much faster convergence, especially for quadratic functions.

Key Features:
- Converges in n steps for n-dimensional quadratic functions
- Much faster than steepest descent for ill-conditioned problems
- No zig-zagging behavior
- Requires only gradient information (no Hessian)
- Memory efficient (only stores previous direction)

Algorithm:
1. d_0 = -∇f(x_0)
2. For k = 0, 1, 2, ...
   - α_k = argmin f(x_k + α d_k)  [line search]
   - x_{k+1} = x_k + α_k d_k
   - β_k = ||∇f(x_{k+1})||² / ||∇f(x_k)||²  [Fletcher-Reeves]
   - d_{k+1} = -∇f(x_{k+1}) + β_k d_k

Common β formulas:
- Fletcher-Reeves (FR): β_k = ||g_{k+1}||² / ||g_k||²
- Polak-Ribière (PR): β_k = g_{k+1}^T(g_{k+1} - g_k) / ||g_k||²
- Hestenes-Stiefel (HS): β_k = g_{k+1}^T(g_{k+1} - g_k) / d_k^T(g_{k+1} - g_k)
- Dai-Yuan (DY): β_k = ||g_{k+1}||² / d_k^T(g_{k+1} - g_k)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class CGConfig:
    """Configuration for Conjugate Gradient method"""
    beta_method: str = 'FR'  # 'FR', 'PR', 'HS', 'DY'
    max_iter: int = 1000
    tol: float = 1e-6
    restart_threshold: Optional[int] = None  # Restart every n iterations
    line_search: str = 'backtracking'  # 'backtracking', 'exact'


class ConjugateGradient:
    """
    Conjugate Gradient optimizer with various β update formulas.
    """
    
    def __init__(self, config: Optional[CGConfig] = None):
        """
        Initialize the conjugate gradient optimizer.
        
        Args:
            config: Configuration object
        """
        self.config = config or CGConfig()
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
        d = -grad.copy()  # Initial search direction
        
        self.history = [{
            'x': x.copy(),
            'f': f(x),
            'grad_norm': np.linalg.norm(grad),
            'direction_norm': np.linalg.norm(d)
        }]
        
        for k in range(self.config.max_iter):
            grad_norm = np.linalg.norm(grad)
            
            # Check convergence
            if grad_norm < self.config.tol:
                print(f"Converged in {k} iterations")
                break
            
            # Check for restart
            if self.config.restart_threshold and k % self.config.restart_threshold == 0 and k > 0:
                d = -grad
                print(f"Restarted at iteration {k}")
            
            # Line search
            if self.config.line_search == 'backtracking':
                alpha = self._backtracking_line_search(f, grad_f, x, d, grad)
            else:
                alpha = self._exact_line_search(f, grad_f, x, d)
            
            # Update position
            x_new = x + alpha * d
            grad_new = grad_f(x_new)
            
            # Compute β (conjugate direction parameter)
            beta = self._compute_beta(grad, grad_new, d)
            
            # Update search direction
            d_new = -grad_new + beta * d
            
            # Store history
            self.history.append({
                'x': x_new.copy(),
                'f': f(x_new),
                'grad_norm': np.linalg.norm(grad_new),
                'step_size': alpha,
                'beta': beta,
                'direction_norm': np.linalg.norm(d_new)
            })
            
            # Update for next iteration
            x = x_new
            grad = grad_new
            d = d_new
            
        return x, self.history
    
    def _compute_beta(self,
                     grad: np.ndarray,
                     grad_new: np.ndarray,
                     d: np.ndarray) -> float:
        """
        Compute β parameter using specified method.
        
        Args:
            grad: Current gradient
            grad_new: New gradient
            d: Current direction
            
        Returns:
            β parameter
        """
        method = self.config.beta_method
        
        if method == 'FR':  # Fletcher-Reeves
            beta = np.dot(grad_new, grad_new) / np.dot(grad, grad)
            
        elif method == 'PR':  # Polak-Ribière
            y = grad_new - grad
            beta = np.dot(grad_new, y) / np.dot(grad, grad)
            beta = max(0, beta)  # PR+ (non-negative)
            
        elif method == 'HS':  # Hestenes-Stiefel
            y = grad_new - grad
            denominator = np.dot(d, y)
            if abs(denominator) < 1e-10:
                beta = 0
            else:
                beta = np.dot(grad_new, y) / denominator
                
        elif method == 'DY':  # Dai-Yuan
            y = grad_new - grad
            denominator = np.dot(d, y)
            if abs(denominator) < 1e-10:
                beta = 0
            else:
                beta = np.dot(grad_new, grad_new) / denominator
        else:
            raise ValueError(f"Unknown beta method: {method}")
        
        return beta
    
    def _backtracking_line_search(self,
                                  f: Callable,
                                  grad_f: Callable,
                                  x: np.ndarray,
                                  d: np.ndarray,
                                  grad: np.ndarray,
                                  c: float = 0.5,
                                  rho: float = 0.8) -> float:
        """Backtracking line search with Armijo condition."""
        alpha = 1.0
        fx = f(x)
        grad_d = np.dot(grad, d)
        
        while f(x + alpha * d) > fx + c * alpha * grad_d:
            alpha *= rho
            if alpha < 1e-10:
                break
                
        return alpha
    
    def _exact_line_search(self,
                          f: Callable,
                          grad_f: Callable,
                          x: np.ndarray,
                          d: np.ndarray) -> float:
        """Approximate exact line search using golden section search."""
        from scipy.optimize import golden
        
        def phi(alpha):
            return f(x + alpha * d)
        
        # Find bracket
        alpha_max = 1.0
        while phi(alpha_max) > phi(0):
            alpha_max *= 2
            if alpha_max > 1e6:
                break
        
        try:
            result = golden(phi, brack=(0, alpha_max/2, alpha_max), tol=1e-6)
            # golden returns a scalar, but type checker doesn't know this
            return float(result) if np.isscalar(result) else 0.1  # type: ignore
        except:
            return 0.1


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])


def quadratic_bowl(x: np.ndarray) -> float:
    """Ill-conditioned quadratic: f(x) = 0.5 * x^T A x"""
    A = np.array([[100, 0], [0, 1]])  # Condition number = 100
    return float(0.5 * x @ A @ x)


def quadratic_bowl_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of ill-conditioned quadratic"""
    A = np.array([[100, 0], [0, 1]])
    return A @ x


def visualize_comparison(histories: dict,
                        f: Callable,
                        xlim: Tuple[float, float] = (-2, 2),
                        ylim: Tuple[float, float] = (-2, 2),
                        title: str = "Conjugate Gradient Comparison"):
    """
    Compare different CG methods.
    
    Args:
        histories: Dictionary of {method_name: history}
        f: Objective function
        xlim: X-axis limits
        ylim: Y-axis limits
        title: Plot title
    """
    fig = plt.figure(figsize=(16, 5))
    
    # Plot 1: Contour with all trajectories
    ax1 = plt.subplot(131)
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    ax1.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    
    colors = ['r', 'b', 'g', 'orange', 'm']
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
    ax2.set_title('Function Value Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gradient norm convergence
    ax3 = plt.subplot(133)
    for idx, (method, history) in enumerate(histories.items()):
        grad_norms = [h['grad_norm'] for h in history]
        ax3.semilogy(range(len(grad_norms)), grad_norms,
                    f'{colors[idx]}-', linewidth=2, label=method)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient Norm (log scale)')
    ax3.set_title('Gradient Norm Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def example_ill_conditioned_quadratic():
    """Example: Compare CG methods on ill-conditioned quadratic"""
    print("=" * 70)
    print("Example 1: Ill-Conditioned Quadratic Function")
    print("=" * 70)
    print("f(x) = 0.5 * x^T A x, where A = diag(100, 1)")
    print("Condition number: 100")
    print("This problem showcases CG's advantage over steepest descent")
    print()
    
    x0 = np.array([1.0, 1.0])
    methods = ['FR', 'PR', 'HS', 'DY']
    histories = {}
    
    for method in methods:
        config = CGConfig(beta_method=method, max_iter=100, tol=1e-8)
        optimizer = ConjugateGradient(config)
        x_opt, history = optimizer.optimize(quadratic_bowl, quadratic_bowl_grad, x0)
        histories[method] = history
        
        print(f"\nMethod: {method}")
        print("-" * 40)
        print(f"Optimal point: {x_opt}")
        print(f"Optimal value: {quadratic_bowl(x_opt):.2e}")
        print(f"Iterations: {len(history) - 1}")
        print(f"Final gradient norm: {history[-1]['grad_norm']:.2e}")
    
    visualize_comparison(histories, quadratic_bowl,
                        xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
                        title="CG Methods on Ill-Conditioned Quadratic")


def example_rosenbrock():
    """Example: CG on Rosenbrock function"""
    print("\n" + "=" * 70)
    print("Example 2: Rosenbrock Function")
    print("=" * 70)
    print("f(x,y) = (1-x)² + 100(y-x²)²")
    print("This is a challenging non-convex optimization problem")
    print()
    
    x0 = np.array([-1.0, 1.0])
    methods = ['FR', 'PR']
    histories = {}
    
    for method in methods:
        config = CGConfig(
            beta_method=method,
            max_iter=1000,
            tol=1e-6,
            restart_threshold=50  # Restart every 50 iterations
        )
        optimizer = ConjugateGradient(config)
        x_opt, history = optimizer.optimize(rosenbrock, rosenbrock_grad, x0)
        histories[method] = history
        
        print(f"\nMethod: {method} (with restart every 50 iterations)")
        print("-" * 40)
        print(f"Optimal point: {x_opt}")
        print(f"Optimal value: {rosenbrock(x_opt):.2e}")
        print(f"Iterations: {len(history) - 1}")
        print(f"Final gradient norm: {history[-1]['grad_norm']:.2e}")
    
    visualize_comparison(histories, rosenbrock,
                        xlim=(-1.5, 1.5), ylim=(-0.5, 1.5),
                        title="CG Methods on Rosenbrock Function")


def demonstrate_cg_efficiency():
    """Demonstrate CG vs Steepest Descent on quadratic"""
    print("\n" + "=" * 70)
    print("Example 3: CG vs Steepest Descent")
    print("=" * 70)
    print("Demonstrating superior convergence of CG on quadratic problems")
    print()
    
    # Import steepest descent for comparison
    from steepest_descent import SteepestDescent
    
    x0 = np.array([1.0, 1.0])
    
    # CG method
    config = CGConfig(beta_method='FR', max_iter=100, tol=1e-8)
    cg_optimizer = ConjugateGradient(config)
    x_cg, history_cg = cg_optimizer.optimize(quadratic_bowl, quadratic_bowl_grad, x0)
    
    # Steepest descent
    sd_optimizer = SteepestDescent(step_size=0.01, max_iter=100, tol=1e-8, line_search='backtracking')
    x_sd, history_sd = sd_optimizer.optimize(quadratic_bowl, quadratic_bowl_grad, x0)
    
    histories = {
        'CG (FR)': history_cg,
        'Steepest Descent': history_sd
    }
    
    print(f"\nConjugate Gradient:")
    print(f"  Iterations: {len(history_cg) - 1}")
    print(f"  Final f(x): {history_cg[-1]['f']:.2e}")
    
    print(f"\nSteepest Descent:")
    print(f"  Iterations: {len(history_sd) - 1}")
    print(f"  Final f(x): {history_sd[-1]['f']:.2e}")
    
    print(f"\nSpeedup: {(len(history_sd) - 1) / (len(history_cg) - 1):.1f}x faster")
    
    visualize_comparison(histories, quadratic_bowl,
                        xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
                        title="CG vs Steepest Descent")


if __name__ == "__main__":
    # Run examples
    example_ill_conditioned_quadratic()
    example_rosenbrock()
    
    # Uncomment to compare with steepest descent
    # demonstrate_cg_efficiency()
    
    print("\n" + "=" * 70)
    print("Key Observations:")
    print("=" * 70)
    print("1. CG converges in n steps for n-dimensional quadratic functions")
    print("2. FR and PR methods perform similarly, PR often slightly better")
    print("3. Restarting CG helps on non-quadratic functions")
    print("4. CG is much faster than steepest descent on ill-conditioned problems")
    print("5. No zig-zagging behavior - conjugate directions are orthogonal")
    print("6. Memory efficient: only stores previous direction")
