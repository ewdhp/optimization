"""
Quasi-Newton Methods (BFGS and Variants)

Quasi-Newton methods approximate the Hessian using only gradient information,
avoiding the O(n³) cost of computing and inverting the true Hessian. They
achieve superlinear convergence, providing a good balance between the speed
of Newton's method and the low cost of gradient descent.

Key Methods:
1. BFGS (Broyden-Fletcher-Goldfarb-Shanno): Most popular quasi-Newton method
2. L-BFGS (Limited-memory BFGS): For large-scale problems
3. DFP (Davidon-Fletcher-Powell): Historical predecessor to BFGS
4. SR1 (Symmetric Rank-1): For non-convex problems

BFGS Update:
    B_{k+1} = B_k - (B_k s_k s_k^T B_k)/(s_k^T B_k s_k) + (y_k y_k^T)/(y_k^T s_k)

where:
- s_k = x_{k+1} - x_k (step taken)
- y_k = ∇f_{k+1} - ∇f_k (gradient change)
- B_k approximates Hessian ∇²f

Convergence:
- Superlinear convergence: lim_{k→∞} ||x_{k+1} - x*|| / ||x_k - x*|| = 0
- Faster than gradient descent, cheaper than Newton
- Requires O(n²) storage for BFGS, O(mn) for L-BFGS
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional, Tuple
import warnings
from collections import deque

warnings.filterwarnings('ignore')

class BFGS:
    """
    BFGS quasi-Newton method implementation.
    """
    
    def __init__(self,
                 objective: Callable[[np.ndarray], float],
                 gradient: Callable[[np.ndarray], np.ndarray],
                 name: str = "Function"):
        """
        Initialize BFGS optimizer.
        
        Args:
            objective: Objective function f(x)
            gradient: Gradient function ∇f(x)
            name: Function name
        """
        self.objective = objective
        self.gradient = gradient
        self.name = name
        
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
                max_iters: int = 1000,
                tolerance: float = 1e-6,
                c1: float = 1e-4,
                c2: float = 0.9,
                verbose: bool = False) -> Dict:
        """
        BFGS optimization with Wolfe line search.
        
        Args:
            x0: Initial point
            max_iters: Maximum iterations
            tolerance: Convergence tolerance
            c1: Armijo constant for line search
            c2: Curvature constant for line search
            verbose: Print progress
            
        Returns:
            Optimization results
        """
        self.reset_history()
        
        x = np.array(x0, dtype=float)
        n = len(x)
        
        # Initialize inverse Hessian approximation as identity
        H_inv = np.eye(n)
        
        grad = self.gradient(x)
        
        for k in range(max_iters):
            grad_norm = np.linalg.norm(grad)
            f_val = self.objective(x)
            
            # Store history
            self.history['x'].append(x.copy())
            self.history['f'].append(f_val)
            self.history['grad_norm'].append(grad_norm)
            
            # Check convergence
            if grad_norm < tolerance:
                if verbose:
                    print(f"Converged in {k} iterations")
                break
            
            # BFGS direction: p = -H_inv * grad
            p = -np.dot(H_inv, grad)
            
            # Wolfe line search
            alpha, grad_new = self._wolfe_line_search(x, p, grad, f_val, c1, c2)
            
            self.history['step_size'].append(alpha)
            
            # Update position
            s = alpha * p
            x_new = x + s
            
            # Gradient change
            y = grad_new - grad
            
            # BFGS update of inverse Hessian
            rho = 1.0 / np.dot(y, s)
            
            if np.isfinite(rho) and rho > 1e-10:
                # Ensure curvature condition: y^T s > 0
                A1 = np.eye(n) - rho * np.outer(s, y)
                A2 = np.eye(n) - rho * np.outer(y, s)
                H_inv = np.dot(A1, np.dot(H_inv, A2)) + rho * np.outer(s, s)
            
            # Update for next iteration
            x = x_new
            grad = grad_new
            
            if verbose and k % 100 == 0:
                print(f"Iter {k}: f = {f_val:.6e}, ||∇f|| = {grad_norm:.6e}, α = {alpha:.6f}")
        
        return {
            'x_optimal': x,
            'f_optimal': self.objective(x),
            'iterations': k + 1,
            'gradient_norm': np.linalg.norm(self.gradient(x)),
            'converged': grad_norm < tolerance,
            'history': self.history
        }
    
    def _wolfe_line_search(self, x, p, grad, f_x, c1, c2, max_iters=50):
        """Wolfe line search implementation."""
        alpha = 1.0
        alpha_low = 0
        alpha_high = np.inf
        
        directional_deriv = np.dot(grad, p)
        
        for i in range(max_iters):
            x_new = x + alpha * p
            f_new = self.objective(x_new)
            grad_new = self.gradient(x_new)
            
            # Check Armijo condition
            armijo = f_new <= f_x + c1 * alpha * directional_deriv
            
            # Check curvature condition
            directional_deriv_new = np.dot(grad_new, p)
            curvature = directional_deriv_new >= c2 * directional_deriv
            
            if armijo and curvature:
                return alpha, grad_new
            
            # Update bounds
            if not armijo:
                alpha_high = alpha
            elif not curvature:
                alpha_low = alpha
            
            # Bisection
            if alpha_high < np.inf:
                alpha = 0.5 * (alpha_low + alpha_high)
            else:
                alpha = 2 * alpha
            
            if alpha < 1e-16:
                return alpha, grad_new
        
        return alpha, self.gradient(x + alpha * p)


class LBFGS:
    """
    Limited-memory BFGS for large-scale optimization.
    """
    
    def __init__(self,
                 objective: Callable[[np.ndarray], float],
                 gradient: Callable[[np.ndarray], np.ndarray],
                 memory_size: int = 10,
                 name: str = "Function"):
        """
        Initialize L-BFGS optimizer.
        
        Args:
            objective: Objective function
            gradient: Gradient function
            memory_size: Number of (s, y) pairs to store
            name: Function name
        """
        self.objective = objective
        self.gradient = gradient
        self.memory_size = memory_size
        self.name = name
        
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
                max_iters: int = 1000,
                tolerance: float = 1e-6,
                c1: float = 1e-4,
                c2: float = 0.9,
                verbose: bool = False) -> Dict:
        """
        L-BFGS optimization.
        
        Args:
            x0: Initial point
            max_iters: Maximum iterations
            tolerance: Convergence tolerance
            c1: Armijo constant
            c2: Curvature constant
            verbose: Print progress
            
        Returns:
            Optimization results
        """
        self.reset_history()
        
        x = np.array(x0, dtype=float)
        
        # Storage for (s, y) pairs
        s_history = deque(maxlen=self.memory_size)
        y_history = deque(maxlen=self.memory_size)
        rho_history = deque(maxlen=self.memory_size)
        
        grad = self.gradient(x)
        
        for k in range(max_iters):
            grad_norm = np.linalg.norm(grad)
            f_val = self.objective(x)
            
            # Store history
            self.history['x'].append(x.copy())
            self.history['f'].append(f_val)
            self.history['grad_norm'].append(grad_norm)
            
            # Check convergence
            if grad_norm < tolerance:
                if verbose:
                    print(f"Converged in {k} iterations")
                break
            
            # Compute search direction using two-loop recursion
            p = self._two_loop_recursion(grad, s_history, y_history, rho_history)
            
            # Line search (simplified backtracking)
            alpha = self._backtracking_line_search(x, p, grad, f_val, c1)
            
            self.history['step_size'].append(alpha)
            
            # Update
            x_new = x + alpha * p
            grad_new = self.gradient(x_new)
            
            # Store s and y for L-BFGS update
            s = x_new - x
            y = grad_new - grad
            
            rho = 1.0 / np.dot(y, s)
            
            if np.isfinite(rho) and rho > 1e-10:
                s_history.append(s)
                y_history.append(y)
                rho_history.append(rho)
            
            x = x_new
            grad = grad_new
            
            if verbose and k % 100 == 0:
                print(f"Iter {k}: f = {f_val:.6e}, ||∇f|| = {grad_norm:.6e}")
        
        return {
            'x_optimal': x,
            'f_optimal': self.objective(x),
            'iterations': k + 1,
            'gradient_norm': np.linalg.norm(self.gradient(x)),
            'converged': grad_norm < tolerance,
            'history': self.history
        }
    
    def _two_loop_recursion(self, grad, s_history, y_history, rho_history):
        """L-BFGS two-loop recursion to compute search direction."""
        q = grad.copy()
        m = len(s_history)
        
        alpha_list = []
        
        # First loop (backward)
        for i in range(m-1, -1, -1):
            alpha_i = rho_history[i] * np.dot(s_history[i], q)
            alpha_list.append(alpha_i)
            q = q - alpha_i * y_history[i]
        
        # Compute initial Hessian approximation H_0 = γI
        if m > 0:
            gamma = np.dot(s_history[-1], y_history[-1]) / np.dot(y_history[-1], y_history[-1])
            r = gamma * q
        else:
            r = q
        
        # Second loop (forward)
        alpha_list.reverse()
        for i in range(m):
            beta = rho_history[i] * np.dot(y_history[i], r)
            r = r + s_history[i] * (alpha_list[i] - beta)
        
        return -r
    
    def _backtracking_line_search(self, x, p, grad, f_x, c1, beta=0.5):
        """Simple backtracking line search."""
        alpha = 1.0
        directional_deriv = np.dot(grad, p)
        
        for _ in range(50):
            x_new = x + alpha * p
            f_new = self.objective(x_new)
            
            if f_new <= f_x + c1 * alpha * directional_deriv:
                return alpha
            
            alpha *= beta
            
            if alpha < 1e-16:
                return alpha
        
        return alpha


def demonstrate_quasi_newton():
    """
    Comprehensive demonstration of quasi-Newton methods.
    """
    print("🔧 QUASI-NEWTON METHODS (BFGS & L-BFGS)")
    print("=" * 60)
    
    # Test Problem 1: Rosenbrock function
    print("\n🎯 EXAMPLE 1: Rosenbrock Function")
    print("-" * 50)
    
    def rosenbrock_obj(x):
        x = np.array(x)
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_grad(x):
        x = np.array(x)
        grad_x = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        grad_y = 200*(x[1] - x[0]**2)
        return np.array([grad_x, grad_y])
    
    bfgs = BFGS(rosenbrock_obj, rosenbrock_grad, "Rosenbrock")
    lbfgs = LBFGS(rosenbrock_obj, rosenbrock_grad, memory_size=5, name="Rosenbrock")
    
    x0 = np.array([-1.0, 2.0])
    print(f"Starting point: {x0}")
    print(f"True minimum: [1, 1]")
    
    # BFGS
    result_bfgs = bfgs.optimize(x0, max_iters=1000, tolerance=1e-6, verbose=False)
    print(f"\nBFGS Results:")
    print(f"  Iterations: {result_bfgs['iterations']}")
    print(f"  Final x: {result_bfgs['x_optimal']}")
    print(f"  Final f: {result_bfgs['f_optimal']:.8f}")
    print(f"  Final ||∇f||: {result_bfgs['gradient_norm']:.2e}")
    
    # L-BFGS
    result_lbfgs = lbfgs.optimize(x0, max_iters=1000, tolerance=1e-6, verbose=False)
    print(f"\nL-BFGS Results:")
    print(f"  Iterations: {result_lbfgs['iterations']}")
    print(f"  Final x: {result_lbfgs['x_optimal']}")
    print(f"  Final f: {result_lbfgs['f_optimal']:.8f}")
    print(f"  Final ||∇f||: {result_lbfgs['gradient_norm']:.2e}")
    
    # Quadratic problem for comparison
    print("\n🎯 EXAMPLE 2: Ill-Conditioned Quadratic")
    print("-" * 50)
    
    Q = np.array([[10, 0], [0, 0.1]])
    b = np.array([1, 1])
    x_star = np.linalg.solve(Q, b)
    
    def quad_obj(x):
        x = np.array(x)
        return 0.5 * np.dot(x, np.dot(Q, x)) - np.dot(b, x)
    
    def quad_grad(x):
        x = np.array(x)
        return np.dot(Q, x) - b
    
    bfgs2 = BFGS(quad_obj, quad_grad, "Quadratic")
    
    x0_quad = np.array([5, 5])
    result_bfgs_quad = bfgs2.optimize(x0_quad, max_iters=200, tolerance=1e-8)
    
    print(f"Condition number: {np.linalg.cond(Q):.1f}")
    print(f"BFGS iterations: {result_bfgs_quad['iterations']}")
    print(f"Final error: {np.linalg.norm(result_bfgs_quad['x_optimal'] - x_star):.2e}")
    
    # Visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Convergence comparison
    ax1 = plt.subplot(2, 3, 1)
    
    ax1.semilogy(range(len(result_bfgs['history']['f'])),
                result_bfgs['history']['f'],
                'b-o', linewidth=2, markersize=4, label='BFGS')
    ax1.semilogy(range(len(result_lbfgs['history']['f'])),
                result_lbfgs['history']['f'],
                'r-s', linewidth=2, markersize=3, label='L-BFGS')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('f(x) (log scale)')
    ax1.set_title('Function Value Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gradient norm convergence
    ax2 = plt.subplot(2, 3, 2)
    
    ax2.semilogy(range(len(result_bfgs['history']['grad_norm'])),
                result_bfgs['history']['grad_norm'],
                'b-o', linewidth=2, markersize=4, label='BFGS')
    ax2.semilogy(range(len(result_lbfgs['history']['grad_norm'])),
                result_lbfgs['history']['grad_norm'],
                'r-s', linewidth=2, markersize=3, label='L-BFGS')
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('||∇f(x)|| (log scale)')
    ax2.set_title('Gradient Norm Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Optimization paths on Rosenbrock
    ax3 = plt.subplot(2, 3, 3)
    
    # Contour plot
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100*(Y - X**2)**2
    
    levels = np.logspace(0, 3, 20)
    contour = ax3.contour(X, Y, Z, levels=levels, alpha=0.6, colors='gray')
    
    # BFGS path
    bfgs_x = [x[0] for x in result_bfgs['history']['x']]
    bfgs_y = [x[1] for x in result_bfgs['history']['x']]
    
    ax3.plot(bfgs_x, bfgs_y, 'b-o', linewidth=2, markersize=3,
            alpha=0.7, label='BFGS path')
    ax3.plot(x0[0], x0[1], 'go', markersize=10, label='Start')
    ax3.plot(1, 1, 'r*', markersize=15, label='Optimum')
    
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    ax3.set_title('BFGS Path on Rosenbrock')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Comparison with gradient descent
    ax4 = plt.subplot(2, 3, 4)
    
    # Simple gradient descent for comparison
    from steepest_descent import SteepestDescent
    gd = SteepestDescent(quad_obj, quad_grad)
    result_gd = gd.optimize_backtracking(x0_quad, max_iters=200, tolerance=1e-8)
    
    errors_bfgs = [np.linalg.norm(x - x_star) 
                   for x in result_bfgs_quad['history']['x']]
    errors_gd = [np.linalg.norm(x - x_star)
                 for x in result_gd['history']['x']]
    
    ax4.semilogy(range(len(errors_bfgs)), errors_bfgs, 'b-o',
                linewidth=2, markersize=4, label='BFGS')
    ax4.semilogy(range(len(errors_gd)), errors_gd, 'r-s',
                linewidth=2, markersize=3, label='Gradient Descent')
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('||x_k - x*|| (log scale)')
    ax4.set_title('BFGS vs Gradient Descent (Ill-Conditioned)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Step sizes
    ax5 = plt.subplot(2, 3, 5)
    
    ax5.plot(range(len(result_bfgs['history']['step_size'])),
            result_bfgs['history']['step_size'],
            'b-o', linewidth=2, markersize=4, label='BFGS')
    ax5.axhline(y=1.0, color='g', linestyle='--', alpha=0.7,
               label='Full step')
    
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Step Size α')
    ax5.set_title('BFGS Step Sizes (Rosenbrock)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Superlinear convergence demonstration
    ax6 = plt.subplot(2, 3, 6)
    
    if len(errors_bfgs) > 5:
        # Plot convergence ratios
        ratios = []
        for i in range(1, min(len(errors_bfgs), 20)):
            if errors_bfgs[i-1] > 1e-15:
                ratio = errors_bfgs[i] / errors_bfgs[i-1]
                ratios.append(ratio)
        
        ax6.plot(range(len(ratios)), ratios, 'b-o', linewidth=2, markersize=6)
        ax6.axhline(y=1.0, color='r', linestyle='--', alpha=0.7,
                   label='Linear convergence threshold')
        
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('e_{k+1} / e_k')
        ax6.set_title('Convergence Ratio (Superlinear if → 0)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def quasi_newton_theory():
    """
    Theory and comparison of quasi-Newton methods.
    """
    print("\n📚 QUASI-NEWTON THEORY")
    print("=" * 60)
    
    print("🔑 KEY IDEA:")
    print("  Approximate Hessian using only gradient information")
    print("  Build approximation iteratively: B_{k+1} from B_k")
    print("  Secant condition: B_{k+1} s_k = y_k")
    print("  where s_k = x_{k+1} - x_k, y_k = ∇f_{k+1} - ∇f_k")
    
    print("\n📊 BFGS UPDATE FORMULA:")
    print("  H_{k+1}^{-1} = (I - ρ_k s_k y_k^T) H_k^{-1} (I - ρ_k y_k s_k^T)")
    print("                 + ρ_k s_k s_k^T")
    print("  where ρ_k = 1/(y_k^T s_k)")
    
    print("\n💡 PROPERTIES:")
    print("1. SUPERLINEAR CONVERGENCE:")
    print("   lim_{k→∞} ||x_{k+1} - x*|| / ||x_k - x*|| = 0")
    print("   Faster than linear, slower than quadratic")
    
    print("\n2. SELF-CORRECTING:")
    print("   Even if H_0 ≠ ∇²f(x_0), BFGS converges")
    print("   Usually start with H_0 = I")
    
    print("\n3. POSITIVE DEFINITENESS:")
    print("   If H_0 ≻ 0 and y_k^T s_k > 0, then H_k ≻ 0 for all k")
    print("   Guaranteed descent directions")
    
    print("\n⚖️  BFGS vs L-BFGS:")
    print("  BFGS:")
    print("    ✓ Full Hessian approximation")
    print("    ✓ Better for small-medium problems (n < 1000)")
    print("    ✗ O(n²) storage")
    print("    ✗ O(n²) computation per iteration")
    
    print("\n  L-BFGS:")
    print("    ✓ Only store m recent (s, y) pairs")
    print("    ✓ O(mn) storage (typically m = 5-20)")
    print("    ✓ O(mn) computation per iteration")
    print("    ✓ Scales to millions of variables")
    print("    ✗ Slightly slower convergence than BFGS")
    
    print("\n🎯 WHEN TO USE QUASI-NEWTON:")
    print("  ✓ Medium to large problems")
    print("  ✓ Hessian unavailable or expensive")
    print("  ✓ Need faster convergence than gradient descent")
    print("  ✓ Can afford O(n²) storage (BFGS) or O(mn) (L-BFGS)")
    print("  ✗ Very large sparse problems (use conjugate gradient)")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_quasi_newton()
    quasi_newton_theory()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Quasi-Newton methods approximate Hessian from gradients only")
    print("- BFGS is the most popular quasi-Newton method")
    print("- Achieves superlinear convergence (between linear and quadratic)")
    print("- O(n²) storage for BFGS, O(mn) for L-BFGS")
    print("- Self-correcting: works even with poor initial Hessian")
    print("- Excellent balance of speed and cost for medium-scale problems")
    print("- L-BFGS is the method of choice for large-scale optimization")
    print("\nQuasi-Newton: The workhorse of practical optimization! 🚀")
