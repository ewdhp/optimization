"""
Newton's Method for Unconstrained Optimization

Newton's method is a second-order optimization algorithm that uses both
the gradient and Hessian of the objective function. It achieves quadratic
convergence near the solution, making it very fast when close to the optimum.

Algorithm:
    x_{k+1} = x_k - [∇²f(x_k)]^{-1} ∇f(x_k)

Equivalently (Newton direction):
    Solve: ∇²f(x_k) p_k = -∇f(x_k)
    Update: x_{k+1} = x_k + p_k

Key Properties:
- Quadratic convergence: ||x_{k+1} - x*|| ≤ C||x_k - x*||²
- Affine invariant: Performance independent of coordinate system
- Requires Hessian computation and inversion (O(n³) cost)

Convergence Theorem:
If f is twice continuously differentiable, ∇²f is Lipschitz continuous,
and x_0 is sufficiently close to x* where ∇²f(x*) is positive definite,
then Newton's method converges quadratically.

Damped Newton:
    x_{k+1} = x_k + α_k p_k
where α_k is chosen via line search for global convergence.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

class NewtonMethod:
    """
    Implementation of Newton's method for unconstrained optimization.
    """
    
    def __init__(self,
                 objective: Callable[[np.ndarray], float],
                 gradient: Callable[[np.ndarray], np.ndarray],
                 hessian: Callable[[np.ndarray], np.ndarray],
                 name: str = "Function"):
        """
        Initialize Newton's method optimizer.
        
        Args:
            objective: Objective function f(x)
            gradient: Gradient function ∇f(x)
            hessian: Hessian function ∇²f(x)
            name: Function name
        """
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.name = name
        
        self.reset_history()
    
    def reset_history(self):
        """Reset optimization history."""
        self.history = {
            'x': [],
            'f': [],
            'grad_norm': [],
            'step_size': [],
            'newton_decrement': [],
            'hessian_condition': []
        }
    
    def compute_newton_direction(self, x: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """
        Compute Newton direction by solving ∇²f(x)p = -∇f(x).
        
        Args:
            x: Current point
            
        Returns:
            (direction, newton_decrement, info_dict)
        """
        x = np.array(x)
        
        grad = self.gradient(x)
        hess = self.hessian(x)
        
        # Check if Hessian is positive definite
        eigenvals = np.linalg.eigvals(hess)
        min_eigenval = np.min(np.real(eigenvals))
        condition_number = np.linalg.cond(hess)
        
        info = {
            'min_eigenvalue': min_eigenval,
            'condition_number': condition_number,
            'positive_definite': min_eigenval > 0
        }
        
        try:
            # Solve Newton system
            if min_eigenval > 1e-10:
                # Hessian is positive definite - use Cholesky
                try:
                    L = np.linalg.cholesky(hess)
                    # Solve L L^T p = -grad
                    y = np.linalg.solve(L, -grad)
                    p = np.linalg.solve(L.T, y)
                    info['method'] = 'cholesky'
                except:
                    # Fall back to regular solve
                    p = np.linalg.solve(hess, -grad)
                    info['method'] = 'direct_solve'
            else:
                # Hessian not positive definite - regularize
                # Add λI to make it positive definite
                lambda_reg = max(1e-4, -2 * min_eigenval)
                hess_reg = hess + lambda_reg * np.eye(len(hess))
                p = np.linalg.solve(hess_reg, -grad)
                info['method'] = 'regularized'
                info['regularization'] = lambda_reg
            
            # Newton decrement: λ = √(∇f^T H^{-1} ∇f)
            newton_decrement = np.sqrt(np.abs(np.dot(grad, p)))
            
            return p, newton_decrement, info
            
        except np.linalg.LinAlgError:
            # Singular Hessian - fall back to gradient direction
            p = -grad
            newton_decrement = np.linalg.norm(grad)
            info['method'] = 'gradient_fallback'
            return p, newton_decrement, info
    
    def optimize(self,
                x0: np.ndarray,
                max_iters: int = 100,
                tolerance: float = 1e-8,
                verbose: bool = False) -> Dict:
        """
        Pure Newton's method (no line search).
        
        Args:
            x0: Initial point
            max_iters: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress
            
        Returns:
            Optimization results
        """
        self.reset_history()
        
        x = np.array(x0, dtype=float)
        
        for k in range(max_iters):
            # Evaluate function and gradient
            f_val = self.objective(x)
            grad = self.gradient(x)
            grad_norm = np.linalg.norm(grad)
            
            # Compute Newton direction
            p, newton_dec, info = self.compute_newton_direction(x)
            
            # Store history
            self.history['x'].append(x.copy())
            self.history['f'].append(f_val)
            self.history['grad_norm'].append(grad_norm)
            self.history['step_size'].append(1.0)  # Pure Newton uses α=1
            self.history['newton_decrement'].append(newton_dec)
            self.history['hessian_condition'].append(info['condition_number'])
            
            # Check convergence
            if grad_norm < tolerance or newton_dec < tolerance:
                if verbose:
                    print(f"Converged in {k} iterations")
                break
            
            # Newton update (full step)
            x = x + p
            
            if verbose and k % 10 == 0:
                print(f"Iter {k}: f = {f_val:.6e}, ||∇f|| = {grad_norm:.6e}, "
                      f"λ = {newton_dec:.6e}")
        
        return {
            'x_optimal': x,
            'f_optimal': self.objective(x),
            'iterations': k + 1,
            'gradient_norm': np.linalg.norm(self.gradient(x)),
            'converged': grad_norm < tolerance,
            'history': self.history
        }
    
    def optimize_damped(self,
                       x0: np.ndarray,
                       backtrack_beta: float = 0.5,
                       backtrack_c: float = 1e-4,
                       max_iters: int = 100,
                       tolerance: float = 1e-8,
                       verbose: bool = False) -> Dict:
        """
        Damped Newton's method with backtracking line search.
        
        Provides global convergence guarantees.
        
        Args:
            x0: Initial point
            backtrack_beta: Backtracking factor
            backtrack_c: Armijo constant
            max_iters: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress
            
        Returns:
            Optimization results
        """
        self.reset_history()
        
        x = np.array(x0, dtype=float)
        
        for k in range(max_iters):
            f_val = self.objective(x)
            grad = self.gradient(x)
            grad_norm = np.linalg.norm(grad)
            
            # Compute Newton direction
            p, newton_dec, info = self.compute_newton_direction(x)
            
            # Check convergence
            if grad_norm < tolerance or newton_dec < tolerance:
                if verbose:
                    print(f"Converged in {k} iterations")
                break
            
            # Backtracking line search
            alpha = 1.0
            directional_deriv = np.dot(grad, p)
            
            # Backtrack if not descent direction or insufficient decrease
            max_backtracks = 50
            for _ in range(max_backtracks):
                x_new = x + alpha * p
                f_new = self.objective(x_new)
                
                # Armijo condition
                if f_new <= f_val + backtrack_c * alpha * directional_deriv:
                    break
                
                alpha *= backtrack_beta
                
                if alpha < 1e-16:
                    break
            
            # Store history
            self.history['x'].append(x.copy())
            self.history['f'].append(f_val)
            self.history['grad_norm'].append(grad_norm)
            self.history['step_size'].append(alpha)
            self.history['newton_decrement'].append(newton_dec)
            self.history['hessian_condition'].append(info['condition_number'])
            
            # Update
            x = x + alpha * p
            
            if verbose and k % 10 == 0:
                print(f"Iter {k}: f = {f_val:.6e}, ||∇f|| = {grad_norm:.6e}, "
                      f"α = {alpha:.6f}, λ = {newton_dec:.6e}")
        
        return {
            'x_optimal': x,
            'f_optimal': self.objective(x),
            'iterations': k + 1,
            'gradient_norm': np.linalg.norm(self.gradient(x)),
            'converged': grad_norm < tolerance,
            'history': self.history
        }


def demonstrate_newton_method():
    """
    Comprehensive demonstration of Newton's method.
    """
    print("🔬 NEWTON'S METHOD FOR OPTIMIZATION")
    print("=" * 60)
    
    # Problem 1: Simple quadratic
    print("\n🎯 EXAMPLE 1: Quadratic Function")
    print("-" * 50)
    
    Q = np.array([[3, 1], [1, 2]])
    b = np.array([1, -1])
    x_star = np.linalg.solve(Q, b)
    
    def quad_obj(x):
        x = np.array(x)
        return 0.5 * np.dot(x, np.dot(Q, x)) - np.dot(b, x)
    
    def quad_grad(x):
        x = np.array(x)
        return np.dot(Q, x) - b
    
    def quad_hess(x):
        return Q
    
    newton1 = NewtonMethod(quad_obj, quad_grad, quad_hess, "Quadratic")
    
    x0 = np.array([5, 5])
    print(f"Starting point: {x0}")
    print(f"True minimum: {x_star}")
    print(f"Initial distance: {np.linalg.norm(x0 - x_star):.6f}")
    
    # Pure Newton
    result_pure = newton1.optimize(x0, max_iters=20, tolerance=1e-12)
    print(f"\nPure Newton's Method:")
    print(f"  Iterations: {result_pure['iterations']}")
    print(f"  Final point: {result_pure['x_optimal']}")
    print(f"  Final error: {np.linalg.norm(result_pure['x_optimal'] - x_star):.2e}")
    print(f"  Expected: 1 iteration (quadratic converges in 1 step)")
    
    # Problem 2: Rosenbrock function
    print("\n🎯 EXAMPLE 2: Rosenbrock Function")
    print("-" * 50)
    
    def rosenbrock_obj(x):
        x = np.array(x)
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_grad(x):
        x = np.array(x)
        grad_x = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        grad_y = 200*(x[1] - x[0]**2)
        return np.array([grad_x, grad_y])
    
    def rosenbrock_hess(x):
        x = np.array(x)
        h11 = 2 + 1200*x[0]**2 - 400*x[1]
        h12 = h21 = -400*x[0]
        h22 = 200
        return np.array([[h11, h12], [h21, h22]])
    
    newton2 = NewtonMethod(rosenbrock_obj, rosenbrock_grad, rosenbrock_hess,
                          "Rosenbrock")
    
    x0_rosen = np.array([-1, 2])
    print(f"Starting point: {x0_rosen}")
    print(f"True minimum: [1, 1]")
    
    # Pure Newton (may not converge from far away)
    result_rosen_pure = newton2.optimize(x0_rosen, max_iters=100, tolerance=1e-6)
    print(f"\nPure Newton:")
    print(f"  Iterations: {result_rosen_pure['iterations']}")
    print(f"  Final point: {result_rosen_pure['x_optimal']}")
    print(f"  Final f: {result_rosen_pure['f_optimal']:.6e}")
    
    # Damped Newton (globally convergent)
    result_rosen_damped = newton2.optimize_damped(x0_rosen, max_iters=100,
                                                  tolerance=1e-6)
    print(f"\nDamped Newton:")
    print(f"  Iterations: {result_rosen_damped['iterations']}")
    print(f"  Final point: {result_rosen_damped['x_optimal']}")
    print(f"  Final f: {result_rosen_damped['f_optimal']:.6e}")
    
    # Problem 3: Demonstrate quadratic convergence
    print("\n🎯 EXAMPLE 3: Quadratic Convergence Demonstration")
    print("-" * 50)
    
    # Start close to optimum
    x0_close = x_star + np.array([0.1, 0.1])
    result_convergence = newton1.optimize(x0_close, max_iters=10, tolerance=1e-14)
    
    print("Newton's method from point close to optimum:")
    errors = [np.linalg.norm(x - x_star) for x in result_convergence['history']['x']]
    
    for k in range(min(5, len(errors))):
        print(f"  Iteration {k}: error = {errors[k]:.2e}")
        if k > 0:
            ratio = errors[k] / errors[k-1]**2
            print(f"    Error ratio e_{k}/e_{k-1}² = {ratio:.4f}")
    
    # Visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Convergence rate comparison
    ax1 = plt.subplot(2, 3, 1)
    
    # Compare with gradient descent
    from steepest_descent import SteepestDescent
    gd = SteepestDescent(quad_obj, quad_grad)
    result_gd = gd.optimize(x0, step_size=0.1, max_iters=50)
    
    errors_newton = [np.linalg.norm(x - x_star) 
                     for x in result_pure['history']['x']]
    errors_gd = [np.linalg.norm(x - x_star) 
                 for x in result_gd['history']['x']]
    
    ax1.semilogy(range(len(errors_newton)), errors_newton, 'r-o',
                linewidth=3, markersize=8, label='Newton')
    ax1.semilogy(range(len(errors_gd)), errors_gd, 'b-s',
                linewidth=2, markersize=4, label='Gradient Descent')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('||x_k - x*|| (log scale)')
    ax1.set_title('Convergence Rate: Newton vs Gradient Descent')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Quadratic convergence demonstration
    ax2 = plt.subplot(2, 3, 2)
    
    if len(errors) > 2:
        # Plot e_{k+1} vs e_k²
        errors_k = errors[:-1]
        errors_k1 = errors[1:]
        errors_k_squared = [e**2 for e in errors_k]
        
        ax2.loglog(errors_k_squared, errors_k1, 'ro-', markersize=8,
                  linewidth=2, label='Actual')
        
        # Reference line e_{k+1} = C·e_k²
        if len(errors_k_squared) > 1:
            C = errors_k1[-1] / errors_k_squared[-1]
            ref_line = [C * e2 for e2 in errors_k_squared]
            ax2.loglog(errors_k_squared, ref_line, 'b--',
                      linewidth=2, label=f'C·e_k² (C={C:.2f})')
        
        ax2.set_xlabel('e_k²')
        ax2.set_ylabel('e_{k+1}')
        ax2.set_title('Quadratic Convergence: e_{k+1} = O(e_k²)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Newton decrement
    ax3 = plt.subplot(2, 3, 3)
    
    if 'newton_decrement' in result_rosen_damped['history']:
        ax3.semilogy(range(len(result_rosen_damped['history']['newton_decrement'])),
                    result_rosen_damped['history']['newton_decrement'],
                    'g-o', linewidth=2, markersize=6)
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Newton Decrement λ (log scale)')
        ax3.set_title('Newton Decrement (Rosenbrock)')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Optimization paths
    ax4 = plt.subplot(2, 3, 4)
    
    # Contour plot
    x_range = np.linspace(-2, 6, 100)
    y_range = np.linspace(-2, 6, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = quad_obj(np.array([X[i, j], Y[i, j]]))
    
    contour = ax4.contour(X, Y, Z, levels=20, alpha=0.6, colors='gray')
    ax4.clabel(contour, inline=True, fontsize=8)
    
    # Newton path
    newton_x = [x[0] for x in result_pure['history']['x']]
    newton_y = [x[1] for x in result_pure['history']['x']]
    
    ax4.plot(newton_x, newton_y, 'r-o', linewidth=3, markersize=8,
            label='Newton path')
    ax4.plot(x0[0], x0[1], 'go', markersize=12, label='Start')
    ax4.plot(x_star[0], x_star[1], 'r*', markersize=15, label='Optimum')
    
    ax4.set_xlabel('x₁')
    ax4.set_ylabel('x₂')
    ax4.set_title('Newton Path on Quadratic')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Rosenbrock paths
    ax5 = plt.subplot(2, 3, 5)
    
    # Rosenbrock contours
    x_r = np.linspace(-2, 2, 100)
    y_r = np.linspace(-1, 3, 100)
    X_r, Y_r = np.meshgrid(x_r, y_r)
    Z_r = (1 - X_r)**2 + 100*(Y_r - X_r**2)**2
    
    levels_r = np.logspace(0, 3, 20)
    contour_r = ax5.contour(X_r, Y_r, Z_r, levels=levels_r, alpha=0.6,
                           colors='gray')
    
    # Damped Newton path
    path_x = [x[0] for x in result_rosen_damped['history']['x']]
    path_y = [x[1] for x in result_rosen_damped['history']['x']]
    
    ax5.plot(path_x, path_y, 'r-o', linewidth=2, markersize=4,
            label='Damped Newton')
    ax5.plot(x0_rosen[0], x0_rosen[1], 'go', markersize=10, label='Start')
    ax5.plot(1, 1, 'r*', markersize=15, label='Optimum')
    
    ax5.set_xlabel('x₁')
    ax5.set_ylabel('x₂')
    ax5.set_title('Damped Newton on Rosenbrock')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Step sizes in damped Newton
    ax6 = plt.subplot(2, 3, 6)
    
    if 'step_size' in result_rosen_damped['history']:
        step_sizes = result_rosen_damped['history']['step_size']
        ax6.plot(range(len(step_sizes)), step_sizes, 'b-o',
                linewidth=2, markersize=6)
        ax6.axhline(y=1.0, color='r', linestyle='--', alpha=0.7,
                   label='Full Newton step')
        
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Step Size α')
        ax6.set_title('Damped Newton Step Sizes')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def newton_theory():
    """
    Theoretical analysis of Newton's method.
    """
    print("\n📚 NEWTON'S METHOD THEORY")
    print("=" * 60)
    
    print("🔑 QUADRATIC CONVERGENCE THEOREM:")
    print("  If f is twice continuously differentiable,")
    print("  ∇²f is Lipschitz continuous with constant M,")
    print("  ∇²f(x*) is positive definite with λ_min > 0,")
    print("  and x_0 is close enough to x*, then:")
    print("  ||x_{k+1} - x*|| ≤ (M/(2λ_min))||x_k - x*||²")
    print("  This is QUADRATIC convergence!")
    
    print("\n💡 KEY PROPERTIES:")
    print("1. AFFINE INVARIANCE:")
    print("   Performance unchanged under linear transformation")
    print("   Makes Newton robust to scaling/rotation")
    
    print("\n2. EXACT FOR QUADRATICS:")
    print("   For f(x) = ½x^TQx + b^Tx + c")
    print("   Newton converges in exactly ONE iteration")
    
    print("\n3. LOCAL vs GLOBAL:")
    print("   ✓ Quadratic convergence near solution")
    print("   ✗ May diverge if started far from optimum")
    print("   → Use damped Newton for global convergence")
    
    print("\n⚠️  COMPUTATIONAL COST:")
    print("  Per iteration:")
    print("  - Gradient: O(n)")
    print("  - Hessian: O(n²) storage, O(n²) to O(n³) computation")
    print("  - Linear solve: O(n³)")
    print("  Total: O(n³) per iteration")
    
    print("\n🎯 WHEN TO USE NEWTON:")
    print("  ✓ Small to medium problems (n < 1000)")
    print("  ✓ Need very high accuracy")
    print("  ✓ Hessian available/computable")
    print("  ✓ Near the solution")
    print("  ✗ Large-scale problems")
    print("  ✗ Hessian too expensive")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_newton_method()
    newton_theory()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Newton's method uses second-order (Hessian) information")
    print("- Achieves quadratic convergence: ||e_{k+1}|| = O(||e_k||²)")
    print("- Converges in 1 step for quadratic functions")
    print("- Affine invariant: robust to coordinate transformations")
    print("- Requires positive definite Hessian for descent")
    print("- Damped Newton adds line search for global convergence")
    print("- O(n³) cost per iteration limits to medium-scale problems")
    print("\nNewton's method: The gold standard for fast local convergence! 🚀")
