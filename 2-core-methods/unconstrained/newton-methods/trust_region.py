"""
Trust Region Methods

Trust region methods are an alternative to line search for step size control.
Instead of choosing a direction and then a step size, trust region methods:
1. Define a region around the current point where the model is trusted
2. Solve a subproblem to optimize within this region
3. Update the trust region radius based on how well the model predicted improvement

Trust Region Subproblem:
    min_{p} m_k(p) = f_k + ∇f_k^T p + ½p^T B_k p
    subject to ||p|| ≤ Δ_k

where:
- Δ_k is the trust region radius
- B_k approximates ∇²f (Hessian or quasi-Newton)
- ||·|| is typically Euclidean norm

Solution Methods:
1. Cauchy Point: Steepest descent direction truncated at boundary
2. Dogleg: Combines Cauchy point and full Newton step
3. Two-Dimensional Subspace: More accurate but expensive
4. Conjugate Gradient-Steihaug: For large-scale problems

Convergence:
- Global convergence to stationary point
- Q-quadratic convergence in two-norm when B_k → ∇²f(x*)
- More robust than line search for non-convex problems
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class TrustRegion:
    """
    Trust region optimization with dogleg method.
    """
    
    def __init__(self,
                 objective: Callable[[np.ndarray], float],
                 gradient: Callable[[np.ndarray], np.ndarray],
                 hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 name: str = "Function"):
        """
        Initialize trust region optimizer.
        
        Args:
            objective: Objective function f(x)
            gradient: Gradient function ∇f(x)
            hessian: Hessian function ∇²f(x) (if None, use BFGS approximation)
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
            'delta': [],
            'rho': [],
            'step_type': []  # 'cauchy', 'newton', 'dogleg'
        }
    
    def optimize(self,
                x0: np.ndarray,
                max_iters: int = 1000,
                tolerance: float = 1e-6,
                delta_init: float = 1.0,
                delta_max: float = 10.0,
                eta: float = 0.15,
                verbose: bool = False) -> Dict:
        """
        Trust region optimization with dogleg method.
        
        Args:
            x0: Initial point
            max_iters: Maximum iterations
            tolerance: Convergence tolerance
            delta_init: Initial trust region radius
            delta_max: Maximum trust region radius
            eta: Acceptance threshold for step
            verbose: Print progress
            
        Returns:
            Optimization results
        """
        self.reset_history()
        
        x = np.array(x0, dtype=float)
        n = len(x)
        delta = delta_init
        
        # Initialize BFGS approximation if Hessian not provided
        if self.hessian is None:
            B = np.eye(n)
            use_bfgs = True
        else:
            use_bfgs = False
        
        for k in range(max_iters):
            f_val = self.objective(x)
            grad = self.gradient(x)
            grad_norm = np.linalg.norm(grad)
            
            # Get Hessian or BFGS approximation
            if use_bfgs:
                H = B
            else:
                H = self.hessian(x)
            
            # Store history
            self.history['x'].append(x.copy())
            self.history['f'].append(f_val)
            self.history['grad_norm'].append(grad_norm)
            self.history['delta'].append(delta)
            
            # Check convergence
            if grad_norm < tolerance:
                if verbose:
                    print(f"Converged in {k} iterations")
                break
            
            # Solve trust region subproblem
            p, step_type = self._dogleg_step(grad, H, delta)
            
            self.history['step_type'].append(step_type)
            
            # Compute actual vs predicted reduction
            f_new = self.objective(x + p)
            actual_reduction = f_val - f_new
            
            # Model reduction
            model_reduction = -(np.dot(grad, p) + 0.5 * np.dot(p, np.dot(H, p)))
            
            # Ratio of actual to predicted reduction
            if model_reduction > 0:
                rho = actual_reduction / model_reduction
            else:
                rho = 0
            
            self.history['rho'].append(rho)
            
            # Update trust region radius and position
            if rho < 0.25:
                # Poor agreement: shrink region
                delta = 0.25 * delta
            elif rho > 0.75 and np.abs(np.linalg.norm(p) - delta) < 1e-10:
                # Excellent agreement at boundary: expand region
                delta = min(2 * delta, delta_max)
            # else: keep delta the same
            
            # Accept or reject step
            if rho > eta:
                x_old = x.copy()
                x = x + p
                
                # BFGS update if using quasi-Newton
                if use_bfgs:
                    grad_new = self.gradient(x)
                    s = x - x_old
                    y = grad_new - grad
                    
                    rho_bfgs = 1.0 / np.dot(y, s)
                    
                    if np.isfinite(rho_bfgs) and rho_bfgs > 1e-10:
                        # BFGS update
                        I = np.eye(n)
                        A1 = I - rho_bfgs * np.outer(s, y)
                        A2 = I - rho_bfgs * np.outer(y, s)
                        B = np.dot(A1, np.dot(B, A2)) + rho_bfgs * np.outer(s, s)
            
            if verbose and k % 100 == 0:
                print(f"Iter {k}: f = {f_val:.6e}, ||∇f|| = {grad_norm:.6e}, "
                      f"Δ = {delta:.3f}, ρ = {rho:.3f}, step = {step_type}")
        
        return {
            'x_optimal': x,
            'f_optimal': self.objective(x),
            'iterations': k + 1,
            'gradient_norm': np.linalg.norm(self.gradient(x)),
            'converged': grad_norm < tolerance,
            'history': self.history
        }
    
    def _dogleg_step(self, grad, H, delta):
        """
        Compute dogleg step for trust region subproblem.
        
        The dogleg path connects:
        1. Origin
        2. Cauchy point (steepest descent minimizer)
        3. Newton point (unconstrained minimizer)
        """
        n = len(grad)
        
        # Try Newton step first
        try:
            p_newton = -np.linalg.solve(H, grad)
            if np.linalg.norm(p_newton) <= delta:
                return p_newton, 'newton'
        except np.linalg.LinAlgError:
            p_newton = None
        
        # Cauchy point: minimize along steepest descent
        grad_norm_sq = np.dot(grad, grad)
        Hg = np.dot(H, grad)
        gHg = np.dot(grad, Hg)
        
        if gHg > 0:
            tau_c = grad_norm_sq / gHg
        else:
            tau_c = 1.0
        
        p_cauchy = -tau_c * grad
        
        if np.linalg.norm(p_cauchy) >= delta:
            # Cauchy point outside region: return truncated
            return -(delta / np.linalg.norm(grad)) * grad, 'cauchy'
        
        # If we have Newton step, use dogleg
        if p_newton is not None:
            # Dogleg path: p(τ) = p_cauchy + (τ-1)(p_newton - p_cauchy) for τ ∈ [1,2]
            # Find τ such that ||p(τ)|| = delta
            
            diff = p_newton - p_cauchy
            a = np.dot(diff, diff)
            b = 2 * np.dot(p_cauchy, diff)
            c = np.dot(p_cauchy, p_cauchy) - delta**2
            
            discriminant = b**2 - 4*a*c
            
            if discriminant >= 0 and a > 1e-10:
                tau = (-b + np.sqrt(discriminant)) / (2*a)
                tau = np.clip(tau, 0, 1)
                p_dogleg = p_cauchy + tau * diff
                return p_dogleg, 'dogleg'
        
        # Fallback: return Cauchy point
        return p_cauchy, 'cauchy'


class CauchyPointTrustRegion:
    """
    Simplified trust region using only Cauchy point.
    """
    
    def __init__(self,
                 objective: Callable[[np.ndarray], float],
                 gradient: Callable[[np.ndarray], np.ndarray],
                 name: str = "Function"):
        """Initialize Cauchy point trust region."""
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
            'delta': []
        }
    
    def optimize(self,
                x0: np.ndarray,
                max_iters: int = 1000,
                tolerance: float = 1e-6,
                delta_init: float = 1.0,
                verbose: bool = False) -> Dict:
        """Optimize using Cauchy point trust region."""
        self.reset_history()
        
        x = np.array(x0, dtype=float)
        delta = delta_init
        
        for k in range(max_iters):
            f_val = self.objective(x)
            grad = self.gradient(x)
            grad_norm = np.linalg.norm(grad)
            
            self.history['x'].append(x.copy())
            self.history['f'].append(f_val)
            self.history['grad_norm'].append(grad_norm)
            self.history['delta'].append(delta)
            
            if grad_norm < tolerance:
                break
            
            # Cauchy point step
            p = -(delta / grad_norm) * grad
            
            # Compute reduction ratio
            f_new = self.objective(x + p)
            actual_reduction = f_val - f_new
            predicted_reduction = -np.dot(grad, p)
            
            if predicted_reduction > 0:
                rho = actual_reduction / predicted_reduction
            else:
                rho = 0
            
            # Update radius
            if rho < 0.25:
                delta *= 0.25
            elif rho > 0.75:
                delta *= 2.0
            
            # Accept step
            if rho > 0.1:
                x = x + p
        
        return {
            'x_optimal': x,
            'f_optimal': self.objective(x),
            'iterations': k + 1,
            'gradient_norm': np.linalg.norm(self.gradient(x)),
            'converged': grad_norm < tolerance,
            'history': self.history
        }


def demonstrate_trust_region():
    """
    Comprehensive demonstration of trust region methods.
    """
    print("🎯 TRUST REGION METHODS")
    print("=" * 60)
    
    # Example 1: Rosenbrock function
    print("\n🌹 EXAMPLE 1: Rosenbrock Function")
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
        h11 = 2 - 400*(x[1] - x[0]**2) + 800*x[0]**2
        h12 = -400*x[0]
        h22 = 200
        return np.array([[h11, h12], [h12, h22]])
    
    tr = TrustRegion(rosenbrock_obj, rosenbrock_grad, rosenbrock_hess, "Rosenbrock")
    
    x0 = np.array([-1.0, 2.0])
    
    result = tr.optimize(x0, max_iters=200, tolerance=1e-6, verbose=False)
    
    print(f"Starting point: {x0}")
    print(f"True minimum: [1, 1]")
    print(f"\nResults:")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Final x: {result['x_optimal']}")
    print(f"  Final f: {result['f_optimal']:.8f}")
    print(f"  Final ||∇f||: {result['gradient_norm']:.2e}")
    
    # Example 2: Quadratic function
    print("\n📐 EXAMPLE 2: Ill-Conditioned Quadratic")
    print("-" * 50)
    
    Q = np.array([[100, 0], [0, 1]])
    b = np.array([1, 1])
    
    def quad_obj(x):
        x = np.array(x)
        return 0.5 * np.dot(x, np.dot(Q, x)) - np.dot(b, x)
    
    def quad_grad(x):
        x = np.array(x)
        return np.dot(Q, x) - b
    
    def quad_hess(x):
        return Q
    
    tr2 = TrustRegion(quad_obj, quad_grad, quad_hess, "Quadratic")
    
    x0_quad = np.array([5, 5])
    result2 = tr2.optimize(x0_quad, max_iters=100, tolerance=1e-8)
    
    x_star = np.linalg.solve(Q, b)
    
    print(f"Condition number: {np.linalg.cond(Q):.1f}")
    print(f"Iterations: {result2['iterations']}")
    print(f"Final error: {np.linalg.norm(result2['x_optimal'] - x_star):.2e}")
    
    # Comparison with Cauchy point only
    print("\n🔄 Comparison: Dogleg vs Cauchy Point")
    print("-" * 50)
    
    cauchy_tr = CauchyPointTrustRegion(rosenbrock_obj, rosenbrock_grad)
    result_cauchy = cauchy_tr.optimize(x0, max_iters=200, tolerance=1e-6)
    
    print(f"Dogleg iterations: {result['iterations']}")
    print(f"Cauchy-only iterations: {result_cauchy['iterations']}")
    
    # Visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Function value convergence
    ax1 = plt.subplot(2, 3, 1)
    
    ax1.semilogy(range(len(result['history']['f'])),
                result['history']['f'],
                'b-o', linewidth=2, markersize=4, label='Trust Region (Dogleg)')
    ax1.semilogy(range(len(result_cauchy['history']['f'])),
                result_cauchy['history']['f'],
                'r-s', linewidth=2, markersize=3, label='Cauchy Point Only')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('f(x) (log scale)')
    ax1.set_title('Function Value Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gradient norm
    ax2 = plt.subplot(2, 3, 2)
    
    ax2.semilogy(range(len(result['history']['grad_norm'])),
                result['history']['grad_norm'],
                'b-o', linewidth=2, markersize=4)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('||∇f(x)|| (log scale)')
    ax2.set_title('Gradient Norm Convergence')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Trust region radius evolution
    ax3 = plt.subplot(2, 3, 3)
    
    ax3.plot(range(len(result['history']['delta'])),
            result['history']['delta'],
            'g-o', linewidth=2, markersize=4)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Trust Region Radius Δ')
    ax3.set_title('Trust Region Radius Evolution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Optimization path
    ax4 = plt.subplot(2, 3, 4)
    
    # Contour plot
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100*(Y - X**2)**2
    
    levels = np.logspace(0, 3, 20)
    ax4.contour(X, Y, Z, levels=levels, alpha=0.6, colors='gray')
    
    # Path
    path_x = [x[0] for x in result['history']['x']]
    path_y = [x[1] for x in result['history']['x']]
    
    ax4.plot(path_x, path_y, 'b-o', linewidth=2, markersize=4,
            alpha=0.7, label='TR path')
    ax4.plot(x0[0], x0[1], 'go', markersize=10, label='Start')
    ax4.plot(1, 1, 'r*', markersize=15, label='Optimum')
    
    ax4.set_xlabel('x₁')
    ax4.set_ylabel('x₂')
    ax4.set_title('Trust Region Path on Rosenbrock')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Reduction ratio ρ
    ax5 = plt.subplot(2, 3, 5)
    
    if 'rho' in result['history'] and len(result['history']['rho']) > 0:
        ax5.plot(range(len(result['history']['rho'])),
                result['history']['rho'],
                'purple', linewidth=2, marker='o', markersize=4)
        
        ax5.axhline(y=0.75, color='g', linestyle='--', alpha=0.7,
                   label='Excellent (ρ > 0.75)')
        ax5.axhline(y=0.25, color='orange', linestyle='--', alpha=0.7,
                   label='Poor (ρ < 0.25)')
        ax5.axhline(y=0, color='r', linestyle='--', alpha=0.7,
                   label='Rejected')
        
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Reduction Ratio ρ')
        ax5.set_title('Actual/Predicted Reduction Ratio')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Step types
    ax6 = plt.subplot(2, 3, 6)
    
    if 'step_type' in result['history']:
        step_types = result['history']['step_type']
        
        cauchy_count = step_types.count('cauchy')
        newton_count = step_types.count('newton')
        dogleg_count = step_types.count('dogleg')
        
        ax6.bar(['Cauchy', 'Newton', 'Dogleg'],
               [cauchy_count, newton_count, dogleg_count],
               color=['orange', 'green', 'blue'], alpha=0.7)
        
        ax6.set_ylabel('Count')
        ax6.set_title('Step Types Used')
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def trust_region_theory():
    """
    Theory of trust region methods.
    """
    print("\n📚 TRUST REGION THEORY")
    print("=" * 60)
    
    print("🎯 CORE IDEA:")
    print("  Instead of: choose direction, then step size (line search)")
    print("  Do: define trusted region, optimize model within it")
    
    print("\n📊 TRUST REGION SUBPROBLEM:")
    print("  min_{p} m_k(p) = f_k + ∇f_k^T p + ½p^T B_k p")
    print("  s.t.    ||p|| ≤ Δ_k")
    print()
    print("  where:")
    print("    m_k(p) = quadratic model of f near x_k")
    print("    Δ_k = trust region radius")
    print("    B_k ≈ ∇²f (Hessian or quasi-Newton)")
    
    print("\n🔄 ALGORITHM FLOW:")
    print("1. Solve subproblem for step p_k")
    print("2. Compute actual reduction: ared = f_k - f(x_k + p_k)")
    print("3. Compute predicted reduction: pred = -m_k(p_k)")
    print("4. Compute ratio: ρ = ared / pred")
    print("5. Update radius based on ρ:")
    print("   - ρ < 0.25: shrink Δ (model is poor)")
    print("   - ρ > 0.75: expand Δ (model is good)")
    print("6. Accept step if ρ > η (typically η = 0.15)")
    
    print("\n🐕 DOGLEG METHOD:")
    print("  Approximate solution path:")
    print("    1. p = 0 (current point)")
    print("    2. p_C = Cauchy point (steepest descent minimizer)")
    print("    3. p_N = Newton point (unconstrained minimizer)")
    print()
    print("  If ||p_N|| ≤ Δ: use Newton step")
    print("  Else if ||p_C|| ≥ Δ: use truncated Cauchy")
    print("  Else: use point on dogleg path p_C → p_N at boundary")
    
    print("\n💡 CONVERGENCE PROPERTIES:")
    print("1. GLOBAL CONVERGENCE:")
    print("   Converges to stationary point from any starting point")
    print("   (assuming gradient is Lipschitz continuous)")
    
    print("\n2. LOCAL CONVERGENCE:")
    print("   If B_k → ∇²f(x*) and ∇²f(x*) ≻ 0:")
    print("   - Eventually all steps accepted (ρ ≈ 1)")
    print("   - Q-quadratic convergence in ||·||₂")
    print("   - ||x_{k+1} - x*|| ≤ C ||x_k - x*||²")
    
    print("\n⚖️  TRUST REGION vs LINE SEARCH:")
    print("  Trust Region:")
    print("    ✓ More robust for non-convex problems")
    print("    ✓ Global convergence easier to prove")
    print("    ✓ Natural handling of indefinite Hessian")
    print("    ✗ Subproblem can be expensive to solve exactly")
    
    print("\n  Line Search:")
    print("    ✓ Simpler to implement")
    print("    ✓ Cheaper per iteration (no subproblem)")
    print("    ✗ May struggle with indefinite Hessian")
    print("    ✗ Zigzagging in narrow valleys")
    
    print("\n🎯 WHEN TO USE TRUST REGION:")
    print("  ✓ Non-convex optimization")
    print("  ✓ Problems with indefinite Hessian")
    print("  ✓ When robustness is more important than speed")
    print("  ✓ Large-scale: use CG-Steihaug for subproblem")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_trust_region()
    trust_region_theory()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Trust region: define region of trust, optimize model within it")
    print("- Update radius based on how well model predicts improvement")
    print("- Dogleg combines Cauchy point and Newton point")
    print("- More robust than line search for non-convex problems")
    print("- Global convergence + local quadratic convergence")
    print("- Natural handling of indefinite Hessian")
    print("\nTrust region: When you can't trust the model everywhere! 🎯")
