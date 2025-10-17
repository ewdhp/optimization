"""
Taylor's Theorem for Multivariable Functions

Taylor's theorem is fundamental to optimization as it provides local approximations
of functions that justify Newton's method, convergence analysis, and second-order
conditions for optimality.

Mathematical Statement:
Let f: R^n → R be k+1 times continuously differentiable in a neighborhood of a.
Then for any x in this neighborhood:

f(x) = f(a) + ∇f(a)ᵀ(x-a) + ½(x-a)ᵀ∇²f(a)(x-a) + ... + Rₖ(x,a)

where Rₖ(x,a) is the remainder term.

Applications in Optimization:
- Newton's method derivation and convergence analysis
- Second-order optimality conditions
- Trust region methods
- Quasi-Newton methods (BFGS, L-BFGS)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Tuple
import warnings

warnings.filterwarnings('ignore')

class TaylorApproximation:
    """
    Implementation of Taylor's theorem for multivariable functions.
    Provides first and second-order approximations with error analysis.
    """
    
    def __init__(self, func: Callable[[np.ndarray], float],
                 gradient: Callable[[np.ndarray], np.ndarray],
                 hessian: Callable[[np.ndarray], np.ndarray]):
        """
        Initialize Taylor approximation for a function.
        
        Args:
            func: Function f(x) to approximate
            gradient: Gradient function ∇f(x)
            hessian: Hessian function ∇²f(x)
        """
        self.func = func
        self.gradient = gradient
        self.hessian = hessian
    
    def first_order_approximation(self, x: np.ndarray, a: np.ndarray) -> float:
        """
        First-order Taylor approximation: f(x) ≈ f(a) + ∇f(a)ᵀ(x-a)
        
        This is the linear approximation used in gradient descent.
        """
        f_a = self.func(a)
        grad_a = self.gradient(a)
        return f_a + np.dot(grad_a, x - a)
    
    def second_order_approximation(self, x: np.ndarray, a: np.ndarray) -> float:
        """
        Second-order Taylor approximation:
        f(x) ≈ f(a) + ∇f(a)ᵀ(x-a) + ½(x-a)ᵀ∇²f(a)(x-a)
        
        This is the quadratic approximation used in Newton's method.
        """
        f_a = self.func(a)
        grad_a = self.gradient(a)
        hess_a = self.hessian(a)
        
        diff = x - a
        return f_a + np.dot(grad_a, diff) + 0.5 * np.dot(diff, np.dot(hess_a, diff))
    
    def approximation_error(self, x: np.ndarray, a: np.ndarray, order: int = 2) -> Tuple[float, float, float]:
        """
        Compute approximation errors for Taylor expansions.
        
        Returns:
            (actual_value, approximation, absolute_error)
        """
        actual = self.func(x)
        
        if order == 1:
            approx = self.first_order_approximation(x, a)
        elif order == 2:
            approx = self.second_order_approximation(x, a)
        else:
            raise ValueError("Only order 1 and 2 supported")
        
        error = abs(actual - approx)
        return actual, approx, error
    
    def convergence_analysis(self, x_sequence: np.ndarray, x_star: np.ndarray) -> dict:
        """
        Analyze convergence using Taylor's theorem.
        
        For Newton's method: ||x_{k+1} - x*|| ≤ C||x_k - x*||²
        """
        n_iter = len(x_sequence)
        errors = []
        ratios = []
        
        for i in range(n_iter):
            error = np.linalg.norm(x_sequence[i] - x_star)
            errors.append(error)
            
            if i > 0 and errors[i-1] > 1e-15:
                ratio = errors[i] / (errors[i-1]**2)
                ratios.append(ratio)
        
        return {
            'errors': np.array(errors),
            'quadratic_ratios': np.array(ratios),
            'is_quadratic': len(ratios) > 0 and np.std(ratios[-5:]) < 0.1 if len(ratios) >= 5 else False
        }


def demonstrate_taylor_theorem():
    """
    Demonstrate Taylor's theorem with visual examples and applications.
    """
    print("🔷 TAYLOR'S THEOREM: Foundation of Optimization Methods")
    print("=" * 60)
    
    # Example function: f(x,y) = x² + xy + y²
    def quadratic_func(x):
        if len(x) == 1:
            return x[0]**2
        return x[0]**2 + x[0]*x[1] + x[1]**2
    
    def quadratic_grad(x):
        if len(x) == 1:
            return np.array([2*x[0]])
        return np.array([2*x[0] + x[1], x[0] + 2*x[1]])
    
    def quadratic_hess(x):
        if len(x) == 1:
            return np.array([[2.0]])
        return np.array([[2.0, 1.0], [1.0, 2.0]])
    
    taylor_approx = TaylorApproximation(quadratic_func, quadratic_grad, quadratic_hess)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 1D Example: Taylor approximations
    ax1 = plt.subplot(2, 3, 1)
    
    # 1D function for visualization
    def f_1d(x):
        return 0.5 * x**2 + 0.1 * x**4
    
    def grad_1d(x):
        return x + 0.4 * x**3
    
    def hess_1d(x):
        return 1 + 1.2 * x**2
    
    taylor_1d = TaylorApproximation(
        lambda x: f_1d(x[0]),
        lambda x: np.array([grad_1d(x[0])]),
        lambda x: np.array([[hess_1d(x[0])]])
    )
    
    x_vals = np.linspace(-2, 2, 200)
    y_vals = [f_1d(x) for x in x_vals]
    
    # Original function
    ax1.plot(x_vals, y_vals, 'blue', linewidth=3, label='f(x) = ½x² + 0.1x⁴')
    
    # Taylor approximations around x = 0.5
    a = np.array([0.5])
    
    # First-order approximation
    y_first = [taylor_1d.first_order_approximation(np.array([x]), a) for x in x_vals]
    ax1.plot(x_vals, y_first, 'red', linewidth=2, linestyle='--', 
             label='1st order (linear)')
    
    # Second-order approximation
    y_second = [taylor_1d.second_order_approximation(np.array([x]), a) for x in x_vals]
    ax1.plot(x_vals, y_second, 'green', linewidth=2, linestyle='--',
             label='2nd order (quadratic)')
    
    # Mark expansion point
    ax1.plot(a[0], f_1d(a[0]), 'ko', markersize=10, label='Expansion point')
    
    ax1.set_xlim(-1.5, 2.0)
    ax1.set_ylim(-0.5, 2.5)
    ax1.set_title('Taylor Approximations (1D)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error analysis
    ax2 = plt.subplot(2, 3, 2)
    
    # Compute approximation errors at different distances
    distances = np.logspace(-2, 0, 50)
    errors_1st = []
    errors_2nd = []
    
    a_point = np.array([0.5])
    
    for d in distances:
        x_test = a_point + d
        
        _, _, error_1st = taylor_1d.approximation_error(x_test, a_point, order=1)
        _, _, error_2nd = taylor_1d.approximation_error(x_test, a_point, order=2)
        
        errors_1st.append(error_1st)
        errors_2nd.append(error_2nd)
    
    ax2.loglog(distances, errors_1st, 'red', linewidth=2, label='1st order error')
    ax2.loglog(distances, errors_2nd, 'green', linewidth=2, label='2nd order error')
    
    # Theoretical scaling
    ax2.loglog(distances, 0.1 * distances**2, 'r--', alpha=0.7, label='O(h²)')
    ax2.loglog(distances, 0.01 * distances**3, 'g--', alpha=0.7, label='O(h³)')
    
    ax2.set_xlabel('Distance from expansion point')
    ax2.set_ylabel('Approximation error')
    ax2.set_title('Taylor Approximation Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Newton's method convergence
    ax3 = plt.subplot(2, 3, 3)
    
    # Newton's method on f(x) = x³ - x - 1
    def newton_func(x):
        return x[0]**3 - x[0] - 1
    
    def newton_grad(x):
        return np.array([3*x[0]**2 - 1])
    
    def newton_hess(x):
        return np.array([[6*x[0]]])
    
    # Newton iteration: x_{k+1} = x_k - H⁻¹∇f(x_k)
    x_current = np.array([2.0])  # Starting point
    x_sequence = [x_current.copy()]
    
    for _ in range(10):
        grad = newton_grad(x_current)
        hess = newton_hess(x_current)
        
        if abs(hess[0, 0]) > 1e-12:
            x_current = x_current - grad / hess[0, 0]
            x_sequence.append(x_current.copy())
        else:
            break
    
    # True root (approximately)
    x_star = np.array([1.3247179572])
    
    # Analyze convergence
    x_sequence = np.array(x_sequence)
    convergence = TaylorApproximation(newton_func, newton_grad, newton_hess).convergence_analysis(
        x_sequence, x_star)
    
    # Plot convergence
    iterations = range(len(convergence['errors']))
    ax3.semilogy(iterations, convergence['errors'], 'bo-', linewidth=2, 
                 markersize=6, label='Newton errors')
    
    # Show quadratic convergence
    if len(convergence['errors']) > 5:
        # Fit quadratic convergence in log space
        log_errors = np.log(convergence['errors'][-5:])
        iterations_fit = np.arange(len(log_errors))
        
        if len(iterations_fit) > 1:
            # Theoretical quadratic: log(e_{k+1}) ≈ log(c) + 2*log(e_k)
            ax3.semilogy(iterations[-5:], np.exp(log_errors), 'r--', 
                         linewidth=2, alpha=0.7, label='Quadratic fit')
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Error ||x_k - x*||')
    ax3.set_title('Newton Method Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 3D Taylor surface (2D function)
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    
    # Create mesh for 2D function
    x_range = np.linspace(-2, 2, 30)
    y_range = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Original function
    Z_orig = X**2 + X*Y + Y**2
    
    # Second-order approximation around (0.5, 0.5)
    a_2d = np.array([0.5, 0.5])
    Z_taylor = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i,j], Y[i,j]])
            Z_taylor[i,j] = taylor_approx.second_order_approximation(point, a_2d)
    
    # Plot surfaces
    ax4.plot_surface(X, Y, Z_orig, alpha=0.6, cmap='viridis', label='Original')
    ax4.plot_surface(X, Y, Z_taylor, alpha=0.6, cmap='plasma', label='Taylor')
    
    # Mark expansion point
    ax4.scatter([a_2d[0]], [a_2d[1]], [quadratic_func(a_2d)], 
               color='red', s=100, label='Expansion point')
    
    ax4.set_xlabel('x₁')
    ax4.set_ylabel('x₂')
    ax4.set_zlabel('f(x₁,x₂)')
    ax4.set_title('2nd Order Taylor Approximation')
    
    # 5. Application: Trust region visualization
    ax5 = plt.subplot(2, 3, 5)
    
    # Trust region method uses Taylor approximation within a region
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Show trust region around expansion point
    trust_radius = 0.8
    trust_x = a_2d[0] + trust_radius * np.cos(theta)
    trust_y = a_2d[1] + trust_radius * np.sin(theta)
    
    # Contour plot of original function
    x_contour = np.linspace(-1, 2, 100)
    y_contour = np.linspace(-1, 2, 100)
    X_cont, Y_cont = np.meshgrid(x_contour, y_contour)
    Z_cont = X_cont**2 + X_cont*Y_cont + Y_cont**2
    
    contours = ax5.contour(X_cont, Y_cont, Z_cont, levels=20, alpha=0.6)
    ax5.clabel(contours, inline=True, fontsize=8)
    
    # Trust region boundary
    ax5.plot(trust_x, trust_y, 'red', linewidth=3, label='Trust region')
    ax5.fill(trust_x, trust_y, 'red', alpha=0.1)
    
    # Expansion point
    ax5.plot(a_2d[0], a_2d[1], 'ro', markersize=10, label='Current point')
    
    # Gradient direction
    grad_current = quadratic_grad(a_2d)
    grad_direction = -grad_current / np.linalg.norm(grad_current)
    ax5.arrow(a_2d[0], a_2d[1], 0.5*grad_direction[0], 0.5*grad_direction[1],
              head_width=0.1, head_length=0.1, fc='blue', ec='blue',
              label='Gradient direction')
    
    ax5.set_xlim(-1, 2)
    ax5.set_ylim(-1, 2)
    ax5.set_title('Trust Region Method')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. Convergence rate comparison
    ax6 = plt.subplot(2, 3, 6)
    
    # Compare different method convergence rates
    iterations = np.arange(1, 21)
    
    # Linear convergence (gradient descent)
    linear_conv = 0.9**iterations
    
    # Superlinear convergence (quasi-Newton)
    superlinear_conv = 0.5**(1.6**iterations) 
    
    # Quadratic convergence (Newton)
    quadratic_conv = 0.1**(2**iterations)
    quadratic_conv[quadratic_conv < 1e-16] = 1e-16  # Numerical floor
    
    ax6.semilogy(iterations, linear_conv, 'blue', linewidth=2, 
                 label='Linear (Gradient Descent)')
    ax6.semilogy(iterations, superlinear_conv, 'green', linewidth=2,
                 label='Superlinear (BFGS)')
    ax6.semilogy(iterations, quadratic_conv, 'red', linewidth=2,
                 label='Quadratic (Newton)')
    
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Error')
    ax6.set_title('Convergence Rate Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Numerical verification
    print("\n📊 NUMERICAL VERIFICATION")
    print("-" * 40)
    
    # Test Taylor approximation accuracy
    print("1. Taylor Approximation Accuracy:")
    test_point = np.array([1.0, 0.5])
    expansion_point = np.array([0.8, 0.4])
    
    actual, approx_1st, error_1st = taylor_approx.approximation_error(
        test_point, expansion_point, order=1)
    actual, approx_2nd, error_2nd = taylor_approx.approximation_error(
        test_point, expansion_point, order=2)
    
    print(f"   Actual value: {actual:.6f}")
    print(f"   1st order approximation: {approx_1st:.6f} (error: {error_1st:.6f})")
    print(f"   2nd order approximation: {approx_2nd:.6f} (error: {error_2nd:.6f})")
    print(f"   Improvement ratio: {error_1st/error_2nd:.1f}x better")
    
    # Test convergence analysis
    print("\n2. Newton Method Convergence Analysis:")
    if len(convergence['quadratic_ratios']) > 0:
        avg_ratio = np.mean(convergence['quadratic_ratios'][-3:])
        print(f"   Quadratic convergence ratio: {avg_ratio:.3f}")
        print(f"   Is quadratically convergent: {convergence['is_quadratic']}")
    
    print(f"   Final error: {convergence['errors'][-1]:.2e}")
    print(f"   Iterations to convergence: {len(convergence['errors'])}")


def taylor_theorem_applications():
    """
    Showcase practical applications of Taylor's theorem in optimization.
    """
    print("\n🎯 TAYLOR'S THEOREM APPLICATIONS")
    print("=" * 40)
    
    print("1. NEWTON'S METHOD:")
    print("   - Uses 2nd order Taylor: x_{k+1} = x_k - [∇²f(x_k)]⁻¹∇f(x_k)")
    print("   - Quadratic convergence near solution")
    print("   - Requires Hessian computation/approximation")
    
    print("\n2. TRUST REGION METHODS:")
    print("   - Quadratic model: m_k(s) = f_k + g_k^T s + ½s^T B_k s")
    print("   - Minimize model within trust region ||s|| ≤ Δ_k")
    print("   - Adjust trust region based on model accuracy")
    
    print("\n3. QUASI-NEWTON METHODS:")
    print("   - Approximate Hessian using gradient information")
    print("   - BFGS: Build up curvature information iteratively")
    print("   - Maintains positive definiteness")
    
    print("\n4. CONVERGENCE ANALYSIS:")
    print("   - Linear convergence: ||e_{k+1}|| ≤ c||e_k||")
    print("   - Quadratic convergence: ||e_{k+1}|| ≤ c||e_k||²")
    print("   - Taylor remainder gives convergence rates")
    
    print("\n5. SENSITIVITY ANALYSIS:")
    print("   - Parameter perturbations: f(x,p+Δp) ≈ f(x,p) + ∇_p f^T Δp")
    print("   - Robustness assessment of optimal solutions")
    print("   - Gradient-based uncertainty propagation")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_taylor_theorem()
    taylor_theorem_applications()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Taylor's theorem provides local approximations essential for optimization")
    print("- 1st order: Linear approximation (gradient descent)")
    print("- 2nd order: Quadratic approximation (Newton's method)")  
    print("- Error scales as O(h^{k+1}) for k-th order approximation")
    print("- Enables convergence analysis and algorithm design")
    print("- Foundation for Newton, quasi-Newton, and trust region methods")
    print("\nTaylor's theorem is the mathematical foundation of modern optimization! 🚀")