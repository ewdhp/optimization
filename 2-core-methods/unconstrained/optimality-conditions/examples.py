"""
Optimality Conditions: Worked Examples

This module provides comprehensive worked examples demonstrating:
1. First-order necessary conditions (FONC)
2. Second-order necessary conditions (SONC)
3. Second-order sufficient conditions (SOSC)
4. Complete classification of critical points
5. Constrained optimization with KKT conditions

Examples range from simple 1D functions to multivariate problems,
illustrating when conditions are satisfied, when they fail, and
practical implications for optimization algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


class OptimalityAnalyzer:
    """
    Comprehensive optimality condition analyzer.
    """
    
    def __init__(self,
                 objective: Callable,
                 gradient: Callable,
                 hessian: Callable,
                 name: str = "Function"):
        """
        Initialize analyzer.
        
        Args:
            objective: f(x)
            gradient: ∇f(x)
            hessian: ∇²f(x)
            name: Function name
        """
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.name = name
    
    def analyze_point(self, x: np.ndarray, tolerance: float = 1e-6) -> Dict:
        """
        Complete analysis of optimality conditions at point.
        
        Args:
            x: Point to analyze
            tolerance: Numerical tolerance
            
        Returns:
            Analysis results
        """
        x = np.array(x)
        
        # Compute function value, gradient, Hessian
        f_val = self.objective(x)
        grad = self.gradient(x)
        hess = self.hessian(x)
        
        grad_norm = np.linalg.norm(grad)
        
        # First-order check
        is_critical = grad_norm < tolerance
        
        # Hessian analysis
        eigenvalues = np.linalg.eigvalsh(hess)
        min_eig = np.min(eigenvalues)
        max_eig = np.max(eigenvalues)
        
        # Classify definiteness
        if min_eig > tolerance:
            definiteness = 'positive_definite'
            hess_symbol = '≻ 0'
        elif min_eig >= -tolerance and max_eig > tolerance:
            definiteness = 'positive_semidefinite'
            hess_symbol = '⪰ 0'
        elif max_eig < -tolerance:
            definiteness = 'negative_definite'
            hess_symbol = '≺ 0'
        elif max_eig <= tolerance and min_eig < -tolerance:
            definiteness = 'negative_semidefinite'
            hess_symbol = '⪯ 0'
        else:
            definiteness = 'indefinite'
            hess_symbol = 'indefinite'
        
        # Classify critical point
        if not is_critical:
            classification = 'not_critical'
            optimality = 'Not a critical point'
        elif definiteness == 'positive_definite':
            classification = 'strict_local_minimum'
            optimality = 'STRICT LOCAL MINIMUM (SOSC satisfied)'
        elif definiteness == 'negative_definite':
            classification = 'strict_local_maximum'
            optimality = 'STRICT LOCAL MAXIMUM'
        elif definiteness == 'indefinite':
            classification = 'saddle_point'
            optimality = 'SADDLE POINT'
        elif definiteness == 'positive_semidefinite':
            classification = 'local_minimum_possible'
            optimality = 'Local minimum possible (need higher order)'
        else:
            classification = 'local_maximum_possible'
            optimality = 'Local maximum possible (need higher order)'
        
        # Condition checks
        fonc = is_critical
        sonc = is_critical and definiteness in ['positive_definite', 'positive_semidefinite']
        sosc = is_critical and definiteness == 'positive_definite'
        
        return {
            'x': x,
            'f': f_val,
            'gradient': grad,
            'gradient_norm': grad_norm,
            'hessian': hess,
            'eigenvalues': eigenvalues,
            'min_eigenvalue': min_eig,
            'max_eigenvalue': max_eig,
            'is_critical': is_critical,
            'definiteness': definiteness,
            'hessian_symbol': hess_symbol,
            'classification': classification,
            'optimality': optimality,
            'FONC': fonc,
            'SONC': sonc,
            'SOSC': sosc
        }
    
    def print_analysis(self, x: np.ndarray):
        """Print formatted analysis."""
        result = self.analyze_point(x)
        
        print(f"\n{'='*60}")
        print(f"OPTIMALITY ANALYSIS: {self.name}")
        print(f"{'='*60}")
        print(f"\n📍 Point: {result['x']}")
        print(f"📊 f(x) = {result['f']:.6f}")
        print(f"\n🔍 FIRST-ORDER CONDITIONS:")
        print(f"  ∇f(x) = {result['gradient']}")
        print(f"  ||∇f(x)|| = {result['gradient_norm']:.2e}")
        print(f"  FONC (∇f=0): {'✓ YES' if result['FONC'] else '✗ NO'}")
        
        if result['is_critical']:
            print(f"\n🔬 SECOND-ORDER CONDITIONS:")
            print(f"  ∇²f(x) eigenvalues: {result['eigenvalues']}")
            print(f"  λ_min = {result['min_eigenvalue']:.4f}")
            print(f"  λ_max = {result['max_eigenvalue']:.4f}")
            print(f"  Definiteness: {result['definiteness']}")
            print(f"  Hessian: {result['hessian_symbol']}")
            print(f"  SONC (∇²f⪰0): {'✓ YES' if result['SONC'] else '✗ NO'}")
            print(f"  SOSC (∇²f≻0): {'✓ YES' if result['SOSC'] else '✗ NO'}")
        
        print(f"\n🎯 CLASSIFICATION: {result['optimality']}")
        print(f"{'='*60}")


def example_1_simple_quadratic():
    """
    Example 1: Simple quadratic bowl (minimum).
    """
    print("\n" + "🎯" * 30)
    print("EXAMPLE 1: SIMPLE QUADRATIC MINIMUM")
    print("🎯" * 30)
    
    # f(x, y) = x² + 2y²
    def f(xy):
        x, y = xy
        return x**2 + 2*y**2
    
    def grad(xy):
        x, y = xy
        return np.array([2*x, 4*y])
    
    def hess(xy):
        return np.array([[2, 0], [0, 4]])
    
    analyzer = OptimalityAnalyzer(f, grad, hess, "f(x,y) = x² + 2y²")
    
    # Analyze critical point
    analyzer.print_analysis(np.array([0.0, 0.0]))
    
    # Analyze non-critical point
    print("\n📌 For comparison, analyze non-critical point (1, 1):")
    analyzer.print_analysis(np.array([1.0, 1.0]))


def example_2_saddle_point():
    """
    Example 2: Saddle point.
    """
    print("\n" + "🎯" * 30)
    print("EXAMPLE 2: SADDLE POINT")
    print("🎯" * 30)
    
    # f(x, y) = x² - y²
    def f(xy):
        x, y = xy
        return x**2 - y**2
    
    def grad(xy):
        x, y = xy
        return np.array([2*x, -2*y])
    
    def hess(xy):
        return np.array([[2, 0], [0, -2]])
    
    analyzer = OptimalityAnalyzer(f, grad, hess, "f(x,y) = x² - y²")
    
    analyzer.print_analysis(np.array([0.0, 0.0]))
    
    print("\n💡 NOTE: FONC satisfied but NOT minimum!")
    print("   Hessian is indefinite → saddle point")


def example_3_rosenbrock():
    """
    Example 3: Rosenbrock function (challenging minimum).
    """
    print("\n" + "🎯" * 30)
    print("EXAMPLE 3: ROSENBROCK FUNCTION")
    print("🎯" * 30)
    
    # f(x, y) = (1-x)² + 100(y-x²)²
    def f(xy):
        x, y = xy
        return (1 - x)**2 + 100*(y - x**2)**2
    
    def grad(xy):
        x, y = xy
        grad_x = -2*(1-x) - 400*x*(y - x**2)
        grad_y = 200*(y - x**2)
        return np.array([grad_x, grad_y])
    
    def hess(xy):
        x, y = xy
        h11 = 2 - 400*(y - x**2) + 800*x**2
        h12 = -400*x
        h22 = 200
        return np.array([[h11, h12], [h12, h22]])
    
    analyzer = OptimalityAnalyzer(f, grad, hess, "Rosenbrock")
    
    # True minimum at (1, 1)
    analyzer.print_analysis(np.array([1.0, 1.0]))
    
    # Saddle point at (0, 0)
    print("\n📌 Also analyze non-optimal point (0, 0):")
    analyzer.print_analysis(np.array([0.0, 0.0]))


def example_4_degenerate_case():
    """
    Example 4: Degenerate case (higher-order needed).
    """
    print("\n" + "🎯" * 30)
    print("EXAMPLE 4: DEGENERATE CASE")
    print("🎯" * 30)
    
    # f(x) = x⁴
    def f(x):
        return x[0]**4
    
    def grad(x):
        return np.array([4*x[0]**3])
    
    def hess(x):
        return np.array([[12*x[0]**2]])
    
    analyzer = OptimalityAnalyzer(f, grad, hess, "f(x) = x⁴")
    
    analyzer.print_analysis(np.array([0.0]))
    
    print("\n💡 NOTE: FONC satisfied, Hessian = 0 (degenerate)")
    print("   Need higher-order analysis: f''''(0) = 24 > 0")
    print("   ∴ x = 0 is actually a MINIMUM")
    print("   But standard conditions inconclusive!")


def example_5_constrained_kkt():
    """
    Example 5: Constrained optimization with KKT conditions.
    """
    print("\n" + "🎯" * 30)
    print("EXAMPLE 5: CONSTRAINED OPTIMIZATION (KKT)")
    print("🎯" * 30)
    
    print("\nProblem:")
    print("  minimize    f(x,y) = (x-2)² + (y-1)²")
    print("  subject to  g(x,y) = x + y - 1 ≤ 0")
    print()
    
    # Functions
    def f(xy):
        x, y = xy
        return (x-2)**2 + (y-1)**2
    
    def grad_f(xy):
        x, y = xy
        return np.array([2*(x-2), 2*(y-1)])
    
    def g(xy):
        x, y = xy
        return x + y - 1
    
    def grad_g(xy):
        return np.array([1, 1])
    
    # Analytical solution: minimize ||x - (2,1)||² on x+y≤1
    # Project (2,1) onto x+y=1: x* = (1.5, -0.5) violates y≥0
    # So use boundary: x+y=1
    
    # Lagrangian: L = f + λg
    # ∇_x L = 0: 2(x-2) + λ = 0, 2(y-1) + λ = 0
    # g = 0: x + y = 1
    # Solve: x = 2 - λ/2, y = 1 - λ/2, x + y = 1
    # ⟹ 3 - λ = 1, λ = 2
    # ⟹ x = 1, y = 0
    
    x_star = np.array([1.0, 0.0])
    lambda_star = 2.0
    
    print("KKT CONDITIONS CHECK:")
    print(f"\nCandidate solution: x* = {x_star}")
    print(f"Lagrange multiplier: λ* = {lambda_star}")
    
    # 1. Stationarity
    grad_f_val = grad_f(x_star)
    grad_g_val = grad_g(x_star)
    stationarity_residual = grad_f_val + lambda_star * grad_g_val
    
    print(f"\n1. Stationarity: ∇f + λ∇g = 0")
    print(f"   ∇f(x*) = {grad_f_val}")
    print(f"   λ∇g(x*) = {lambda_star * grad_g_val}")
    print(f"   Residual = {stationarity_residual}")
    print(f"   ||Residual|| = {np.linalg.norm(stationarity_residual):.2e}")
    print(f"   ✓ Satisfied" if np.linalg.norm(stationarity_residual) < 1e-6 else "   ✗ Violated")
    
    # 2. Primal feasibility
    g_val = g(x_star)
    print(f"\n2. Primal Feasibility: g(x*) ≤ 0")
    print(f"   g(x*) = {g_val:.6f}")
    print(f"   ✓ Satisfied" if g_val <= 1e-6 else "   ✗ Violated")
    
    # 3. Dual feasibility
    print(f"\n3. Dual Feasibility: λ ≥ 0")
    print(f"   λ = {lambda_star:.6f}")
    print(f"   ✓ Satisfied" if lambda_star >= -1e-6 else "   ✗ Violated")
    
    # 4. Complementarity
    complementarity = lambda_star * g_val
    print(f"\n4. Complementarity: λg(x*) = 0")
    print(f"   λg(x*) = {complementarity:.2e}")
    print(f"   ✓ Satisfied" if abs(complementarity) < 1e-6 else "   ✗ Violated")
    
    print(f"\n🎯 RESULT: All KKT conditions satisfied!")
    print(f"   x* = {x_star} is optimal for constrained problem")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Contours with constraint
    ax1 = axes[0]
    
    x_range = np.linspace(-1, 3, 100)
    y_range = np.linspace(-1, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (X-2)**2 + (Y-1)**2
    
    contour = ax1.contour(X, Y, Z, levels=20, alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    
    # Constraint boundary
    x_constraint = np.linspace(-1, 3, 100)
    y_constraint = 1 - x_constraint
    ax1.plot(x_constraint, y_constraint, 'r-', linewidth=3, label='g(x,y)=0')
    
    # Feasible region
    ax1.fill_between(x_constraint, y_constraint, -1, alpha=0.2, color='green',
                     label='Feasible region')
    
    # Unconstrained minimum
    ax1.plot(2, 1, 'b*', markersize=15, label='Unconstrained min')
    
    # Constrained minimum
    ax1.plot(x_star[0], x_star[1], 'r*', markersize=15, label='Constrained min')
    
    # Gradients at optimum
    scale = 0.3
    ax1.arrow(x_star[0], x_star[1],
             scale*grad_f_val[0], scale*grad_f_val[1],
             head_width=0.1, head_length=0.1,
             fc='blue', ec='blue', linewidth=2,
             label='∇f(x*)')
    ax1.arrow(x_star[0], x_star[1],
             -scale*lambda_star*grad_g_val[0], -scale*lambda_star*grad_g_val[1],
             head_width=0.1, head_length=0.1,
             fc='red', ec='red', linewidth=2,
             label='-λ∇g(x*)')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Constrained Optimization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Distance from constraint
    ax2 = axes[1]
    
    t_vals = np.linspace(-1, 3, 100)
    x_vals = t_vals
    y_vals = 1 - t_vals
    
    f_vals = [(x-2)**2 + (y-1)**2 for x, y in zip(x_vals, y_vals)]
    
    ax2.plot(t_vals, f_vals, 'b-', linewidth=2)
    ax2.plot(x_star[0], f(x_star), 'r*', markersize=15,
            label=f'Minimum at x={x_star[0]:.1f}')
    
    ax2.set_xlabel('x (on constraint x+y=1)')
    ax2.set_ylabel('f(x, 1-x)')
    ax2.set_title('Objective along Constraint')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def comprehensive_visualization():
    """
    Comprehensive visualization of all examples.
    """
    print("\n" + "📊" * 30)
    print("COMPREHENSIVE VISUALIZATION")
    print("📊" * 30)
    
    examples = [
        # (f, grad, hess, x_crit, title, x_range, y_range)
        (lambda xy: xy[0]**2 + 2*xy[1]**2,
         lambda xy: np.array([2*xy[0], 4*xy[1]]),
         lambda xy: np.array([[2, 0], [0, 4]]),
         np.array([0, 0]), "Minimum", (-2, 2), (-2, 2)),
        
        (lambda xy: -xy[0]**2 - xy[1]**2,
         lambda xy: np.array([-2*xy[0], -2*xy[1]]),
         lambda xy: np.array([[-2, 0], [0, -2]]),
         np.array([0, 0]), "Maximum", (-2, 2), (-2, 2)),
        
        (lambda xy: xy[0]**2 - xy[1]**2,
         lambda xy: np.array([2*xy[0], -2*xy[1]]),
         lambda xy: np.array([[2, 0], [0, -2]]),
         np.array([0, 0]), "Saddle", (-2, 2), (-2, 2)),
    ]
    
    fig = plt.figure(figsize=(18, 10))
    
    for idx, (f, grad, hess, x_crit, title, x_range, y_range) in enumerate(examples):
        # 3D surface
        ax_3d = fig.add_subplot(2, 3, idx+1, projection='3d')
        
        x_vals = np.linspace(x_range[0], x_range[1], 50)
        y_vals = np.linspace(y_range[0], y_range[1], 50)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        Z = np.array([[f(np.array([x, y])) for x in x_vals] for y in y_vals])
        
        surf = ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax_3d.plot([x_crit[0]], [x_crit[1]], [f(x_crit)],
                  'r*', markersize=15)
        
        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('f')
        ax_3d.set_title(f'{title} (3D)')
        
        # Contour with Hessian eigenvectors
        ax_contour = fig.add_subplot(2, 3, idx+4)
        
        contour = ax_contour.contour(X, Y, Z, levels=20, cmap='viridis')
        ax_contour.plot(x_crit[0], x_crit[1], 'r*', markersize=15)
        
        # Hessian eigenvectors
        H = hess(x_crit)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        for eval, evec in zip(eigenvalues, eigenvectors.T):
            if abs(eval) > 1e-6:
                color = 'green' if eval > 0 else 'red'
                scale = 0.5
                ax_contour.arrow(x_crit[0], x_crit[1],
                               scale*evec[0], scale*evec[1],
                               head_width=0.15, head_length=0.1,
                               fc=color, ec=color, alpha=0.7,
                               linewidth=2)
        
        ax_contour.set_xlabel('x')
        ax_contour.set_ylabel('y')
        ax_contour.set_title(f'{title} (Eigenvectors)')
        ax_contour.grid(True, alpha=0.3)
        ax_contour.axis('equal')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run all examples
    example_1_simple_quadratic()
    example_2_saddle_point()
    example_3_rosenbrock()
    example_4_degenerate_case()
    example_5_constrained_kkt()
    comprehensive_visualization()
    
    print("\n" + "="*60)
    print("🎓 SUMMARY: OPTIMALITY CONDITIONS")
    print("="*60)
    print("\n✅ NECESSARY CONDITIONS (must hold at optimum):")
    print("   • FONC: ∇f(x*) = 0")
    print("   • SONC: ∇²f(x*) ⪰ 0")
    print()
    print("✅ SUFFICIENT CONDITIONS (guarantee optimum):")
    print("   • SOSC: ∇f(x*) = 0  AND  ∇²f(x*) ≻ 0")
    print()
    print("🔍 CLASSIFICATION:")
    print("   • ∇²f ≻ 0 → strict local minimum")
    print("   • ∇²f ≺ 0 → strict local maximum")
    print("   • ∇²f indefinite → saddle point")
    print("   • ∇²f = 0 → degenerate (need higher order)")
    print()
    print("🔒 CONSTRAINED (KKT CONDITIONS):")
    print("   1. Stationarity: ∇_x L = 0")
    print("   2. Primal feasibility: g_i(x) ≤ 0, h_j(x) = 0")
    print("   3. Dual feasibility: λ_i ≥ 0")
    print("   4. Complementarity: λ_i g_i(x) = 0")
    print("\n" + "="*60)
