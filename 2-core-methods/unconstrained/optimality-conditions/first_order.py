"""
First-Order Optimality Conditions

First-order necessary conditions characterize local optima using gradient information.
These conditions are fundamental to all gradient-based optimization algorithms.

UNCONSTRAINED OPTIMIZATION:
    Necessary Condition (FONC):
        If x* is a local minimum, then ∇f(x*) = 0
    
    Critical Points:
        Points where ∇f(x) = 0 are called stationary or critical points
        Can be minima, maxima, or saddle points

CONSTRAINED OPTIMIZATION:
    For problem: min f(x)  s.t.  g_i(x) ≤ 0, h_j(x) = 0
    
    KKT Conditions (necessary):
        1. Stationarity: ∇f(x*) + Σ λ_i ∇g_i(x*) + Σ μ_j ∇h_j(x*) = 0
        2. Primal feasibility: g_i(x*) ≤ 0, h_j(x*) = 0
        3. Dual feasibility: λ_i ≥ 0
        4. Complementarity: λ_i g_i(x*) = 0

SUFFICIENT CONDITIONS:
    First-order conditions alone are NOT sufficient for optimality
    (except in convex case: ∇f(x*) = 0 + convexity ⟹ global minimum)
    
    Need second-order conditions (Hessian) for general case

Key Results:
1. Fermat's Theorem: Gradient vanishes at interior local extrema
2. Convex functions: FONC is necessary and sufficient
3. Descent direction: d is descent if ∇f^T d < 0
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def find_critical_points_1d(f: Callable, df: Callable, 
                           x_range: Tuple[float, float],
                           num_starts: int = 20) -> List[float]:
    """
    Find critical points of 1D function by solving ∇f = 0.
    
    Args:
        f: Objective function
        df: Derivative function
        x_range: Search range (a, b)
        num_starts: Number of random starting points
        
    Returns:
        List of critical points
    """
    from scipy.optimize import fsolve
    
    critical_points = []
    a, b = x_range
    
    # Try multiple starting points
    for _ in range(num_starts):
        x0 = np.random.uniform(a, b)
        
        try:
            x_crit = fsolve(df, x0)[0]
            
            # Verify it's in range and actually critical
            if a <= x_crit <= b and abs(df(x_crit)) < 1e-6:
                # Check if already found
                is_new = all(abs(x_crit - xc) > 1e-4 for xc in critical_points)
                if is_new:
                    critical_points.append(x_crit)
        except:
            pass
    
    return sorted(critical_points)


def classify_critical_point_1d(f: Callable, df: Callable, d2f: Callable,
                               x_crit: float) -> str:
    """
    Classify critical point using second derivative.
    
    Args:
        f: Function
        df: First derivative
        d2f: Second derivative
        x_crit: Critical point
        
    Returns:
        Classification: 'minimum', 'maximum', 'inflection', or 'degenerate'
    """
    # Verify it's critical
    if abs(df(x_crit)) > 1e-4:
        return 'not_critical'
    
    # Check second derivative
    second_deriv = d2f(x_crit)
    
    if abs(second_deriv) < 1e-8:
        return 'degenerate'
    elif second_deriv > 0:
        return 'minimum'
    else:
        return 'maximum'


def verify_fonc_unconstrained(f: Callable, grad: Callable, 
                              x_star: np.ndarray,
                              tolerance: float = 1e-6) -> bool:
    """
    Verify first-order necessary condition for unconstrained problem.
    
    FONC: ∇f(x*) = 0
    
    Args:
        f: Objective function
        grad: Gradient function
        x_star: Candidate optimal point
        tolerance: Tolerance for gradient norm
        
    Returns:
        True if FONC satisfied
    """
    gradient = grad(x_star)
    grad_norm = np.linalg.norm(gradient)
    
    return grad_norm < tolerance


def verify_descent_direction(grad: np.ndarray, direction: np.ndarray) -> bool:
    """
    Verify if direction is a descent direction.
    
    Direction d is descent if ∇f^T d < 0
    
    Args:
        grad: Gradient at current point
        direction: Proposed direction
        
    Returns:
        True if descent direction
    """
    return np.dot(grad, direction) < 0


def verify_kkt_conditions(x: np.ndarray,
                         f_grad: np.ndarray,
                         g_vals: List[float],
                         g_grads: List[np.ndarray],
                         lambdas: np.ndarray,
                         tolerance: float = 1e-6) -> Dict[str, bool]:
    """
    Verify KKT conditions for inequality-constrained problem.
    
    Problem: min f(x)  s.t.  g_i(x) ≤ 0
    
    KKT Conditions:
    1. Stationarity: ∇f + Σ λ_i ∇g_i = 0
    2. Primal feasibility: g_i(x) ≤ 0
    3. Dual feasibility: λ_i ≥ 0
    4. Complementarity: λ_i g_i(x) = 0
    
    Args:
        x: Candidate point
        f_grad: Gradient of f at x
        g_vals: Values of constraints g_i(x)
        g_grads: Gradients of constraints
        lambdas: Lagrange multipliers
        tolerance: Numerical tolerance
        
    Returns:
        Dictionary with verification results
    """
    # 1. Stationarity
    lagrangian_grad = f_grad.copy()
    for i, g_grad in enumerate(g_grads):
        lagrangian_grad += lambdas[i] * g_grad
    
    stationarity = np.linalg.norm(lagrangian_grad) < tolerance
    
    # 2. Primal feasibility
    primal_feasible = all(g_val <= tolerance for g_val in g_vals)
    
    # 3. Dual feasibility
    dual_feasible = all(lam >= -tolerance for lam in lambdas)
    
    # 4. Complementarity
    complementarity_slacks = [abs(lambdas[i] * g_vals[i]) 
                             for i in range(len(g_vals))]
    complementarity = all(slack < tolerance for slack in complementarity_slacks)
    
    return {
        'stationarity': stationarity,
        'primal_feasible': primal_feasible,
        'dual_feasible': dual_feasible,
        'complementarity': complementarity,
        'all_satisfied': all([stationarity, primal_feasible, 
                             dual_feasible, complementarity])
    }


def demonstrate_fonc_1d():
    """
    Demonstrate first-order conditions in 1D.
    """
    print("📈 FIRST-ORDER CONDITIONS: 1D EXAMPLES")
    print("=" * 60)
    
    # Example 1: Simple quadratic
    print("\n🎯 EXAMPLE 1: Quadratic f(x) = x²")
    print("-" * 50)
    
    f1 = lambda x: x**2
    df1 = lambda x: 2*x
    d2f1 = lambda x: 2.0
    
    critical_1 = find_critical_points_1d(f1, df1, (-5, 5))
    print(f"Critical points: {critical_1}")
    
    for x_c in critical_1:
        classification = classify_critical_point_1d(f1, df1, d2f1, x_c)
        print(f"  x = {x_c:.4f}: {classification}, f(x) = {f1(x_c):.4f}")
    
    # Example 2: Cubic with multiple critical points
    print("\n🎯 EXAMPLE 2: Cubic f(x) = x³ - 3x")
    print("-" * 50)
    
    f2 = lambda x: x**3 - 3*x
    df2 = lambda x: 3*x**2 - 3
    d2f2 = lambda x: 6*x
    
    critical_2 = find_critical_points_1d(f2, df2, (-3, 3))
    print(f"Critical points: {critical_2}")
    
    for x_c in critical_2:
        classification = classify_critical_point_1d(f2, df2, d2f2, x_c)
        print(f"  x = {x_c:.4f}: {classification}, f(x) = {f2(x_c):.4f}")
    
    # Example 3: Sinusoidal
    print("\n🎯 EXAMPLE 3: Sinusoidal f(x) = sin(x) + 0.1x")
    print("-" * 50)
    
    f3 = lambda x: np.sin(x) + 0.1*x
    df3 = lambda x: np.cos(x) + 0.1
    d2f3 = lambda x: -np.sin(x)
    
    critical_3 = find_critical_points_1d(f3, df3, (0, 4*np.pi))
    print(f"Critical points: {critical_3}")
    
    for x_c in critical_3:
        classification = classify_critical_point_1d(f3, df3, d2f3, x_c)
        print(f"  x = {x_c:.4f}: {classification}, f(x) = {f3(x_c):.4f}")
    
    # Visualization
    fig = plt.figure(figsize=(18, 10))
    
    # Plot quadratic
    ax1 = plt.subplot(2, 3, 1)
    x_vals = np.linspace(-5, 5, 200)
    ax1.plot(x_vals, [f1(x) for x in x_vals], 'b-', linewidth=2)
    
    for x_c in critical_1:
        classification = classify_critical_point_1d(f1, df1, d2f1, x_c)
        color = 'g' if 'min' in classification else 'r'
        marker = 'o' if 'min' in classification else '^'
        ax1.plot(x_c, f1(x_c), color=color, marker=marker, 
                markersize=12, label=f'{classification}')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('f(x) = x²')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot cubic
    ax2 = plt.subplot(2, 3, 2)
    x_vals = np.linspace(-3, 3, 200)
    ax2.plot(x_vals, [f2(x) for x in x_vals], 'b-', linewidth=2)
    
    for x_c in critical_2:
        classification = classify_critical_point_1d(f2, df2, d2f2, x_c)
        if 'min' in classification:
            color, marker = 'g', 'o'
        elif 'max' in classification:
            color, marker = 'r', '^'
        else:
            color, marker = 'orange', 's'
        
        ax2.plot(x_c, f2(x_c), color=color, marker=marker,
                markersize=12, label=f'{classification}')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title('f(x) = x³ - 3x')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot sinusoidal
    ax3 = plt.subplot(2, 3, 3)
    x_vals = np.linspace(0, 4*np.pi, 200)
    ax3.plot(x_vals, [f3(x) for x in x_vals], 'b-', linewidth=2)
    
    for x_c in critical_3:
        classification = classify_critical_point_1d(f3, df3, d2f3, x_c)
        if 'min' in classification:
            color, marker = 'g', 'o'
        elif 'max' in classification:
            color, marker = 'r', '^'
        else:
            color, marker = 'orange', 's'
        
        ax3.plot(x_c, f3(x_c), color=color, marker=marker,
                markersize=10, label=f'{classification}')
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('f(x)')
    ax3.set_title('f(x) = sin(x) + 0.1x')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot derivatives for cubic
    ax4 = plt.subplot(2, 3, 4)
    x_vals = np.linspace(-3, 3, 200)
    ax4.plot(x_vals, [df2(x) for x in x_vals], 'b-', linewidth=2, 
            label="f'(x) = 3x² - 3")
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    for x_c in critical_2:
        ax4.plot(x_c, 0, 'ro', markersize=10)
    
    ax4.set_xlabel('x')
    ax4.set_ylabel("f'(x)")
    ax4.set_title("First Derivative (zeros = critical points)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot second derivative for cubic
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(x_vals, [d2f2(x) for x in x_vals], 'g-', linewidth=2,
            label="f''(x) = 6x")
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    for x_c in critical_2:
        second_val = d2f2(x_c)
        color = 'g' if second_val > 0 else 'r'
        ax5.plot(x_c, second_val, 'o', color=color, markersize=10)
    
    ax5.set_xlabel('x')
    ax5.set_ylabel("f''(x)")
    ax5.set_title("Second Derivative (sign = curvature)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Descent directions example
    ax6 = plt.subplot(2, 3, 6)
    
    # At point x = 2 for cubic
    x_test = 2.0
    grad_test = df2(x_test)
    
    # Test various directions
    directions = np.linspace(-2, 2, 100)
    directional_derivs = directions * grad_test
    
    ax6.plot(directions, directional_derivs, 'b-', linewidth=2)
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Shade descent region
    descent_dirs = directions[directional_derivs < 0]
    if len(descent_dirs) > 0:
        ax6.fill_between(descent_dirs,
                        grad_test * descent_dirs,
                        alpha=0.3, color='green',
                        label='Descent directions')
    
    ax6.set_xlabel('Direction d')
    ax6.set_ylabel("∇f·d")
    ax6.set_title(f'Descent Directions at x = {x_test}')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_fonc_2d():
    """
    Demonstrate first-order conditions in 2D.
    """
    print("\n🗺️  FIRST-ORDER CONDITIONS: 2D EXAMPLES")
    print("=" * 60)
    
    # Example: Quadratic bowl
    print("\n🎯 EXAMPLE: Quadratic f(x,y) = x² + 2y²")
    print("-" * 50)
    
    def f(xy):
        x, y = xy
        return x**2 + 2*y**2
    
    def grad_f(xy):
        x, y = xy
        return np.array([2*x, 4*y])
    
    # Critical point
    x_star = np.array([0.0, 0.0])
    
    print(f"Candidate minimum: {x_star}")
    print(f"Function value: {f(x_star):.4f}")
    print(f"Gradient: {grad_f(x_star)}")
    print(f"||∇f||: {np.linalg.norm(grad_f(x_star)):.2e}")
    
    fonc_satisfied = verify_fonc_unconstrained(f, grad_f, x_star)
    print(f"FONC satisfied: {fonc_satisfied}")
    
    # Test descent directions
    print("\n📐 Descent Directions at (1, 1):")
    x_test = np.array([1.0, 1.0])
    grad_test = grad_f(x_test)
    
    test_directions = [
        (np.array([-1, 0]), "Negative x"),
        (np.array([0, -1]), "Negative y"),
        (np.array([-1, -1]), "Negative gradient"),
        (np.array([1, 1]), "Positive gradient"),
        (np.array([1, -0.5]), "Mixed")
    ]
    
    for direction, name in test_directions:
        direction_norm = direction / np.linalg.norm(direction)
        is_descent = verify_descent_direction(grad_test, direction_norm)
        directional_deriv = np.dot(grad_test, direction_norm)
        print(f"  {name:20s}: {'✓ Descent' if is_descent else '✗ Ascent':12s} "
              f"(∇f·d = {directional_deriv:6.2f})")
    
    # Visualization
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Contour plot with gradient field
    ax1 = plt.subplot(1, 3, 1)
    
    x_range = np.linspace(-3, 3, 50)
    y_range = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + 2*Y**2
    
    contour = ax1.contour(X, Y, Z, levels=20, alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    
    # Gradient field
    x_sparse = np.linspace(-3, 3, 10)
    y_sparse = np.linspace(-3, 3, 10)
    X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse)
    
    U = 2*X_sparse
    V = 4*Y_sparse
    
    ax1.quiver(X_sparse, Y_sparse, U, V, alpha=0.5)
    
    # Critical point
    ax1.plot(0, 0, 'r*', markersize=20, label='Critical point')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('f(x,y) = x² + 2y² with Gradient Field')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Level sets
    ax2 = plt.subplot(1, 3, 2)
    
    levels_plot = [0.5, 1, 2, 4, 8, 16]
    contour = ax2.contour(X, Y, Z, levels=levels_plot, alpha=0.8)
    ax2.clabel(contour, inline=True, fontsize=10)
    
    ax2.plot(0, 0, 'r*', markersize=20)
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Level Sets (∇f ⊥ level curves)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot 3: Descent cone
    ax3 = plt.subplot(1, 3, 3)
    
    # At point (1, 1)
    x_pt, y_pt = 1.0, 1.0
    grad_at_pt = grad_f(np.array([x_pt, y_pt]))
    
    # Plot many directions
    angles = np.linspace(0, 2*np.pi, 100)
    descent_angles = []
    ascent_angles = []
    
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        if verify_descent_direction(grad_at_pt, direction):
            descent_angles.append(angle)
        else:
            ascent_angles.append(angle)
    
    # Plot descent cone
    for angle in descent_angles:
        ax3.plot([0, np.cos(angle)], [0, np.sin(angle)],
                'g-', alpha=0.3, linewidth=1)
    
    # Plot ascent cone
    for angle in ascent_angles:
        ax3.plot([0, np.cos(angle)], [0, np.sin(angle)],
                'r-', alpha=0.3, linewidth=1)
    
    # Plot gradient direction
    grad_dir = grad_at_pt / np.linalg.norm(grad_at_pt)
    ax3.arrow(0, 0, grad_dir[0], grad_dir[1],
             head_width=0.1, head_length=0.1,
             fc='blue', ec='blue', linewidth=2,
             label='Gradient')
    
    # Plot negative gradient
    ax3.arrow(0, 0, -grad_dir[0], -grad_dir[1],
             head_width=0.1, head_length=0.1,
             fc='darkgreen', ec='darkgreen', linewidth=2,
             label='Steepest descent')
    
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_xlabel('d_x')
    ax3.set_ylabel('d_y')
    ax3.set_title(f'Descent Cone at ({x_pt}, {y_pt})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.show()


def first_order_theory():
    """
    Summary of first-order optimality theory.
    """
    print("\n📚 FIRST-ORDER OPTIMALITY THEORY")
    print("=" * 60)
    
    print("🎯 UNCONSTRAINED OPTIMIZATION:")
    print("  Problem: min f(x)")
    print()
    print("  NECESSARY CONDITION (Fermat's Theorem):")
    print("    If x* is a local minimum, then ∇f(x*) = 0")
    print()
    print("  Note: NOT SUFFICIENT in general!")
    print("  Counterexample: f(x) = x³ at x = 0")
    print("    ∇f(0) = 0, but 0 is inflection point, not minimum")
    
    print("\n🔒 CONSTRAINED OPTIMIZATION:")
    print("  Problem: min f(x)  s.t.  g_i(x) ≤ 0, h_j(x) = 0")
    print()
    print("  KKT CONDITIONS (necessary if constraint qualification holds):")
    print("    1. Stationarity: ∇_x L(x*, λ*, μ*) = 0")
    print("       where L = f + Σ λ_i g_i + Σ μ_j h_j")
    print("    2. Primal feasibility: g_i(x*) ≤ 0, h_j(x*) = 0")
    print("    3. Dual feasibility: λ_i* ≥ 0")
    print("    4. Complementarity: λ_i* g_i(x*) = 0")
    
    print("\n💡 CONVEX CASE (Special!):")
    print("  If f is convex:")
    print("    ∇f(x*) = 0  ⟺  x* is GLOBAL minimum")
    print()
    print("  If f is convex and constraints are convex:")
    print("    KKT conditions are NECESSARY and SUFFICIENT")
    
    print("\n📐 DESCENT DIRECTIONS:")
    print("  Direction d is a descent direction at x if:")
    print("    ∇f(x)^T d < 0")
    print()
    print("  Steepest descent direction:")
    print("    d* = -∇f(x) / ||∇f(x)||")
    print("    Maximizes rate of decrease per unit step")
    
    print("\n🔍 PRACTICAL IMPLICATIONS:")
    print("1. Gradient descent uses: x_{k+1} = x_k - α_k ∇f(x_k)")
    print("   Guaranteed to find stationary point (if converges)")
    print()
    print("2. Need second-order test to verify it's minimum")
    print()
    print("3. For constrained problems, use KKT conditions")
    print("   to check optimality and find multipliers")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_fonc_1d()
    demonstrate_fonc_2d()
    first_order_theory()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- FONC: ∇f(x*) = 0 at local minimum (necessary, not sufficient)")
    print("- Critical points can be minima, maxima, or saddle points")
    print("- Descent direction: ∇f^T d < 0")
    print("- Steepest descent: d = -∇f (best local linear approximation)")
    print("- Convex case: FONC is necessary AND sufficient")
    print("- KKT conditions generalize FONC to constrained problems")
    print("\nFirst-order: Where the gradient points the way! 🧭")
