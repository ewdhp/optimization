"""
Jensen's Inequality and Convexity

Jensen's inequality is the fundamental characterization of convex functions
and the cornerstone of convex optimization theory. It provides both the
definition of convexity and practical tools for optimization.

Mathematical Statement:
For a convex function f and any convex combination:

f(λ₁x₁ + λ₂x₂ + ... + λₙxₙ) ≤ λ₁f(x₁) + λ₂f(x₂) + ... + λₙf(xₙ)

where λᵢ ≥ 0 and Σλᵢ = 1.

General Form (Probability Measures):
If f is convex and X is a random variable, then:
f(E[X]) ≤ E[f(X)]

Key Implications:
- Defines convex functions
- Any local minimum of a convex function is global
- Enables powerful optimization algorithms
- Foundation for concentration inequalities
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class ConvexFunction:
    """
    Implementation of convex functions with Jensen's inequality verification.
    """
    
    def __init__(self, func: Callable[[np.ndarray], float], 
                 name: str = "f", 
                 domain_check: Optional[Callable[[np.ndarray], bool]] = None):
        """
        Initialize convex function.
        
        Args:
            func: Function to evaluate
            name: Name for plotting and display
            domain_check: Function to check if point is in domain
        """
        self.func = func
        self.name = name
        self.domain_check = domain_check if domain_check else lambda x: True
    
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate function at point x."""
        x = np.array(x)
        if not self.domain_check(x):
            return np.inf
        return self.func(x)
    
    def verify_jensen_inequality(self, points: List[np.ndarray], 
                                weights: np.ndarray, 
                                tol: float = 1e-12) -> Tuple[bool, float, float]:
        """
        Verify Jensen's inequality for given points and weights.
        
        Returns:
            (inequality_holds, lhs_value, rhs_value)
        """
        points = [np.array(p) for p in points]
        weights = np.array(weights)
        
        # Check weight constraints
        if abs(np.sum(weights) - 1.0) > tol or np.any(weights < -tol):
            raise ValueError("Weights must be non-negative and sum to 1")
        
        # Compute convex combination of points
        convex_combination = np.sum([w * p for w, p in zip(weights, points)], axis=0)
        
        # Left side: f(convex combination)
        lhs = self(convex_combination)
        
        # Right side: convex combination of function values
        rhs = np.sum([w * self(p) for w, p in zip(weights, points)])
        
        # Jensen's inequality: f(convex_combo) ≤ convex_combo(f(points))
        inequality_holds = lhs <= rhs + tol
        
        return inequality_holds, lhs, rhs
    
    def test_convexity(self, n_tests: int = 1000, 
                      domain_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> dict:
        """
        Test convexity using random Jensen's inequality tests.
        
        Args:
            n_tests: Number of random tests to perform
            domain_bounds: (lower_bounds, upper_bounds) for sampling
            
        Returns:
            Dictionary with test results
        """
        if domain_bounds is None:
            lower_bounds = np.array([-2.0])
            upper_bounds = np.array([2.0])
        else:
            lower_bounds, upper_bounds = domain_bounds
        
        dim = len(lower_bounds)
        violations = 0
        max_violation = 0.0
        
        for _ in range(n_tests):
            # Generate two random points
            x1 = np.random.uniform(lower_bounds, upper_bounds)
            x2 = np.random.uniform(lower_bounds, upper_bounds)
            
            # Random weight
            lam = np.random.random()
            weights = np.array([lam, 1 - lam])
            
            try:
                inequality_holds, lhs, rhs = self.verify_jensen_inequality(
                    [x1, x2], weights)
                
                if not inequality_holds:
                    violations += 1
                    violation_amount = lhs - rhs
                    max_violation = max(max_violation, violation_amount)
                    
            except:
                # Skip if points are outside domain or other issues
                continue
        
        return {
            'total_tests': n_tests,
            'violations': violations,
            'violation_rate': violations / n_tests,
            'max_violation': max_violation,
            'is_convex': violations == 0
        }
    
    def epigraph_points(self, x_range: Tuple[float, float], 
                       n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generate points for epigraph visualization."""
        x_vals = np.linspace(x_range[0], x_range[1], n_points)
        
        if isinstance(x_vals[0], (int, float)):
            # 1D function
            y_vals = np.array([self(np.array([x])) for x in x_vals])
        else:
            # Multi-dimensional - take slice
            y_vals = np.array([self(x) for x in x_vals])
        
        return x_vals, y_vals


class JensenInequalityExamples:
    """
    Collection of important convex functions and Jensen's inequality examples.
    """
    
    @staticmethod
    def quadratic_function(A: Optional[np.ndarray] = None) -> ConvexFunction:
        """Quadratic function: f(x) = ½xᵀAx (convex if A ⪰ 0)."""
        if A is None:
            A = np.array([[1.0]])
        
        A = np.array(A)
        
        def func(x):
            x = np.array(x)
            return 0.5 * np.dot(x, np.dot(A, x))
        
        return ConvexFunction(func, f"½xᵀAx (eigenvals: {np.linalg.eigvals(A)})")
    
    @staticmethod
    def exponential_function(a: float = 1.0) -> ConvexFunction:
        """Exponential function: f(x) = exp(ax) (convex for any a)."""
        def func(x):
            x = np.array(x)
            return np.exp(a * np.sum(x))
        
        return ConvexFunction(func, f"exp({a}x)")
    
    @staticmethod
    def log_sum_exp() -> ConvexFunction:
        """Log-sum-exp: f(x) = log(Σᵢ exp(xᵢ)) (convex)."""
        def func(x):
            x = np.array(x)
            # Numerically stable computation
            x_max = np.max(x)
            return x_max + np.log(np.sum(np.exp(x - x_max)))
        
        return ConvexFunction(func, "log(Σexp(xᵢ))")
    
    @staticmethod
    def negative_log_function() -> ConvexFunction:
        """Negative log: f(x) = -log(x) (convex on x > 0)."""
        def func(x):
            x = np.array(x)
            if np.any(x <= 0):
                return np.inf
            return -np.sum(np.log(x))
        
        def domain_check(x):
            return np.all(np.array(x) > 0)
        
        return ConvexFunction(func, "-log(x)", domain_check)
    
    @staticmethod
    def norm_function(p: float = 2.0) -> ConvexFunction:
        """p-norm: f(x) = ||x||_p (convex for p ≥ 1)."""
        def func(x):
            x = np.array(x)
            if p == 2:
                return np.linalg.norm(x, 2)
            elif p == 1:
                return np.sum(np.abs(x))
            elif p == np.inf:
                return np.max(np.abs(x))
            else:
                return np.sum(np.abs(x)**p)**(1/p)
        
        return ConvexFunction(func, f"||x||_{p}")
    
    @staticmethod
    def non_convex_function() -> ConvexFunction:
        """Example non-convex function: f(x) = x⁴ - 2x² (for comparison)."""
        def func(x):
            x = np.array(x)
            return x[0]**4 - 2*x[0]**2
        
        return ConvexFunction(func, "x⁴ - 2x² (non-convex)")


def demonstrate_jensen_inequality():
    """
    Demonstrate Jensen's inequality with comprehensive visual examples.
    """
    print("🔷 JENSEN'S INEQUALITY: The Heart of Convexity")
    print("=" * 55)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Classic Jensen's inequality visualization
    ax1 = plt.subplot(2, 3, 1)
    
    # Convex function: f(x) = x²
    convex_func = JensenInequalityExamples.quadratic_function(np.array([[2.0]]))
    
    x_vals = np.linspace(-2, 2, 200)
    y_vals = [convex_func(np.array([x])) for x in x_vals]
    
    ax1.plot(x_vals, y_vals, 'blue', linewidth=3, label='f(x) = x²')
    
    # Show Jensen's inequality for specific points
    x1, x2 = -1.5, 1.0
    y1, y2 = convex_func(np.array([x1])), convex_func(np.array([x2]))
    
    # Different weights
    for lam, color, style in [(0.3, 'red', '-'), (0.6, 'green', '--'), (0.5, 'orange', ':')]:
        # Convex combination of points
        x_combo = lam * x1 + (1 - lam) * x2
        y_combo = convex_func(np.array([x_combo]))
        
        # Linear combination of function values
        y_linear = lam * y1 + (1 - lam) * y2
        
        # Plot points and connections
        ax1.plot([x1, x2], [y1, y2], color=color, linestyle=style, alpha=0.7)
        ax1.plot(x_combo, y_combo, 'o', color=color, markersize=8, 
                 label=f'λ={lam:.1f}: f({x_combo:.2f})={y_combo:.2f}')
        ax1.plot(x_combo, y_linear, 's', color=color, markersize=8,
                 label=f'λf(x₁)+(1-λ)f(x₂)={y_linear:.2f}')
        
        # Show inequality gap
        ax1.plot([x_combo, x_combo], [y_combo, y_linear], 
                 color=color, linewidth=3, alpha=0.8)
    
    # Mark original points
    ax1.plot(x1, y1, 'ko', markersize=10, label='x₁, x₂')
    ax1.plot(x2, y2, 'ko', markersize=10)
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(0, 5)
    ax1.set_title('Jensen\'s Inequality Visualization')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Multiple point Jensen's inequality
    ax2 = plt.subplot(2, 3, 2)
    
    # Exponential function
    exp_func = JensenInequalityExamples.exponential_function(0.5)
    
    x_range = np.linspace(-2, 3, 200)
    y_range = [exp_func(np.array([x])) for x in x_range]
    
    ax2.plot(x_range, y_range, 'blue', linewidth=3, label='f(x) = exp(0.5x)')
    
    # Multiple points with weights
    points = [-1.5, 0, 1.5, 2.5]
    weights = np.array([0.2, 0.3, 0.3, 0.2])
    colors = ['red', 'green', 'purple', 'orange']
    
    # Plot individual points
    for i, (x, w, color) in enumerate(zip(points, weights, colors)):
        y = exp_func(np.array([x]))
        ax2.plot(x, y, 'o', color=color, markersize=10, 
                 label=f'x_{i+1}={x}, w_{i+1}={w}')
    
    # Compute and plot convex combination
    x_combo = np.sum([w * x for w, x in zip(weights, points)])
    y_combo = exp_func(np.array([x_combo]))
    y_weighted_avg = np.sum([w * exp_func(np.array([x])) for w, x in zip(weights, points)])
    
    ax2.plot(x_combo, y_combo, 'rs', markersize=12, 
             label=f'f(Σwᵢxᵢ)={y_combo:.2f}')
    ax2.plot(x_combo, y_weighted_avg, 'bs', markersize=12,
             label=f'Σwᵢf(xᵢ)={y_weighted_avg:.2f}')
    
    # Show the gap
    ax2.plot([x_combo, x_combo], [y_combo, y_weighted_avg], 
             'k-', linewidth=4, alpha=0.7, label='Jensen gap')
    
    ax2.set_xlim(-2, 3)
    ax2.set_ylim(0, 8)
    ax2.set_title('Multi-Point Jensen\'s Inequality')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Probability interpretation
    ax3 = plt.subplot(2, 3, 3)
    
    # Show Jensen for random variables
    log_func = JensenInequalityExamples.negative_log_function()
    
    # Random variable X with discrete distribution
    x_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    probabilities = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    
    # E[X] and E[f(X)]
    expected_x = np.sum(probabilities * x_values)
    expected_fx = np.sum(probabilities * [log_func(np.array([x])) for x in x_values])
    f_expected_x = log_func(np.array([expected_x]))
    
    # Plot function
    x_plot = np.linspace(0.2, 3, 200)
    y_plot = [log_func(np.array([x])) for x in x_plot]
    ax3.plot(x_plot, y_plot, 'blue', linewidth=3, label='f(x) = -log(x)')
    
    # Plot distribution points
    for x, p in zip(x_values, probabilities):
        y = log_func(np.array([x]))
        ax3.plot(x, y, 'ro', markersize=10*p*10, alpha=0.7)
        ax3.text(x, y+0.1, f'P={p}', ha='center', fontsize=8)
    
    # Plot Jensen's inequality
    ax3.plot(expected_x, f_expected_x, 'gs', markersize=12, 
             label=f'f(E[X])={f_expected_x:.2f}')
    ax3.axhline(expected_fx, color='red', linestyle='--', 
                label=f'E[f(X)]={expected_fx:.2f}')
    ax3.axvline(expected_x, color='green', linestyle='--', alpha=0.5)
    
    ax3.set_xlim(0.2, 3)
    ax3.set_ylim(-1, 3)
    ax3.set_title('Jensen\'s for Random Variables')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Convexity testing results
    ax4 = plt.subplot(2, 3, 4)
    
    # Test different functions for convexity
    functions_to_test = [
        JensenInequalityExamples.quadratic_function(np.array([[2.0]])),
        JensenInequalityExamples.exponential_function(1.0),
        JensenInequalityExamples.norm_function(2.0),
        JensenInequalityExamples.negative_log_function(),
        JensenInequalityExamples.non_convex_function()
    ]
    
    function_names = ['x²', 'exp(x)', '||x||₂', '-log(x)', 'x⁴-2x² (non-convex)']
    
    violation_rates = []
    for func in functions_to_test:
        if 'log' in func.name:
            # Positive domain for log function
            domain_bounds = (np.array([0.1]), np.array([3.0]))
        else:
            domain_bounds = (np.array([-2.0]), np.array([2.0]))
        
        results = func.test_convexity(n_tests=1000, domain_bounds=domain_bounds)
        violation_rates.append(results['violation_rate'])
    
    colors = ['green' if rate < 0.01 else 'red' for rate in violation_rates]
    bars = ax4.bar(range(len(function_names)), violation_rates, color=colors, alpha=0.7)
    
    ax4.set_ylim(0, max(max(violation_rates), 0.1))
    ax4.set_ylabel('Jensen Violation Rate')
    ax4.set_title('Convexity Test Results')
    ax4.set_xticks(range(len(function_names)))
    ax4.set_xticklabels(function_names, rotation=45, ha='right')
    
    # Add value labels
    for bar, rate in zip(bars, violation_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{rate:.3f}', ha='center', va='bottom')
    
    ax4.grid(True, alpha=0.3)
    
    # 5. 2D Jensen's inequality
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    
    # 2D quadratic function
    def func_2d(x):
        return x[0]**2 + x[1]**2 + x[0]*x[1]
    
    # Create surface
    x1_range = np.linspace(-2, 2, 30)
    x2_range = np.linspace(-2, 2, 30)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = X1**2 + X2**2 + X1*X2
    
    ax5.plot_surface(X1, X2, Z, alpha=0.6, cmap='viridis')
    
    # Show Jensen's inequality for triangle of points
    points_2d = [np.array([-1, -1]), np.array([1.5, 0]), np.array([0, 1.5])]
    weights_2d = np.array([0.3, 0.4, 0.3])
    
    # Plot individual points
    for i, point in enumerate(points_2d):
        z_val = func_2d(point)
        ax5.scatter([point[0]], [point[1]], [z_val], 
                   color='red', s=100, label=f'Point {i+1}' if i == 0 else '')
    
    # Convex combination
    combo_2d = np.sum([w * p for w, p in zip(weights_2d, points_2d)], axis=0)
    combo_z = func_2d(combo_2d)
    ax5.scatter([combo_2d[0]], [combo_2d[1]], [combo_z], 
               color='blue', s=150, label='f(convex combo)')
    
    # Weighted average of function values
    avg_z = np.sum([w * func_2d(p) for w, p in zip(weights_2d, points_2d)])
    ax5.scatter([combo_2d[0]], [combo_2d[1]], [avg_z],
               color='green', s=150, label='weighted avg')
    
    # Connect the gap
    ax5.plot([combo_2d[0], combo_2d[0]], [combo_2d[1], combo_2d[1]], 
             [combo_z, avg_z], 'k-', linewidth=4, alpha=0.8)
    
    ax5.set_xlabel('x₁')
    ax5.set_ylabel('x₂')
    ax5.set_zlabel('f(x₁,x₂)')
    ax5.set_title('2D Jensen\'s Inequality')
    ax5.legend()
    
    # 6. Applications in optimization
    ax6 = plt.subplot(2, 3, 6)
    
    # Show convergence guarantees for convex vs non-convex functions
    
    # Convex function: guaranteed global minimum
    x_conv = np.linspace(-3, 3, 200)
    y_conv = x_conv**2 + 0.1*x_conv  # Convex quadratic
    
    ax6.plot(x_conv, y_conv, 'green', linewidth=3, label='Convex: global minimum')
    ax6.plot(0, min(y_conv), 'go', markersize=10, label='Unique global min')
    
    # Non-convex function: multiple local minima
    y_nonconv = 0.3*x_conv**4 - 2*x_conv**2 + 1  # Non-convex quartic
    
    ax6.plot(x_conv, y_nonconv, 'red', linewidth=3, label='Non-convex: local minima')
    
    # Find local minima approximately
    local_mins_x = [-1.8, 1.8]  # Approximate locations
    for x_min in local_mins_x:
        y_min = 0.3*x_min**4 - 2*x_min**2 + 1
        ax6.plot(x_min, y_min, 'ro', markersize=8)
    
    ax6.set_xlim(-3, 3)
    ax6.set_ylim(-2, 3)
    ax6.set_title('Convex vs Non-Convex Optimization')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add annotations
    ax6.annotate('Jensen\'s inequality\nguarantees no local\nminima trapped',
                xy=(0, 0), xytext=(1, 2),
                arrowprops=dict(arrowstyle='->', color='green'),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    ax6.annotate('Multiple local minima\nviolate Jensen\'s inequality',
                xy=(-1.8, -1.3), xytext=(-1, 1),
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    plt.tight_layout()
    plt.show()
    
    # Numerical verification
    print("\n📊 NUMERICAL VERIFICATION")
    print("-" * 40)
    
    # Test Jensen's inequality for specific functions
    test_functions = [
        (JensenInequalityExamples.quadratic_function(), [-2, 2]),
        (JensenInequalityExamples.exponential_function(), [-2, 2]),
        (JensenInequalityExamples.negative_log_function(), [0.1, 3]),
        (JensenInequalityExamples.non_convex_function(), [-2, 2])
    ]
    
    for i, (func, domain) in enumerate(test_functions, 1):
        print(f"{i}. Function: {func.name}")
        
        domain_bounds = (np.array([domain[0]]), np.array([domain[1]]))
        results = func.test_convexity(n_tests=10000, domain_bounds=domain_bounds)
        
        print(f"   Tests performed: {results['total_tests']}")
        print(f"   Violations: {results['violations']}")
        print(f"   Violation rate: {results['violation_rate']:.6f}")
        print(f"   Max violation: {results['max_violation']:.6f}")
        print(f"   Is convex: {results['is_convex']}")
        
        # Specific Jensen test
        if 'log' in func.name:
            x1, x2 = np.array([0.5]), np.array([2.0])
        else:
            x1, x2 = np.array([-1.0]), np.array([1.5])
        
        weights = np.array([0.6, 0.4])
        inequality_holds, lhs, rhs = func.verify_jensen_inequality([x1, x2], weights)
        
        print(f"   Specific test: f({0.6}·{x1[0]:.1f} + {0.4}·{x2[0]:.1f}) = {lhs:.4f}")
        print(f"                 {0.6}·f({x1[0]:.1f}) + {0.4}·f({x2[0]:.1f}) = {rhs:.4f}")
        print(f"                 Jensen holds: {inequality_holds}")
        print()


def jensen_applications():
    """
    Showcase applications of Jensen's inequality in optimization and beyond.
    """
    print("\n🎯 JENSEN'S INEQUALITY APPLICATIONS")
    print("=" * 40)
    
    print("1. CONVEX OPTIMIZATION:")
    print("   - Any local minimum is global minimum")
    print("   - Efficient algorithms (gradient descent, interior point)")
    print("   - Duality theory and optimality conditions")
    
    print("\n2. MACHINE LEARNING:")
    print("   - Convex loss functions (logistic, SVM, least squares)")
    print("   - EM algorithm convergence guarantees")
    print("   - Regularization and sparsity")
    
    print("\n3. PROBABILITY AND STATISTICS:")
    print("   - Information theory (entropy inequalities)")
    print("   - Concentration inequalities")
    print("   - Method of moments bounds")
    
    print("\n4. ECONOMICS:")
    print("   - Risk aversion and utility theory")
    print("   - Portfolio theory (mean-variance analysis)")
    print("   - Market equilibrium existence")
    
    print("\n5. SIGNAL PROCESSING:")
    print("   - Convex relaxations of non-convex problems")
    print("   - Compressed sensing and sparse recovery")
    print("   - Filter design optimization")
    
    print("\n6. CONTROL THEORY:")
    print("   - Linear Matrix Inequalities (LMIs)")
    print("   - Robust control synthesis")
    print("   - Model predictive control")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_jensen_inequality()
    jensen_applications()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Jensen's inequality defines convex functions")
    print("- f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y) for convex f")
    print("- Extends to: f(E[X]) ≤ E[f(X)] for random variables")
    print("- Guarantees global optimality in convex optimization")
    print("- Foundation for efficient optimization algorithms")
    print("- Critical for machine learning theory and applications")
    print("\nJensen's inequality is the mathematical heart of modern optimization! 🚀")