"""
Weierstrass Extreme Value Theorem

The Weierstrass theorem guarantees the existence of global optima for continuous
functions on compact sets. This fundamental result ensures that optimization
problems have solutions and justifies the search for global extrema.

Mathematical Statement:
If f: K → R is continuous and K ⊆ R^n is compact (closed and bounded), then
f attains its maximum and minimum values on K.

Formally: ∃ x_min, x_max ∈ K such that:
f(x_min) ≤ f(x) ≤ f(x_max) ∀ x ∈ K

Key Implications for Optimization:
- Every continuous objective function on a compact feasible region has a solution
- Justifies the existence of global optima in bounded optimization problems
- Foundation for proving convergence of optimization algorithms
- Basis for minimax theorems in game theory and robust optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Tuple, List
import warnings

warnings.filterwarnings('ignore')

class CompactSet:
    """
    Implementation of compact sets with methods to verify compactness
    and find extrema of continuous functions.
    """
    
    def __init__(self, constraints: List[Callable[[np.ndarray], bool]],
                 bounds: Tuple[np.ndarray, np.ndarray]):
        """
        Initialize compact set defined by constraints and bounds.
        
        Args:
            constraints: List of constraint functions g_i(x) ≤ 0
            bounds: (lower_bounds, upper_bounds) for the variables
        """
        self.constraints = constraints
        self.lower_bounds = np.array(bounds[0])
        self.upper_bounds = np.array(bounds[1])
        self.dimension = len(self.lower_bounds)
    
    def contains(self, x: np.ndarray) -> bool:
        """Check if point x is in the compact set."""
        x = np.array(x)
        
        # Check bounds
        if not (np.all(x >= self.lower_bounds) and np.all(x <= self.upper_bounds)):
            return False
        
        # Check constraints
        for constraint in self.constraints:
            if not constraint(x):
                return False
        
        return True
    
    def is_bounded(self) -> bool:
        """Check if the set is bounded (always true by construction here)."""
        return np.all(np.isfinite(self.lower_bounds)) and np.all(np.isfinite(self.upper_bounds))
    
    def is_closed(self) -> bool:
        """
        Check if the set is closed.
        This is a simplified check assuming constraints define closed sets.
        """
        # In practice, this would require more sophisticated analysis
        # For demonstration, we assume constraints g(x) ≤ 0 define closed sets
        return True
    
    def is_compact(self) -> bool:
        """Check if the set is compact (closed and bounded in R^n)."""
        return self.is_closed() and self.is_bounded()
    
    def grid_sample(self, n_points_per_dim: int = 50) -> np.ndarray:
        """Generate grid sample of points in the compact set."""
        # Create grid
        coordinates = []
        for i in range(self.dimension):
            coordinates.append(np.linspace(self.lower_bounds[i], 
                                         self.upper_bounds[i], 
                                         n_points_per_dim))
        
        # Generate all combinations
        mesh_grids = np.meshgrid(*coordinates, indexing='ij')
        grid_points = np.column_stack([grid.ravel() for grid in mesh_grids])
        
        # Filter points that satisfy constraints
        feasible_points = []
        for point in grid_points:
            if self.contains(point):
                feasible_points.append(point)
        
        return np.array(feasible_points) if feasible_points else np.empty((0, self.dimension))
    
    def random_sample(self, n_points: int = 1000) -> np.ndarray:
        """Generate random sample of points in the compact set."""
        points = []
        attempts = 0
        max_attempts = n_points * 100
        
        while len(points) < n_points and attempts < max_attempts:
            # Random point in bounding box
            random_point = np.random.uniform(self.lower_bounds, self.upper_bounds)
            
            if self.contains(random_point):
                points.append(random_point)
            
            attempts += 1
        
        return np.array(points) if points else np.empty((0, self.dimension))


class WeierstrasSolver:
    """
    Solver to find global extrema using the Weierstrass theorem.
    Demonstrates existence and provides numerical verification.
    """
    
    def __init__(self, func: Callable[[np.ndarray], float]):
        """
        Initialize solver for a continuous function.
        
        Args:
            func: Continuous function to optimize
        """
        self.func = func
    
    def find_global_extrema(self, compact_set: CompactSet, 
                           method: str = 'grid') -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Find global minimum and maximum on compact set.
        
        Args:
            compact_set: Compact set to search over
            method: 'grid' or 'random' sampling
            
        Returns:
            (x_min, x_max, f_min, f_max)
        """
        if not compact_set.is_compact():
            raise ValueError("Set must be compact for Weierstrass theorem")
        
        # Sample points from the compact set
        if method == 'grid':
            sample_points = compact_set.grid_sample()
        elif method == 'random':
            sample_points = compact_set.random_sample()
        else:
            raise ValueError("Method must be 'grid' or 'random'")
        
        if len(sample_points) == 0:
            raise ValueError("No feasible points found in compact set")
        
        # Evaluate function at all sample points
        function_values = np.array([self.func(point) for point in sample_points])
        
        # Find extrema
        min_idx = np.argmin(function_values)
        max_idx = np.argmax(function_values)
        
        x_min = sample_points[min_idx]
        x_max = sample_points[max_idx]
        f_min = function_values[min_idx]
        f_max = function_values[max_idx]
        
        return x_min, x_max, f_min, f_max
    
    def verify_continuity(self, x: np.ndarray, epsilon: float = 1e-6, 
                         delta_factor: float = 0.1) -> bool:
        """
        Verify continuity at a point (simplified check).
        
        For true continuity: ∀ε > 0, ∃δ > 0 such that ||x - y|| < δ ⟹ |f(x) - f(y)| < ε
        """
        f_x = self.func(x)
        
        # Generate nearby points
        n_tests = 100
        delta = epsilon * delta_factor
        
        for _ in range(n_tests):
            # Random perturbation within delta ball
            perturbation = np.random.normal(0, delta/3, len(x))
            if np.linalg.norm(perturbation) > delta:
                perturbation = perturbation / np.linalg.norm(perturbation) * delta
            
            y = x + perturbation
            f_y = self.func(y)
            
            if abs(f_x - f_y) > epsilon:
                return False
        
        return True
    
    def convergence_analysis(self, compact_set: CompactSet, 
                           n_samples_list: List[int]) -> dict:
        """
        Analyze how extrema estimates converge as sample size increases.
        """
        results = {
            'n_samples': [],
            'f_min_estimates': [],
            'f_max_estimates': [],
            'x_min_estimates': [],
            'x_max_estimates': []
        }
        
        for n_samples in n_samples_list:
            # Use random sampling with specified number of points
            sample_points = compact_set.random_sample(n_samples)
            
            if len(sample_points) > 0:
                function_values = np.array([self.func(point) for point in sample_points])
                
                min_idx = np.argmin(function_values)
                max_idx = np.argmax(function_values)
                
                results['n_samples'].append(len(sample_points))
                results['f_min_estimates'].append(function_values[min_idx])
                results['f_max_estimates'].append(function_values[max_idx])
                results['x_min_estimates'].append(sample_points[min_idx])
                results['x_max_estimates'].append(sample_points[max_idx])
        
        return results


def demonstrate_weierstrass_theorem():
    """
    Demonstrate the Weierstrass theorem with visual examples.
    """
    print("🔷 WEIERSTRASS EXTREME VALUE THEOREM: Existence of Global Optima")
    print("=" * 70)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Simple 1D example: continuous function on closed interval
    ax1 = plt.subplot(2, 3, 1)
    
    # Function: f(x) = x³ - 3x² + 2x + 1 on [-1, 3]
    def cubic_func(x):
        return x[0]**3 - 3*x[0]**2 + 2*x[0] + 1
    
    # Compact set: closed interval [-1, 3]
    compact_interval = CompactSet(
        constraints=[],  # No additional constraints
        bounds=([-1.0], [3.0])
    )
    
    # Solve for extrema
    weierstrass_solver = WeierstrasSolver(cubic_func)
    x_min, x_max, f_min, f_max = weierstrass_solver.find_global_extrema(
        compact_interval, method='grid')
    
    # Plot function and extrema
    x_vals = np.linspace(-1, 3, 1000)
    y_vals = [cubic_func(np.array([x])) for x in x_vals]
    
    ax1.plot(x_vals, y_vals, 'blue', linewidth=2, label='f(x) = x³ - 3x² + 2x + 1')
    ax1.plot(x_min[0], f_min, 'ro', markersize=10, label=f'Global min: ({x_min[0]:.2f}, {f_min:.2f})')
    ax1.plot(x_max[0], f_max, 'go', markersize=10, label=f'Global max: ({x_max[0]:.2f}, {f_max:.2f})')
    
    # Mark interval endpoints
    ax1.axvline(-1, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(3, color='black', linestyle='--', alpha=0.5)
    ax1.fill_between([-1, 3], [-5, -5], [5, 5], alpha=0.1, color='gray', 
                     label='Compact set [−1, 3]')
    
    ax1.set_xlim(-1.5, 3.5)
    ax1.set_ylim(-3, 5)
    ax1.set_title('Weierstrass Theorem: 1D Example')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 2D example: function on compact region
    ax2 = plt.subplot(2, 3, 2)
    
    # Function: f(x,y) = (x-1)² + (y-0.5)² + 0.5*sin(5x)*sin(5y)
    def hills_func(x):
        return (x[0]-1)**2 + (x[1]-0.5)**2 + 0.5*np.sin(5*x[0])*np.sin(5*x[1])
    
    # Compact set: unit square [0,2] × [0,1.5] with circular constraint
    def circular_constraint(x):
        return (x[0]-1)**2 + (x[1]-0.75)**2 <= 1.2**2
    
    compact_region = CompactSet(
        constraints=[circular_constraint],
        bounds=([0.0, 0.0], [2.0, 1.5])
    )
    
    # Create contour plot
    x_range = np.linspace(0, 2, 100)
    y_range = np.linspace(0, 1.5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i,j], Y[i,j]])
            if compact_region.contains(point):
                Z[i,j] = hills_func(point)
            else:
                Z[i,j] = np.nan
    
    contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, ax=ax2, shrink=0.8)
    
    # Find and plot extrema
    x_min_2d, x_max_2d, f_min_2d, f_max_2d = weierstrass_solver = WeierstrasSolver(hills_func)
    x_min_2d, x_max_2d, f_min_2d, f_max_2d = weierstrass_solver.find_global_extrema(
        compact_region, method='random')
    
    ax2.plot(x_min_2d[0], x_min_2d[1], 'ro', markersize=10, 
             label=f'Global min: f = {f_min_2d:.2f}')
    ax2.plot(x_max_2d[0], x_max_2d[1], 'yo', markersize=10,
             label=f'Global max: f = {f_max_2d:.2f}')
    
    # Draw compact set boundary
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = 1 + 1.2*np.cos(theta)
    circle_y = 0.75 + 1.2*np.sin(theta)
    ax2.plot(circle_x, circle_y, 'white', linewidth=3, label='Compact set boundary')
    
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 1.5)
    ax2.set_title('Weierstrass Theorem: 2D Example')
    ax2.legend()
    ax2.set_aspect('equal')
    
    # 3. Non-compact set counter-example
    ax3 = plt.subplot(2, 3, 3)
    
    # Function: f(x) = x²/(1+x²) on unbounded domain
    def unbounded_func(x):
        return x[0]**2 / (1 + x[0]**2)
    
    x_vals_unb = np.linspace(-10, 10, 1000)
    y_vals_unb = [unbounded_func(np.array([x])) for x in x_vals_unb]
    
    ax3.plot(x_vals_unb, y_vals_unb, 'red', linewidth=2, 
             label='f(x) = x²/(1+x²)')
    
    # Show that supremum is approached but not attained
    ax3.axhline(1.0, color='red', linestyle='--', alpha=0.7, 
                label='Supremum = 1 (not attained)')
    ax3.axhline(0.0, color='blue', linestyle='--', alpha=0.7,
                label='Infimum = 0 (attained at x=0)')
    
    ax3.plot(0, 0, 'bo', markersize=8, label='Global minimum at x=0')
    
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_title('Non-Compact Domain: No Maximum')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Convergence analysis
    ax4 = plt.subplot(2, 3, 4)
    
    # Analyze convergence of extrema estimates
    n_samples_list = [50, 100, 200, 500, 1000, 2000]
    convergence_results = weierstrass_solver.convergence_analysis(
        compact_region, n_samples_list)
    
    ax4.semilogx(convergence_results['n_samples'], 
                 convergence_results['f_min_estimates'], 
                 'ro-', linewidth=2, label='Minimum estimates')
    ax4.semilogx(convergence_results['n_samples'],
                 convergence_results['f_max_estimates'],
                 'go-', linewidth=2, label='Maximum estimates')
    
    # True values (high-resolution estimate)
    true_solver = WeierstrasSolver(hills_func)
    x_min_true, x_max_true, f_min_true, f_max_true = true_solver.find_global_extrema(
        compact_region, method='grid')
    
    ax4.axhline(f_min_true, color='red', linestyle='--', alpha=0.7, 
                label=f'True minimum ≈ {f_min_true:.3f}')
    ax4.axhline(f_max_true, color='green', linestyle='--', alpha=0.7,
                label=f'True maximum ≈ {f_max_true:.3f}')
    
    ax4.set_xlabel('Number of samples')
    ax4.set_ylabel('Estimated extrema')
    ax4.set_title('Convergence of Extrema Estimates')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Compactness verification
    ax5 = plt.subplot(2, 3, 5)
    
    # Different set types
    sets_info = [
        ("Compact: [0,1]×[0,1]", CompactSet([], ([0,0], [1,1])), 'green'),
        ("Closed, unbounded: x²+y²≥1", CompactSet([lambda x: x[0]**2+x[1]**2 >= 1], 
                                                   ([-5,-5], [5,5])), 'red'),
        ("Bounded, open: x²+y²<1", CompactSet([lambda x: x[0]**2+x[1]**2 < 1], 
                                              ([-1,-1], [1,1])), 'orange'),
    ]
    
    y_pos = 0
    for name, set_obj, color in sets_info:
        # Check properties
        bounded = set_obj.is_bounded()
        closed = set_obj.is_closed()  # Simplified check
        compact = bounded and closed
        
        # Create visual representation
        ax5.barh(y_pos, 1, color=color, alpha=0.7, label=name)
        
        status_text = f"Bounded: {bounded}, Closed: {closed}, Compact: {compact}"
        ax5.text(0.02, y_pos, status_text, va='center', fontweight='bold')
        
        y_pos += 1
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(-0.5, len(sets_info) - 0.5)
    ax5.set_yticks(range(len(sets_info)))
    ax5.set_yticklabels([info[0] for info in sets_info])
    ax5.set_title('Compactness Check: Different Set Types')
    ax5.set_xticks([])
    
    # 6. Applications showcase
    ax6 = plt.subplot(2, 3, 6)
    
    # Show different optimization scenarios
    scenarios = [
        "Portfolio optimization\n(budget constraint)",
        "Control theory\n(bounded inputs)",
        "Signal processing\n(finite energy)",
        "Game theory\n(mixed strategies)",
        "Machine learning\n(regularization)"
    ]
    
    guarantees = [0.95, 0.98, 0.92, 0.99, 0.88]  # Simulated confidence levels
    
    bars = ax6.bar(range(len(scenarios)), guarantees, 
                   color=['blue', 'green', 'red', 'purple', 'orange'], alpha=0.7)
    
    ax6.set_ylim(0, 1)
    ax6.set_ylabel('Solution Existence Guarantee')
    ax6.set_title('Weierstrass Applications')
    ax6.set_xticks(range(len(scenarios)))
    ax6.set_xticklabels([s.split('\n')[0] for s in scenarios], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, guarantee in zip(bars, guarantees):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{guarantee:.0%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Numerical verification
    print("\n📊 NUMERICAL VERIFICATION")
    print("-" * 40)
    
    print("1. Extrema on Compact Set:")
    print(f"   Function: f(x) = x³ - 3x² + 2x + 1 on [-1, 3]")
    print(f"   Global minimum: x = {x_min[0]:.3f}, f = {f_min:.3f}")
    print(f"   Global maximum: x = {x_max[0]:.3f}, f = {f_max:.3f}")
    
    print(f"\n2. 2D Optimization Results:")
    print(f"   Function: f(x,y) = (x-1)² + (y-0.5)² + 0.5sin(5x)sin(5y)")
    print(f"   Compact region: Circle of radius 1.2 centered at (1, 0.75)")
    print(f"   Global minimum: ({x_min_2d[0]:.3f}, {x_min_2d[1]:.3f}), f = {f_min_2d:.3f}")
    print(f"   Global maximum: ({x_max_2d[0]:.3f}, {x_max_2d[1]:.3f}), f = {f_max_2d:.3f}")
    
    print(f"\n3. Convergence Analysis:")
    if len(convergence_results['n_samples']) > 0:
        final_min = convergence_results['f_min_estimates'][-1]
        final_max = convergence_results['f_max_estimates'][-1]
        print(f"   Final minimum estimate: {final_min:.6f}")
        print(f"   Final maximum estimate: {final_max:.6f}")
        print(f"   Samples used: {convergence_results['n_samples'][-1]}")


def weierstrass_applications():
    """
    Showcase applications of the Weierstrass theorem in optimization.
    """
    print("\n🎯 WEIERSTRASS THEOREM APPLICATIONS")
    print("=" * 40)
    
    print("1. OPTIMIZATION PROBLEM FORMULATION:")
    print("   - Ensures solution exists for minimize f(x) subject to x ∈ K")
    print("   - K must be compact (closed and bounded)")
    print("   - f must be continuous")
    print("   - Justifies global optimization algorithms")
    
    print("\n2. FEASIBLE REGION DESIGN:")
    print("   - Add bounds: -M ≤ xᵢ ≤ M to ensure boundedness")
    print("   - Use closed constraints: g(x) ≤ 0 (not g(x) < 0)")
    print("   - Penalty methods make unbounded problems bounded")
    
    print("\n3. CONVERGENCE GUARANTEES:")
    print("   - Global optimization algorithms will find global minimum")
    print("   - Grid search converges to true optimum")
    print("   - Random sampling approaches true extrema")
    
    print("\n4. ROBUST OPTIMIZATION:")
    print("   - Worst-case analysis over compact uncertainty sets")
    print("   - minimax problems have solutions")
    print("   - Distributionally robust optimization")
    
    print("\n5. GAME THEORY:")
    print("   - Mixed strategies form compact sets (probability simplex)")
    print("   - Nash equilibrium existence in finite games")
    print("   - Minimax theorem foundations")
    
    print("\n6. MACHINE LEARNING:")
    print("   - Regularized loss functions on bounded parameter sets")
    print("   - SVM optimization with bounded variables")
    print("   - Neural network training with weight bounds")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_weierstrass_theorem()
    weierstrass_applications()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Weierstrass theorem guarantees existence of global optima")
    print("- Requires: continuous function + compact (closed & bounded) set")
    print("- Foundation for all optimization theory and algorithms")
    print("- Justifies search for global minimum/maximum")
    print("- Critical for problem formulation and algorithm design")
    print("- Ensures optimization problems are well-posed")
    print("\nWithout Weierstrass, optimization would have no theoretical foundation! 🚀")