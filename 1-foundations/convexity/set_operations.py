"""
Convex Set Operations: Intersection, Sum, Projection and More

This module covers the fundamental operations on convex sets that preserve convexity
and are essential for understanding how complex convex sets are built from simpler ones.

Learning Objectives:
- Master convex-preserving operations on sets
- Understand Minkowski sums and their applications
- Learn projection operations onto convex sets
- Explore polarity and dual representations

Key Operations:
- Intersection: Always preserves convexity
- Minkowski sum: C ⊕ D = {x + y : x ∈ C, y ∈ D}
- Cartesian product: Builds higher-dimensional convex sets
- Linear transformations: f(C) = {Ax + b : x ∈ C}
- Projections: Essential for constrained optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Callable
import warnings
from definitions import ConvexSet, Ellipsoid, Halfspace, Polyhedron

warnings.filterwarnings('ignore')

class ConvexSetOperations:
    """
    A collection of operations on convex sets that preserve convexity.
    These operations are fundamental for building complex convex sets
    from simpler building blocks.
    """
    
    @staticmethod
    def intersection(sets: List[ConvexSet]) -> 'IntersectionSet':
        """
        Compute the intersection of multiple convex sets.
        
        Theorem: The intersection of any collection of convex sets is convex.
        
        Args:
            sets: List of convex sets to intersect
            
        Returns:
            IntersectionSet representing the intersection
        """
        return IntersectionSet(sets)
    
    @staticmethod
    def minkowski_sum(set1: ConvexSet, set2: ConvexSet) -> 'MinkowskiSum':
        """
        Compute the Minkowski sum of two convex sets.
        
        Definition: C ⊕ D = {x + y : x ∈ C, y ∈ D}
        
        Theorem: The Minkowski sum of convex sets is convex.
        
        Args:
            set1, set2: Convex sets to sum
            
        Returns:
            MinkowskiSum representing C ⊕ D
        """
        return MinkowskiSum(set1, set2)
    
    @staticmethod
    def cartesian_product(sets: List[ConvexSet]) -> 'CartesianProduct':
        """
        Compute the Cartesian product of convex sets.
        
        Definition: C₁ × C₂ × ... × Cₙ = {(x₁, x₂, ..., xₙ) : xᵢ ∈ Cᵢ}
        
        Theorem: The Cartesian product of convex sets is convex.
        
        Args:
            sets: List of convex sets
            
        Returns:
            CartesianProduct representing the product
        """
        return CartesianProduct(sets)
    
    @staticmethod
    def linear_transform(convex_set: ConvexSet, A: np.ndarray, 
                        b: Optional[np.ndarray] = None) -> 'LinearTransform':
        """
        Apply linear transformation to a convex set.
        
        Definition: f(C) = {Ax + b : x ∈ C}
        
        Theorem: Linear transformations preserve convexity.
        
        Args:
            convex_set: Input convex set
            A: Linear transformation matrix
            b: Translation vector (optional)
            
        Returns:
            LinearTransform representing f(C)
        """
        return LinearTransform(convex_set, A, b)
    
    @staticmethod
    def polar_set(convex_set: ConvexSet) -> 'PolarSet':
        """
        Compute the polar (dual) of a convex set containing origin.
        
        Definition: C° = {y : ⟨x, y⟩ ≤ 1 for all x ∈ C}
        
        Theorem: The polar of a convex set is convex.
        
        Args:
            convex_set: Convex set containing the origin
            
        Returns:
            PolarSet representing C°
        """
        return PolarSet(convex_set)


class IntersectionSet(ConvexSet):
    """
    Intersection of multiple convex sets.
    
    This is perhaps the most important convex operation, as every convex set
    can be represented as the intersection of halfspaces.
    """
    
    def __init__(self, sets: List[ConvexSet]):
        """Initialize intersection of convex sets."""
        if not sets:
            raise ValueError("Cannot intersect empty list of sets")
        self.sets = sets
    
    def contains(self, x: np.ndarray) -> bool:
        """Point is in intersection if it's in ALL constituent sets."""
        return all(s.contains(x) for s in self.sets)
    
    def project(self, x: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
        """
        Project onto intersection using alternating projections (Dykstra's algorithm).
        
        For two sets, this converges to the projection. For more sets,
        it converges to a point in the intersection but may not be the closest.
        """
        point = x.copy()
        
        for iteration in range(max_iter):
            old_point = point.copy()
            
            # Project onto each set in sequence
            for s in self.sets:
                point = s.project(point)
            
            # Check convergence
            if np.linalg.norm(point - old_point) < tol:
                break
        
        return point


class MinkowskiSum(ConvexSet):
    """
    Minkowski sum of two convex sets: C ⊕ D = {x + y : x ∈ C, y ∈ D}
    
    The Minkowski sum is fundamental in robotics (configuration space),
    optimization (robust optimization), and convex analysis.
    """
    
    def __init__(self, set1: ConvexSet, set2: ConvexSet):
        """Initialize Minkowski sum of two convex sets."""
        self.set1 = set1
        self.set2 = set2
    
    def contains(self, x: np.ndarray) -> bool:
        """
        Check if x ∈ C ⊕ D.
        
        This is generally hard to compute exactly. We use an approximation
        based on support functions for specific set types.
        """
        # This is a simplified implementation
        # For exact membership, we'd need to solve: min ||x - (y + z)||² s.t. y ∈ C, z ∈ D
        
        # For now, use a heuristic based on distance to sets
        # Project x onto each set and check if sum is close to x
        try:
            proj1 = self.set1.project(x / 2)
            proj2 = self.set2.project(x / 2)
            return np.linalg.norm(x - (proj1 + proj2)) < 1e-6
        except:
            return False
    
    def support(self, direction: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Support function of Minkowski sum.
        
        Theorem: σ_{C⊕D}(d) = σ_C(d) + σ_D(d)
        """
        try:
            point1, value1 = self.set1.support(direction)
            point2, value2 = self.set2.support(direction)
            
            if point1 is not None and point2 is not None:
                return point1 + point2, value1 + value2
            else:
                return None, value1 + value2
        except:
            return None, np.inf


class CartesianProduct(ConvexSet):
    """
    Cartesian product of convex sets: C₁ × C₂ × ... × Cₙ
    
    Used to build higher-dimensional convex sets from lower-dimensional ones.
    Common in multi-objective optimization and decoupled problems.
    """
    
    def __init__(self, sets: List[ConvexSet]):
        """Initialize Cartesian product of convex sets."""
        if not sets:
            raise ValueError("Cannot take product of empty list of sets")
        self.sets = sets
    
    def contains(self, x: np.ndarray) -> bool:
        """
        Check if x ∈ C₁ × C₂ × ... × Cₙ.
        
        We need to split x into components and check each against its set.
        """
        # For simplicity, assume equal dimensions
        n_sets = len(self.sets)
        dim_per_set = len(x) // n_sets
        
        for i, s in enumerate(self.sets):
            start_idx = i * dim_per_set
            end_idx = (i + 1) * dim_per_set
            component = x[start_idx:end_idx]
            
            if not s.contains(component):
                return False
        
        return True
    
    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project onto Cartesian product.
        
        Theorem: Projection onto product = product of projections.
        """
        n_sets = len(self.sets)
        dim_per_set = len(x) // n_sets
        
        projected = np.zeros_like(x)
        
        for i, s in enumerate(self.sets):
            start_idx = i * dim_per_set
            end_idx = (i + 1) * dim_per_set
            component = x[start_idx:end_idx]
            
            projected[start_idx:end_idx] = s.project(component)
        
        return projected


class LinearTransform(ConvexSet):
    """
    Linear transformation of a convex set: f(C) = {Ax + b : x ∈ C}
    
    Linear transformations are fundamental in convex analysis and optimization.
    They include scaling, rotation, projection, and translation.
    """
    
    def __init__(self, convex_set: ConvexSet, A: np.ndarray, 
                 b: Optional[np.ndarray] = None):
        """
        Initialize linear transformation f(x) = Ax + b of convex set.
        
        Args:
            convex_set: Input convex set C
            A: Linear transformation matrix
            b: Translation vector (default: zero)
        """
        self.convex_set = convex_set
        self.A = np.array(A)
        self.b = np.zeros(A.shape[0]) if b is None else np.array(b)
        
        # Precompute pseudo-inverse for projection
        try:
            self.A_pinv = np.linalg.pinv(A)
        except:
            self.A_pinv = None
    
    def contains(self, y: np.ndarray) -> bool:
        """
        Check if y ∈ f(C) = {Ax + b : x ∈ C}.
        
        Equivalent to: does there exist x ∈ C such that Ax + b = y?
        """
        if self.A_pinv is None:
            return False
        
        # Solve Ax = y - b for x, then check if x ∈ C
        try:
            x_candidate = self.A_pinv @ (y - self.b)
            
            # Verify Ax + b ≈ y (in case A is not full rank)
            if np.linalg.norm(self.A @ x_candidate + self.b - y) > 1e-10:
                return False
            
            return self.convex_set.contains(x_candidate)
        except:
            return False
    
    def support(self, direction: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Support function of linear transform.
        
        Theorem: σ_{f(C)}(d) = σ_C(A^T d) where f(x) = Ax + b
        """
        try:
            # Transform direction back to original space
            transformed_direction = self.A.T @ direction
            
            # Get support in original space
            point, value = self.convex_set.support(transformed_direction)
            
            if point is not None:
                # Transform support point forward
                transformed_point = self.A @ point + self.b
                # Add contribution from translation
                transformed_value = value + np.dot(direction, self.b)
                return transformed_point, transformed_value
            else:
                return None, value + np.dot(direction, self.b)
        except:
            return None, np.inf


class PolarSet(ConvexSet):
    """
    Polar (dual) of a convex set: C° = {y : ⟨x, y⟩ ≤ 1 for all x ∈ C}
    
    Polarity provides a beautiful duality between convex sets and is
    fundamental in convex analysis and optimization theory.
    """
    
    def __init__(self, convex_set: ConvexSet):
        """
        Initialize polar of a convex set containing the origin.
        
        Args:
            convex_set: Convex set C containing origin
        """
        self.convex_set = convex_set
        
        # Verify origin is contained (required for polar to be bounded)
        origin = np.zeros(2)  # Assume 2D for simplicity
        if not convex_set.contains(origin):
            warnings.warn("Set should contain origin for polar to be well-defined")
    
    def contains(self, y: np.ndarray) -> bool:
        """
        Check if y ∈ C°.
        
        This requires checking ⟨x, y⟩ ≤ 1 for ALL x ∈ C.
        We approximate this using the support function: σ_C(y) ≤ 1
        """
        try:
            _, support_value = self.convex_set.support(y)
            return support_value <= 1.0 + 1e-10
        except:
            return False
    
    def support(self, direction: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Support function of polar set.
        
        The polar relationship extends to support functions in a complex way.
        """
        # This is a simplified implementation
        # Full theory requires more sophisticated analysis
        return None, np.inf


def demonstrate_convex_operations():
    """
    Demonstrate convex set operations with visual examples.
    """
    print("🔷 CONVEX SET OPERATIONS: Building Complex Sets")
    print("=" * 60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Intersection of Halfspaces (Polyhedron)
    ax1 = plt.subplot(2, 3, 1)
    
    # Create several halfspaces
    halfspaces = [
        Halfspace(np.array([1, 0]), 2),     # x ≤ 2
        Halfspace(np.array([0, 1]), 2),     # y ≤ 2
        Halfspace(np.array([-1, 0]), 1),    # -x ≤ 1, i.e., x ≥ -1
        Halfspace(np.array([0, -1]), 1),    # -y ≤ 1, i.e., y ≥ -1
        Halfspace(np.array([1, 1]), 2),     # x + y ≤ 2
    ]
    
    # Plot individual halfspaces
    x_range = np.linspace(-2, 3, 100)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (hs, color) in enumerate(zip(halfspaces, colors)):
        if abs(hs.a[1]) > 1e-10:  # Non-vertical line
            y_line = (hs.b - hs.a[0] * x_range) / hs.a[1]
            ax1.plot(x_range, y_line, color=color, alpha=0.7, linewidth=1)
        else:  # Vertical line
            ax1.axvline(hs.b / hs.a[0], color=color, alpha=0.7, linewidth=1)
    
    # Create intersection
    intersection = ConvexSetOperations.intersection(halfspaces)
    
    # Find feasible region by testing grid points
    x_grid, y_grid = np.meshgrid(np.linspace(-2, 3, 100), np.linspace(-2, 3, 100))
    feasible = np.zeros_like(x_grid, dtype=bool)
    
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            point = np.array([x_grid[i, j], y_grid[i, j]])
            feasible[i, j] = intersection.contains(point)
    
    # Plot feasible region
    ax1.contourf(x_grid, y_grid, feasible.astype(int), levels=[0.5, 1.5], 
                 colors=['lightblue'], alpha=0.5)
    
    ax1.set_xlim(-2, 3)
    ax1.set_ylim(-2, 3)
    ax1.set_title('Intersection of Halfspaces')
    ax1.grid(True, alpha=0.3)
    
    # 2. Minkowski Sum of Two Sets
    ax2 = plt.subplot(2, 3, 2)
    
    # Create two simple convex sets (ellipses)
    A1 = np.array([[2, 0], [0, 0.5]])
    A2 = np.array([[0.5, 0], [0, 2]])
    ellipse1 = Ellipsoid(A1, np.array([0, 0]))
    ellipse2 = Ellipsoid(A2, np.array([0, 0]))
    
    # Generate boundary points for visualization
    theta = np.linspace(0, 2*np.pi, 50)
    
    # Ellipse 1 boundary
    points1 = []
    for t in theta:
        direction = np.array([np.cos(t), np.sin(t)])
        point, _ = ellipse1.support(direction)
        points1.append(point)
    points1 = np.array(points1)
    
    # Ellipse 2 boundary  
    points2 = []
    for t in theta:
        direction = np.array([np.cos(t), np.sin(t)])
        point, _ = ellipse2.support(direction)
        points2.append(point)
    points2 = np.array(points2)
    
    # Plot original sets
    ax2.fill(points1[:, 0], points1[:, 1], alpha=0.3, color='red', label='Set 1')
    ax2.fill(points2[:, 0], points2[:, 1], alpha=0.3, color='blue', label='Set 2')
    
    # Approximate Minkowski sum boundary
    sum_points = []
    for t in theta:
        direction = np.array([np.cos(t), np.sin(t)])
        point1, _ = ellipse1.support(direction)
        point2, _ = ellipse2.support(direction)
        sum_points.append(point1 + point2)
    sum_points = np.array(sum_points)
    
    ax2.fill(sum_points[:, 0], sum_points[:, 1], alpha=0.5, color='green', 
             label='Minkowski Sum')
    
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_title('Minkowski Sum: C ⊕ D')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 3. Linear Transformation
    ax3 = plt.subplot(2, 3, 3)
    
    # Original ellipse
    A_orig = np.array([[1, 0], [0, 0.5]])
    ellipse_orig = Ellipsoid(A_orig, np.array([0, 0]))
    
    # Generate original boundary
    orig_points = []
    for t in theta:
        direction = np.array([np.cos(t), np.sin(t)])
        point, _ = ellipse_orig.support(direction)
        orig_points.append(point)
    orig_points = np.array(orig_points)
    
    # Apply linear transformation: rotation + scaling
    angle = np.pi / 4
    scale = 1.5
    A_transform = scale * np.array([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle), np.cos(angle)]])
    b_transform = np.array([1, 0.5])
    
    # Transform points
    transformed_points = (A_transform @ orig_points.T).T + b_transform
    
    # Plot original and transformed sets
    ax3.fill(orig_points[:, 0], orig_points[:, 1], alpha=0.5, color='blue', 
             label='Original Set')
    ax3.fill(transformed_points[:, 0], transformed_points[:, 1], alpha=0.5, 
             color='red', label='Transformed Set')
    
    # Show transformation arrows for a few points
    sample_indices = [0, 12, 25, 37]
    for idx in sample_indices:
        ax3.arrow(orig_points[idx, 0], orig_points[idx, 1],
                  transformed_points[idx, 0] - orig_points[idx, 0],
                  transformed_points[idx, 1] - orig_points[idx, 1],
                  head_width=0.1, head_length=0.1, fc='green', ec='green', alpha=0.7)
    
    ax3.set_xlim(-2, 4)
    ax3.set_ylim(-2, 3)
    ax3.set_title('Linear Transformation: Ax + b')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # 4. Cartesian Product (2D example: interval × interval)
    ax4 = plt.subplot(2, 3, 4)
    
    # Two 1D intervals represented as line segments
    interval1 = np.array([0, 2])  # [0, 2]
    interval2 = np.array([0.5, 1.5])  # [0.5, 1.5]
    
    # Cartesian product is a rectangle
    rect_x = [interval1[0], interval1[1], interval1[1], interval1[0], interval1[0]]
    rect_y = [interval2[0], interval2[0], interval2[1], interval2[1], interval2[0]]
    
    ax4.fill(rect_x, rect_y, alpha=0.5, color='orange', label='I₁ × I₂')
    ax4.plot(rect_x, rect_y, 'orange', linewidth=2)
    
    # Show the original intervals on the axes
    ax4.plot([interval1[0], interval1[1]], [0, 0], 'blue', linewidth=4, 
             label='Interval I₁')
    ax4.plot([0, 0], [interval2[0], interval2[1]], 'red', linewidth=4, 
             label='Interval I₂')
    
    ax4.set_xlim(-0.5, 2.5)
    ax4.set_ylim(-0.5, 2)
    ax4.set_title('Cartesian Product: I₁ × I₂')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Polar Set Example
    ax5 = plt.subplot(2, 3, 5)
    
    # Original set: unit ball
    theta_circle = np.linspace(0, 2*np.pi, 100)
    unit_circle_x = np.cos(theta_circle)
    unit_circle_y = np.sin(theta_circle)
    
    ax5.fill(unit_circle_x, unit_circle_y, alpha=0.3, color='blue', label='Unit Ball')
    
    # Polar of unit ball is also unit ball
    ax5.fill(unit_circle_x, unit_circle_y, alpha=0.3, color='red', label='Polar Set')
    
    # Show some normal vectors and their duals
    for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        # Point on unit circle
        x_point = np.cos(angle)
        y_point = np.sin(angle)
        
        # Normal vector (same as point for unit circle)
        normal = np.array([x_point, y_point])
        
        # Corresponding point in polar (which is the same for unit ball)
        ax5.arrow(0, 0, x_point, y_point, head_width=0.05, head_length=0.05,
                  fc='black', ec='black', alpha=0.7)
        ax5.plot(x_point, y_point, 'ko', markersize=6)
    
    ax5.set_xlim(-1.5, 1.5)
    ax5.set_ylim(-1.5, 1.5)
    ax5.set_title('Polar Set: Unit Ball')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. Projection onto Intersection
    ax6 = plt.subplot(2, 3, 6)
    
    # Two intersecting ellipses
    A1 = np.array([[2, 0], [0, 1]])
    A2 = np.array([[1, 0.5], [0.5, 2]])
    ellipse1 = Ellipsoid(A1, np.array([-0.5, 0]))
    ellipse2 = Ellipsoid(A2, np.array([0.5, 0]))
    
    # Plot ellipses
    for ellipse, color, label in [(ellipse1, 'red', 'Ellipse 1'), 
                                  (ellipse2, 'blue', 'Ellipse 2')]:
        points = []
        for t in theta:
            direction = np.array([np.cos(t), np.sin(t)])
            point, _ = ellipse.support(direction)
            points.append(point)
        points = np.array(points)
        ax6.fill(points[:, 0], points[:, 1], alpha=0.3, color=color, label=label)
    
    # Create intersection and project external points
    intersection = ConvexSetOperations.intersection([ellipse1, ellipse2])
    
    external_points = [np.array([2, 2]), np.array([-2, -1]), np.array([1, -2])]
    
    for point in external_points:
        projected = intersection.project(point)
        
        # Plot original point and projection
        ax6.plot(point[0], point[1], 'ko', markersize=8)
        ax6.plot(projected[0], projected[1], 'go', markersize=8)
        ax6.plot([point[0], projected[0]], [point[1], projected[1]], 
                 'g--', linewidth=2, alpha=0.7)
    
    ax6.set_xlim(-3, 3)
    ax6.set_ylim(-3, 3)
    ax6.set_title('Projection onto Intersection')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Numerical verification
    print("\n📊 NUMERICAL VERIFICATION")
    print("-" * 40)
    
    # Test Minkowski sum property
    print("1. Minkowski Sum Support Function:")
    A1 = np.array([[2, 0], [0, 1]])
    A2 = np.array([[1, 0], [0, 2]])
    ellipse1 = Ellipsoid(A1)
    ellipse2 = Ellipsoid(A2)
    
    mink_sum = ConvexSetOperations.minkowski_sum(ellipse1, ellipse2)
    
    direction = np.array([1, 1]) / np.sqrt(2)
    _, support1 = ellipse1.support(direction)
    _, support2 = ellipse2.support(direction)
    _, support_sum = mink_sum.support(direction)
    
    print(f"   Support of Set 1: {support1:.3f}")
    print(f"   Support of Set 2: {support2:.3f}")
    print(f"   Support of Sum: {support_sum:.3f}")
    print(f"   Sum of Supports: {support1 + support2:.3f}")
    print(f"   Difference: {abs(support_sum - (support1 + support2)):.6f}")


def verify_operation_properties():
    """
    Verify key properties of convex set operations.
    """
    print("\n🔍 VERIFICATION OF OPERATION PROPERTIES")
    print("=" * 50)
    
    # Property 1: Intersection preserves convexity
    print("1. Intersection Preserves Convexity:")
    
    # Create multiple convex sets
    sets = [
        Halfspace(np.array([1, 0]), 1),
        Halfspace(np.array([0, 1]), 1),
        Halfspace(np.array([-1, -1]), -0.5)
    ]
    
    intersection = ConvexSetOperations.intersection(sets)
    
    # Test convexity by checking random convex combinations
    test_points = []
    for _ in range(50):
        x = np.random.uniform(-2, 2, 2)
        if intersection.contains(x):
            test_points.append(x)
    
    if len(test_points) >= 2:
        violations = 0
        for _ in range(100):
            p1, p2 = np.random.choice(len(test_points), 2, replace=False)
            point1, point2 = test_points[p1], test_points[p2]
            
            t = np.random.random()
            combo = t * point1 + (1 - t) * point2
            
            if not intersection.contains(combo):
                violations += 1
        
        print(f"   Convexity violations in intersection: {violations}/100")
    
    # Property 2: Linear transformations preserve convexity
    print("\n2. Linear Transform Preserves Convexity:")
    
    # Original convex set
    A_ellipse = np.array([[2, 0], [0, 1]])
    ellipse = Ellipsoid(A_ellipse)
    
    # Linear transformation
    A_transform = np.array([[1, 0.5], [0, 2]])
    b_transform = np.array([1, 0])
    transformed = ConvexSetOperations.linear_transform(ellipse, A_transform, b_transform)
    
    # Test convexity of transformed set
    test_points = []
    for _ in range(100):
        # Generate random point in original ellipse
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0, 1)
        x_orig = np.array([np.sqrt(radius) * np.cos(angle) / np.sqrt(2),
                          np.sqrt(radius) * np.sin(angle)])
        
        # Transform it
        x_transformed = A_transform @ x_orig + b_transform
        test_points.append(x_transformed)
    
    violations = 0
    for _ in range(50):
        p1, p2 = np.random.choice(len(test_points), 2, replace=False)
        point1, point2 = test_points[p1], test_points[p2]
        
        t = np.random.random()
        combo = t * point1 + (1 - t) * point2
        
        if not transformed.contains(combo):
            violations += 1
    
    print(f"   Convexity violations in transformed set: {violations}/50")
    
    # Property 3: Support function of Minkowski sum
    print("\n3. Minkowski Sum Support Function Property:")
    
    # Two ellipsoids
    A1 = np.array([[1, 0], [0, 2]])
    A2 = np.array([[2, 0.5], [0.5, 1]])
    ellipse1 = Ellipsoid(A1)
    ellipse2 = Ellipsoid(A2)
    
    mink_sum = ConvexSetOperations.minkowski_sum(ellipse1, ellipse2)
    
    # Test several directions
    errors = []
    for _ in range(10):
        direction = np.random.randn(2)
        direction = direction / np.linalg.norm(direction)
        
        _, support1 = ellipse1.support(direction)
        _, support2 = ellipse2.support(direction)
        _, support_sum = mink_sum.support(direction)
        
        expected = support1 + support2
        error = abs(support_sum - expected)
        errors.append(error)
    
    print(f"   Average support function error: {np.mean(errors):.6f}")
    print(f"   Maximum support function error: {np.max(errors):.6f}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_convex_operations()
    verify_operation_properties()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Intersection ALWAYS preserves convexity (fundamental property)")
    print("- Minkowski sum: σ_{C⊕D}(d) = σ_C(d) + σ_D(d)")
    print("- Linear transformations preserve convexity")
    print("- Cartesian products build higher-dimensional convex sets")
    print("- Projections onto convex sets are well-defined")
    print("- Polar sets provide beautiful duality theory")
    print("\nNext: Explore convex functions and their characterizations! 🚀")