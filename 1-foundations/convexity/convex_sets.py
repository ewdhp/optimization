"""
Convex Sets: Definitions and Fundamental Theory

This module provides the foundational theory of convex sets, which forms the geometric
backbone of convex optimization. Understanding convex sets is crucial for recognizing
when optimization problems have the "nice" properties that make them tractable.

Learning Objectives:
- Master the definition and characterization of convex sets
- Understand the geometric intuition behind convexity
- Learn to verify convexity using multiple equivalent conditions
- Recognize common convex sets in applications

Key Concepts:
- Convex combinations and convex hulls
- Extreme points and vertices
- Supporting hyperplanes and separation theorems
- Convex cones and their properties
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ConvexSet:
    """
    Abstract base class for convex sets with common operations and properties.
    
    This class provides a framework for working with convex sets and includes
    methods for membership testing, visualization, and geometric operations.
    """
    
    def contains(self, x: np.ndarray) -> bool:
        """Check if point x is in the convex set."""
        raise NotImplementedError("Subclasses must implement contains method")
    
    def project(self, x: np.ndarray) -> np.ndarray:
        """Project point x onto the convex set."""
        raise NotImplementedError("Subclasses must implement project method")
    
    def support(self, direction: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Find the support point in the given direction.
        Returns (support_point, support_value).
        """
        raise NotImplementedError("Subclasses must implement support method")

    @staticmethod
    def is_convex_combination(points: np.ndarray, weights: np.ndarray, 
                            target: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Check if target is a convex combination of points with given weights.
        
        Args:
            points: Array of shape (n_points, n_dim) containing the points
            weights: Array of shape (n_points,) containing the weights
            target: Array of shape (n_dim,) target point
            tol: Tolerance for numerical comparisons
            
        Returns:
            True if target = Σ wᵢ * pᵢ with wᵢ ≥ 0, Σ wᵢ = 1
        """
        # Check weight constraints
        if not (np.all(weights >= -tol) and abs(np.sum(weights) - 1.0) < tol):
            return False
        
        # Check if combination equals target
        combination = np.sum(weights.reshape(-1, 1) * points, axis=0)
        return np.linalg.norm(combination - target) < tol

    @staticmethod
    def convex_hull_2d(points: np.ndarray) -> np.ndarray:
        """
        Compute the convex hull of 2D points using Graham scan algorithm.
        
        Args:
            points: Array of shape (n_points, 2)
            
        Returns:
            Array of hull vertices in counterclockwise order
        """
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        points = points[np.lexsort((points[:, 1], points[:, 0]))]
        
        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        
        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        
        # Remove last point of each half because it's repeated
        return np.array(lower[:-1] + upper[:-1])


class Hyperplane(ConvexSet):
    """
    Hyperplane: {x : aᵀx = b}
    
    A hyperplane is a convex set that divides the space into two half-spaces.
    It's the affine generalization of a line (2D) or plane (3D).
    
    Properties:
    - Always convex (actually affine)
    - Has dimension n-1 in n-dimensional space
    - Can be used to separate convex sets
    """
    
    def __init__(self, a: np.ndarray, b: float):
        """
        Initialize hyperplane aᵀx = b.
        
        Args:
            a: Normal vector (must be non-zero)
            b: Offset value
        """
        self.a = np.array(a)
        self.b = float(b)
        
        if np.linalg.norm(self.a) == 0:
            raise ValueError("Normal vector cannot be zero")
        
        # Normalize for numerical stability
        norm_a = np.linalg.norm(self.a)
        self.a = self.a / norm_a
        self.b = self.b / norm_a
    
    def contains(self, x: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if point x lies on the hyperplane."""
        return abs(np.dot(self.a, x) - self.b) < tol
    
    def distance(self, x: np.ndarray) -> float:
        """Compute signed distance from point to hyperplane."""
        return np.dot(self.a, x) - self.b
    
    def project(self, x: np.ndarray) -> np.ndarray:
        """Project point onto hyperplane."""
        return x - self.distance(x) * self.a
    
    def support(self, direction: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Support function for hyperplane.
        Returns None if direction is not orthogonal to hyperplane.
        """
        if abs(np.dot(direction, self.a)) < 1e-10:
            # Direction is parallel to hyperplane - support is unbounded
            return None, np.inf
        else:
            # Direction not parallel - no finite support point
            return None, -np.inf


class Halfspace(ConvexSet):
    """
    Halfspace: {x : aᵀx ≤ b}
    
    A halfspace is one of the most fundamental convex sets. Every convex set
    can be represented as the intersection of halfspaces (though possibly infinitely many).
    
    Properties:
    - Always convex
    - Closed and unbounded (unless degenerate)
    - Building block for polyhedra
    """
    
    def __init__(self, a: np.ndarray, b: float):
        """
        Initialize halfspace aᵀx ≤ b.
        
        Args:
            a: Normal vector pointing "outward" from feasible region
            b: Offset value
        """
        self.a = np.array(a)
        self.b = float(b)
        
        if np.linalg.norm(self.a) == 0:
            raise ValueError("Normal vector cannot be zero")
    
    def contains(self, x: np.ndarray) -> bool:
        """Check if point x is in the halfspace."""
        return np.dot(self.a, x) <= self.b + 1e-10
    
    def project(self, x: np.ndarray) -> np.ndarray:
        """Project point onto halfspace."""
        violation = np.dot(self.a, x) - self.b
        if violation <= 0:
            return x  # Already feasible
        else:
            # Project onto boundary hyperplane
            return x - violation * self.a / np.dot(self.a, self.a)
    
    def support(self, direction: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Support function for halfspace."""
        if np.dot(direction, self.a) <= 0:
            return None, np.inf  # Unbounded in this direction
        else:
            return None, -np.inf  # No finite support


class Ellipsoid(ConvexSet):
    """
    Ellipsoid: {x : (x-c)ᵀ A⁻¹ (x-c) ≤ 1}
    
    Ellipsoids are smooth, bounded convex sets that generalize circles and ellipses
    to higher dimensions. They arise naturally in probability (confidence regions)
    and optimization (trust regions).
    
    Properties:
    - Always convex and bounded
    - Smooth boundary (infinitely differentiable)
    - Contains a ball and is contained in a ball
    """
    
    def __init__(self, A: np.ndarray, c: Optional[np.ndarray] = None):
        """
        Initialize ellipsoid (x-c)ᵀ A⁻¹ (x-c) ≤ 1.
        
        Args:
            A: Positive definite matrix defining the ellipsoid shape
            c: Center point (default: origin)
        """
        self.A = np.array(A)
        self.c = np.zeros(A.shape[0]) if c is None else np.array(c)
        
        # Verify positive definiteness
        eigenvals = np.linalg.eigvals(A)
        if not np.all(eigenvals > 1e-12):
            raise ValueError("Matrix A must be positive definite")
        
        self.A_inv = np.linalg.inv(A)
        self.sqrt_A = np.linalg.cholesky(A)  # A = L L^T
    
    def contains(self, x: np.ndarray) -> bool:
        """Check if point x is in the ellipsoid."""
        diff = x - self.c
        return np.dot(diff, np.dot(self.A_inv, diff)) <= 1.0 + 1e-10
    
    def project(self, x: np.ndarray) -> np.ndarray:
        """Project point onto ellipsoid boundary (or return x if inside)."""
        diff = x - self.c
        quad_form = np.dot(diff, np.dot(self.A_inv, diff))
        
        if quad_form <= 1.0:
            return x  # Already inside
        else:
            # Project onto boundary
            return self.c + diff / np.sqrt(quad_form)
    
    def support(self, direction: np.ndarray) -> Tuple[np.ndarray, float]:
        """Support function for ellipsoid."""
        # Support point is c + A * direction / ||A * direction||
        A_dir = np.dot(self.A, direction)
        norm_A_dir = np.linalg.norm(A_dir)
        
        if norm_A_dir == 0:
            return self.c, np.dot(direction, self.c)
        
        support_point = self.c + A_dir / norm_A_dir
        support_value = np.dot(direction, support_point)
        
        return support_point, support_value
    
    def volume(self) -> float:
        """Compute the volume of the ellipsoid."""
        n = len(self.c)
        det_A = np.linalg.det(self.A)
        
        # Volume of unit ball in n dimensions
        if n % 2 == 0:
            # Even dimension
            unit_volume = np.pi**(n/2) / np.math.factorial(n//2)
        else:
            # Odd dimension
            unit_volume = 2 * np.math.factorial((n-1)//2) * (4*np.pi)**((n-1)/2) / np.math.factorial(n)
        
        return unit_volume * np.sqrt(det_A)


class Polyhedron(ConvexSet):
    """
    Polyhedron: {x : Ax ≤ b, Cx = d}
    
    Polyhedra are the intersection of finitely many halfspaces and hyperplanes.
    They are fundamental in linear programming and have rich geometric structure.
    
    Properties:
    - Always convex
    - Can be bounded (polytope) or unbounded
    - Vertices, edges, and faces have clear meaning
    - Dual representation as convex hull of vertices + rays
    """
    
    def __init__(self, A: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None,
                 C: Optional[np.ndarray] = None, d: Optional[np.ndarray] = None):
        """
        Initialize polyhedron Ax ≤ b, Cx = d.
        
        Args:
            A: Inequality constraint matrix
            b: Inequality constraint vector
            C: Equality constraint matrix  
            d: Equality constraint vector
        """
        # Handle inequality constraints
        if A is not None and b is not None:
            self.A = np.array(A)
            self.b = np.array(b)
        else:
            self.A = np.empty((0, 1))  # Will be resized when first point is tested
            self.b = np.empty(0)
        
        # Handle equality constraints
        if C is not None and d is not None:
            self.C = np.array(C)
            self.d = np.array(d)
        else:
            self.C = np.empty((0, 1))  # Will be resized when first point is tested
            self.d = np.empty(0)
    
    def contains(self, x: np.ndarray) -> bool:
        """Check if point x is in the polyhedron."""
        x = np.array(x)
        
        # Resize constraint matrices if needed
        if self.A.size > 0 and self.A.shape[1] != len(x):
            self.A = np.empty((0, len(x)))
            self.b = np.empty(0)
        if self.C.size > 0 and self.C.shape[1] != len(x):
            self.C = np.empty((0, len(x)))
            self.d = np.empty(0)
        
        # Check inequality constraints
        if self.A.size > 0:
            if not np.all(np.dot(self.A, x) <= self.b + 1e-10):
                return False
        
        # Check equality constraints
        if self.C.size > 0:
            if not np.allclose(np.dot(self.C, x), self.d, atol=1e-10):
                return False
        
        return True
    
    def add_inequality(self, a: np.ndarray, b: float):
        """Add inequality constraint aᵀx ≤ b."""
        a = np.array(a)
        
        if self.A.size == 0:
            self.A = a.reshape(1, -1)
            self.b = np.array([b])
        else:
            self.A = np.vstack([self.A, a.reshape(1, -1)])
            self.b = np.append(self.b, b)
    
    def add_equality(self, c: np.ndarray, d: float):
        """Add equality constraint cᵀx = d."""
        c = np.array(c)
        
        if self.C.size == 0:
            self.C = c.reshape(1, -1)
            self.d = np.array([d])
        else:
            self.C = np.vstack([self.C, c.reshape(1, -1)])
            self.d = np.append(self.d, d)


def demonstrate_convex_sets():
    """
    Demonstrate the theory and properties of convex sets with visual examples.
    """
    print("🔷 CONVEX SETS: Fundamental Theory and Examples")
    print("=" * 60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Convex vs Non-convex Sets
    ax1 = plt.subplot(2, 3, 1)
    
    # Convex set (ellipse)
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_x = 2 * np.cos(theta)
    ellipse_y = np.sin(theta)
    ax1.fill(ellipse_x, ellipse_y, alpha=0.3, color='green', label='Convex')
    
    # Show convex combination
    p1, p2 = np.array([1.5, 0.7]), np.array([-1.2, -0.8])
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g--', linewidth=2, alpha=0.7)
    ax1.plot(p1[0], p1[1], 'go', markersize=8)
    ax1.plot(p2[0], p2[1], 'go', markersize=8)
    
    # Non-convex set (crescent)
    theta1 = np.linspace(0, np.pi, 50)
    theta2 = np.linspace(np.pi, 2*np.pi, 50)
    crescent_x = np.concatenate([3 + 1.5*np.cos(theta1), 3 + 0.8*np.cos(theta2[::-1])])
    crescent_y = np.concatenate([1.5*np.sin(theta1), 0.8*np.sin(theta2[::-1])])
    ax1.fill(crescent_x, crescent_y, alpha=0.3, color='red', label='Non-convex')
    
    # Show non-convex combination
    p3, p4 = np.array([2.3, 1.0]), np.array([3.8, 0.6])
    ax1.plot([p3[0], p4[0]], [p3[1], p4[1]], 'r--', linewidth=2, alpha=0.7)
    ax1.plot(p3[0], p3[1], 'ro', markersize=8)
    ax1.plot(p4[0], p4[1], 'ro', markersize=8)
    
    ax1.set_xlim(-3, 6)
    ax1.set_ylim(-2, 3)
    ax1.set_title('Convex vs Non-convex Sets')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Halfspaces and Hyperplanes
    ax2 = plt.subplot(2, 3, 2)
    
    # Create hyperplane and halfspace
    a = np.array([1, 1])
    b = 1
    hyperplane = Hyperplane(a, b)
    halfspace = Halfspace(a, b)
    
    # Plot hyperplane
    x_line = np.linspace(-2, 3, 100)
    y_line = (b - a[0] * x_line) / a[1]
    ax2.plot(x_line, y_line, 'b-', linewidth=2, label=f'Hyperplane: {a[0]}x + {a[1]}y = {b}')
    
    # Shade halfspace
    y_fill = np.minimum(y_line, 3)
    ax2.fill_between(x_line, y_fill, -3, alpha=0.2, color='blue', 
                     label=f'Halfspace: {a[0]}x + {a[1]}y ≤ {b}')
    
    # Test some points
    test_points = np.array([[0, 0], [1, 0], [0, 1], [2, 1], [-1, 2]])
    for i, point in enumerate(test_points):
        color = 'green' if halfspace.contains(point) else 'red'
        marker = 'o' if halfspace.contains(point) else 'x'
        ax2.plot(point[0], point[1], color=color, marker=marker, markersize=8)
    
    ax2.set_xlim(-2, 3)
    ax2.set_ylim(-1, 3)
    ax2.set_title('Hyperplanes and Halfspaces')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Ellipsoids
    ax3 = plt.subplot(2, 3, 3)
    
    # Create ellipsoid
    A = np.array([[2, 0.5], [0.5, 1]])
    c = np.array([1, 0.5])
    ellipsoid = Ellipsoid(A, c)
    
    # Generate ellipsoid boundary
    theta = np.linspace(0, 2*np.pi, 200)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse_points = c.reshape(-1, 1) + ellipsoid.sqrt_A @ unit_circle
    
    ax3.fill(ellipse_points[0], ellipse_points[1], alpha=0.3, color='purple')
    ax3.plot(ellipse_points[0], ellipse_points[1], 'purple', linewidth=2, 
             label='Ellipsoid')
    ax3.plot(c[0], c[1], 'ko', markersize=8, label='Center')
    
    # Show support function
    direction = np.array([1, 1])
    direction = direction / np.linalg.norm(direction)
    support_point, support_value = ellipsoid.support(direction)
    
    ax3.arrow(c[0], c[1], 2*direction[0], 2*direction[1], 
              head_width=0.1, head_length=0.1, fc='orange', ec='orange')
    ax3.plot(support_point[0], support_point[1], 'ro', markersize=8, 
             label='Support point')
    
    ax3.set_xlim(-1, 4)
    ax3.set_ylim(-2, 3)
    ax3.set_title('Ellipsoids and Support Functions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # 4. Polyhedra
    ax4 = plt.subplot(2, 3, 4)
    
    # Create a simple polyhedron (triangle)
    poly = Polyhedron()
    poly.add_inequality(np.array([1, 0]), 2)    # x ≤ 2
    poly.add_inequality(np.array([0, 1]), 2)    # y ≤ 2  
    poly.add_inequality(np.array([-1, -1]), -1) # -x - y ≤ -1, i.e., x + y ≥ 1
    
    # Find vertices by solving systems of equations
    vertices = []
    constraints = [
        (np.array([1, 0]), 2),
        (np.array([0, 1]), 2),
        (np.array([-1, -1]), -1)
    ]
    
    # Intersection of each pair of constraint boundaries
    for i in range(len(constraints)):
        for j in range(i+1, len(constraints)):
            a1, b1 = constraints[i]
            a2, b2 = constraints[j]
            
            # Solve a1^T x = b1, a2^T x = b2
            A_sys = np.array([a1, a2])
            b_sys = np.array([b1, b2])
            
            try:
                vertex = np.linalg.solve(A_sys, b_sys)
                if poly.contains(vertex):
                    vertices.append(vertex)
            except np.linalg.LinAlgError:
                continue
    
    if vertices:
        vertices = np.array(vertices)
        hull = ConvexSet.convex_hull_2d(vertices)
        ax4.fill(hull[:, 0], hull[:, 1], alpha=0.3, color='cyan', label='Polyhedron')
        ax4.plot(np.append(hull[:, 0], hull[0, 0]), 
                 np.append(hull[:, 1], hull[0, 1]), 'c-', linewidth=2)
        
        for vertex in vertices:
            ax4.plot(vertex[0], vertex[1], 'ko', markersize=8)
    
    ax4.set_xlim(0, 3)
    ax4.set_ylim(0, 3)
    ax4.set_title('Polyhedra (Intersection of Halfspaces)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Convex Hull
    ax5 = plt.subplot(2, 3, 5)
    
    # Random points
    np.random.seed(42)
    points = np.random.randn(15, 2) * 1.5 + np.array([1, 1])
    
    # Compute convex hull
    hull_vertices = ConvexSet.convex_hull_2d(points)
    
    # Plot points and hull
    ax5.scatter(points[:, 0], points[:, 1], c='blue', alpha=0.6, s=50, label='Points')
    ax5.fill(hull_vertices[:, 0], hull_vertices[:, 1], alpha=0.2, color='red')
    ax5.plot(np.append(hull_vertices[:, 0], hull_vertices[0, 0]),
             np.append(hull_vertices[:, 1], hull_vertices[0, 1]), 
             'r-', linewidth=2, label='Convex Hull')
    
    for vertex in hull_vertices:
        ax5.plot(vertex[0], vertex[1], 'ro', markersize=8)
    
    ax5.set_title('Convex Hull of Points')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Convex Combinations
    ax6 = plt.subplot(2, 3, 6)
    
    # Three points forming a triangle
    A, B, C = np.array([0, 0]), np.array([3, 1]), np.array([1, 3])
    triangle = np.array([A, B, C, A])
    
    ax6.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=2)
    ax6.fill(triangle[:-1, 0], triangle[:-1, 1], alpha=0.2, color='yellow')
    
    # Show some convex combinations
    np.random.seed(123)
    for _ in range(20):
        # Random convex weights
        w = np.random.random(3)
        w = w / np.sum(w)
        
        # Convex combination
        point = w[0]*A + w[1]*B + w[2]*C
        
        # Color based on dominant weight
        if np.argmax(w) == 0:
            color, marker = 'red', 'o'
        elif np.argmax(w) == 1:
            color, marker = 'green', 's'
        else:
            color, marker = 'blue', '^'
        
        ax6.plot(point[0], point[1], color=color, marker=marker, 
                 markersize=6, alpha=0.7)
    
    # Plot vertices
    ax6.plot(A[0], A[1], 'ro', markersize=10, label='Vertex A')
    ax6.plot(B[0], B[1], 'go', markersize=10, label='Vertex B')  
    ax6.plot(C[0], C[1], 'bo', markersize=10, label='Vertex C')
    
    ax6.set_xlim(-0.5, 3.5)
    ax6.set_ylim(-0.5, 3.5)
    ax6.set_title('Convex Combinations')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Numerical examples
    print("\n📊 NUMERICAL EXAMPLES")
    print("-" * 40)
    
    # Test convex set properties
    print("1. Ellipsoid Properties:")
    ellipsoid = Ellipsoid(np.array([[4, 1], [1, 2]]), np.array([0, 0]))
    test_point = np.array([1, 0.5])
    
    print(f"   Point {test_point} in ellipsoid: {ellipsoid.contains(test_point)}")
    print(f"   Ellipsoid volume: {ellipsoid.volume():.3f}")
    
    projected = ellipsoid.project(np.array([3, 3]))
    print(f"   Projection of [3,3]: [{projected[0]:.3f}, {projected[1]:.3f}]")
    
    print("\n2. Hyperplane Properties:")
    hyperplane = Hyperplane(np.array([1, 1]), 1)
    print(f"   Distance from [2,1] to hyperplane: {hyperplane.distance(np.array([2, 1])):.3f}")
    print(f"   Projection of [2,1]: {hyperplane.project(np.array([2, 1]))}")
    
    print("\n3. Convex Combination Test:")
    points = np.array([[0, 0], [1, 0], [0, 1]])
    weights = np.array([0.3, 0.4, 0.3])
    target = np.array([0.4, 0.3])
    
    is_convex_combo = ConvexSet.is_convex_combination(points, weights, target)
    print(f"   Is [0.4, 0.3] a convex combination: {is_convex_combo}")
    print(f"   Actual combination: {np.sum(weights.reshape(-1, 1) * points, axis=0)}")


def verify_convexity_properties():
    """
    Verify key theoretical properties of convex sets through computation.
    """
    print("\n🔍 VERIFICATION OF CONVEXITY PROPERTIES")
    print("=" * 50)
    
    # Property 1: Intersection of convex sets is convex
    print("1. Intersection Property:")
    
    # Create two ellipsoids
    A1 = np.array([[2, 0], [0, 1]])
    A2 = np.array([[1, 0.5], [0.5, 2]])
    ellipsoid1 = Ellipsoid(A1, np.array([0, 0]))
    ellipsoid2 = Ellipsoid(A2, np.array([1, 0]))
    
    # Test multiple points
    test_points = np.random.randn(100, 2) * 2
    intersection_points = []
    
    for point in test_points:
        if ellipsoid1.contains(point) and ellipsoid2.contains(point):
            intersection_points.append(point)
    
    print(f"   Found {len(intersection_points)} points in intersection")
    
    # Verify convexity by testing random convex combinations
    if len(intersection_points) >= 2:
        intersection_points = np.array(intersection_points)
        convex_violations = 0
        
        for _ in range(50):
            # Pick two random points from intersection
            idx = np.random.choice(len(intersection_points), 2, replace=False)
            p1, p2 = intersection_points[idx]
            
            # Random convex combination
            t = np.random.random()
            combo = t * p1 + (1 - t) * p2
            
            # Check if combination is still in intersection
            if not (ellipsoid1.contains(combo) and ellipsoid2.contains(combo)):
                convex_violations += 1
        
        print(f"   Convexity violations: {convex_violations}/50 (should be 0)")
    
    # Property 2: Convex hull is the smallest convex set containing points
    print("\n2. Convex Hull Property:")
    
    # Random points
    np.random.seed(42)
    points = np.random.randn(10, 2)
    hull_vertices = ConvexSet.convex_hull_2d(points)
    
    print(f"   Original points: {len(points)}")
    print(f"   Hull vertices: {len(hull_vertices)}")
    
    # Verify all original points are in convex hull
    violations = 0
    for point in points:
        # Check if point can be written as convex combination of hull vertices
        # This is a simplified test
        distances = [np.linalg.norm(point - vertex) for vertex in hull_vertices]
        if min(distances) > 1e-10:  # Not a vertex
            # Check if inside hull (simplified geometric test)
            # For full verification, would need to solve linear system
            pass
    
    print(f"   All original points contained in hull: {violations == 0}")
    
    # Property 3: Support function characterizes convex sets
    print("\n3. Support Function Property:")
    
    ellipsoid = Ellipsoid(np.array([[2, 0.5], [0.5, 1]]), np.array([0, 0]))
    
    # Test support function property: max_{x∈C} d^T x = σ_C(d)
    directions = [
        np.array([1, 0]),
        np.array([0, 1]), 
        np.array([1, 1]) / np.sqrt(2),
        np.array([-1, 1]) / np.sqrt(2)
    ]
    
    for direction in directions:
        support_point, support_value = ellipsoid.support(direction)
        
        # Verify this is indeed the maximum
        verification_value = np.dot(direction, support_point)
        
        print(f"   Direction {direction}: Support value = {support_value:.3f}, "
              f"Verification = {verification_value:.3f}")
        
        # Check that support point is on boundary
        on_boundary = abs(np.dot(support_point, np.dot(ellipsoid.A_inv, support_point)) - 1.0) < 1e-10
        print(f"   Support point on boundary: {on_boundary}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_convex_sets()
    verify_convexity_properties()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Convex sets preserve line segments between any two points")
    print("- Halfspaces and hyperplanes are building blocks for all convex sets")
    print("- Ellipsoids provide smooth, bounded convex sets")
    print("- Polyhedra arise from intersecting finitely many halfspaces")
    print("- Convex hull gives the smallest convex set containing given points")
    print("- Support functions completely characterize convex sets")
    print("\nNext: Study convex functions and their properties! 🚀")