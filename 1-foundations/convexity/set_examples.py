"""
Convex Sets: Concrete Examples and Applications

This module provides a rich collection of important convex sets that appear
frequently in optimization, machine learning, and engineering applications.
Understanding these examples builds intuition for recognizing convexity.

Learning Objectives:
- Recognize important families of convex sets
- Understand geometric properties of each set type
- Learn applications where each set type appears
- Develop intuition for convex set recognition

Key Examples:
- Polyhedra: Linear programming feasible regions
- Ellipsoids: Confidence regions, trust regions
- Cones: Positive orthants, second-order cones
- Simplices: Probability distributions, barycentric coordinates
- Epigraphs: Connection between sets and functions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Union, Callable
import warnings
from definitions import ConvexSet, Ellipsoid, Halfspace, Polyhedron

warnings.filterwarnings('ignore')

class Simplex(ConvexSet):
    """
    Unit simplex: {x : x ≥ 0, Σxᵢ = 1}
    
    The simplex is fundamental in probability (probability distributions),
    game theory (mixed strategies), and optimization (feasible regions).
    
    Properties:
    - Vertices are standard basis vectors
    - Faces correspond to setting some coordinates to zero
    - Volume is 1/√n! in n dimensions
    """
    
    def __init__(self, n: int):
        """
        Initialize n-dimensional unit simplex.
        
        Args:
            n: Dimension of the simplex
        """
        self.n = n
    
    def contains(self, x: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if point x is in the unit simplex."""
        x = np.array(x)
        if len(x) != self.n:
            return False
        
        # Check non-negativity
        if not np.all(x >= -tol):
            return False
        
        # Check sum constraint
        if abs(np.sum(x) - 1.0) > tol:
            return False
        
        return True
    
    def project(self, x: np.ndarray) -> np.ndarray:
        """
        Project point onto simplex using Michelot's algorithm.
        
        This is the projection onto {x : x ≥ 0, Σxᵢ = 1}.
        """
        x = np.array(x, dtype=float)
        n = len(x)
        
        # Sort elements in descending order
        sorted_indices = np.argsort(x)[::-1]
        x_sorted = x[sorted_indices]
        
        # Find the largest k such that x_k + (1 - sum(x[1:k+1]))/k > 0
        cumsum = 0.0
        k = 0
        
        for i in range(n):
            cumsum += x_sorted[i]
            if x_sorted[i] + (1.0 - cumsum) / (i + 1) > 0:
                k = i + 1
            else:
                break
        
        # Compute the threshold
        theta = (1.0 - np.sum(x_sorted[:k])) / k
        
        # Project
        projected = np.maximum(x + theta, 0.0)
        
        return projected
    
    def vertices(self) -> np.ndarray:
        """Return vertices of the simplex (standard basis vectors)."""
        return np.eye(self.n)
    
    def volume(self) -> float:
        """Compute volume of n-dimensional simplex."""
        return 1.0 / np.math.factorial(self.n)


class SecondOrderCone(ConvexSet):
    """
    Second-order cone (Lorentz cone): {(x, t) : ||x||₂ ≤ t}
    
    SOCs are fundamental in robust optimization, engineering design,
    and semidefinite programming. They generalize linear programming.
    
    Properties:
    - Self-dual cone
    - Smooth except at origin
    - Relates to hyperbolic geometry
    """
    
    def __init__(self, n: int):
        """
        Initialize n-dimensional second-order cone.
        
        Args:
            n: Dimension of x vector (total dimension is n+1)
        """
        self.n = n
    
    def contains(self, y: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Check if point y = (x, t) is in the second-order cone.
        
        Args:
            y: Point of dimension n+1, where y = [x₁, ..., xₙ, t]
        """
        y = np.array(y)
        if len(y) != self.n + 1:
            return False
        
        x = y[:-1]  # First n components
        t = y[-1]   # Last component
        
        return np.linalg.norm(x) <= t + tol
    
    def project(self, y: np.ndarray) -> np.ndarray:
        """
        Project point onto second-order cone.
        
        Closed-form projection formula for SOC.
        """
        y = np.array(y, dtype=float)
        x = y[:-1]
        t = y[-1]
        
        norm_x = np.linalg.norm(x)
        
        if norm_x <= t:
            # Already in cone
            return y
        elif norm_x <= -t:
            # Project to origin
            return np.zeros_like(y)
        else:
            # Project to boundary
            alpha = (norm_x + t) / (2 * norm_x)
            projected_x = alpha * x
            projected_t = norm_x * alpha
            return np.concatenate([projected_x, [projected_t]])
    
    def dual_cone(self) -> 'SecondOrderCone':
        """Return dual cone (which is the same for SOC)."""
        return SecondOrderCone(self.n)


class PositiveOrthant(ConvexSet):
    """
    Positive orthant: {x : x ≥ 0}
    
    The positive orthant is the simplest cone and appears everywhere
    in optimization (non-negativity constraints, linear programming).
    
    Properties:
    - Polyhedral cone
    - Self-dual
    - Extreme rays are coordinate axes
    """
    
    def __init__(self, n: int):
        """
        Initialize n-dimensional positive orthant.
        
        Args:
            n: Dimension of the space
        """
        self.n = n
    
    def contains(self, x: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if point x is in positive orthant."""
        x = np.array(x)
        return len(x) == self.n and np.all(x >= -tol)
    
    def project(self, x: np.ndarray) -> np.ndarray:
        """Project onto positive orthant (componentwise maximum with zero)."""
        return np.maximum(x, 0.0)
    
    def support(self, direction: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Support function for positive orthant."""
        if np.any(direction < -1e-10):
            return None, -np.inf  # Unbounded in negative directions
        
        # Support point has infinite components where direction > 0
        return None, np.inf if np.any(direction > 1e-10) else 0.0


class SemidefiniteCone(ConvexSet):
    """
    Semidefinite cone: {X : X ⪰ 0} (positive semidefinite matrices)
    
    The PSD cone is central to semidefinite programming and has deep
    connections to linear algebra and optimization theory.
    
    Properties:
    - Self-dual cone
    - Defined by eigenvalue constraints
    - Interior consists of positive definite matrices
    """
    
    def __init__(self, n: int):
        """
        Initialize semidefinite cone for n×n matrices.
        
        Args:
            n: Size of the matrices
        """
        self.n = n
    
    def contains(self, X: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Check if matrix X is positive semidefinite.
        
        Args:
            X: Square matrix to test
        """
        X = np.array(X)
        if X.shape != (self.n, self.n):
            return False
        
        # Check symmetry
        if not np.allclose(X, X.T, atol=tol):
            return False
        
        # Check positive semidefiniteness via eigenvalues
        eigenvals = np.linalg.eigvals(X)
        return np.all(eigenvals >= -tol)
    
    def project(self, X: np.ndarray) -> np.ndarray:
        """
        Project matrix onto semidefinite cone.
        
        Projection is via eigenvalue decomposition: keep positive eigenvalues.
        """
        X = np.array(X, dtype=float)
        
        # Symmetrize first
        X_sym = (X + X.T) / 2
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(X_sym)
        
        # Keep only positive eigenvalues
        eigenvals_proj = np.maximum(eigenvals, 0.0)
        
        # Reconstruct matrix
        return eigenvecs @ np.diag(eigenvals_proj) @ eigenvecs.T
    
    def trace_inner_product(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute trace inner product ⟨X, Y⟩ = tr(X^T Y)."""
        return np.trace(X.T @ Y)


class Epigraph(ConvexSet):
    """
    Epigraph of a function: epi(f) = {(x, t) : t ≥ f(x)}
    
    Epigraphs connect convex sets and convex functions, providing
    the geometric foundation for convex analysis.
    
    Properties:
    - Function is convex iff epigraph is convex
    - Projection onto epigraph related to proximal operators
    - Fundamental in variational analysis
    """
    
    def __init__(self, func: Callable[[np.ndarray], float], 
                 gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """
        Initialize epigraph of a function.
        
        Args:
            func: Function f(x) whose epigraph to represent
            gradient: Gradient of f (optional, for projections)
        """
        self.func = func
        self.gradient = gradient
    
    def contains(self, y: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Check if point (x, t) is in epigraph.
        
        Args:
            y: Point where y[:-1] is x and y[-1] is t
        """
        x = y[:-1]
        t = y[-1]
        
        try:
            return t >= self.func(x) - tol
        except:
            return False
    
    def project(self, y: np.ndarray, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
        """
        Project point onto epigraph.
        
        This is complex in general and requires iterative methods.
        """
        x = y[:-1]
        t = y[-1]
        
        func_value = self.func(x)
        
        if t >= func_value:
            # Already in epigraph
            return y
        else:
            # Project onto graph of function
            # For general functions, this requires solving optimization problem
            # Here we use a simple heuristic
            return np.concatenate([x, [func_value]])


def demonstrate_convex_examples():
    """
    Demonstrate important examples of convex sets with visualizations.
    """
    print("🔷 CONVEX SET EXAMPLES: Important Families")
    print("=" * 60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Unit Simplex in 2D (triangle)
    ax1 = plt.subplot(2, 3, 1)
    
    simplex = Simplex(3)  # 3D simplex projected to 2D
    
    # Vertices of 2D simplex (probability simplex for 3 outcomes)
    vertices = np.array([[1, 0], [0, 1], [0, 0]])
    triangle = np.array([vertices[0], vertices[1], vertices[2], vertices[0]])
    
    ax1.fill(triangle[:, 0], triangle[:, 1], alpha=0.3, color='blue', label='2-Simplex')
    ax1.plot(triangle[:, 0], triangle[:, 1], 'b-', linewidth=2)
    
    # Plot vertices
    for i, vertex in enumerate(vertices):
        ax1.plot(vertex[0], vertex[1], 'ro', markersize=8)
        ax1.annotate(f'e{i+1}', vertex + 0.05, fontsize=12)
    
    # Show some probability distributions (points in simplex)
    np.random.seed(42)
    for _ in range(20):
        # Random probability distribution
        probs = np.random.random(3)
        probs = probs / np.sum(probs)
        
        # Convert to 2D (drop last coordinate since they sum to 1)
        point_2d = probs[:2]
        ax1.plot(point_2d[0], point_2d[1], 'go', alpha=0.6, markersize=4)
    
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_title('Unit Simplex (Probability Distributions)')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Second-Order Cone in 2D
    ax2 = plt.subplot(2, 3, 2)
    
    soc = SecondOrderCone(1)  # 2D SOC: |x| ≤ t
    
    # Generate cone boundary
    t_vals = np.linspace(0, 3, 100)
    x_pos = t_vals
    x_neg = -t_vals
    
    ax2.fill_between(x_pos, t_vals, 0, alpha=0.3, color='orange', label='SOC')
    ax2.fill_between(x_neg, t_vals, 0, alpha=0.3, color='orange')
    ax2.plot(x_pos, t_vals, 'orange', linewidth=2)
    ax2.plot(x_neg, t_vals, 'orange', linewidth=2)
    
    # Test some points
    test_points = [np.array([1, 2]), np.array([2, 1]), np.array([0.5, 1]), np.array([1.5, 1])]
    for point in test_points:
        color = 'green' if soc.contains(point) else 'red'
        marker = 'o' if soc.contains(point) else 'x'
        ax2.plot(point[0], point[1], color=color, marker=marker, markersize=8)
    
    # Show projection of an external point
    external_point = np.array([2, 1])
    projected = soc.project(external_point)
    ax2.plot([external_point[0], projected[0]], [external_point[1], projected[1]], 
             'g--', linewidth=2, alpha=0.7)
    
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 3)
    ax2.set_title('Second-Order Cone: ||x|| ≤ t')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Positive Orthant in 2D
    ax3 = plt.subplot(2, 3, 3)
    
    orthant = PositiveOrthant(2)
    
    # Shade positive quadrant
    x_range = np.linspace(0, 3, 100)
    y_range = np.linspace(0, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    ax3.fill_between([0, 3], [0, 0], [3, 3], alpha=0.3, color='green', label='R₊²')
    ax3.axhline(y=0, color='black', linewidth=2)
    ax3.axvline(x=0, color='black', linewidth=2)
    
    # Test points and projections
    test_points = [np.array([-1, 2]), np.array([1, -1]), np.array([-0.5, -0.5])]
    for point in test_points:
        projected = orthant.project(point)
        
        # Plot original and projected points
        ax3.plot(point[0], point[1], 'ro', markersize=8)
        ax3.plot(projected[0], projected[1], 'go', markersize=8)
        ax3.plot([point[0], projected[0]], [point[1], projected[1]], 
                 'b--', linewidth=2, alpha=0.7)
    
    ax3.set_xlim(-2, 3)
    ax3.set_ylim(-2, 3)
    ax3.set_title('Positive Orthant: x ≥ 0')
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Semidefinite Cone (2×2 matrices)
    ax4 = plt.subplot(2, 3, 4)
    
    # For 2×2 symmetric matrices, PSD cone is {[a c; c b] : a,b ≥ 0, ab ≥ c²}
    # Visualize in (a, b, c) space projected to (a, b) with c fixed
    
    a_vals = np.linspace(0, 3, 100)
    b_vals = np.linspace(0, 3, 100)
    A, B = np.meshgrid(a_vals, b_vals)
    
    # For c = 0.5, constraint is ab ≥ 0.25
    c_fixed = 0.5
    feasible = A * B >= c_fixed**2
    
    ax4.contour(A, B, A * B, levels=[c_fixed**2], colors=['red'], linewidths=2)
    ax4.contourf(A, B, feasible.astype(int), levels=[0.5, 1.5], colors=['lightblue'], alpha=0.5)
    
    # Test some matrices
    test_matrices = [
        np.array([[2, 0.5], [0.5, 1]]),      # PSD
        np.array([[1, 1.5], [1.5, 1]]),      # Not PSD
        np.array([[2, 0.3], [0.3, 0.5]]),    # PSD
    ]
    
    sdc = SemidefiniteCone(2)
    for matrix in test_matrices:
        a, b, c = matrix[0, 0], matrix[1, 1], matrix[0, 1]
        color = 'green' if sdc.contains(matrix) else 'red'
        marker = 'o' if sdc.contains(matrix) else 'x'
        ax4.plot(a, b, color=color, marker=marker, markersize=10)
    
    ax4.set_xlim(0, 3)
    ax4.set_ylim(0, 3)
    ax4.set_title(f'PSD Cone (2×2): ab ≥ c² (c={c_fixed})')
    ax4.set_xlabel('a (diagonal element)')
    ax4.set_ylabel('b (diagonal element)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Epigraph of a Convex Function
    ax5 = plt.subplot(2, 3, 5)
    
    # Convex function: f(x) = x²
    def quadratic(x):
        return x[0]**2 if len(x) == 1 else np.sum(x**2)
    
    epigraph = Epigraph(quadratic)
    
    # Plot function and its epigraph
    x_vals = np.linspace(-2, 2, 100)
    y_vals = x_vals**2
    
    ax5.plot(x_vals, y_vals, 'blue', linewidth=3, label='f(x) = x²')
    
    # Shade epigraph
    y_upper = np.full_like(x_vals, 5)
    ax5.fill_between(x_vals, y_vals, y_upper, alpha=0.3, color='lightblue', 
                     label='Epigraph')
    
    # Test some points
    test_points = [
        np.array([1, 2]),    # In epigraph
        np.array([1, 0.5]),  # Not in epigraph
        np.array([-1, 3]),   # In epigraph
        np.array([0, -1])    # Not in epigraph
    ]
    
    for point in test_points:
        color = 'green' if epigraph.contains(point) else 'red'
        marker = 'o' if epigraph.contains(point) else 'x'
        ax5.plot(point[0], point[1], color=color, marker=marker, markersize=8)
    
    ax5.set_xlim(-2, 2)
    ax5.set_ylim(-1, 5)
    ax5.set_title('Epigraph of f(x) = x²')
    ax5.set_xlabel('x')
    ax5.set_ylabel('t')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Norm Balls (different norms)
    ax6 = plt.subplot(2, 3, 6)
    
    # Different p-norm balls
    theta = np.linspace(0, 2*np.pi, 1000)
    
    # L2 ball (circle)
    l2_x = np.cos(theta)
    l2_y = np.sin(theta)
    ax6.plot(l2_x, l2_y, 'blue', linewidth=2, label='L₂ ball')
    ax6.fill(l2_x, l2_y, alpha=0.2, color='blue')
    
    # L1 ball (diamond)
    l1_vertices = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    ax6.plot(l1_vertices[:, 0], l1_vertices[:, 1], 'red', linewidth=2, label='L₁ ball')
    ax6.fill(l1_vertices[:, 0], l1_vertices[:, 1], alpha=0.2, color='red')
    
    # L∞ ball (square)
    linf_vertices = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])
    ax6.plot(linf_vertices[:, 0], linf_vertices[:, 1], 'green', linewidth=2, label='L∞ ball')
    ax6.fill(linf_vertices[:, 0], linf_vertices[:, 1], alpha=0.2, color='green')
    
    ax6.set_xlim(-1.5, 1.5)
    ax6.set_ylim(-1.5, 1.5)
    ax6.set_title('Unit Balls for Different Norms')
    ax6.set_xlabel('x₁')
    ax6.set_ylabel('x₂')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Numerical examples and properties
    print("\n📊 NUMERICAL EXAMPLES AND PROPERTIES")
    print("-" * 50)
    
    # Simplex properties
    print("1. Simplex Properties:")
    simplex_3d = Simplex(3)
    vertices = simplex_3d.vertices()
    print(f"   Vertices: \n{vertices}")
    print(f"   Volume: {simplex_3d.volume():.6f}")
    
    # Test projection
    point = np.array([0.5, 0.3, -0.1])  # Outside simplex
    projected = simplex_3d.project(point)
    print(f"   Point: {point}")
    print(f"   Projected: {projected}")
    print(f"   Sum after projection: {np.sum(projected):.6f}")
    
    # SOC properties
    print("\n2. Second-Order Cone Properties:")
    soc_3d = SecondOrderCone(2)  # 3D SOC
    
    test_points = [
        np.array([1, 1, 2]),     # Inside
        np.array([2, 1, 2]),     # Outside
        np.array([0, 0, 0]),     # On boundary (apex)
    ]
    
    for point in test_points:
        inside = soc_3d.contains(point)
        projected = soc_3d.project(point)
        print(f"   Point {point}: Inside = {inside}")
        print(f"     Projected: {projected}")
    
    # PSD cone properties
    print("\n3. Semidefinite Cone Properties:")
    psd_cone = SemidefiniteCone(2)
    
    test_matrices = [
        np.array([[2, 1], [1, 1]]),       # PSD (eigenvals: ~2.6, ~0.4)
        np.array([[1, 2], [2, 1]]),       # Not PSD (eigenvals: 3, -1)
        np.array([[1, 0], [0, 1]]),       # PSD (identity)
    ]
    
    for i, matrix in enumerate(test_matrices):
        is_psd = psd_cone.contains(matrix)
        eigenvals = np.linalg.eigvals(matrix)
        projected = psd_cone.project(matrix)
        
        print(f"   Matrix {i+1}:")
        print(f"     {matrix}")
        print(f"     PSD: {is_psd}, Eigenvalues: {eigenvals}")
        print(f"     Projected eigenvalues: {np.linalg.eigvals(projected)}")


def applications_showcase():
    """
    Showcase applications where these convex sets appear.
    """
    print("\n🎯 APPLICATIONS OF CONVEX SETS")
    print("=" * 40)
    
    print("1. LINEAR PROGRAMMING:")
    print("   - Feasible region: Intersection of halfspaces (polyhedron)")
    print("   - Standard form: {x : Ax = b, x ≥ 0} (positive orthant)")
    print("   - Vertices of feasible region are candidate optimal solutions")
    
    print("\n2. PORTFOLIO OPTIMIZATION:")
    print("   - Weights constraint: w ∈ simplex (portfolio weights sum to 1)")
    print("   - Long-only: w ∈ positive orthant (no short selling)")
    print("   - Risk ellipsoids: confidence regions for returns")
    
    print("\n3. ROBUST OPTIMIZATION:")
    print("   - Uncertainty sets: ellipsoids, polyhedra, norm balls")
    print("   - Worst-case constraints: sup over uncertainty set")
    print("   - Distributionally robust: Wasserstein balls")
    
    print("\n4. SEMIDEFINITE PROGRAMMING:")
    print("   - Matrix variables: X ∈ semidefinite cone")
    print("   - Relaxations of combinatorial problems")
    print("   - Control theory: Lyapunov matrix inequalities")
    
    print("\n5. MACHINE LEARNING:")
    print("   - SVM margin: second-order cone constraints")
    print("   - Regularization: norm ball constraints")
    print("   - Probability distributions: simplex constraints")
    
    print("\n6. SIGNAL PROCESSING:")
    print("   - Compressed sensing: L1 ball (sparsity)")
    print("   - Filter design: frequency response constraints")
    print("   - Robust estimation: Huber penalty (epigraph)")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_convex_examples()
    applications_showcase()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Simplex: Probability distributions, convex combinations")
    print("- SOC: Robust optimization, ||·||₂ constraints")
    print("- Positive orthant: Non-negativity, LP feasible regions")  
    print("- PSD cone: Matrix optimization, control theory")
    print("- Epigraphs: Bridge between sets and functions")
    print("- Norm balls: Regularization, uncertainty sets")
    print("\nThese examples appear everywhere in optimization! 🚀")