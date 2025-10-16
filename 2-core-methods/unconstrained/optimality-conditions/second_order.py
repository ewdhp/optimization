"""
Second-Order Optimality Conditions

Second-order conditions use the Hessian matrix to determine whether critical
points are local minima, maxima, or saddle points. These conditions provide
SUFFICIENT conditions for local optimality.

UNCONSTRAINED OPTIMIZATION:
    Second-Order Necessary Condition (SONC):
        If x* is a local minimum, then:
        1. ∇f(x*) = 0                    (FONC)
        2. ∇²f(x*) ⪰ 0                   (positive semidefinite)
    
    Second-Order Sufficient Condition (SOSC):
        If:
        1. ∇f(x*) = 0                    (critical point)
        2. ∇²f(x*) ≻ 0                   (positive definite)
        Then x* is a STRICT local minimum

HESSIAN ANALYSIS:
    For function f: ℝⁿ → ℝ, Hessian ∇²f(x) is n×n matrix of second derivatives:
        [∂²f/∂x_i∂x_j]
    
    Definiteness:
    - Positive definite (PD): all eigenvalues > 0 → strict local min
    - Positive semidefinite (PSD): all eigenvalues ≥ 0 → local min possible
    - Negative definite (ND): all eigenvalues < 0 → strict local max
    - Negative semidefinite (NSD): all eigenvalues ≤ 0 → local max possible
    - Indefinite: mixed signs → saddle point

CONSTRAINED OPTIMIZATION:
    Second-order conditions involve projected Hessian on feasible directions
    More complex - typically use numerical methods for verification

Key Tests:
1. Eigenvalue test: Check sign of all eigenvalues
2. Sylvester's criterion: Check leading principal minors
3. Cholesky decomposition: Exists ⟺ positive definite
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


def classify_matrix_definiteness(matrix: np.ndarray, 
                                 tolerance: float = 1e-10) -> str:
    """
    Classify matrix definiteness using eigenvalues.
    
    Args:
        matrix: Symmetric matrix to classify
        tolerance: Numerical tolerance
        
    Returns:
        Classification string
    """
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(matrix)
    
    min_eig = np.min(eigenvalues)
    max_eig = np.max(eigenvalues)
    
    if min_eig > tolerance:
        return 'positive_definite'
    elif min_eig >= -tolerance and max_eig > tolerance:
        return 'positive_semidefinite'
    elif max_eig < -tolerance:
        return 'negative_definite'
    elif max_eig <= tolerance and min_eig < -tolerance:
        return 'negative_semidefinite'
    else:
        return 'indefinite'


def verify_sonc(grad: np.ndarray, 
                hessian: np.ndarray,
                tolerance: float = 1e-6) -> Tuple[bool, str]:
    """
    Verify Second-Order Necessary Condition.
    
    SONC: ∇f(x*) = 0 AND ∇²f(x*) ⪰ 0
    
    Args:
        grad: Gradient at point
        hessian: Hessian at point
        tolerance: Numerical tolerance
        
    Returns:
        (satisfied, explanation)
    """
    # Check FONC
    grad_norm = np.linalg.norm(grad)
    fonc_satisfied = grad_norm < tolerance
    
    if not fonc_satisfied:
        return False, f"FONC violated: ||∇f|| = {grad_norm:.2e} > {tolerance}"
    
    # Check Hessian positive semidefinite
    definiteness = classify_matrix_definiteness(hessian, tolerance)
    
    if definiteness in ['positive_definite', 'positive_semidefinite']:
        return True, f"SONC satisfied: ∇f=0, H is {definiteness}"
    else:
        return False, f"SONC violated: H is {definiteness}"


def verify_sosc(grad: np.ndarray,
                hessian: np.ndarray,
                tolerance: float = 1e-6) -> Tuple[bool, str]:
    """
    Verify Second-Order Sufficient Condition.
    
    SOSC: ∇f(x*) = 0 AND ∇²f(x*) ≻ 0
    
    Args:
        grad: Gradient at point
        hessian: Hessian at point
        tolerance: Numerical tolerance
        
    Returns:
        (satisfied, explanation)
    """
    # Check FONC
    grad_norm = np.linalg.norm(grad)
    fonc_satisfied = grad_norm < tolerance
    
    if not fonc_satisfied:
        return False, f"FONC violated: ||∇f|| = {grad_norm:.2e}"
    
    # Check Hessian positive definite (all eigenvalues > 0)
    definiteness = classify_matrix_definiteness(hessian, tolerance)
    
    if definiteness == 'positive_definite':
        eigenvalues = np.linalg.eigvalsh(hessian)
        return True, f"SOSC satisfied: ∇f=0, H ≻ 0, λ_min = {np.min(eigenvalues):.4f}"
    else:
        return False, f"SOSC not satisfied: H is {definiteness}"


def classify_critical_point(grad: np.ndarray,
                           hessian: np.ndarray,
                           tolerance: float = 1e-6) -> Dict[str, any]:
    """
    Classify a critical point using second-order conditions.
    
    Args:
        grad: Gradient at point
        hessian: Hessian at point
        tolerance: Numerical tolerance
        
    Returns:
        Classification dictionary
    """
    # Check if critical point
    grad_norm = np.linalg.norm(grad)
    is_critical = grad_norm < tolerance
    
    if not is_critical:
        return {
            'is_critical': False,
            'gradient_norm': grad_norm,
            'classification': 'not_critical'
        }
    
    # Classify based on Hessian
    definiteness = classify_matrix_definiteness(hessian, tolerance)
    eigenvalues = np.linalg.eigvalsh(hessian)
    
    classification_map = {
        'positive_definite': 'strict_local_minimum',
        'positive_semidefinite': 'local_minimum_possible',
        'negative_definite': 'strict_local_maximum',
        'negative_semidefinite': 'local_maximum_possible',
        'indefinite': 'saddle_point'
    }
    
    return {
        'is_critical': True,
        'gradient_norm': grad_norm,
        'hessian_definiteness': definiteness,
        'classification': classification_map[definiteness],
        'eigenvalues': eigenvalues,
        'min_eigenvalue': np.min(eigenvalues),
        'max_eigenvalue': np.max(eigenvalues)
    }


def demonstrate_2d_critical_points():
    """
    Demonstrate second-order conditions on 2D examples.
    """
    print("🔍 SECOND-ORDER CONDITIONS: 2D CRITICAL POINTS")
    print("=" * 60)
    
    # Example 1: Paraboloid (minimum)
    print("\n🎯 EXAMPLE 1: Paraboloid f(x,y) = x² + 2y²")
    print("-" * 50)
    
    def f1(xy):
        x, y = xy
        return x**2 + 2*y**2
    
    def grad1(xy):
        x, y = xy
        return np.array([2*x, 4*y])
    
    def hess1(xy):
        return np.array([[2, 0], [0, 4]])
    
    x1 = np.array([0.0, 0.0])
    g1 = grad1(x1)
    h1 = hess1(x1)
    
    result1 = classify_critical_point(g1, h1)
    print(f"Point: {x1}")
    print(f"Gradient: {g1}")
    print(f"Hessian:\n{h1}")
    print(f"Classification: {result1['classification']}")
    print(f"Eigenvalues: {result1['eigenvalues']}")
    
    # Example 2: Inverted paraboloid (maximum)
    print("\n🎯 EXAMPLE 2: Inverted Paraboloid f(x,y) = -x² - y²")
    print("-" * 50)
    
    def f2(xy):
        x, y = xy
        return -x**2 - y**2
    
    def grad2(xy):
        x, y = xy
        return np.array([-2*x, -2*y])
    
    def hess2(xy):
        return np.array([[-2, 0], [0, -2]])
    
    x2 = np.array([0.0, 0.0])
    g2 = grad2(x2)
    h2 = hess2(x2)
    
    result2 = classify_critical_point(g2, h2)
    print(f"Point: {x2}")
    print(f"Classification: {result2['classification']}")
    print(f"Eigenvalues: {result2['eigenvalues']}")
    
    # Example 3: Saddle point
    print("\n🎯 EXAMPLE 3: Saddle f(x,y) = x² - y²")
    print("-" * 50)
    
    def f3(xy):
        x, y = xy
        return x**2 - y**2
    
    def grad3(xy):
        x, y = xy
        return np.array([2*x, -2*y])
    
    def hess3(xy):
        return np.array([[2, 0], [0, -2]])
    
    x3 = np.array([0.0, 0.0])
    g3 = grad3(x3)
    h3 = hess3(x3)
    
    result3 = classify_critical_point(g3, h3)
    print(f"Point: {x3}")
    print(f"Classification: {result3['classification']}")
    print(f"Eigenvalues: {result3['eigenvalues']}")
    
    # Example 4: Monkey saddle (degenerate)
    print("\n🎯 EXAMPLE 4: Monkey Saddle f(x,y) = x³ - 3xy²")
    print("-" * 50)
    
    def f4(xy):
        x, y = xy
        return x**3 - 3*x*y**2
    
    def grad4(xy):
        x, y = xy
        return np.array([3*x**2 - 3*y**2, -6*x*y])
    
    def hess4(xy):
        x, y = xy
        return np.array([[6*x, -6*y], [-6*y, -6*x]])
    
    x4 = np.array([0.0, 0.0])
    g4 = grad4(x4)
    h4 = hess4(x4)
    
    result4 = classify_critical_point(g4, h4)
    print(f"Point: {x4}")
    print(f"Classification: {result4['classification']}")
    print(f"Eigenvalues: {result4['eigenvalues']}")
    print("Note: Higher-order terms needed for classification!")
    
    # Visualization
    fig = plt.figure(figsize=(20, 10))
    
    # Common setup
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    examples = [
        (f1, grad1, hess1, x1, "Minimum", 1),
        (f2, grad2, hess2, x2, "Maximum", 2),
        (f3, grad3, hess3, x3, "Saddle", 3),
        (f4, grad4, hess4, x4, "Monkey Saddle", 4)
    ]
    
    for f, grad, hess, x_crit, title, idx in examples:
        # 3D surface
        ax_3d = fig.add_subplot(2, 4, idx, projection='3d')
        
        Z = np.array([[f(np.array([x, y])) for x in x_range] for y in y_range])
        
        surf = ax_3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax_3d.plot([x_crit[0]], [x_crit[1]], [f(x_crit)], 
                  'r*', markersize=15)
        
        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('f(x,y)')
        ax_3d.set_title(f'{title} (3D)')
        
        # Contour plot
        ax_contour = fig.add_subplot(2, 4, idx + 4)
        
        levels = 30
        contour = ax_contour.contour(X, Y, Z, levels=levels, cmap='viridis')
        ax_contour.plot(x_crit[0], x_crit[1], 'r*', markersize=15)
        
        # Add eigenvector directions if not degenerate
        h_mat = hess(x_crit)
        eigenvalues, eigenvectors = np.linalg.eigh(h_mat)
        
        for i, (eval, evec) in enumerate(zip(eigenvalues, eigenvectors.T)):
            if abs(eval) > 1e-6:
                color = 'green' if eval > 0 else 'red'
                scale = 0.5
                ax_contour.arrow(x_crit[0], x_crit[1],
                               scale*evec[0], scale*evec[1],
                               head_width=0.1, head_length=0.1,
                               fc=color, ec=color, alpha=0.7,
                               linewidth=2)
        
        ax_contour.set_xlabel('x')
        ax_contour.set_ylabel('y')
        ax_contour.set_title(f'{title} (Contour)')
        ax_contour.grid(True, alpha=0.3)
        ax_contour.axis('equal')
    
    plt.tight_layout()
    plt.show()


def demonstrate_hessian_tests():
    """
    Demonstrate various Hessian definiteness tests.
    """
    print("\n🧪 HESSIAN DEFINITENESS TESTS")
    print("=" * 60)
    
    # Example matrices
    matrices = {
        'Positive Definite': np.array([[2, 0], [0, 3]]),
        'Positive Semidefinite': np.array([[1, 0], [0, 0]]),
        'Negative Definite': np.array([[-2, 0], [0, -3]]),
        'Indefinite (Saddle)': np.array([[2, 0], [0, -3]]),
        'Indefinite (Complex)': np.array([[1, 2], [2, 1]])
    }
    
    print("\n📊 Matrix Classifications:")
    print("-" * 50)
    
    for name, matrix in matrices.items():
        eigenvalues = np.linalg.eigvalsh(matrix)
        classification = classify_matrix_definiteness(matrix)
        
        print(f"\n{name}:")
        print(f"  Matrix:\n{matrix}")
        print(f"  Eigenvalues: {eigenvalues}")
        print(f"  Classification: {classification}")
        
        # Try Cholesky (works only for PD)
        try:
            L = np.linalg.cholesky(matrix)
            print(f"  Cholesky: SUCCESS (confirms PD)")
        except np.linalg.LinAlgError:
            print(f"  Cholesky: FAILED (not PD)")
    
    # Visualization of eigenvalue distribution
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (name, matrix) in enumerate(matrices.items()):
        ax = axes[idx]
        
        eigenvalues = np.linalg.eigvalsh(matrix)
        
        # Plot eigenvalues
        ax.bar(range(len(eigenvalues)), eigenvalues, 
              color=['green' if e > 0 else 'red' for e in eigenvalues],
              alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Eigenvalue Index')
        ax.set_ylabel('Eigenvalue')
        ax.set_title(f'{name}\nλ = {eigenvalues}')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Summary plot
    ax_summary = axes[-1]
    ax_summary.axis('off')
    
    summary_text = """
    DEFINITENESS RULES:
    
    Positive Definite:
      All λ > 0
      ⟹ Strict local minimum
    
    Positive Semidefinite:
      All λ ≥ 0, some λ = 0
      ⟹ Minimum possible
    
    Negative Definite:
      All λ < 0
      ⟹ Strict local maximum
    
    Indefinite:
      Mixed signs
      ⟹ Saddle point
    """
    
    ax_summary.text(0.1, 0.5, summary_text, 
                   fontsize=11, family='monospace',
                   verticalalignment='center')
    
    plt.tight_layout()
    plt.show()


def second_order_theory():
    """
    Summary of second-order optimality theory.
    """
    print("\n📚 SECOND-ORDER OPTIMALITY THEORY")
    print("=" * 60)
    
    print("🎯 SECOND-ORDER NECESSARY CONDITION (SONC):")
    print("  If x* is a local minimum, then:")
    print("    1. ∇f(x*) = 0           (first-order)")
    print("    2. ∇²f(x*) ⪰ 0          (positive semidefinite)")
    print()
    print("  Interpretation: At minimum, curvature is non-negative")
    print("                 in all directions")
    
    print("\n✅ SECOND-ORDER SUFFICIENT CONDITION (SOSC):")
    print("  If:")
    print("    1. ∇f(x*) = 0           (critical point)")
    print("    2. ∇²f(x*) ≻ 0          (positive definite)")
    print("  Then:")
    print("    x* is a STRICT local minimum")
    print()
    print("  Interpretation: Positive curvature in all directions")
    print("                 guarantees local bowl shape")
    
    print("\n🔬 CHECKING DEFINITENESS:")
    print("  Method 1 - Eigenvalues:")
    print("    Compute eigenvalues of Hessian")
    print("    PD: all λ > 0,  PSD: all λ ≥ 0")
    print("    ND: all λ < 0,  NSD: all λ ≤ 0")
    print("    Indefinite: mixed signs")
    print()
    print("  Method 2 - Cholesky:")
    print("    Try Cholesky decomposition H = LL^T")
    print("    Success ⟺ Positive definite")
    print()
    print("  Method 3 - Sylvester's Criterion:")
    print("    Check leading principal minors")
    print("    PD: all > 0,  ND: alternating signs starting <0")
    
    print("\n📊 CLASSIFICATION TABLE:")
    print("  ∇f(x*)  |  ∇²f(x*)  |  Classification")
    print("  --------|-----------|----------------")
    print("    ≠ 0   |    any    |  Not critical")
    print("    = 0   |    ≻ 0    |  Strict local min")
    print("    = 0   |    ≺ 0    |  Strict local max")
    print("    = 0   | Indefinite|  Saddle point")
    print("    = 0   |    ⪰ 0    |  Min possible*")
    print("    = 0   |    ⪯ 0    |  Max possible*")
    print()
    print("  * Need higher-order analysis to confirm")
    
    print("\n💡 PRACTICAL IMPLICATIONS:")
    print("1. Newton's method uses Hessian for direction:")
    print("   p = -[∇²f(x)]^{-1} ∇f(x)")
    print("   Requires Hessian positive definite")
    print()
    print("2. Quasi-Newton methods approximate Hessian")
    print("   ensuring positive definiteness")
    print()
    print("3. Trust region methods handle indefinite Hessian")
    print("   by restricting step size")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_2d_critical_points()
    demonstrate_hessian_tests()
    second_order_theory()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- FONC (∇f=0) finds critical points, but can be min/max/saddle")
    print("- SONC (∇²f⪰0) necessary for minimum, but not sufficient")
    print("- SOSC (∇²f≻0) sufficient for strict local minimum")
    print("- Hessian eigenvalues determine critical point type:")
    print("  • All positive → local minimum")
    print("  • All negative → local maximum")
    print("  • Mixed signs → saddle point")
    print("- Cholesky decomposition tests positive definiteness")
    print("\nSecond-order: When the gradient isn't enough! 🎯")
