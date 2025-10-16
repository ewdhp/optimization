"""
Convex Functions: Characterizations and Properties

This module covers the fundamental theory of convex functions, which are the
objective functions that make optimization problems tractable. Understanding
convex functions is crucial for recognizing when problems have nice properties.

Learning Objectives:
- Master multiple characterizations of convex functions
- Understand first and second-order conditions for convexity
- Learn about epigraphs and their geometric interpretation
- Explore Jensen's inequality and its applications

Key Concepts:
- Definition via Jensen's inequality
- First-order characterization (supporting hyperplanes)
- Second-order characterization (positive semidefinite Hessian)
- Epigraph characterization (convex set above function)
- Strong convexity and smoothness
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Callable
import warnings
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')

class ConvexFunction:
    """
    Base class for convex functions with characterization methods.
    
    This class provides a framework for working with convex functions and
    includes methods for testing convexity using different characterizations.
    """
    
    def __init__(self, func: Callable[[np.ndarray], float],
                 gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 domain: Optional[Callable[[np.ndarray], bool]] = None):
        """
        Initialize convex function.
        
        Args:
            func: Function f(x) to evaluate
            gradient: Gradient function ∇f(x) (optional)
            hessian: Hessian function ∇²f(x) (optional)
            domain: Domain indicator function (optional)
        """
        self.func = func
        self.gradient = gradient
        self.hessian = hessian
        self.domain = domain if domain is not None else lambda x: True
    
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate function at point x."""
        x = np.array(x)
        if not self.domain(x):
            return np.inf
        return self.func(x)
    
    def grad(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient at point x."""
        if self.gradient is None:
            # Numerical gradient
            return self._numerical_gradient(x)
        return self.gradient(x)
    
    def hess(self, x: np.ndarray) -> np.ndarray:
        """Compute Hessian at point x."""
        if self.hessian is None:
            # Numerical Hessian
            return self._numerical_hessian(x)
        return self.hessian(x)
    
    def _numerical_gradient(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Compute numerical gradient using finite differences."""
        x = np.array(x, dtype=float)
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            grad[i] = (self(x_plus) - self(x_minus)) / (2 * eps)
        
        return grad
    
    def _numerical_hessian(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute numerical Hessian using finite differences."""
        x = np.array(x, dtype=float)
        n = len(x)
        hess = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal element: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h))/h²
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += eps
                    x_minus[i] -= eps
                    
                    hess[i, i] = (self(x_plus) - 2*self(x) + self(x_minus)) / (eps**2)
                else:
                    # Off-diagonal: mixed partial derivative
                    x_pp = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_mm = x.copy()
                    
                    x_pp[[i, j]] += eps
                    x_pm[i] += eps; x_pm[j] -= eps
                    x_mp[i] -= eps; x_mp[j] += eps
                    x_mm[[i, j]] -= eps
                    
                    hess[i, j] = (self(x_pp) - self(x_pm) - self(x_mp) + self(x_mm)) / (4 * eps**2)
        
        return hess
    
    def is_convex_jensen(self, x1: np.ndarray, x2: np.ndarray, 
                        num_tests: int = 100, tol: float = 1e-10) -> bool:
        """
        Test convexity using Jensen's inequality.
        
        Test: f(λx₁ + (1-λ)x₂) ≤ λf(x₁) + (1-λ)f(x₂) for λ ∈ [0,1]
        """
        violations = 0
        
        for _ in range(num_tests):
            lam = np.random.random()
            x_combo = lam * x1 + (1 - lam) * x2
            
            lhs = self(x_combo)
            rhs = lam * self(x1) + (1 - lam) * self(x2)
            
            if lhs > rhs + tol:
                violations += 1
        
        return violations == 0
    
    def is_convex_first_order(self, x1: np.ndarray, x2: np.ndarray, 
                             tol: float = 1e-10) -> bool:
        """
        Test convexity using first-order condition.
        
        Test: f(x₂) ≥ f(x₁) + ∇f(x₁)ᵀ(x₂ - x₁)
        """
        try:
            grad_x1 = self.grad(x1)
            
            lhs = self(x2)
            rhs = self(x1) + np.dot(grad_x1, x2 - x1)
            
            return lhs >= rhs - tol
        except:
            return False
    
    def is_convex_second_order(self, x: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Test convexity using second-order condition.
        
        Test: ∇²f(x) ⪰ 0 (positive semidefinite)
        """
        try:
            hess_x = self.hess(x)
            eigenvals = np.linalg.eigvals(hess_x)
            return np.all(eigenvals >= -tol)
        except:
            return False
    
    def epigraph_points(self, x_range: Tuple[float, float], 
                       num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate points on the epigraph boundary for visualization.
        
        Returns:
            x_values, function_values for plotting
        """
        x_min, x_max = x_range
        x_values = np.linspace(x_min, x_max, num_points)
        
        if len(x_values[0:1]) == 1:  # 1D function
            func_values = [self(np.array([x])) for x in x_values]
        else:
            func_values = [self(x) for x in x_values]
        
        return x_values, np.array(func_values)


class QuadraticFunction(ConvexFunction):
    """
    Quadratic function: f(x) = ½xᵀQx + cᵀx + d
    
    The most important class of convex functions, appearing everywhere
    in optimization, statistics, and engineering.
    """
    
    def __init__(self, Q: np.ndarray, c: Optional[np.ndarray] = None, d: float = 0.0):
        """
        Initialize quadratic function ½xᵀQx + cᵀx + d.
        
        Args:
            Q: Quadratic term matrix (should be positive semidefinite for convexity)
            c: Linear term vector (optional, default: zero)
            d: Constant term (optional, default: 0)
        """
        self.Q = np.array(Q)
        self.c = np.zeros(Q.shape[0]) if c is None else np.array(c)
        self.d = float(d)
        
        # Verify matrix dimensions
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be square matrix")
        if len(self.c) != Q.shape[0]:
            raise ValueError("c must have same dimension as Q")
        
        # Symmetrize Q for numerical stability
        self.Q = (self.Q + self.Q.T) / 2
        
        def func(x):
            x = np.array(x)
            return 0.5 * np.dot(x, np.dot(self.Q, x)) + np.dot(self.c, x) + self.d
        
        def gradient(x):
            x = np.array(x)
            return np.dot(self.Q, x) + self.c
        
        def hessian(x):
            return self.Q
        
        super().__init__(func, gradient, hessian)
    
    def is_convex(self) -> bool:
        """Check if quadratic function is convex (Q ⪰ 0)."""
        eigenvals = np.linalg.eigvals(self.Q)
        return np.all(eigenvals >= -1e-12)
    
    def is_strongly_convex(self) -> Tuple[bool, float]:
        """
        Check if function is strongly convex and return strong convexity parameter.
        
        Returns:
            (is_strongly_convex, mu) where mu is the strong convexity parameter
        """
        eigenvals = np.linalg.eigvals(self.Q)
        mu = np.min(eigenvals)
        return mu > 1e-12, max(mu, 0.0)
    
    def condition_number(self) -> float:
        """Compute condition number of the Hessian."""
        eigenvals = np.linalg.eigvals(self.Q)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove near-zero eigenvalues
        
        if len(eigenvals) == 0:
            return np.inf
        
        return np.max(eigenvals) / np.min(eigenvals)


class NormFunction(ConvexFunction):
    """
    Norm function: f(x) = ||x||_p
    
    Norms are fundamental convex functions that appear in regularization,
    robust optimization, and geometry.
    """
    
    def __init__(self, p: Union[float, str] = 2):
        """
        Initialize p-norm function.
        
        Args:
            p: Norm type (1, 2, 'inf', or float > 1)
        """
        self.p = p
        
        if p == 1:
            func = lambda x: np.sum(np.abs(x))
            # Gradient is not differentiable at zero, but subdifferential exists
            gradient = None
        elif p == 2:
            func = lambda x: np.linalg.norm(x)
            gradient = lambda x: x / (np.linalg.norm(x) + 1e-12)  # Add small epsilon
        elif p == 'inf':
            func = lambda x: np.max(np.abs(x))
            gradient = None  # Not differentiable
        elif isinstance(p, (int, float)) and p > 1:
            func = lambda x: np.sum(np.abs(x)**p)**(1/p)
            # For p > 1, gradient exists except at zero
            def grad(x):
                norm_val = func(x)
                if norm_val < 1e-12:
                    return np.zeros_like(x)
                return np.sign(x) * np.abs(x)**(p-1) * (norm_val**(1-p))
            gradient = grad
        else:
            raise ValueError("p must be >= 1 or 'inf'")
        
        super().__init__(func, gradient)


class ExponentialFunction(ConvexFunction):
    """
    Exponential function: f(x) = eᵃˣ (univariate) or f(x) = Σᵢ eᵃⁱˣⁱ
    
    Exponential functions are convex and appear in entropy, maximum likelihood,
    and logistic regression.
    """
    
    def __init__(self, a: Union[float, np.ndarray] = 1.0):
        """
        Initialize exponential function.
        
        Args:
            a: Scaling parameter(s)
        """
        if np.isscalar(a):
            self.a = float(a)
            func = lambda x: np.exp(self.a * x[0]) if len(x) == 1 else np.exp(self.a * np.sum(x))
            gradient = lambda x: self.a * np.exp(self.a * x[0]) * np.ones_like(x) if len(x) == 1 else self.a * np.exp(self.a * np.sum(x)) * np.ones_like(x)
            hessian = lambda x: self.a**2 * np.exp(self.a * x[0]) * np.ones((len(x), len(x))) if len(x) == 1 else self.a**2 * np.exp(self.a * np.sum(x)) * np.ones((len(x), len(x)))
        else:
            self.a = np.array(a)
            func = lambda x: np.sum(np.exp(self.a * x))
            gradient = lambda x: self.a * np.exp(self.a * x)
            hessian = lambda x: np.diag(self.a**2 * np.exp(self.a * x))
        
        super().__init__(func, gradient, hessian)


class LogSumExpFunction(ConvexFunction):
    """
    Log-sum-exp function: f(x) = log(Σᵢ eˣⁱ)
    
    This is a smooth approximation to the max function and appears in
    machine learning (softmax) and convex optimization.
    """
    
    def __init__(self):
        """Initialize log-sum-exp function."""
        
        def func(x):
            x = np.array(x)
            # Numerically stable computation
            x_max = np.max(x)
            return x_max + np.log(np.sum(np.exp(x - x_max)))
        
        def gradient(x):
            x = np.array(x)
            exp_x = np.exp(x - np.max(x))  # Numerical stability
            return exp_x / np.sum(exp_x)
        
        def hessian(x):
            x = np.array(x)
            exp_x = np.exp(x - np.max(x))
            sum_exp = np.sum(exp_x)
            prob = exp_x / sum_exp
            
            # Hessian is diag(prob) - prob * prob^T
            return np.diag(prob) - np.outer(prob, prob)
        
        super().__init__(func, gradient, hessian)


def demonstrate_convex_functions():
    """
    Demonstrate convex function theory with visual examples.
    """
    print("🔷 CONVEX FUNCTIONS: Theory and Characterizations")
    print("=" * 60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Jensen's Inequality Illustration
    ax1 = plt.subplot(2, 3, 1)
    
    # Convex function: f(x) = x²
    x_vals = np.linspace(-2, 2, 100)
    y_vals = x_vals**2
    
    ax1.plot(x_vals, y_vals, 'blue', linewidth=2, label='f(x) = x²')
    
    # Show Jensen's inequality
    x1, x2 = -1.5, 1.0
    y1, y2 = x1**2, x2**2
    lam = 0.3
    
    # Points on function
    ax1.plot(x1, y1, 'ro', markersize=8, label='f(x₁)')
    ax1.plot(x2, y2, 'ro', markersize=8, label='f(x₂)')
    
    # Convex combination point
    x_combo = lam * x1 + (1 - lam) * x2
    y_combo = x_combo**2
    y_linear = lam * y1 + (1 - lam) * y2
    
    ax1.plot(x_combo, y_combo, 'go', markersize=8, label='f(λx₁ + (1-λ)x₂)')
    ax1.plot(x_combo, y_linear, 'bo', markersize=8, label='λf(x₁) + (1-λ)f(x₂)')
    
    # Connect points to show inequality
    ax1.plot([x1, x2], [y1, y2], 'r--', alpha=0.7, linewidth=2)
    ax1.plot([x_combo, x_combo], [y_combo, y_linear], 'g-', linewidth=3, alpha=0.8)
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(0, 4)
    ax1.set_title('Jensen\'s Inequality')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. First-Order Condition
    ax2 = plt.subplot(2, 3, 2)
    
    # Convex function and its tangent
    quadratic = QuadraticFunction(np.array([[2.0]]), np.array([0.0]), 0.0)
    
    x_vals = np.linspace(-2, 2, 100)
    y_vals = [quadratic(np.array([x])) for x in x_vals]
    
    ax2.plot(x_vals, y_vals, 'blue', linewidth=2, label='f(x) = x²')
    
    # Point and tangent line
    x0 = -0.5
    y0 = quadratic(np.array([x0]))
    grad0 = quadratic.grad(np.array([x0]))[0]
    
    # Tangent line: y = f(x0) + f'(x0)(x - x0)
    tangent_y = y0 + grad0 * (x_vals - x0)
    ax2.plot(x_vals, tangent_y, 'red', linewidth=2, linestyle='--', 
             label='Tangent line')
    ax2.plot(x0, y0, 'ro', markersize=8, label='Point (x₀, f(x₀))')
    
    # Show that function lies above tangent
    ax2.fill_between(x_vals, y_vals, tangent_y, where=(np.array(y_vals) >= tangent_y), 
                     alpha=0.3, color='green', label='f(x) ≥ tangent')
    
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(0, 4)
    ax2.set_title('First-Order Condition')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Second-Order Condition (Hessian)
    ax3 = plt.subplot(2, 3, 3)
    
    # Show different quadratic functions with different curvatures
    x_vals = np.linspace(-2, 2, 100)
    
    # Convex: positive definite Hessian
    Q_convex = np.array([[2.0]])
    convex_func = QuadraticFunction(Q_convex)
    y_convex = [convex_func(np.array([x])) for x in x_vals]
    ax3.plot(x_vals, y_convex, 'green', linewidth=2, label='Convex (Q > 0)')
    
    # Non-convex: negative definite Hessian
    Q_concave = np.array([[-1.0]])
    concave_func = QuadraticFunction(Q_concave)
    y_concave = [concave_func(np.array([x])) for x in x_vals]
    ax3.plot(x_vals, y_concave, 'red', linewidth=2, label='Concave (Q < 0)')
    
    # Flat: zero Hessian
    Q_flat = np.array([[0.0]])
    flat_func = QuadraticFunction(Q_flat, np.array([1.0]), 0.0)
    y_flat = [flat_func(np.array([x])) for x in x_vals]
    ax3.plot(x_vals, y_flat, 'blue', linewidth=2, label='Linear (Q = 0)')
    
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 4)
    ax3.set_title('Second-Order Condition')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Epigraph of Convex Function
    ax4 = plt.subplot(2, 3, 4)
    
    # 3D plot of epigraph
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    
    # Create 2D quadratic function for visualization
    x_range = np.linspace(-2, 2, 30)
    y_range = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Function: f(x,y) = x² + y²
    Z = X**2 + Y**2
    
    # Plot function surface
    ax4.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
    
    # Show epigraph by adding points above surface
    Z_upper = Z + 2
    ax4.plot_surface(X, Y, Z_upper, alpha=0.3, color='lightblue')
    
    ax4.set_xlabel('x₁')
    ax4.set_ylabel('x₂')
    ax4.set_zlabel('t')
    ax4.set_title('Epigraph: {(x,t) : t ≥ f(x)}')
    
    # 5. Norm Functions
    ax5 = plt.subplot(2, 3, 5)
    
    # Different norm functions in 2D
    theta = np.linspace(0, 2*np.pi, 100)
    
    # L1 norm level set
    l1_norm = NormFunction(1)
    # For level set ||x||₁ = 1, we use parametric form
    t_vals = np.linspace(0, 4, 100)
    l1_x = np.concatenate([1-t_vals[t_vals<=1], -(t_vals[t_vals<=1]-1)[::-1], 
                          -(1-t_vals[t_vals<=1]), (t_vals[t_vals<=1]-1)[::-1]])
    l1_y = np.concatenate([t_vals[t_vals<=1], (1-t_vals[t_vals<=1])[::-1], 
                          -t_vals[t_vals<=1], -(1-t_vals[t_vals<=1])[::-1]])
    
    # Simplified: just plot unit balls
    l1_vertices = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    ax5.plot(l1_vertices[:, 0], l1_vertices[:, 1], 'red', linewidth=2, label='||x||₁ = 1')
    
    # L2 norm level set (circle)
    l2_x = np.cos(theta)
    l2_y = np.sin(theta)
    ax5.plot(l2_x, l2_y, 'blue', linewidth=2, label='||x||₂ = 1')
    
    # L∞ norm level set (square)
    linf_vertices = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])
    ax5.plot(linf_vertices[:, 0], linf_vertices[:, 1], 'green', linewidth=2, label='||x||∞ = 1')
    
    ax5.set_xlim(-1.5, 1.5)
    ax5.set_ylim(-1.5, 1.5)
    ax5.set_title('Level Sets of Norm Functions')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # 6. Log-Sum-Exp Function
    ax6 = plt.subplot(2, 3, 6)
    
    # Compare max function with log-sum-exp approximation
    x1_vals = np.linspace(-3, 3, 100)
    x2_fixed = 0.0
    
    # Max function
    max_vals = np.maximum(x1_vals, x2_fixed)
    
    # Log-sum-exp with different temperature parameters
    logsumexp = LogSumExpFunction()
    
    for scale in [0.1, 0.5, 1.0, 2.0]:
        lse_vals = []
        for x1 in x1_vals:
            x = np.array([x1/scale, x2_fixed/scale])
            lse_vals.append(scale * logsumexp(x))
        
        ax6.plot(x1_vals, lse_vals, linewidth=2, alpha=0.7, 
                label=f'LSE (τ={scale})')
    
    ax6.plot(x1_vals, max_vals, 'black', linewidth=3, linestyle='--', 
             label='max(x₁, 0)')
    
    ax6.set_xlim(-3, 3)
    ax6.set_ylim(-1, 3)
    ax6.set_title('Log-Sum-Exp Approximation to Max')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Numerical verification of properties
    print("\n📊 NUMERICAL VERIFICATION OF CONVEXITY")
    print("-" * 50)
    
    # Test different characterizations
    print("1. Quadratic Function Tests:")
    Q = np.array([[2, 0.5], [0.5, 1]])
    quadratic_2d = QuadraticFunction(Q)
    
    print(f"   Matrix Q eigenvalues: {np.linalg.eigvals(Q)}")
    print(f"   Is convex (eigenvalues): {quadratic_2d.is_convex()}")
    print(f"   Condition number: {quadratic_2d.condition_number():.3f}")
    
    is_strongly_convex, mu = quadratic_2d.is_strongly_convex()
    print(f"   Is strongly convex: {is_strongly_convex} (μ = {mu:.3f})")
    
    # Test Jensen's inequality
    x1 = np.array([1.0, 0.5])
    x2 = np.array([-0.5, 1.0])
    jensen_test = quadratic_2d.is_convex_jensen(x1, x2, num_tests=1000)
    print(f"   Jensen's inequality test: {jensen_test}")
    
    # Test first-order condition
    first_order_test = quadratic_2d.is_convex_first_order(x1, x2)
    print(f"   First-order condition test: {first_order_test}")
    
    # Test second-order condition
    second_order_test = quadratic_2d.is_convex_second_order(x1)
    print(f"   Second-order condition test: {second_order_test}")
    
    print("\n2. Norm Function Tests:")
    
    # Test different norms
    norms = [NormFunction(1), NormFunction(2), NormFunction('inf')]
    norm_names = ['L1', 'L2', 'L∞']
    test_point = np.array([1.0, -0.5])
    
    for norm_func, name in zip(norms, norm_names):
        value = norm_func(test_point)
        print(f"   {name} norm of {test_point}: {value:.3f}")
    
    print("\n3. Log-Sum-Exp Properties:")
    lse = LogSumExpFunction()
    test_vector = np.array([1.0, 2.0, 0.5])
    
    lse_value = lse(test_vector)
    max_value = np.max(test_vector)
    
    print(f"   Input vector: {test_vector}")
    print(f"   Max value: {max_value:.3f}")
    print(f"   Log-sum-exp: {lse_value:.3f}")
    print(f"   Difference: {lse_value - max_value:.3f}")
    
    # Gradient (should be softmax)
    gradient = lse.grad(test_vector)
    print(f"   Gradient (softmax): {gradient}")
    print(f"   Gradient sum: {np.sum(gradient):.6f} (should be 1)")


def verify_function_properties():
    """
    Verify key properties of convex functions through computation.
    """
    print("\n🔍 VERIFICATION OF CONVEX FUNCTION PROPERTIES")
    print("=" * 55)
    
    # Property 1: Composition with increasing convex function
    print("1. Composition Property:")
    
    # f(x) = ||x||₂ (convex), g(t) = t² (convex, increasing for t≥0)
    # h(x) = g(f(x)) = ||x||₂² should be convex
    
    def norm_squared(x):
        return np.linalg.norm(x)**2
    
    # Test Jensen's inequality for h(x) = ||x||₂²
    x1 = np.array([1.0, 0.5])
    x2 = np.array([-0.5, 1.5])
    
    violations = 0
    for _ in range(100):
        lam = np.random.random()
        x_combo = lam * x1 + (1 - lam) * x2
        
        lhs = norm_squared(x_combo)
        rhs = lam * norm_squared(x1) + (1 - lam) * norm_squared(x2)
        
        if lhs > rhs + 1e-10:
            violations += 1
    
    print(f"   Composition h(x) = ||x||₂² Jensen violations: {violations}/100")
    
    # Property 2: Pointwise maximum preserves convexity
    print("\n2. Pointwise Maximum Property:")
    
    # f₁(x) = x², f₂(x) = (x-1)², h(x) = max(f₁(x), f₂(x))
    def f1(x): return x**2
    def f2(x): return (x-1)**2
    def h_max(x): return max(f1(x), f2(x))
    
    # Test convexity of pointwise max
    violations = 0
    for _ in range(100):
        x1_scalar = np.random.uniform(-2, 3)
        x2_scalar = np.random.uniform(-2, 3)
        lam = np.random.random()
        
        x_combo = lam * x1_scalar + (1 - lam) * x2_scalar
        
        lhs = h_max(x_combo)
        rhs = lam * h_max(x1_scalar) + (1 - lam) * h_max(x2_scalar)
        
        if lhs > rhs + 1e-10:
            violations += 1
    
    print(f"   Pointwise max Jensen violations: {violations}/100")
    
    # Property 3: Perspective function preserves convexity
    print("\n3. Perspective Function Property:")
    
    # If f is convex, then g(x,t) = t·f(x/t) is convex on {(x,t) : t > 0}
    # Example: f(x) = x², g(x,t) = t·(x/t)² = x²/t
    
    def perspective_func(x, t):
        if t <= 0:
            return np.inf
        return (x**2) / t
    
    # Test Jensen's inequality
    violations = 0
    for _ in range(100):
        # Generate positive t values
        t1 = np.random.uniform(0.1, 2.0)
        t2 = np.random.uniform(0.1, 2.0)
        x1_scalar = np.random.uniform(-2, 2)
        x2_scalar = np.random.uniform(-2, 2)
        
        lam = np.random.random()
        
        x_combo = lam * x1_scalar + (1 - lam) * x2_scalar
        t_combo = lam * t1 + (1 - lam) * t2
        
        lhs = perspective_func(x_combo, t_combo)
        rhs = lam * perspective_func(x1_scalar, t1) + (1 - lam) * perspective_func(x2_scalar, t2)
        
        if lhs > rhs + 1e-10:
            violations += 1
    
    print(f"   Perspective function Jensen violations: {violations}/100")
    
    # Property 4: Strong convexity implies unique minimum
    print("\n4. Strong Convexity Property:")
    
    # Strongly convex function: f(x) = x² + 0.1||x||²
    mu = 0.1  # Strong convexity parameter
    Q_strong = np.eye(2) * (1 + mu)
    strongly_convex = QuadraticFunction(Q_strong)
    
    is_strong, mu_computed = strongly_convex.is_strongly_convex()
    print(f"   Is strongly convex: {is_strong}")
    print(f"   Strong convexity parameter μ: {mu_computed:.3f}")
    
    # Find minimum (should be unique at origin)
    gradient_at_origin = strongly_convex.grad(np.array([0.0, 0.0]))
    print(f"   Gradient at origin: {gradient_at_origin}")
    print(f"   Minimum is at origin: {np.allclose(gradient_at_origin, 0)}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_convex_functions()
    verify_function_properties()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Jensen's inequality: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)")
    print("- First-order: f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)")
    print("- Second-order: ∇²f(x) ⪰ 0")
    print("- Epigraph characterization connects sets and functions")
    print("- Operations preserving convexity: +, max, composition, perspective")
    print("- Strong convexity guarantees unique global minimum")
    print("\nNext: Explore convex function operations and transformations! 🚀")