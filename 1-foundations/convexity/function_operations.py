"""
Convex Function Operations: Building Complex Functions

This module covers the fundamental operations on convex functions that preserve
convexity. These operations are essential for constructing complex objective
functions from simpler building blocks.

Learning Objectives:
- Master convexity-preserving operations on functions
- Understand how to build complex convex functions
- Learn about infimal convolution and its applications
- Explore envelope functions and variational representations

Key Operations:
- Addition: f + g (always preserves convexity)
- Positive scaling: αf for α ≥ 0
- Pointwise maximum: max{f₁, f₂, ...}
- Composition with affine functions: f(Ax + b)
- Composition with convex increasing functions
- Infimal convolution: (f □ g)(x) = inf_y {f(y) + g(x-y)}
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Callable
import warnings
from characterizations import ConvexFunction, QuadraticFunction

warnings.filterwarnings('ignore')

class FunctionOperations:
    """
    Collection of operations on convex functions that preserve convexity.
    These operations allow building complex convex functions from simple ones.
    """
    
    @staticmethod
    def add_functions(functions: List[ConvexFunction], 
                     weights: Optional[np.ndarray] = None) -> 'SumFunction':
        """
        Weighted sum of convex functions: Σᵢ αᵢfᵢ(x) where αᵢ ≥ 0.
        
        Theorem: Positive weighted sums of convex functions are convex.
        
        Args:
            functions: List of convex functions
            weights: Non-negative weights (default: equal weights)
            
        Returns:
            SumFunction representing the weighted sum
        """
        if weights is None:
            weights = np.ones(len(functions))
        
        if not np.all(weights >= 0):
            raise ValueError("Weights must be non-negative")
        
        return SumFunction(functions, weights)
    
    @staticmethod
    def pointwise_maximum(functions: List[ConvexFunction]) -> 'MaxFunction':
        """
        Pointwise maximum of convex functions: max{f₁(x), f₂(x), ...}.
        
        Theorem: The pointwise maximum of convex functions is convex.
        
        Args:
            functions: List of convex functions
            
        Returns:
            MaxFunction representing the pointwise maximum
        """
        return MaxFunction(functions)
    
    @staticmethod
    def compose_affine(func: ConvexFunction, A: np.ndarray, 
                      b: Optional[np.ndarray] = None) -> 'AffineComposition':
        """
        Composition with affine function: f(Ax + b).
        
        Theorem: f(Ax + b) is convex if f is convex.
        
        Args:
            func: Convex function f
            A: Linear transformation matrix
            b: Translation vector (optional)
            
        Returns:
            AffineComposition representing f(Ax + b)
        """
        return AffineComposition(func, A, b)
    
    @staticmethod
    def compose_increasing(inner_func: ConvexFunction, 
                          outer_func: ConvexFunction) -> 'CompositeFunction':
        """
        Composition g(f(x)) where g is convex and increasing, f is convex.
        
        Theorem: If g is convex and increasing, and f is convex,
                then g∘f is convex.
        
        Args:
            inner_func: Inner function f (convex)
            outer_func: Outer function g (convex and increasing)
            
        Returns:
            CompositeFunction representing g(f(x))
        """
        return CompositeFunction(inner_func, outer_func)
    
    @staticmethod
    def infimal_convolution(f: ConvexFunction, g: ConvexFunction) -> 'InfimalConvolution':
        """
        Infimal convolution: (f □ g)(x) = inf_y {f(y) + g(x-y)}.
        
        Theorem: Infimal convolution of convex functions is convex.
        
        Args:
            f, g: Convex functions
            
        Returns:
            InfimalConvolution representing f □ g
        """
        return InfimalConvolution(f, g)


class SumFunction(ConvexFunction):
    """
    Weighted sum of convex functions: h(x) = Σᵢ αᵢfᵢ(x) where αᵢ ≥ 0.
    
    This is the most basic operation preserving convexity and is used
    everywhere in multi-objective optimization and regularization.
    """
    
    def __init__(self, functions: List[ConvexFunction], weights: np.ndarray):
        """Initialize weighted sum of functions."""
        self.functions = functions
        self.weights = np.array(weights)
        
        if len(functions) != len(weights):
            raise ValueError("Number of functions must match number of weights")
        
        def func(x):
            return sum(w * f(x) for w, f in zip(self.weights, self.functions))
        
        def gradient(x):
            return sum(w * f.grad(x) for w, f in zip(self.weights, self.functions))
        
        def hessian(x):
            return sum(w * f.hess(x) for w, f in zip(self.weights, self.functions))
        
        super().__init__(func, gradient, hessian)


class MaxFunction(ConvexFunction):
    """
    Pointwise maximum: h(x) = max{f₁(x), f₂(x), ...}.
    
    The pointwise maximum is fundamental in robust optimization,
    minimax problems, and piecewise linear approximations.
    """
    
    def __init__(self, functions: List[ConvexFunction]):
        """Initialize pointwise maximum of functions."""
        self.functions = functions
        
        def func(x):
            return max(f(x) for f in self.functions)
        
        # Gradient is not everywhere differentiable, but subdifferential exists
        def gradient(x):
            # Find active functions (those achieving the maximum)
            values = [f(x) for f in self.functions]
            max_val = max(values)
            active_indices = [i for i, val in enumerate(values) if abs(val - max_val) < 1e-12]
            
            # Gradient is convex combination of gradients of active functions
            if len(active_indices) == 1:
                return self.functions[active_indices[0]].grad(x)
            else:
                # For simplicity, use uniform weights among active functions
                weights = np.zeros(len(self.functions))
                weights[active_indices] = 1.0 / len(active_indices)
                
                grad = np.zeros_like(x)
                for i, f in enumerate(self.functions):
                    if weights[i] > 0:
                        grad += weights[i] * f.grad(x)
                return grad
        
        super().__init__(func, gradient)


class AffineComposition(ConvexFunction):
    """
    Affine composition: h(x) = f(Ax + b).
    
    This operation is fundamental for change of variables,
    constraints, and building structured optimization problems.
    """
    
    def __init__(self, func: ConvexFunction, A: np.ndarray, 
                 b: Optional[np.ndarray] = None):
        """Initialize affine composition f(Ax + b)."""
        self.func = func
        self.A = np.array(A)
        self.b = np.zeros(A.shape[0]) if b is None else np.array(b)
        
        def composed_func(x):
            return self.func(self.A @ x + self.b)
        
        def composed_gradient(x):
            # Chain rule: ∇h(x) = Aᵀ∇f(Ax + b)
            return self.A.T @ self.func.grad(self.A @ x + self.b)
        
        def composed_hessian(x):
            # Chain rule: ∇²h(x) = Aᵀ∇²f(Ax + b)A
            return self.A.T @ self.func.hess(self.A @ x + self.b) @ self.A
        
        super().__init__(composed_func, composed_gradient, composed_hessian)


class CompositeFunction(ConvexFunction):
    """
    Composite function: h(x) = g(f(x)) where g is convex increasing, f is convex.
    
    This preserves convexity under specific conditions and is useful
    for building complex objective functions.
    """
    
    def __init__(self, inner_func: ConvexFunction, outer_func: ConvexFunction):
        """Initialize composite function g(f(x))."""
        self.inner_func = inner_func  # f
        self.outer_func = outer_func  # g
        
        def composed_func(x):
            inner_val = self.inner_func(x)
            return self.outer_func(np.array([inner_val]))
        
        def composed_gradient(x):
            # Chain rule: ∇h(x) = g'(f(x))∇f(x)
            inner_val = self.inner_func(x)
            outer_grad = self.outer_func.grad(np.array([inner_val]))[0]
            inner_grad = self.inner_func.grad(x)
            return outer_grad * inner_grad
        
        def composed_hessian(x):
            # Second-order chain rule
            inner_val = self.inner_func(x)
            inner_grad = self.inner_func.grad(x)
            inner_hess = self.inner_func.hess(x)
            
            outer_grad = self.outer_func.grad(np.array([inner_val]))[0]
            outer_hess = self.outer_func.hess(np.array([inner_val]))[0, 0]
            
            # ∇²h(x) = g'(f(x))∇²f(x) + g''(f(x))∇f(x)∇f(x)ᵀ
            return (outer_grad * inner_hess + 
                   outer_hess * np.outer(inner_grad, inner_grad))
        
        super().__init__(composed_func, composed_gradient, composed_hessian)


class InfimalConvolution(ConvexFunction):
    """
    Infimal convolution: (f □ g)(x) = inf_y {f(y) + g(x-y)}.
    
    This operation is fundamental in convex analysis and appears in
    regularization, optimal transport, and variational problems.
    """
    
    def __init__(self, f: ConvexFunction, g: ConvexFunction):
        """Initialize infimal convolution f □ g."""
        self.f = f
        self.g = g
        
        def inf_conv_func(x):
            # Approximate infimal convolution using grid search
            # For practical implementation, this would use sophisticated optimization
            y_values = np.linspace(-5, 5, 100)  # Simplified grid
            if len(x) > 1:
                # For multi-dimensional case, this becomes much more complex
                return min(self.f(np.array([y])) + self.g(x - np.array([y])) 
                          for y in y_values)
            else:
                return min(self.f(np.array([y])) + self.g(x - np.array([y])) 
                          for y in y_values)
        
        super().__init__(inf_conv_func)


class PerspectiveFunction(ConvexFunction):
    """
    Perspective function: P(x, t) = t·f(x/t) for t > 0.
    
    The perspective function preserves convexity and is fundamental
    in conic optimization and financial mathematics.
    """
    
    def __init__(self, func: ConvexFunction):
        """Initialize perspective of function f."""
        self.func = func
        
        def perspective(xt):
            # xt = [x₁, ..., xₙ, t] where x ∈ Rⁿ, t ∈ R
            x = xt[:-1]
            t = xt[-1]
            
            if t <= 0:
                return np.inf
            
            return t * self.func(x / t)
        
        def perspective_gradient(xt):
            x = xt[:-1]
            t = xt[-1]
            
            if t <= 0:
                return np.full_like(xt, np.inf)
            
            f_val = self.func(x / t)
            f_grad = self.func.grad(x / t)
            
            # Gradient w.r.t. x: ∇ₓP(x,t) = ∇f(x/t)
            grad_x = f_grad
            
            # Gradient w.r.t. t: ∇ₜP(x,t) = f(x/t) - (x/t)ᵀ∇f(x/t)
            grad_t = f_val - np.dot(x / t, f_grad)
            
            return np.concatenate([grad_x, [grad_t]])
        
        super().__init__(perspective, perspective_gradient)


def demonstrate_function_operations():
    """
    Demonstrate convex function operations with visual examples.
    """
    print("🔷 CONVEX FUNCTION OPERATIONS: Building Complex Functions")
    print("=" * 70)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Addition of Functions
    ax1 = plt.subplot(2, 3, 1)
    
    x_vals = np.linspace(-3, 3, 100)
    
    # Two quadratic functions
    f1 = QuadraticFunction(np.array([[1.0]]), np.array([0.0]), 0.0)
    f2 = QuadraticFunction(np.array([[0.5]]), np.array([1.0]), 0.5)
    
    # Sum function
    sum_func = FunctionOperations.add_functions([f1, f2], np.array([1.0, 1.0]))
    
    y1_vals = [f1(np.array([x])) for x in x_vals]
    y2_vals = [f2(np.array([x])) for x in x_vals]
    y_sum_vals = [sum_func(np.array([x])) for x in x_vals]
    
    ax1.plot(x_vals, y1_vals, 'blue', linewidth=2, label='f₁(x) = x²')
    ax1.plot(x_vals, y2_vals, 'red', linewidth=2, label='f₂(x) = ½(x+1)² + 0.5')
    ax1.plot(x_vals, y_sum_vals, 'green', linewidth=3, label='f₁ + f₂')
    
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 8)
    ax1.set_title('Addition of Convex Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Pointwise Maximum
    ax2 = plt.subplot(2, 3, 2)
    
    # Multiple linear functions (convex)
    functions = []
    colors = ['blue', 'red', 'green', 'orange']
    slopes = [-1, 0.5, -0.5, 1]
    intercepts = [2, -1, 3, -2]
    
    for i, (slope, intercept) in enumerate(zip(slopes, intercepts)):
        # Linear function: f(x) = slope * x + intercept
        linear_func = QuadraticFunction(np.array([[0.0]]), np.array([slope]), intercept)
        functions.append(linear_func)
        
        y_vals = [linear_func(np.array([x])) for x in x_vals]
        ax2.plot(x_vals, y_vals, colors[i], linewidth=2, alpha=0.7, 
                label=f'f{i+1}(x) = {slope}x + {intercept}')
    
    # Pointwise maximum
    max_func = FunctionOperations.pointwise_maximum(functions)
    y_max_vals = [max_func(np.array([x])) for x in x_vals]
    
    ax2.plot(x_vals, y_max_vals, 'black', linewidth=3, label='max{fᵢ}')
    
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 5)
    ax2.set_title('Pointwise Maximum (Piecewise Linear)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Affine Composition
    ax3 = plt.subplot(2, 3, 3)
    
    # Original function: f(x) = x²
    original_func = QuadraticFunction(np.array([[1.0]]), np.array([0.0]), 0.0)
    
    # Affine transformations
    transformations = [
        (np.array([[1.0]]), np.array([0.0]), 'f(x) = x²'),
        (np.array([[2.0]]), np.array([0.0]), 'f(2x)'),
        (np.array([[1.0]]), np.array([1.0]), 'f(x+1)'),
        (np.array([[0.5]]), np.array([-0.5]), 'f(0.5x-0.5)'),
    ]
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (A, b, label) in enumerate(transformations):
        if i == 0:
            # Original function
            y_vals = [original_func(np.array([x])) for x in x_vals]
        else:
            # Composed function
            composed = FunctionOperations.compose_affine(original_func, A, b)
            y_vals = [composed(np.array([x])) for x in x_vals]
        
        ax3.plot(x_vals, y_vals, colors[i], linewidth=2, label=label)
    
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(0, 6)
    ax3.set_title('Affine Composition: f(Ax + b)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Composition with Increasing Function
    ax4 = plt.subplot(2, 3, 4)
    
    # Inner function: f(x) = x² (convex)
    inner = QuadraticFunction(np.array([[1.0]]), np.array([0.0]), 0.0)
    
    # Outer function: g(t) = √t (concave but increasing for t≥0)
    # Instead, use g(t) = t² (convex and increasing for t≥0)
    outer = QuadraticFunction(np.array([[1.0]]), np.array([0.0]), 0.0)
    
    # Composite: h(x) = g(f(x)) = (x²)² = x⁴
    composite = FunctionOperations.compose_increasing(inner, outer)
    
    x_vals_pos = np.linspace(-2, 2, 100)
    y_inner = [inner(np.array([x])) for x in x_vals_pos]
    y_composite = [x**4 for x in x_vals_pos]  # Direct computation for comparison
    y_composite_func = [composite(np.array([x])) for x in x_vals_pos]
    
    ax4.plot(x_vals_pos, y_inner, 'blue', linewidth=2, label='f(x) = x²')
    ax4.plot(x_vals_pos, y_composite, 'red', linewidth=2, label='h(x) = x⁴', linestyle='--')
    ax4.plot(x_vals_pos, y_composite_func, 'green', linewidth=2, label='g(f(x))')
    
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(0, 8)
    ax4.set_title('Composition: g(f(x)) = (x²)²')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Regularization Example
    ax5 = plt.subplot(2, 3, 5)
    
    # Data fitting + regularization: f(x) = ||Ax - b||² + λ||x||²
    # This is sum of two convex functions
    
    # Simulate data fitting term ||Ax - b||²
    A_matrix = np.array([[1.0]])
    b_vector = np.array([1.0])
    
    def data_fit_func(x):
        return np.linalg.norm(A_matrix @ x - b_vector)**2
    
    data_fit = ConvexFunction(data_fit_func)
    
    # Regularization term λ||x||²
    lambda_reg = 0.5
    regularizer = QuadraticFunction(np.array([[lambda_reg]]), np.array([0.0]), 0.0)
    
    # Total objective
    total_objective = FunctionOperations.add_functions([data_fit, regularizer], 
                                                      np.array([1.0, 1.0]))
    
    x_vals = np.linspace(-1, 3, 100)
    y_data = [data_fit(np.array([x])) for x in x_vals]
    y_reg = [regularizer(np.array([x])) for x in x_vals]
    y_total = [total_objective(np.array([x])) for x in x_vals]
    
    ax5.plot(x_vals, y_data, 'blue', linewidth=2, label='||Ax - b||²')
    ax5.plot(x_vals, y_reg, 'red', linewidth=2, label='λ||x||²')
    ax5.plot(x_vals, y_total, 'green', linewidth=3, label='Total objective')
    
    # Show minimum
    min_idx = np.argmin(y_total)
    ax5.plot(x_vals[min_idx], y_total[min_idx], 'ro', markersize=8, label='Minimum')
    
    ax5.set_xlim(-1, 3)
    ax5.set_ylim(0, 6)
    ax5.set_title('Regularized Least Squares')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Perspective Function
    ax6 = plt.subplot(2, 3, 6, projection='3d')
    
    # 2D visualization of perspective function
    # f(x) = x², P(x,t) = t·f(x/t) = t·(x/t)² = x²/t
    
    def simple_quadratic(x):
        return x[0]**2
    
    original_2d = ConvexFunction(simple_quadratic)
    perspective = PerspectiveFunction(original_2d)
    
    x_range = np.linspace(-2, 2, 30)
    t_range = np.linspace(0.1, 2, 30)
    X, T = np.meshgrid(x_range, t_range)
    
    Z = X**2 / T  # Perspective function values
    
    ax6.plot_surface(X, T, Z, alpha=0.7, cmap='viridis')
    
    ax6.set_xlabel('x')
    ax6.set_ylabel('t')
    ax6.set_zlabel('P(x,t) = x²/t')
    ax6.set_title('Perspective Function')
    
    plt.tight_layout()
    plt.show()
    
    # Numerical verification
    print("\n📊 NUMERICAL VERIFICATION")
    print("-" * 40)
    
    # Test operation properties
    print("1. Function Addition Properties:")
    f1 = QuadraticFunction(np.array([[2.0]]))
    f2 = QuadraticFunction(np.array([[1.0]]), np.array([1.0]))
    sum_func = FunctionOperations.add_functions([f1, f2])
    
    test_point = np.array([1.5])
    val1 = f1(test_point)
    val2 = f2(test_point)
    val_sum = sum_func(test_point)
    
    print(f"   f₁(1.5) = {val1:.3f}")
    print(f"   f₂(1.5) = {val2:.3f}")
    print(f"   (f₁ + f₂)(1.5) = {val_sum:.3f}")
    print(f"   Sum matches: {abs(val_sum - (val1 + val2)) < 1e-10}")
    
    print("\n2. Composition Chain Rule:")
    # Test gradient of composition
    A = np.array([[2.0]])
    composed = FunctionOperations.compose_affine(f1, A)
    
    # Gradient: d/dx f(2x) = 2 * f'(2x)
    grad_direct = 2 * f1.grad(2 * test_point)
    grad_composed = composed.grad(test_point)
    
    print(f"   Direct gradient: {grad_direct}")
    print(f"   Composed gradient: {grad_composed}")
    print(f"   Chain rule matches: {np.allclose(grad_direct, grad_composed)}")


def applications_showcase():
    """
    Show applications of function operations in optimization.
    """
    print("\n🎯 APPLICATIONS OF FUNCTION OPERATIONS")
    print("=" * 45)
    
    print("1. REGULARIZED OPTIMIZATION:")
    print("   - Ridge regression: ||Ax - b||² + λ||x||²")
    print("   - LASSO: ||Ax - b||² + λ||x||₁")
    print("   - Elastic net: ||Ax - b||² + λ₁||x||₁ + λ₂||x||²")
    print("   → Addition of convex functions")
    
    print("\n2. ROBUST OPTIMIZATION:")
    print("   - Worst-case objective: max_{ξ∈U} f(x, ξ)")
    print("   - Chance constraints: max probability")
    print("   - Minimax problems: min_x max_y f(x,y)")
    print("   → Pointwise maximum operations")
    
    print("\n3. MULTI-OBJECTIVE OPTIMIZATION:")
    print("   - Weighted sum: Σᵢ wᵢfᵢ(x)")
    print("   - Scalarization methods")
    print("   - Pareto efficiency via weighted sums")
    print("   → Positive weighted combinations")
    
    print("\n4. MACHINE LEARNING:")
    print("   - Loss + regularization: L(θ) + λR(θ)")
    print("   - Neural networks: composition of activations")
    print("   - Support vector machines: hinge loss + regularization")
    print("   → Function composition and addition")
    
    print("\n5. FINANCIAL OPTIMIZATION:")
    print("   - Portfolio risk: x^T Σ x (quadratic)")
    print("   - Transaction costs: perspective functions")
    print("   - Robust portfolio: worst-case scenarios")
    print("   → Complex function constructions")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_function_operations()
    applications_showcase()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Addition: Always preserves convexity with non-negative weights")
    print("- Maximum: Creates piecewise convex functions (robust optimization)")
    print("- Affine composition: f(Ax + b) preserves convexity")
    print("- Increasing composition: g(f(x)) convex if g convex increasing")
    print("- Perspective: Fundamental in conic optimization")
    print("- These operations are building blocks for complex objectives!")
    print("\nNext: Explore convex conjugate functions and duality! 🚀")