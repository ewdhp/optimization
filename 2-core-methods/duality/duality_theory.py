"""
Duality Theory in Optimization

Duality theory provides fundamental insights into optimization by associating
with every optimization problem (the primal) a related problem (the dual).
This relationship offers computational advantages and theoretical insights.

Mathematical Foundation:

Primal Problem (P):
    minimize f(x)
    subject to g_i(x) ≤ 0, i = 1,...,m
               h_j(x) = 0, j = 1,...,p

Lagrangian:
    L(x, λ, μ) = f(x) + Σλᵢgᵢ(x) + Σμⱼhⱼ(x)

Dual Function:
    g(λ, μ) = inf_x L(x, λ, μ)

Dual Problem (D):
    maximize g(λ, μ)
    subject to λ ≥ 0

Key Theorems:
- Weak Duality: g(λ, μ) ≤ p* for all λ ≥ 0, μ
- Strong Duality: Under constraint qualifications, g* = p*
- Complementary Slackness: λᵢgᵢ(x*) = 0 at optimality
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

class DualityProblem:
    """
    Represents a primal-dual optimization problem pair.
    """
    
    def __init__(self, 
                 primal_objective: Callable[[np.ndarray], float],
                 primal_objective_grad: Callable[[np.ndarray], np.ndarray],
                 inequality_constraints: Optional[List[Callable]] = None,
                 inequality_grads: Optional[List[Callable]] = None,
                 equality_constraints: Optional[List[Callable]] = None,
                 equality_grads: Optional[List[Callable]] = None,
                 name: str = "Duality Problem"):
        """
        Initialize primal-dual problem.
        
        Args:
            primal_objective: Primal objective function f(x)
            primal_objective_grad: Gradient ∇f(x)
            inequality_constraints: g_i(x) ≤ 0 constraints
            inequality_grads: Constraint gradients ∇g_i(x)
            equality_constraints: h_j(x) = 0 constraints  
            equality_grads: Equality gradients ∇h_j(x)
            name: Problem description
        """
        self.primal_objective = primal_objective
        self.primal_objective_grad = primal_objective_grad
        
        self.inequality_constraints = inequality_constraints or []
        self.inequality_grads = inequality_grads or []
        self.equality_constraints = equality_constraints or []
        self.equality_grads = equality_grads or []
        
        self.name = name
    
    def lagrangian(self, x: np.ndarray, 
                   lambda_ineq: np.ndarray, 
                   mu_eq: np.ndarray) -> float:
        """
        Compute Lagrangian L(x, λ, μ) = f(x) + Σλᵢgᵢ(x) + Σμⱼhⱼ(x).
        """
        x = np.array(x)
        lambda_ineq = np.array(lambda_ineq)
        mu_eq = np.array(mu_eq)
        
        # Objective function
        L = self.primal_objective(x)
        
        # Inequality constraints
        for i, (lam, g) in enumerate(zip(lambda_ineq, self.inequality_constraints)):
            L += lam * g(x)
        
        # Equality constraints
        for j, (mu, h) in enumerate(zip(mu_eq, self.equality_constraints)):
            L += mu * h(x)
        
        return L
    
    def lagrangian_grad_x(self, x: np.ndarray,
                         lambda_ineq: np.ndarray,
                         mu_eq: np.ndarray) -> np.ndarray:
        """
        Compute gradient of Lagrangian with respect to x.
        """
        x = np.array(x)
        lambda_ineq = np.array(lambda_ineq)
        mu_eq = np.array(mu_eq)
        
        # Gradient of objective
        grad_L = self.primal_objective_grad(x)
        
        # Add inequality constraint gradients
        for lam, grad_g in zip(lambda_ineq, self.inequality_grads):
            grad_L += lam * grad_g(x)
        
        # Add equality constraint gradients
        for mu, grad_h in zip(mu_eq, self.equality_grads):
            grad_L += mu * grad_h(x)
        
        return grad_L
    
    def dual_function(self, lambda_ineq: np.ndarray, 
                     mu_eq: np.ndarray,
                     x_domain: Tuple[np.ndarray, np.ndarray] = None,
                     n_samples: int = 1000) -> float:
        """
        Compute dual function g(λ, μ) = inf_x L(x, λ, μ).
        Uses grid search over specified domain.
        """
        lambda_ineq = np.array(lambda_ineq)
        mu_eq = np.array(mu_eq)
        
        # Default domain if not specified
        if x_domain is None:
            lower_bounds = np.array([-10.0])
            upper_bounds = np.array([10.0])
        else:
            lower_bounds, upper_bounds = x_domain
        
        # For analytical problems, try to solve ∇_x L = 0
        # For now, use grid search as approximation
        min_value = np.inf
        
        # Generate random sample points
        dim = len(lower_bounds)
        for _ in range(n_samples):
            x_sample = np.random.uniform(lower_bounds, upper_bounds, dim)
            lagrangian_val = self.lagrangian(x_sample, lambda_ineq, mu_eq)
            min_value = min(min_value, lagrangian_val)
        
        # Also check some structured points
        for i in range(dim):
            # Check domain boundaries
            x_min = lower_bounds.copy()
            x_max = upper_bounds.copy()
            
            lagrangian_min = self.lagrangian(x_min, lambda_ineq, mu_eq)
            lagrangian_max = self.lagrangian(x_max, lambda_ineq, mu_eq)
            
            min_value = min(min_value, lagrangian_min, lagrangian_max)
        
        return min_value if np.isfinite(min_value) else -np.inf
    
    def compute_duality_gap(self, x_primal: np.ndarray,
                           lambda_ineq: np.ndarray,
                           mu_eq: np.ndarray) -> Dict:
        """
        Compute primal-dual gap and related quantities.
        """
        x_primal = np.array(x_primal)
        lambda_ineq = np.array(lambda_ineq)
        mu_eq = np.array(mu_eq)
        
        # Primal objective (if feasible)
        primal_value = self.primal_objective(x_primal)
        
        # Check primal feasibility
        primal_feasible = True
        ineq_violations = []
        for g in self.inequality_constraints:
            g_val = g(x_primal)
            ineq_violations.append(max(0, g_val))
            if g_val > 1e-8:
                primal_feasible = False
        
        eq_violations = []
        for h in self.equality_constraints:
            h_val = h(x_primal)
            eq_violations.append(abs(h_val))
            if abs(h_val) > 1e-8:
                primal_feasible = False
        
        # Dual value
        dual_value = self.dual_function(lambda_ineq, mu_eq)
        
        # Duality gap
        if primal_feasible and np.isfinite(dual_value):
            duality_gap = primal_value - dual_value
        else:
            duality_gap = np.inf
        
        return {
            'primal_value': primal_value,
            'dual_value': dual_value,
            'duality_gap': duality_gap,
            'primal_feasible': primal_feasible,
            'inequality_violations': np.array(ineq_violations),
            'equality_violations': np.array(eq_violations),
            'max_ineq_violation': np.max(ineq_violations) if ineq_violations else 0,
            'max_eq_violation': np.max(eq_violations) if eq_violations else 0
        }


class DualityExamples:
    """
    Collection of optimization problems with tractable dual problems.
    """
    
    @staticmethod
    def quadratic_programming():
        """
        Quadratic Programming Example:
        minimize ½xᵀPx + qᵀx
        subject to Ax ≤ b
        
        This has a well-known dual formulation.
        """
        # Problem data
        P = np.array([[2, 1], [1, 2]])  # Positive definite
        q = np.array([1, 1])
        A = np.array([[1, 1], [1, -1], [-1, 0], [0, -1]])  # Constraint matrix
        b = np.array([1, 0, 0, 0])  # RHS
        
        def primal_objective(x):
            x = np.array(x)
            return 0.5 * np.dot(x, np.dot(P, x)) + np.dot(q, x)
        
        def primal_objective_grad(x):
            x = np.array(x)
            return np.dot(P, x) + q
        
        # Inequality constraints: Ax ≤ b  =>  (Ax - b) ≤ 0
        constraints = []
        constraint_grads = []
        
        for i in range(len(b)):
            def make_constraint(row_idx):
                def constraint(x):
                    return np.dot(A[row_idx], x) - b[row_idx]
                return constraint
            
            def make_constraint_grad(row_idx):
                def constraint_grad(x):
                    return A[row_idx]
                return constraint_grad
            
            constraints.append(make_constraint(i))
            constraint_grads.append(make_constraint_grad(i))
        
        return DualityProblem(
            primal_objective=primal_objective,
            primal_objective_grad=primal_objective_grad,
            inequality_constraints=constraints,
            inequality_grads=constraint_grads,
            name="Quadratic Programming"
        )
    
    @staticmethod
    def lp_standard_form():
        """
        Linear Programming in Standard Form:
        minimize cᵀx
        subject to Ax = b, x ≥ 0
        
        Dual: maximize bᵀy subject to Aᵀy ≤ c
        """
        # Problem data (simple 2D example)
        c = np.array([1, 2])  # Objective coefficients
        A = np.array([[1, 1]])  # Equality constraint matrix  
        b = np.array([1])  # RHS
        
        def primal_objective(x):
            x = np.array(x)
            return np.dot(c, x)
        
        def primal_objective_grad(x):
            return c
        
        # Equality constraint: Ax = b
        def equality_constraint(x):
            x = np.array(x)
            return np.dot(A, x) - b
        
        def equality_constraint_grad(x):
            return A.flatten()  # For single constraint
        
        # Non-negativity constraints: x ≥ 0  =>  -x ≤ 0
        def nonnegativity1(x):
            return -x[0]
        
        def nonnegativity2(x):
            return -x[1]
        
        def grad_nonneg1(x):
            return np.array([-1, 0])
        
        def grad_nonneg2(x):
            return np.array([0, -1])
        
        return DualityProblem(
            primal_objective=primal_objective,
            primal_objective_grad=primal_objective_grad,
            inequality_constraints=[nonnegativity1, nonnegativity2],
            inequality_grads=[grad_nonneg1, grad_nonneg2],
            equality_constraints=[equality_constraint],
            equality_grads=[equality_constraint_grad],
            name="Linear Programming (Standard Form)"
        )
    
    @staticmethod
    def svm_dual_problem():
        """
        Support Vector Machine Dual Problem:
        Primal: minimize ½||w||² + C Σξᵢ
                subject to yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
        
        Dual: maximize Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼ(xᵢᵀxⱼ)
              subject to Σαᵢyᵢ = 0, 0 ≤ αᵢ ≤ C
        """
        # Simple 2D dataset
        X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, -1, -1])  # Labels
        C = 1.0  # Regularization parameter
        
        n_samples = len(X)
        
        def gram_matrix():
            """Compute Gram matrix K[i,j] = xᵢᵀxⱼ."""
            K = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = np.dot(X[i], X[j])
            return K
        
        K = gram_matrix()
        
        def dual_objective(alpha):
            """SVM dual objective (we minimize, so negate)."""
            alpha = np.array(alpha)
            linear_term = np.sum(alpha)
            quadratic_term = 0.5 * np.sum(alpha[:, None] * alpha * (y[:, None] * y) * K)
            return -(linear_term - quadratic_term)  # Negate for minimization
        
        def dual_objective_grad(alpha):
            """Gradient of dual objective."""
            alpha = np.array(alpha)
            grad = -np.ones(n_samples)  # From linear term
            for i in range(n_samples):
                for j in range(n_samples):
                    grad[i] += alpha[j] * y[i] * y[j] * K[i, j]
            return -grad  # Negate for minimization
        
        # Constraints for dual problem
        # 1. Σαᵢyᵢ = 0 (equality constraint)
        def balance_constraint(alpha):
            return np.dot(alpha, y)
        
        def balance_constraint_grad(alpha):
            return y
        
        # 2. αᵢ ≥ 0 (non-negativity)
        ineq_constraints = []
        ineq_grads = []
        
        for i in range(n_samples):
            def make_nonneg_constraint(idx):
                def constraint(alpha):
                    return -alpha[idx]  # -αᵢ ≤ 0
                return constraint
            
            def make_nonneg_grad(idx):
                def grad(alpha):
                    g = np.zeros(n_samples)
                    g[idx] = -1
                    return g
                return grad
            
            ineq_constraints.append(make_nonneg_constraint(i))
            ineq_grads.append(make_nonneg_grad(i))
        
        # 3. αᵢ ≤ C (upper bounds)
        for i in range(n_samples):
            def make_upper_constraint(idx):
                def constraint(alpha):
                    return alpha[idx] - C  # αᵢ - C ≤ 0
                return constraint
            
            def make_upper_grad(idx):
                def grad(alpha):
                    g = np.zeros(n_samples)
                    g[idx] = 1
                    return g
                return grad
            
            ineq_constraints.append(make_upper_constraint(i))
            ineq_grads.append(make_upper_grad(i))
        
        return DualityProblem(
            primal_objective=dual_objective,
            primal_objective_grad=dual_objective_grad,
            inequality_constraints=ineq_constraints,
            inequality_grads=ineq_grads,
            equality_constraints=[balance_constraint],
            equality_grads=[balance_constraint_grad],
            name="SVM Dual Problem"
        )


def demonstrate_duality_theory():
    """
    Comprehensive demonstration of duality theory in optimization.
    """
    print("🔄 DUALITY THEORY: Primal-Dual Relationships")
    print("=" * 55)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Example 1: Quadratic Programming
    print("\n📊 EXAMPLE 1: Quadratic Programming Duality")
    print("-" * 50)
    
    qp_problem = DualityExamples.quadratic_programming()
    
    # Test primal-dual relationships at various points
    print(f"Problem: {qp_problem.name}")
    
    # Visualize Example 1: Primal problem geometry
    ax1 = plt.subplot(2, 3, 1)
    
    # Plot feasible region for QP problem
    x1_range = np.linspace(-1, 2, 100)
    x2_range = np.linspace(-1, 2, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    
    # Objective function contours
    Z = 0.5 * (2*X1**2 + 2*X1*X2 + 2*X2**2) + X1 + X2
    
    contour = ax1.contour(X1, X2, Z, levels=15, alpha=0.6, colors='blue')
    ax1.clabel(contour, inline=True, fontsize=8)
    
    # Constraint boundaries
    # x₁ + x₂ ≤ 1
    constraint1_x1 = np.linspace(-1, 2, 100)
    constraint1_x2 = 1 - constraint1_x1
    ax1.plot(constraint1_x1, constraint1_x2, 'r-', linewidth=2, label='x₁ + x₂ ≤ 1')
    
    # x₁ - x₂ ≤ 0  =>  x₂ ≥ x₁
    constraint2_x1 = np.linspace(-1, 2, 100)
    constraint2_x2 = constraint2_x1
    ax1.plot(constraint2_x1, constraint2_x2, 'g-', linewidth=2, label='x₁ ≤ x₂')
    
    # x₁ ≥ 0, x₂ ≥ 0
    ax1.axhline(y=0, color='orange', linewidth=2, alpha=0.7, label='x₂ ≥ 0')
    ax1.axvline(x=0, color='purple', linewidth=2, alpha=0.7, label='x₁ ≥ 0')
    
    # Shade feasible region
    # This is the intersection of all constraints
    x_feas = np.array([[0, 0], [0, 0.5], [0.5, 0.5], [0, 0]])  # Approximate feasible region
    ax1.fill(x_feas[:, 0], x_feas[:, 1], alpha=0.3, color='lightblue', label='Feasible region')
    
    # Mark approximate optimal point
    x_opt_approx = np.array([0, 0])  # Origin is likely optimal for this problem
    ax1.plot(x_opt_approx[0], x_opt_approx[1], 'ro', markersize=10, label='Approximate x*')
    
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title('Quadratic Programming: Primal Problem')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Example 2: Dual function visualization
    ax2 = plt.subplot(2, 3, 2)
    
    # Compute dual function values for different dual variables
    lambda_range = np.linspace(0, 3, 20)
    dual_values = []
    
    for lam in lambda_range:
        # For QP, we test with first constraint multiplier
        lambda_vec = np.array([lam, 0, 0, 0])  # Only first constraint active
        mu_vec = np.array([])  # No equality constraints
        
        dual_val = qp_problem.dual_function(lambda_vec, mu_vec, 
                                          x_domain=(np.array([-2, -2]), np.array([2, 2])),
                                          n_samples=500)
        dual_values.append(dual_val)
    
    ax2.plot(lambda_range, dual_values, 'b-', linewidth=3, label='Dual function g(λ)')
    
    # Mark maximum (dual optimal)
    max_idx = np.argmax(dual_values)
    if np.isfinite(dual_values[max_idx]):
        ax2.plot(lambda_range[max_idx], dual_values[max_idx], 'ro', 
                markersize=10, label=f'Dual optimum: λ*≈{lambda_range[max_idx]:.2f}')
    
    ax2.set_xlabel('λ₁ (Dual variable)')
    ax2.set_ylabel('g(λ₁) (Dual function value)')
    ax2.set_title('Dual Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Example 3: Duality gap analysis
    ax3 = plt.subplot(2, 3, 3)
    
    # Compute duality gaps for various primal points
    test_points = [
        np.array([0.0, 0.0]),
        np.array([0.1, 0.1]), 
        np.array([0.2, 0.3]),
        np.array([0.3, 0.3]),
        np.array([0.4, 0.4])
    ]
    
    duality_gaps = []
    primal_values = []
    dual_values_computed = []
    
    for x_test in test_points:
        # Use simple dual variables for testing
        lambda_test = np.array([1.0, 0.5, 0.1, 0.1])
        mu_test = np.array([])
        
        gap_info = qp_problem.compute_duality_gap(x_test, lambda_test, mu_test)
        
        if gap_info['primal_feasible'] and np.isfinite(gap_info['dual_value']):
            duality_gaps.append(gap_info['duality_gap'])
            primal_values.append(gap_info['primal_value'])
            dual_values_computed.append(gap_info['dual_value'])
        else:
            duality_gaps.append(np.inf)
            primal_values.append(gap_info['primal_value'])
            dual_values_computed.append(-np.inf)
    
    # Plot primal and dual values
    point_labels = [f'({x[0]:.1f},{x[1]:.1f})' for x in test_points]
    x_positions = range(len(test_points))
    
    # Filter finite values for plotting
    finite_mask = np.isfinite(duality_gaps)
    if np.any(finite_mask):
        finite_indices = np.where(finite_mask)[0]
        ax3.bar([x_positions[i] - 0.2 for i in finite_indices], 
               [primal_values[i] for i in finite_indices], 
               width=0.4, alpha=0.7, label='Primal value', color='blue')
        ax3.bar([x_positions[i] + 0.2 for i in finite_indices], 
               [dual_values_computed[i] for i in finite_indices], 
               width=0.4, alpha=0.7, label='Dual value', color='red')
    
    ax3.set_xlabel('Test Points')
    ax3.set_ylabel('Objective Value')
    ax3.set_title('Primal vs Dual Values')
    ax3.set_xticks(x_positions)
    ax3.set_xticklabels(point_labels, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Linear Programming Example
    print("\n📊 EXAMPLE 2: Linear Programming Duality")
    print("-" * 45)
    
    lp_problem = DualityExamples.lp_standard_form()
    
    ax4 = plt.subplot(2, 3, 4)
    
    # Visualize LP problem
    # Objective: minimize x₁ + 2x₂
    # Subject to: x₁ + x₂ = 1, x₁ ≥ 0, x₂ ≥ 0
    
    # Plot feasible line: x₁ + x₂ = 1, x₁,x₂ ≥ 0
    x1_lp = np.linspace(0, 1, 100)
    x2_lp = 1 - x1_lp
    
    # Only plot where both are non-negative
    valid_mask = (x1_lp >= 0) & (x2_lp >= 0)
    ax4.plot(x1_lp[valid_mask], x2_lp[valid_mask], 'r-', linewidth=4, label='Feasible line: x₁+x₂=1')
    
    # Plot objective function contours: x₁ + 2x₂ = constant
    X1_lp, X2_lp = np.meshgrid(np.linspace(-0.2, 1.2, 50), np.linspace(-0.2, 1.2, 50))
    Z_lp = X1_lp + 2*X2_lp
    
    contour_lp = ax4.contour(X1_lp, X2_lp, Z_lp, levels=10, alpha=0.6, colors='blue')
    ax4.clabel(contour_lp, inline=True, fontsize=8)
    
    # Mark optimal point: (1, 0) minimizes x₁ + 2x₂ = 1 + 0 = 1
    ax4.plot(1, 0, 'go', markersize=12, label='Optimal: (1,0)')
    
    # Mark feasible region boundaries
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='x₂ ≥ 0')
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='x₁ ≥ 0')
    
    ax4.set_xlim(-0.2, 1.2)
    ax4.set_ylim(-0.2, 1.2)
    ax4.set_xlabel('x₁')
    ax4.set_ylabel('x₂')
    ax4.set_title('Linear Programming: Primal Problem')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Weak Duality Illustration
    ax5 = plt.subplot(2, 3, 5)
    
    # Demonstrate weak duality theorem
    # For various dual multiplier values, show that dual ≤ primal
    
    n_tests = 50
    lambda_tests = np.random.exponential(1.0, (n_tests, 2))  # λ ≥ 0
    mu_tests = np.random.normal(0, 1, n_tests)  # μ unrestricted
    
    primal_vals_weak = []
    dual_vals_weak = []
    
    # Fixed primal point for comparison
    x_primal_fixed = np.array([0.3, 0.7])  # On constraint boundary
    
    for i in range(n_tests):
        lambda_test = lambda_tests[i]
        mu_test = np.array([mu_tests[i]])
        
        # Primal value
        primal_val = lp_problem.primal_objective(x_primal_fixed)
        
        # Dual value
        dual_val = lp_problem.dual_function(lambda_test, mu_test,
                                          x_domain=(np.array([-2, -2]), np.array([3, 3])),
                                          n_samples=200)
        
        if np.isfinite(dual_val):
            primal_vals_weak.append(primal_val)
            dual_vals_weak.append(dual_val)
    
    # Plot weak duality relationship
    if len(dual_vals_weak) > 0:
        ax5.scatter(dual_vals_weak, primal_vals_weak, alpha=0.6, color='blue', 
                   label='(Dual, Primal) pairs')
        
        # Plot diagonal line y = x
        min_val = min(min(dual_vals_weak), min(primal_vals_weak))
        max_val = max(max(dual_vals_weak), max(primal_vals_weak))
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', 
                linewidth=2, label='Primal = Dual (strong duality)')
        
        # Shade region where weak duality holds
        ax5.fill_between([min_val, max_val], [min_val, max_val], [max_val, max_val], 
                        alpha=0.2, color='green', label='Weak duality region')
    
    ax5.set_xlabel('Dual Value g(λ,μ)')
    ax5.set_ylabel('Primal Value f(x)')
    ax5.set_title('Weak Duality: g(λ,μ) ≤ p*')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Complementary Slackness Visualization
    ax6 = plt.subplot(2, 3, 6)
    
    # Show complementary slackness for QP problem
    # At optimum: λᵢ * gᵢ(x*) = 0
    
    # Test different points along constraint boundary
    t_vals = np.linspace(0, 1, 50)
    complementary_products = []
    constraint_values = []
    multiplier_values = []
    
    for t in t_vals:
        # Point on or near first constraint boundary
        x_test = np.array([t*0.5, (1-t)*0.5])
        
        # Evaluate first constraint: g₁(x) = x₁ + x₂ - 1
        g1_val = qp_problem.inequality_constraints[0](x_test)
        constraint_values.append(abs(g1_val))
        
        # Estimate multiplier from stationarity (approximate)
        grad_f = qp_problem.primal_objective_grad(x_test)
        grad_g1 = qp_problem.inequality_grads[0](x_test)
        
        # From stationarity: ∇f + λ₁∇g₁ = 0  =>  λ₁ = -∇f·∇g₁/||∇g₁||²
        lambda1_est = -np.dot(grad_f, grad_g1) / np.dot(grad_g1, grad_g1)
        lambda1_est = max(0, lambda1_est)  # Dual feasibility
        
        multiplier_values.append(lambda1_est)
        
        # Complementary slackness product
        comp_product = lambda1_est * abs(g1_val)
        complementary_products.append(comp_product)
    
    # Plot complementary slackness components
    ax6.plot(t_vals, constraint_values, 'b-', linewidth=2, label='|g₁(x)|')
    ax6.plot(t_vals, multiplier_values, 'r--', linewidth=2, label='λ₁')
    ax6.plot(t_vals, complementary_products, 'g:', linewidth=3, label='λ₁·|g₁(x)|')
    
    # Highlight where complementary slackness is satisfied (product ≈ 0)
    slack_satisfied = np.array(complementary_products) < 0.1
    if np.any(slack_satisfied):
        ax6.scatter(t_vals[slack_satisfied], np.zeros(np.sum(slack_satisfied)), 
                   color='gold', s=100, marker='*', 
                   label='Complementary slackness satisfied', zorder=5)
    
    ax6.set_xlabel('Parameter t')
    ax6.set_ylabel('Value')
    ax6.set_title('Complementary Slackness: λᵢ·gᵢ(x) = 0')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Numerical verification
    print(f"\n📋 NUMERICAL RESULTS:")
    print(f"QP Problem - Primal objective at origin: {qp_problem.primal_objective(np.array([0, 0])):.4f}")
    
    # Test specific dual variables
    lambda_test = np.array([0.5, 0, 0, 0])
    mu_test = np.array([])
    dual_val_test = qp_problem.dual_function(lambda_test, mu_test)
    print(f"QP Problem - Dual value with λ=[0.5,0,0,0]: {dual_val_test:.4f}")
    
    # LP Problem
    x_lp_opt = np.array([1, 0])
    print(f"LP Problem - Primal objective at (1,0): {lp_problem.primal_objective(x_lp_opt):.4f}")


def duality_applications():
    """
    Applications and importance of duality theory.
    """
    print("\n🎯 DUALITY THEORY APPLICATIONS")
    print("=" * 40)
    
    print("📚 COMPUTATIONAL ADVANTAGES:")
    print("1. INTERIOR POINT METHODS:")
    print("   - Solve primal and dual simultaneously")
    print("   - Use duality gap as stopping criterion")
    print("   - Path-following algorithms")
    
    print("\n2. SENSITIVITY ANALYSIS:")
    print("   - Dual variables = shadow prices")
    print("   - Marginal value of constraint relaxation")
    print("   - Economic interpretation of multipliers")
    
    print("\n3. BOUNDS AND APPROXIMATION:")
    print("   - Dual provides lower bound for minimization")
    print("   - Lagrangian relaxation in integer programming")
    print("   - Branch-and-bound algorithms")
    
    print("\n🎯 THEORETICAL INSIGHTS:")
    print("1. WEAK DUALITY THEOREM:")
    print("   - g(λ,μ) ≤ p* always holds")
    print("   - No constraint qualifications needed")
    print("   - Provides universal bounds")
    
    print("\n2. STRONG DUALITY:")
    print("   - g* = p* under constraint qualifications")
    print("   - Slater condition for convex problems")
    print("   - LICQ for general nonlinear problems")
    
    print("\n3. COMPLEMENTARY SLACKNESS:")
    print("   - λᵢ > 0 ⟹ gᵢ(x*) = 0 (constraint active)")
    print("   - gᵢ(x*) < 0 ⟹ λᵢ = 0 (constraint inactive)")
    print("   - Identifies active constraints at optimum")
    
    print("\n🔧 ALGORITHMIC APPLICATIONS:")
    print("1. DUAL DECOMPOSITION:")
    print("   - Separate large problems into subproblems")
    print("   - Parallel and distributed optimization")
    print("   - Network optimization")
    
    print("\n2. AUGMENTED LAGRANGIAN:")
    print("   - Combine penalty method with dual updates")
    print("   - Better convergence than pure penalty")
    print("   - Handle ill-conditioned problems")
    
    print("\n3. DUAL ASCENT/GRADIENT METHODS:")
    print("   - Maximize dual function")
    print("   - Subgradient methods for non-smooth duals")
    print("   - Coordinate ascent for separable problems")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_duality_theory()
    duality_applications()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Duality associates every optimization problem with a related dual problem")
    print("- Weak duality: dual value ≤ primal value (universal bound)")
    print("- Strong duality: dual value = primal value (under constraint qualifications)")
    print("- Complementary slackness: λᵢ·gᵢ(x*) = 0 identifies active constraints")
    print("- Dual multipliers provide sensitivity/shadow price information")
    print("- Foundation for interior point methods and decomposition algorithms")
    print("\nDuality theory bridges optimization theory and computational practice! 🚀")