"""
KKT Conditions (Karush-Kuhn-Tucker Optimality Conditions)

The KKT conditions provide necessary (and sometimes sufficient) conditions
for optimality in constrained nonlinear programming problems. They extend
the Lagrange multiplier method to handle inequality constraints.

Mathematical Statement:
For the optimization problem:
    minimize f(x)
    subject to g_i(x) ≤ 0, i = 1,...,m
               h_j(x) = 0, j = 1,...,p

The KKT conditions are:
1. Stationarity: ∇f(x*) + Σλᵢ∇gᵢ(x*) + Σμⱼ∇hⱼ(x*) = 0
2. Primal feasibility: gᵢ(x*) ≤ 0, hⱼ(x*) = 0
3. Dual feasibility: λᵢ ≥ 0
4. Complementary slackness: λᵢgᵢ(x*) = 0

Under constraint qualifications (LICQ, MFCQ, etc.), these conditions
are necessary for optimality. For convex problems, they are also sufficient.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from typing import Callable, List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

class OptimizationProblem:
    """
    Represents a constrained optimization problem and its KKT conditions.
    """
    
    def __init__(self, 
                 objective: Callable[[np.ndarray], float],
                 objective_grad: Callable[[np.ndarray], np.ndarray],
                 objective_hess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 inequality_constraints: Optional[List[Callable]] = None,
                 inequality_grads: Optional[List[Callable]] = None,
                 equality_constraints: Optional[List[Callable]] = None,
                 equality_grads: Optional[List[Callable]] = None,
                 name: str = "Problem"):
        """
        Initialize constrained optimization problem.
        
        Args:
            objective: Objective function f(x)
            objective_grad: Gradient of objective ∇f(x)
            objective_hess: Hessian of objective ∇²f(x)
            inequality_constraints: List of g_i(x) ≤ 0 constraints
            inequality_grads: List of constraint gradients ∇g_i(x)
            equality_constraints: List of h_j(x) = 0 constraints
            equality_grads: List of equality constraint gradients ∇h_j(x)
            name: Problem description
        """
        self.objective = objective
        self.objective_grad = objective_grad
        self.objective_hess = objective_hess
        
        self.inequality_constraints = inequality_constraints or []
        self.inequality_grads = inequality_grads or []
        self.equality_constraints = equality_constraints or []
        self.equality_grads = equality_grads or []
        
        self.name = name
        
        # Validate inputs
        if len(self.inequality_constraints) != len(self.inequality_grads):
            raise ValueError("Number of inequality constraints must match number of gradients")
        if len(self.equality_constraints) != len(self.equality_grads):
            raise ValueError("Number of equality constraints must match number of gradients")
    
    def evaluate_constraints(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate all constraints at point x.
        
        Returns:
            (inequality_values, equality_values)
        """
        x = np.array(x)
        
        ineq_vals = np.array([g(x) for g in self.inequality_constraints])
        eq_vals = np.array([h(x) for h in self.equality_constraints])
        
        return ineq_vals, eq_vals
    
    def is_feasible(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        """Check if point x is feasible."""
        ineq_vals, eq_vals = self.evaluate_constraints(x)
        
        # Inequality constraints: g_i(x) ≤ 0
        ineq_feasible = np.all(ineq_vals <= tol)
        
        # Equality constraints: h_j(x) = 0
        eq_feasible = np.all(np.abs(eq_vals) <= tol)
        
        return ineq_feasible and eq_feasible
    
    def active_inequality_constraints(self, x: np.ndarray, tol: float = 1e-8) -> List[int]:
        """Find active inequality constraints at x."""
        ineq_vals, _ = self.evaluate_constraints(x)
        return [i for i, val in enumerate(ineq_vals) if abs(val) <= tol]
    
    def check_kkt_conditions(self, x: np.ndarray, 
                           lambda_ineq: np.ndarray, 
                           mu_eq: np.ndarray,
                           tol: float = 1e-6) -> Dict:
        """
        Check KKT conditions at point x with multipliers.
        
        Returns:
            Dictionary with condition checks and violations
        """
        x = np.array(x)
        lambda_ineq = np.array(lambda_ineq)
        mu_eq = np.array(mu_eq)
        
        results = {}
        
        # 1. Stationarity condition
        grad_f = self.objective_grad(x)
        
        # Add inequality constraint contributions
        stationarity = grad_f.copy()
        for i, (lam, grad_g) in enumerate(zip(lambda_ineq, self.inequality_grads)):
            stationarity += lam * grad_g(x)
        
        # Add equality constraint contributions
        for j, (mu, grad_h) in enumerate(zip(mu_eq, self.equality_grads)):
            stationarity += mu * grad_h(x)
        
        stationarity_violation = np.linalg.norm(stationarity)
        results['stationarity_satisfied'] = stationarity_violation <= tol
        results['stationarity_violation'] = stationarity_violation
        results['stationarity_gradient'] = stationarity
        
        # 2. Primal feasibility
        ineq_vals, eq_vals = self.evaluate_constraints(x)
        
        # Inequality constraints: g_i(x) ≤ 0
        ineq_violations = np.maximum(ineq_vals, 0)
        max_ineq_violation = np.max(ineq_violations) if len(ineq_violations) > 0 else 0
        
        # Equality constraints: h_j(x) = 0
        eq_violations = np.abs(eq_vals)
        max_eq_violation = np.max(eq_violations) if len(eq_violations) > 0 else 0
        
        results['primal_feasible'] = max_ineq_violation <= tol and max_eq_violation <= tol
        results['inequality_violations'] = ineq_violations
        results['equality_violations'] = eq_violations
        results['max_primal_violation'] = max(max_ineq_violation, max_eq_violation)
        
        # 3. Dual feasibility: λᵢ ≥ 0
        dual_violations = np.maximum(-lambda_ineq, 0)
        max_dual_violation = np.max(dual_violations) if len(dual_violations) > 0 else 0
        
        results['dual_feasible'] = max_dual_violation <= tol
        results['dual_violations'] = dual_violations
        results['max_dual_violation'] = max_dual_violation
        
        # 4. Complementary slackness: λᵢ * gᵢ(x) = 0
        if len(ineq_vals) > 0:
            comp_slack_violations = np.abs(lambda_ineq * ineq_vals)
            max_comp_slack_violation = np.max(comp_slack_violations)
        else:
            comp_slack_violations = np.array([])
            max_comp_slack_violation = 0
        
        results['complementary_slackness_satisfied'] = max_comp_slack_violation <= tol
        results['complementary_slackness_violations'] = comp_slack_violations
        results['max_complementary_slackness_violation'] = max_comp_slack_violation
        
        # Overall KKT satisfaction
        results['kkt_satisfied'] = (results['stationarity_satisfied'] and 
                                   results['primal_feasible'] and 
                                   results['dual_feasible'] and 
                                   results['complementary_slackness_satisfied'])
        
        results['total_violation'] = (stationarity_violation + 
                                    results['max_primal_violation'] +
                                    max_dual_violation + 
                                    max_comp_slack_violation)
        
        return results
    
    def solve_with_scipy(self, x0: np.ndarray, method: str = 'SLSQP') -> Dict:
        """
        Solve optimization problem using scipy and extract KKT information.
        """
        x0 = np.array(x0)
        
        # Set up constraints for scipy
        constraints = []
        
        # Inequality constraints: g_i(x) ≤ 0  (scipy uses ≥ 0, so we negate)
        for i, g in enumerate(self.inequality_constraints):
            def constraint_func(x, idx=i):
                return -self.inequality_constraints[idx](x)
            def constraint_jac(x, idx=i):
                return -self.inequality_grads[idx](x)
            
            constraints.append({
                'type': 'ineq',
                'fun': constraint_func,
                'jac': constraint_jac
            })
        
        # Equality constraints: h_j(x) = 0
        for j, h in enumerate(self.equality_constraints):
            def constraint_func(x, idx=j):
                return self.equality_constraints[idx](x)
            def constraint_jac(x, idx=j):
                return self.equality_grads[idx](x)
            
            constraints.append({
                'type': 'eq', 
                'fun': constraint_func,
                'jac': constraint_jac
            })
        
        # Solve optimization problem
        result = minimize(
            fun=self.objective,
            x0=x0,
            method=method,
            jac=self.objective_grad,
            hess=self.objective_hess,
            constraints=constraints,
            options={'ftol': 1e-12, 'disp': False}
        )
        
        # Extract multipliers (if available)
        if hasattr(result, 'v') and result.v is not None:
            # For SLSQP, result.v contains constraint multipliers
            multipliers = result.v
            n_ineq = len(self.inequality_constraints)
            
            if len(multipliers) >= n_ineq:
                lambda_ineq = multipliers[:n_ineq]
                mu_eq = multipliers[n_ineq:] if len(multipliers) > n_ineq else np.array([])
            else:
                lambda_ineq = multipliers
                mu_eq = np.array([])
        else:
            # Estimate multipliers if not provided
            lambda_ineq = np.zeros(len(self.inequality_constraints))
            mu_eq = np.zeros(len(self.equality_constraints))
        
        return {
            'solution': result,
            'x_optimal': result.x,
            'f_optimal': result.fun,
            'lambda_ineq': lambda_ineq,
            'mu_eq': mu_eq,
            'success': result.success,
            'message': result.message
        }


class KKTExamples:
    """
    Collection of optimization problems with known KKT solutions.
    """
    
    @staticmethod
    def quadratic_with_linear_constraint():
        """
        Minimize f(x) = ½(x₁² + x₂²)
        Subject to g(x) = x₁ + x₂ - 1 ≤ 0
        
        Analytical solution: x* = (0.5, 0.5), λ* = 1
        """
        def objective(x):
            return 0.5 * (x[0]**2 + x[1]**2)
        
        def objective_grad(x):
            return np.array([x[0], x[1]])
        
        def objective_hess(x):
            return np.array([[1, 0], [0, 1]])
        
        def inequality_constraint(x):
            return x[0] + x[1] - 1  # g(x) ≤ 0
        
        def inequality_grad(x):
            return np.array([1, 1])
        
        return OptimizationProblem(
            objective=objective,
            objective_grad=objective_grad,
            objective_hess=objective_hess,
            inequality_constraints=[inequality_constraint],
            inequality_grads=[inequality_grad],
            name="Quadratic with Linear Constraint"
        )
    
    @staticmethod
    def circle_in_square():
        """
        Maximize circle area inside unit square.
        Minimize f(x,y,r) = -πr²
        Subject to: x - r ≥ 0, y - r ≥ 0, x + r ≤ 1, y + r ≤ 1
        
        Reformulated as minimization with ≤ 0 constraints.
        """
        def objective(x):
            # x = [center_x, center_y, radius]
            return -np.pi * x[2]**2  # Maximize area = minimize negative area
        
        def objective_grad(x):
            return np.array([0, 0, -2*np.pi*x[2]])
        
        def objective_hess(x):
            return np.array([[0, 0, 0], [0, 0, 0], [0, 0, -2*np.pi]])
        
        # Constraints: ensure circle stays within unit square
        def constraint1(x):  # r - x ≤ 0 (left boundary)
            return x[2] - x[0]
        
        def constraint2(x):  # r - y ≤ 0 (bottom boundary)
            return x[2] - x[1]
        
        def constraint3(x):  # x + r - 1 ≤ 0 (right boundary)
            return x[0] + x[2] - 1
        
        def constraint4(x):  # y + r - 1 ≤ 0 (top boundary)
            return x[1] + x[2] - 1
        
        def constraint5(x):  # r ≥ 0 → -r ≤ 0
            return -x[2]
        
        def grad1(x):
            return np.array([-1, 0, 1])
        
        def grad2(x):
            return np.array([0, -1, 1])
        
        def grad3(x):
            return np.array([1, 0, 1])
        
        def grad4(x):
            return np.array([0, 1, 1])
        
        def grad5(x):
            return np.array([0, 0, -1])
        
        return OptimizationProblem(
            objective=objective,
            objective_grad=objective_grad,
            objective_hess=objective_hess,
            inequality_constraints=[constraint1, constraint2, constraint3, constraint4, constraint5],
            inequality_grads=[grad1, grad2, grad3, grad4, grad5],
            name="Maximum Circle in Unit Square"
        )
    
    @staticmethod
    def portfolio_optimization():
        """
        Markowitz portfolio optimization with budget constraint.
        Minimize f(w) = ½wᵀΣw  (portfolio risk)
        Subject to: 1ᵀw = 1 (budget constraint)
                   w ≥ 0 (no short selling)
        """
        # Sample covariance matrix (2 assets)
        Sigma = np.array([[0.04, 0.01], [0.01, 0.02]])
        
        def objective(w):
            return 0.5 * np.dot(w, np.dot(Sigma, w))
        
        def objective_grad(w):
            return np.dot(Sigma, w)
        
        def objective_hess(w):
            return Sigma
        
        def budget_constraint(w):
            return np.sum(w) - 1  # Σwᵢ = 1
        
        def budget_grad(w):
            return np.ones(len(w))
        
        # Non-negativity constraints: -wᵢ ≤ 0
        def nonnegativity1(w):
            return -w[0]
        
        def nonnegativity2(w):
            return -w[1]
        
        def grad_nonneg1(w):
            return np.array([-1, 0])
        
        def grad_nonneg2(w):
            return np.array([0, -1])
        
        return OptimizationProblem(
            objective=objective,
            objective_grad=objective_grad,
            objective_hess=objective_hess,
            inequality_constraints=[nonnegativity1, nonnegativity2],
            inequality_grads=[grad_nonneg1, grad_nonneg2],
            equality_constraints=[budget_constraint],
            equality_grads=[budget_grad],
            name="Portfolio Optimization (2 assets)"
        )


def demonstrate_kkt_conditions():
    """
    Comprehensive demonstration of KKT conditions.
    """
    print("🎯 KKT CONDITIONS: Constrained Optimization Theory")
    print("=" * 55)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Example 1: Quadratic with linear constraint
    print("\n📊 EXAMPLE 1: Quadratic with Linear Constraint")
    print("-" * 50)
    
    problem1 = KKTExamples.quadratic_with_linear_constraint()
    
    # Solve the problem
    x0 = np.array([0.0, 0.0])
    solution1 = problem1.solve_with_scipy(x0)
    
    print(f"Problem: {problem1.name}")
    print(f"Optimal solution: x* = {solution1['x_optimal']}")
    print(f"Optimal value: f* = {solution1['f_optimal']:.6f}")
    print(f"Lagrange multiplier: λ* = {solution1['lambda_ineq']}")
    
    # Check KKT conditions
    kkt_results1 = problem1.check_kkt_conditions(
        solution1['x_optimal'], 
        solution1['lambda_ineq'], 
        solution1['mu_eq']
    )
    
    print(f"KKT satisfied: {kkt_results1['kkt_satisfied']}")
    print(f"Total violation: {kkt_results1['total_violation']:.2e}")
    
    # Visualize Example 1
    ax1 = plt.subplot(2, 3, 1)
    
    # Create contour plot of objective function
    x1_range = np.linspace(-0.5, 1.5, 100)
    x2_range = np.linspace(-0.5, 1.5, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = 0.5 * (X1**2 + X2**2)
    
    contour = ax1.contour(X1, X2, Z, levels=20, alpha=0.6, colors='blue')
    ax1.clabel(contour, inline=True, fontsize=8)
    
    # Plot constraint boundary: x₁ + x₂ = 1
    constraint_x1 = np.linspace(-0.5, 1.5, 100)
    constraint_x2 = 1 - constraint_x1
    ax1.plot(constraint_x1, constraint_x2, 'r-', linewidth=3, label='x₁ + x₂ = 1')
    
    # Shade feasible region: x₁ + x₂ ≤ 1
    ax1.fill_between(constraint_x1, constraint_x2, -0.5, alpha=0.2, color='red', label='Feasible region')
    
    # Plot optimal point
    x_opt = solution1['x_optimal']
    ax1.plot(x_opt[0], x_opt[1], 'go', markersize=12, label=f'x* = ({x_opt[0]:.2f}, {x_opt[1]:.2f})')
    
    # Plot gradient vectors at optimal point
    grad_f = problem1.objective_grad(x_opt)
    grad_g = problem1.inequality_grads[0](x_opt)
    
    # Scale gradients for visualization
    scale = 0.2
    ax1.arrow(x_opt[0], x_opt[1], scale*grad_f[0], scale*grad_f[1], 
             head_width=0.03, head_length=0.05, fc='blue', ec='blue',
             label='∇f(x*)')
    ax1.arrow(x_opt[0], x_opt[1], -scale*solution1['lambda_ineq'][0]*grad_g[0], 
             -scale*solution1['lambda_ineq'][0]*grad_g[1],
             head_width=0.03, head_length=0.05, fc='red', ec='red',
             label='-λ*∇g(x*)')
    
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title('Quadratic with Linear Constraint')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Example 2: Portfolio optimization
    print("\n📊 EXAMPLE 2: Portfolio Optimization")
    print("-" * 40)
    
    problem2 = KKTExamples.portfolio_optimization()
    
    # Solve portfolio problem
    x0_portfolio = np.array([0.5, 0.5])  # Equal weights initial guess
    solution2 = problem2.solve_with_scipy(x0_portfolio)
    
    print(f"Problem: {problem2.name}")
    print(f"Optimal weights: w* = {solution2['x_optimal']}")
    print(f"Portfolio risk: σ² = {solution2['f_optimal']:.6f}")
    print(f"Inequality multipliers (non-negativity): λ* = {solution2['lambda_ineq']}")
    print(f"Equality multiplier (budget): μ* = {solution2['mu_eq']}")
    
    # Check KKT conditions
    kkt_results2 = problem2.check_kkt_conditions(
        solution2['x_optimal'], 
        solution2['lambda_ineq'], 
        solution2['mu_eq']
    )
    
    print(f"KKT satisfied: {kkt_results2['kkt_satisfied']}")
    print(f"Total violation: {kkt_results2['total_violation']:.2e}")
    
    # Check which constraints are active
    active_ineq = problem2.active_inequality_constraints(solution2['x_optimal'])
    print(f"Active inequality constraints: {active_ineq}")
    
    # Visualize Example 2
    ax2 = plt.subplot(2, 3, 2)
    
    # Plot efficient frontier and feasible region
    w1_range = np.linspace(-0.1, 1.1, 100)
    w2_range = 1 - w1_range  # Budget constraint: w₁ + w₂ = 1
    
    # Compute risk for each portfolio on budget line
    risks = []
    for w1, w2 in zip(w1_range, w2_range):
        if w1 >= 0 and w2 >= 0:  # Feasible region
            w = np.array([w1, w2])
            risk = problem2.objective(w)
            risks.append(risk)
        else:
            risks.append(np.inf)
    
    # Plot efficient frontier
    valid_indices = np.isfinite(risks)
    ax2.plot(w1_range[valid_indices], np.array(risks)[valid_indices], 'b-', 
            linewidth=3, label='Risk curve')
    
    # Plot optimal portfolio
    w_opt = solution2['x_optimal']
    risk_opt = solution2['f_optimal']
    ax2.plot(w_opt[0], risk_opt, 'ro', markersize=12, 
            label=f'Optimal: w*=({w_opt[0]:.3f},{w_opt[1]:.3f})')
    
    # Mark feasible region boundaries
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='w₁ ≥ 0')
    ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.7, label='w₂ ≥ 0')
    
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(0, 0.05)
    ax2.set_xlabel('Weight in Asset 1 (w₁)')
    ax2.set_ylabel('Portfolio Risk (σ²)')
    ax2.set_title('Portfolio Optimization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # KKT Condition Analysis
    ax3 = plt.subplot(2, 3, 3)
    
    # Show KKT condition violations for both problems
    problems = [problem1, problem2]
    solutions = [solution1, solution2]
    kkt_results = [kkt_results1, kkt_results2]
    
    condition_names = ['Stationarity', 'Primal Feas.', 'Dual Feas.', 'Comp. Slack.']
    
    for i, (prob, sol, kkt) in enumerate(zip(problems, solutions, kkt_results)):
        violations = [
            kkt['stationarity_violation'],
            kkt['max_primal_violation'],
            kkt['max_dual_violation'],
            kkt['max_complementary_slackness_violation']
        ]
        
        x_pos = np.arange(len(condition_names)) + i*0.35
        bars = ax3.bar(x_pos, violations, width=0.35, 
                      label=f'Problem {i+1}', alpha=0.7)
        
        # Color bars based on satisfaction (green if < 1e-6, red otherwise)
        for bar, violation in zip(bars, violations):
            if violation < 1e-6:
                bar.set_color('green')
            else:
                bar.set_color('red')
    
    ax3.set_yscale('log')
    ax3.set_ylabel('Violation Magnitude (log scale)')
    ax3.set_title('KKT Condition Violations')
    ax3.set_xticks(np.arange(len(condition_names)) + 0.175)
    ax3.set_xticklabels(condition_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Complementary slackness illustration
    ax4 = plt.subplot(2, 3, 4)
    
    # Show complementary slackness graphically for Problem 1
    x_test_range = np.linspace(0, 1, 50)
    lambdas = []
    constraint_vals = []
    products = []
    
    # For each point on constraint boundary, show λ and g(x)
    for x1 in x_test_range:
        x2 = 1 - x1  # On constraint boundary
        x = np.array([x1, x2])
        
        # Constraint value (should be 0 on boundary)
        g_val = problem1.inequality_constraints[0](x)
        constraint_vals.append(g_val)
        
        # Compute required λ from stationarity condition
        grad_f = problem1.objective_grad(x)
        grad_g = problem1.inequality_grads[0](x)
        # ∇f + λ∇g = 0  =>  λ = -∇f·∇g / ||∇g||²
        lambda_val = -np.dot(grad_f, grad_g) / np.dot(grad_g, grad_g)
        lambdas.append(max(0, lambda_val))  # Dual feasibility: λ ≥ 0
        
        products.append(lambdas[-1] * abs(constraint_vals[-1]))
    
    ax4.plot(x_test_range, lambdas, 'b-', linewidth=2, label='λ(x)')
    ax4.plot(x_test_range, np.abs(constraint_vals), 'r--', linewidth=2, label='|g(x)|')
    ax4.plot(x_test_range, products, 'g:', linewidth=3, label='λ·|g(x)|')
    
    # Mark optimal point
    opt_x1 = solution1['x_optimal'][0]
    opt_idx = np.argmin(np.abs(x_test_range - opt_x1))
    ax4.plot(opt_x1, lambdas[opt_idx], 'go', markersize=10, label='Optimal point')
    
    ax4.set_xlabel('x₁ (along constraint boundary)')
    ax4.set_ylabel('Value')
    ax4.set_title('Complementary Slackness: λ·g(x) = 0')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Active Set Evolution
    ax5 = plt.subplot(2, 3, 5)
    
    # Show how active set changes along solution path
    # For problem 1, trace path from interior to boundary
    
    alphas = np.linspace(0, 2, 100)
    active_set_sizes = []
    objective_values = []
    
    for alpha in alphas:
        # Test point: move from (0,0) toward (0.5, 0.5)
        x_test = alpha * np.array([0.25, 0.25])
        
        # Check if feasible
        if problem1.is_feasible(x_test):
            # Count active constraints
            active = problem1.active_inequality_constraints(x_test)
            active_set_sizes.append(len(active))
            objective_values.append(problem1.objective(x_test))
        else:
            active_set_sizes.append(np.nan)
            objective_values.append(np.nan)
    
    ax5_twin = ax5.twinx()
    
    line1 = ax5.plot(alphas, active_set_sizes, 'b-', linewidth=2, label='Active Set Size')
    line2 = ax5_twin.plot(alphas, objective_values, 'r--', linewidth=2, label='Objective Value')
    
    ax5.set_xlabel('Parameter α')
    ax5.set_ylabel('Active Set Size', color='blue')
    ax5_twin.set_ylabel('Objective Value', color='red')
    ax5.set_title('Active Set Evolution')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, loc='upper left')
    
    ax5.grid(True, alpha=0.3)
    
    # KKT Interpretation
    ax6 = plt.subplot(2, 3, 6)
    
    # Geometric interpretation of KKT conditions
    # Show normal cones and gradient alignment
    
    x_opt = solution1['x_optimal']
    
    # Plot feasible region again
    x1_viz = np.linspace(-0.2, 1.2, 100)
    x2_viz = 1 - x1_viz
    ax6.fill_between(x1_viz, x2_viz, -0.2, alpha=0.2, color='lightblue', 
                     label='Feasible region')
    ax6.plot(x1_viz, x2_viz, 'k-', linewidth=2, label='Active constraint')
    
    # Plot optimal point
    ax6.plot(x_opt[0], x_opt[1], 'ro', markersize=12, label='x*')
    
    # Show gradient of objective
    grad_f = problem1.objective_grad(x_opt)
    ax6.arrow(x_opt[0], x_opt[1], 0.3*grad_f[0], 0.3*grad_f[1],
             head_width=0.03, head_length=0.05, fc='blue', ec='blue',
             linewidth=3, label='∇f(x*)')
    
    # Show normal to constraint (outward)
    grad_g = problem1.inequality_grads[0](x_opt)
    ax6.arrow(x_opt[0], x_opt[1], -0.3*grad_g[0], -0.3*grad_g[1],
             head_width=0.03, head_length=0.05, fc='red', ec='red',
             linewidth=3, label='Normal to constraint')
    
    # Show that they are aligned (KKT condition)
    ax6.text(0.7, 0.8, 'KKT: ∇f(x*) = λ*∇g(x*)\n(Anti-parallel alignment)', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
             fontsize=10)
    
    ax6.set_xlim(-0.2, 1.2)
    ax6.set_ylim(-0.2, 1.2)
    ax6.set_xlabel('x₁')
    ax6.set_ylabel('x₂')
    ax6.set_title('Geometric KKT Interpretation')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()


def kkt_theory_and_applications():
    """
    Theoretical background and applications of KKT conditions.
    """
    print("\n🎓 KKT THEORY AND APPLICATIONS")
    print("=" * 40)
    
    print("📚 THEORETICAL FOUNDATIONS:")
    print("1. NECESSITY:")
    print("   - Under constraint qualifications (LICQ, MFCQ, etc.)")
    print("   - Local optimizers satisfy KKT conditions")
    print("   - Generalization of Lagrange multipliers")
    
    print("\n2. SUFFICIENCY:")
    print("   - For convex problems: KKT conditions are sufficient")
    print("   - f convex, g_i convex, h_j affine => KKT point is global optimum")
    print("   - Second-order conditions for non-convex problems")
    
    print("\n3. CONSTRAINT QUALIFICATIONS:")
    print("   - LICQ: Linear Independence of Constraint Qualifications")
    print("   - MFCQ: Mangasarian-Fromovitz Constraint Qualification")
    print("   - Slater condition (for convex problems)")
    print("   - Ensure KKT conditions are necessary")
    
    print("\n🎯 APPLICATIONS:")
    print("1. PORTFOLIO OPTIMIZATION:")
    print("   - Modern portfolio theory")
    print("   - Risk-return trade-offs")
    print("   - Capital allocation models")
    
    print("\n2. MACHINE LEARNING:")
    print("   - Support Vector Machines (SVM)")
    print("   - Neural network training")
    print("   - Regularized optimization")
    
    print("\n3. ENGINEERING DESIGN:")
    print("   - Structural optimization")
    print("   - Control system design")
    print("   - Resource allocation")
    
    print("\n4. ECONOMICS:")
    print("   - Utility maximization")
    print("   - Production optimization")
    print("   - Market equilibrium")
    
    print("\n⚡ COMPUTATIONAL ASPECTS:")
    print("1. ACTIVE SET METHODS:")
    print("   - Sequential quadratic programming (SQP)")
    print("   - Identify active constraints iteratively")
    print("   - Solve sequence of quadratic subproblems")
    
    print("\n2. INTERIOR POINT METHODS:")
    print("   - Barrier/penalty functions")
    print("   - Newton's method for KKT system")
    print("   - Polynomial-time complexity for convex problems")
    
    print("\n3. AUGMENTED LAGRANGIAN:")
    print("   - Penalty method + Lagrange multipliers")
    print("   - Better conditioning than pure penalty")
    print("   - Convergence guarantees under mild assumptions")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_kkt_conditions()
    kkt_theory_and_applications()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- KKT conditions extend Lagrange multipliers to inequality constraints")
    print("- Four conditions: stationarity, primal/dual feasibility, complementary slackness")
    print("- Necessary under constraint qualifications, sufficient for convex problems")
    print("- Foundation for modern constrained optimization algorithms")
    print("- Critical for SVM, portfolio theory, engineering design")
    print("- Active set interpretation guides computational methods")
    print("\nKKT conditions are the cornerstone of constrained optimization! 🚀")