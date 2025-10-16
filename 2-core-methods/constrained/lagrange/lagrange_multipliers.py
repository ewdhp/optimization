"""
Lagrange Multiplier Theorem

The Lagrange multiplier theorem provides necessary conditions for optimality
in constrained optimization problems. It's the foundation for understanding
how constraints affect optimal solutions and leads to the KKT conditions.

Mathematical Statement:
Consider the problem:
    minimize    f(x)
    subject to  g_i(x) = 0,  i = 1,...,m

If x* is a local minimum and the constraint gradients ∇g_i(x*) are linearly
independent, then there exist multipliers λ*_i such that:

∇f(x*) + Σᵢ λ*_i ∇g_i(x*) = 0

Geometric Interpretation:
At the optimum, the gradient of the objective function is a linear combination
of the constraint gradients. This means the objective gradient lies in the
span of the constraint gradients.

Applications:
- Constrained optimization problems
- Utility maximization with budget constraints (economics)
- Lagrangian mechanics and optimal control
- Support vector machines and machine learning
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Tuple, List, Optional
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

class LagrangeMultiplierProblem:
    """
    Implementation of constrained optimization using Lagrange multipliers.
    Provides methods to find critical points and verify optimality conditions.
    """
    
    def __init__(self, objective: Callable[[np.ndarray], float],
                 objective_grad: Callable[[np.ndarray], np.ndarray],
                 constraints: List[Callable[[np.ndarray], float]],
                 constraint_grads: List[Callable[[np.ndarray], np.ndarray]]):
        """
        Initialize Lagrange multiplier problem.
        
        Args:
            objective: Objective function f(x)
            objective_grad: Gradient of objective ∇f(x)
            constraints: List of equality constraints gᵢ(x) = 0
            constraint_grads: List of constraint gradients ∇gᵢ(x)
        """
        self.objective = objective
        self.objective_grad = objective_grad
        self.constraints = constraints
        self.constraint_grads = constraint_grads
        self.n_constraints = len(constraints)
    
    def lagrangian(self, x: np.ndarray, lam: np.ndarray) -> float:
        """
        Evaluate Lagrangian: L(x,λ) = f(x) + Σᵢ λᵢ gᵢ(x)
        """
        L = self.objective(x)
        for i, constraint in enumerate(self.constraints):
            L += lam[i] * constraint(x)
        return L
    
    def lagrangian_gradient(self, x: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """
        Gradient of Lagrangian with respect to x: ∇ₓL = ∇f + Σᵢ λᵢ ∇gᵢ
        """
        grad_L = self.objective_grad(x).copy()
        for i, constraint_grad in enumerate(self.constraint_grads):
            grad_L += lam[i] * constraint_grad(x)
        return grad_L
    
    def constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian matrix of constraints: J[i,j] = ∂gᵢ/∂xⱼ
        """
        n_vars = len(x)
        jacobian = np.zeros((self.n_constraints, n_vars))
        
        for i, constraint_grad in enumerate(self.constraint_grads):
            jacobian[i] = constraint_grad(x)
        
        return jacobian
    
    def check_licq(self, x: np.ndarray, tol: float = 1e-12) -> bool:
        """
        Check Linear Independence Constraint Qualification (LICQ).
        
        LICQ requires that constraint gradients be linearly independent.
        """
        jacobian = self.constraint_jacobian(x)
        
        if self.n_constraints == 0:
            return True
        
        # Check rank of constraint Jacobian
        rank = np.linalg.matrix_rank(jacobian, tol=tol)
        return rank == self.n_constraints
    
    def solve_kkt_system(self, x_guess: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Solve KKT system for critical point.
        
        System:
        ∇f(x) + Σᵢ λᵢ ∇gᵢ(x) = 0
        gᵢ(x) = 0  for all i
        
        Returns:
            (x_optimal, lambda_optimal, success)
        """
        def kkt_residual(vars):
            n_vars = len(x_guess)
            x = vars[:n_vars]
            lam = vars[n_vars:]
            
            # Stationarity condition: ∇ₓL = 0
            grad_L = self.lagrangian_gradient(x, lam)
            
            # Constraint equations: g(x) = 0
            constraint_vals = np.array([constraint(x) for constraint in self.constraints])
            
            return np.concatenate([grad_L, constraint_vals])
        
        # Initial guess for (x, λ)
        initial_lam = np.zeros(self.n_constraints)
        initial_vars = np.concatenate([x_guess, initial_lam])
        
        # Solve nonlinear system
        from scipy.optimize import fsolve
        try:
            solution = fsolve(kkt_residual, initial_vars, xtol=1e-12)
            n_vars = len(x_guess)
            x_opt = solution[:n_vars]
            lam_opt = solution[n_vars:]
            
            # Check if solution satisfies KKT conditions
            residual = kkt_residual(solution)
            success = np.linalg.norm(residual) < 1e-8
            
            return x_opt, lam_opt, success
            
        except:
            return x_guess, np.zeros(self.n_constraints), False
    
    def compute_multipliers_analytical(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute Lagrange multipliers analytically when possible.
        
        If LICQ holds: λ = -(∇g(x))⁻¹ ∇f(x)
        where ∇g(x) is the constraint Jacobian.
        """
        if not self.check_licq(x):
            return None
        
        jacobian = self.constraint_jacobian(x)
        obj_grad = self.objective_grad(x)
        
        try:
            # Solve: jacobian^T λ = -obj_grad
            multipliers = -np.linalg.solve(jacobian, obj_grad)
            return multipliers
        except np.linalg.LinAlgError:
            return None
    
    def second_order_conditions(self, x: np.ndarray, lam: np.ndarray,
                               hessian_f: Optional[Callable] = None,
                               hessian_g: Optional[List[Callable]] = None) -> Tuple[bool, np.ndarray]:
        """
        Check second-order sufficiency conditions.
        
        The bordered Hessian must have the right inertia.
        """
        if hessian_f is None or hessian_g is None:
            return False, np.array([])
        
        # Compute Hessian of Lagrangian
        H_L = hessian_f(x)
        for i in range(self.n_constraints):
            H_L += lam[i] * hessian_g[i](x)
        
        # Bordered Hessian
        jacobian = self.constraint_jacobian(x)
        n_vars = len(x)
        
        bordered_hessian = np.zeros((n_vars + self.n_constraints, n_vars + self.n_constraints))
        bordered_hessian[:n_vars, :n_vars] = H_L
        bordered_hessian[:n_vars, n_vars:] = jacobian.T
        bordered_hessian[n_vars:, :n_vars] = jacobian
        
        # Check eigenvalues for sufficient conditions
        eigenvalues = np.linalg.eigvals(bordered_hessian)
        
        # For minimum: last n-m eigenvalues should be positive
        # (where n = variables, m = constraints)
        n_positive = np.sum(eigenvalues > 1e-12)
        n_negative = np.sum(eigenvalues < -1e-12)
        
        is_minimum = n_negative == self.n_constraints
        
        return is_minimum, eigenvalues


def demonstrate_lagrange_multipliers():
    """
    Demonstrate Lagrange multiplier theorem with visual examples.
    """
    print("🔷 LAGRANGE MULTIPLIER THEOREM: Constrained Optimization")
    print("=" * 65)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Classic 2D example: minimize x² + y² subject to x + y = 1
    ax1 = plt.subplot(2, 3, 1)
    
    # Problem setup
    def objective_1(x):
        return x[0]**2 + x[1]**2
    
    def objective_grad_1(x):
        return np.array([2*x[0], 2*x[1]])
    
    def constraint_1(x):
        return x[0] + x[1] - 1
    
    def constraint_grad_1(x):
        return np.array([1.0, 1.0])
    
    lagrange_problem_1 = LagrangeMultiplierProblem(
        objective_1, objective_grad_1, [constraint_1], [constraint_grad_1])
    
    # Create visualization
    x_range = np.linspace(-1, 2, 100)
    y_range = np.linspace(-1, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + Y**2
    
    # Plot objective function contours
    contours = ax1.contour(X, Y, Z, levels=15, colors='blue', alpha=0.6)
    ax1.clabel(contours, inline=True, fontsize=8)
    
    # Plot constraint line
    constraint_x = np.linspace(-0.5, 1.5, 100)
    constraint_y = 1 - constraint_x
    ax1.plot(constraint_x, constraint_y, 'red', linewidth=3, label='Constraint: x + y = 1')
    
    # Solve for optimum
    x_opt, lam_opt, success = lagrange_problem_1.solve_kkt_system(np.array([0.5, 0.5]))
    
    if success:
        ax1.plot(x_opt[0], x_opt[1], 'ro', markersize=12, label=f'Optimum: ({x_opt[0]:.2f}, {x_opt[1]:.2f})')
        
        # Show gradients at optimum
        obj_grad = objective_grad_1(x_opt)
        const_grad = constraint_grad_1(x_opt)
        
        scale = 0.3
        ax1.arrow(x_opt[0], x_opt[1], scale*obj_grad[0], scale*obj_grad[1],
                  head_width=0.05, head_length=0.05, fc='blue', ec='blue',
                  label='∇f')
        ax1.arrow(x_opt[0], x_opt[1], -scale*lam_opt[0]*const_grad[0], 
                  -scale*lam_opt[0]*const_grad[1],
                  head_width=0.05, head_length=0.05, fc='green', ec='green',
                  label=f'−λ∇g (λ={lam_opt[0]:.2f})')
    
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_title('Minimize x² + y² s.t. x + y = 1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. 3D visualization of Lagrangian
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    
    # Create surface plot of constraint and objective level sets
    x_3d = np.linspace(0, 1, 30)
    y_3d = np.linspace(0, 1, 30)
    X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
    
    # Objective function surface
    Z_obj = X_3d**2 + Y_3d**2
    ax2.plot_surface(X_3d, Y_3d, Z_obj, alpha=0.3, color='blue', label='Objective')
    
    # Constraint surface (plane)
    Z_constraint = np.ones_like(X_3d) - X_3d - Y_3d
    ax2.plot_surface(X_3d, Y_3d, Z_constraint, alpha=0.3, color='red', label='Constraint')
    
    # Mark optimum
    if success:
        ax2.scatter([x_opt[0]], [x_opt[1]], [objective_1(x_opt)], 
                   color='black', s=100, label='Optimum')
    
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_zlabel('Value')
    ax2.set_title('3D View: Objective and Constraint')
    
    # 3. Multiple constraints example
    ax3 = plt.subplot(2, 3, 3)
    
    # Problem: minimize (x-2)² + (y-1)² subject to x² + y² = 1 and x ≥ 0
    def objective_2(x):
        return (x[0] - 2)**2 + (x[1] - 1)**2
    
    def objective_grad_2(x):
        return np.array([2*(x[0] - 2), 2*(x[1] - 1)])
    
    def constraint_2a(x):
        return x[0]**2 + x[1]**2 - 1
    
    def constraint_grad_2a(x):
        return np.array([2*x[0], 2*x[1]])
    
    lagrange_problem_2 = LagrangeMultiplierProblem(
        objective_2, objective_grad_2, [constraint_2a], [constraint_grad_2a])
    
    # Plot unit circle constraint
    theta = np.linspace(0, 2*np.pi, 200)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax3.plot(circle_x, circle_y, 'red', linewidth=3, label='Constraint: x² + y² = 1')
    
    # Plot objective contours
    x_range_2 = np.linspace(-2, 3, 100)
    y_range_2 = np.linspace(-2, 2, 100)
    X2, Y2 = np.meshgrid(x_range_2, y_range_2)
    Z2 = (X2 - 2)**2 + (Y2 - 1)**2
    
    contours_2 = ax3.contour(X2, Y2, Z2, levels=15, colors='blue', alpha=0.6)
    ax3.clabel(contours_2, inline=True, fontsize=8)
    
    # Find optimum on circle
    x_opt_2, lam_opt_2, success_2 = lagrange_problem_2.solve_kkt_system(np.array([0.8, 0.6]))
    
    if success_2:
        ax3.plot(x_opt_2[0], x_opt_2[1], 'ro', markersize=12, 
                 label=f'Optimum: ({x_opt_2[0]:.2f}, {x_opt_2[1]:.2f})')
        
        # Show center of objective circles
        ax3.plot(2, 1, 'bs', markersize=10, label='Unconstrained optimum (2,1)')
        
        # Connect to show constraint effect
        ax3.plot([2, x_opt_2[0]], [1, x_opt_2[1]], 'g--', linewidth=2, alpha=0.7,
                 label='Effect of constraint')
    
    ax3.set_xlim(-1.5, 2.5)
    ax3.set_ylim(-1.5, 2)
    ax3.set_title('Minimize (x-2)² + (y-1)² s.t. x² + y² = 1')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # 4. Sensitivity analysis
    ax4 = plt.subplot(2, 3, 4)
    
    # Analyze how optimal value changes with constraint parameter
    # Problem: minimize x² + y² subject to x + y = c
    constraint_params = np.linspace(0.5, 2.0, 50)
    optimal_values = []
    multipliers = []
    
    for c in constraint_params:
        def constraint_param(x):
            return x[0] + x[1] - c
        
        problem_param = LagrangeMultiplierProblem(
            objective_1, objective_grad_1, [constraint_param], [constraint_grad_1])
        
        x_opt_param, lam_opt_param, success_param = problem_param.solve_kkt_system(np.array([c/2, c/2]))
        
        if success_param:
            optimal_values.append(objective_1(x_opt_param))
            multipliers.append(lam_opt_param[0])
        else:
            optimal_values.append(np.nan)
            multipliers.append(np.nan)
    
    ax4.plot(constraint_params, optimal_values, 'b-', linewidth=2, label='Optimal value')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(constraint_params, multipliers, 'r--', linewidth=2, label='Multiplier λ')
    
    ax4.set_xlabel('Constraint parameter c')
    ax4.set_ylabel('Optimal value', color='blue')
    ax4_twin.set_ylabel('Multiplier λ', color='red')
    ax4.set_title('Sensitivity Analysis: ∂f*/∂c = λ*')
    ax4.grid(True, alpha=0.3)
    
    # Show that slope of optimal value equals multiplier
    ax4.text(0.7, 0.8, 'Envelope theorem:\n∂f*/∂c = λ*', 
             transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 5. LICQ violation example
    ax5 = plt.subplot(2, 3, 5)
    
    # Problem where LICQ fails: constraints g₁(x,y) = x² and g₂(x,y) = y
    def constraint_licq_1(x):
        return x[0]**2
    
    def constraint_licq_2(x):
        return x[1]
    
    def constraint_grad_licq_1(x):
        return np.array([2*x[0], 0])
    
    def constraint_grad_licq_2(x):
        return np.array([0, 1])
    
    # At origin, both gradients become [0,0] and [0,1] - not full rank
    test_points = [np.array([0, 0]), np.array([1, 0]), np.array([0.5, 0])]
    
    problem_licq = LagrangeMultiplierProblem(
        objective_1, objective_grad_1, 
        [constraint_licq_1, constraint_licq_2],
        [constraint_grad_licq_1, constraint_grad_licq_2])
    
    for i, point in enumerate(test_points):
        licq_satisfied = problem_licq.check_licq(point)
        jacobian = problem_licq.constraint_jacobian(point)
        
        color = 'green' if licq_satisfied else 'red'
        marker = 'o' if licq_satisfied else 'x'
        
        ax5.plot(point[0], point[1], color=color, marker=marker, markersize=10,
                 label=f'Point {i+1}: LICQ {"OK" if licq_satisfied else "FAIL"}')
        
        # Show constraint gradients
        if i == 0:  # At origin
            ax5.annotate(f'∇g₁ = [{jacobian[0,0]:.1f}, {jacobian[0,1]:.1f}]',
                        xy=point, xytext=(point[0]+0.3, point[1]+0.2),
                        arrowprops=dict(arrowstyle='->', color='red'))
            ax5.annotate(f'∇g₂ = [{jacobian[1,0]:.1f}, {jacobian[1,1]:.1f}]',
                        xy=point, xytext=(point[0]+0.3, point[1]-0.2),
                        arrowprops=dict(arrowstyle='->', color='red'))
    
    # Plot constraints
    x_licq = np.linspace(-1, 1, 100)
    ax5.axhline(0, color='blue', linewidth=2, label='g₂(x,y) = y = 0')
    ax5.axvline(0, color='purple', linewidth=2, label='g₁(x,y) = x² = 0')
    
    ax5.set_xlim(-0.8, 1.2)
    ax5.set_ylim(-0.5, 0.5)
    ax5.set_title('LICQ Violation at Origin')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Economic interpretation
    ax6 = plt.subplot(2, 3, 6)
    
    # Utility maximization with budget constraint
    # maximize U(x,y) = x^α * y^β subject to px*x + py*y = I
    alpha, beta = 0.6, 0.4
    px, py, I = 2, 1, 10
    
    def utility(x):
        if x[0] <= 0 or x[1] <= 0:
            return -np.inf
        return x[0]**alpha * x[1]**beta
    
    def utility_grad(x):
        if x[0] <= 0 or x[1] <= 0:
            return np.array([0, 0])
        return np.array([alpha * x[0]**(alpha-1) * x[1]**beta,
                        beta * x[0]**alpha * x[1]**(beta-1)])
    
    def budget_constraint(x):
        return px*x[0] + py*x[1] - I
    
    def budget_grad(x):
        return np.array([px, py])
    
    # Analytical solution: x* = αI/px, y* = βI/py
    x_analytical = alpha * I / px
    y_analytical = beta * I / py
    
    # Plot budget line
    x_budget = np.linspace(0, I/px, 100)
    y_budget = (I - px*x_budget) / py
    ax6.plot(x_budget, y_budget, 'red', linewidth=3, label=f'Budget: {px}x + {py}y = {I}')
    
    # Plot indifference curves (utility level sets)
    x_range_econ = np.linspace(0.1, 5, 100)
    y_range_econ = np.linspace(0.1, 10, 100)
    X_econ, Y_econ = np.meshgrid(x_range_econ, y_range_econ)
    
    # Avoid log of zero by masking
    mask = (X_econ > 0) & (Y_econ > 0)
    U_econ = np.full_like(X_econ, np.nan)
    U_econ[mask] = X_econ[mask]**alpha * Y_econ[mask]**beta
    
    contours_econ = ax6.contour(X_econ, Y_econ, U_econ, levels=10, colors='blue', alpha=0.6)
    ax6.clabel(contours_econ, inline=True, fontsize=8)
    
    # Plot optimal point
    ax6.plot(x_analytical, y_analytical, 'ro', markersize=12, 
             label=f'Optimum: ({x_analytical:.2f}, {y_analytical:.2f})')
    
    # Show tangency condition (MRS = price ratio)
    mrs = (alpha/beta) * (y_analytical/x_analytical)  # Marginal rate of substitution
    price_ratio = px / py
    ax6.text(2, 8, f'MRS = {mrs:.2f}\nPrice ratio = {price_ratio:.2f}',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax6.set_xlim(0, 6)
    ax6.set_ylim(0, 12)
    ax6.set_title('Utility Maximization (Economics)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Numerical verification
    print("\n📊 NUMERICAL VERIFICATION")
    print("-" * 40)
    
    print("1. Simple Quadratic Problem:")
    print(f"   Problem: minimize x² + y² subject to x + y = 1")
    if success:
        print(f"   Solution: x* = ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
        print(f"   Multiplier: λ* = {lam_opt[0]:.6f}")
        print(f"   Optimal value: f* = {objective_1(x_opt):.6f}")
        
        # Verify KKT conditions
        grad_L = lagrange_problem_1.lagrangian_gradient(x_opt, lam_opt)
        constraint_violation = abs(constraint_1(x_opt))
        print(f"   ∇L residual: {np.linalg.norm(grad_L):.2e}")
        print(f"   Constraint violation: {constraint_violation:.2e}")
    
    print("\n2. Circle Constraint Problem:")
    if success_2:
        print(f"   Problem: minimize (x-2)² + (y-1)² subject to x² + y² = 1")
        print(f"   Solution: x* = ({x_opt_2[0]:.6f}, {x_opt_2[1]:.6f})")
        print(f"   Multiplier: λ* = {lam_opt_2[0]:.6f}")
        print(f"   Optimal value: f* = {objective_2(x_opt_2):.6f}")
        
        # Distance from unconstrained optimum
        unconstrained_opt = np.array([2, 1])
        distance = np.linalg.norm(x_opt_2 - unconstrained_opt)
        print(f"   Distance from unconstrained optimum: {distance:.6f}")
    
    print("\n3. Economic Problem:")
    print(f"   Utility maximization with budget constraint")
    print(f"   Optimal consumption: ({x_analytical:.3f}, {y_analytical:.3f})")
    print(f"   Maximum utility: {utility(np.array([x_analytical, y_analytical])):.6f}")
    print(f"   Budget exhausted: ${px*x_analytical + py*y_analytical:.2f} = ${I}")


def lagrange_applications():
    """
    Showcase applications of Lagrange multiplier theorem.
    """
    print("\n🎯 LAGRANGE MULTIPLIER APPLICATIONS")
    print("=" * 40)
    
    print("1. ECONOMICS - Consumer Theory:")
    print("   - Utility maximization subject to budget constraints")
    print("   - Demand function derivation")
    print("   - Slutsky equation and substitution effects")
    
    print("\n2. PORTFOLIO OPTIMIZATION:")
    print("   - Minimize risk subject to target return")
    print("   - Budget constraint: Σwᵢ = 1")
    print("   - Long-only constraint: wᵢ ≥ 0")
    
    print("\n3. MACHINE LEARNING:")
    print("   - Support Vector Machines (SVM)")
    print("   - Principal Component Analysis (PCA)")
    print("   - Regularized regression (Ridge, Lasso)")
    
    print("\n4. ENGINEERING DESIGN:")
    print("   - Minimum weight structures")
    print("   - Optimal control problems")
    print("   - Signal processing with power constraints")
    
    print("\n5. PHYSICS:")
    print("   - Lagrangian mechanics")
    print("   - Principle of least action")
    print("   - Constrained motion on surfaces")
    
    print("\n6. OPERATIONS RESEARCH:")
    print("   - Resource allocation problems")
    print("   - Production planning with capacity constraints")
    print("   - Network flow optimization")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_lagrange_multipliers()
    lagrange_applications()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Lagrange multipliers provide necessary optimality conditions")
    print("- ∇f + Σλᵢ∇gᵢ = 0 (stationarity condition)")
    print("- Requires Linear Independence Constraint Qualification (LICQ)")
    print("- Multipliers have economic interpretation (shadow prices)")
    print("- Foundation for KKT conditions and modern optimization")
    print("- Essential for constrained optimization in all fields")
    print("\nLagrange multipliers bridge geometry and algebra in optimization! 🚀")