"""
Convergence Theory for Optimization Algorithms

Convergence theory provides mathematical foundations for understanding
when and how fast optimization algorithms reach optimal solutions.
This module covers key convergence results for major algorithm classes.

Key Convergence Types:
1. Global Convergence: Algorithm finds global optimum from any starting point
2. Local Convergence: Algorithm converges to local optimum from nearby starts  
3. Linear Convergence: ||x_{k+1} - x*|| ≤ ρ||x_k - x*|| where 0 < ρ < 1
4. Superlinear: lim_{k→∞} ||x_{k+1} - x*|| / ||x_k - x*|| = 0
5. Quadratic: ||x_{k+1} - x*|| ≤ C||x_k - x*||²

Major Theorems:
- Descent Lemma: f(x + αd) ≤ f(x) + cα∇f(x)ᵀd for step sizes
- Armijo-Goldstein: Line search convergence conditions
- BFGS Convergence: Superlinear convergence under standard assumptions
- Newton Convergence: Quadratic convergence near optimum
- Gradient Descent: Linear convergence for strongly convex functions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

class ConvergenceAnalyzer:
    """
    Analyzes convergence properties of optimization algorithms.
    """
    
    def __init__(self, 
                 objective: Callable[[np.ndarray], float],
                 gradient: Callable[[np.ndarray], np.ndarray],
                 hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 true_optimum: Optional[np.ndarray] = None,
                 name: str = "Function"):
        """
        Initialize convergence analyzer.
        
        Args:
            objective: Objective function f(x)
            gradient: Gradient function ∇f(x)
            hessian: Hessian function ∇²f(x)
            true_optimum: Known optimal point x*
            name: Function name for display
        """
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.true_optimum = true_optimum
        self.name = name
    
    def compute_convergence_rate(self, sequence: List[np.ndarray]) -> Dict:
        """
        Analyze convergence rate of optimization sequence.
        
        Args:
            sequence: List of iterates [x_0, x_1, ..., x_k]
            
        Returns:
            Dictionary with convergence analysis
        """
        if len(sequence) < 3:
            return {'error': 'Need at least 3 iterates for analysis'}
        
        # Convert to numpy arrays
        iterates = [np.array(x) for x in sequence]
        n_iters = len(iterates)
        
        # Compute error norms (if true optimum known)
        error_norms = []
        if self.true_optimum is not None:
            x_star = np.array(self.true_optimum)
            error_norms = [np.linalg.norm(x - x_star) for x in iterates]
        else:
            # Use final iterate as proxy for optimum
            x_final = iterates[-1]
            error_norms = [np.linalg.norm(x - x_final) for x in iterates[:-1]]
            error_norms.append(0.0)  # Final error is zero by definition
        
        # Function value convergence
        function_values = [self.objective(x) for x in iterates]
        if self.true_optimum is not None:
            f_star = self.objective(self.true_optimum)
        else:
            f_star = function_values[-1]
        
        function_errors = [abs(f - f_star) for f in function_values]
        
        # Gradient norm convergence
        gradient_norms = [np.linalg.norm(self.gradient(x)) for x in iterates]
        
        # Analyze convergence rates
        convergence_analysis = {}
        
        # 1. Linear convergence test
        linear_rates = []
        for k in range(1, n_iters-1):
            if error_norms[k] > 1e-12 and error_norms[k-1] > 1e-12:
                rate = error_norms[k+1] / error_norms[k]
                linear_rates.append(rate)
        
        if linear_rates:
            avg_linear_rate = np.mean(linear_rates)
            is_linear = len([r for r in linear_rates if 0.1 < r < 0.9]) > len(linear_rates) // 2
            convergence_analysis['linear_rate'] = avg_linear_rate
            convergence_analysis['is_linear_convergent'] = is_linear
        else:
            convergence_analysis['linear_rate'] = None
            convergence_analysis['is_linear_convergent'] = False
        
        # 2. Superlinear convergence test
        superlinear_ratios = []
        for k in range(1, n_iters-1):
            if error_norms[k] > 1e-12:
                ratio = error_norms[k+1] / error_norms[k]
                superlinear_ratios.append(ratio)
        
        if len(superlinear_ratios) > 3:
            # Superlinear if ratios are decreasing toward 0
            is_superlinear = (len(superlinear_ratios) > 1 and 
                            superlinear_ratios[-1] < superlinear_ratios[0] * 0.5)
        else:
            is_superlinear = False
        
        convergence_analysis['superlinear_ratios'] = superlinear_ratios
        convergence_analysis['is_superlinear'] = is_superlinear
        
        # 3. Quadratic convergence test
        quadratic_constants = []
        for k in range(n_iters-1):
            if error_norms[k] > 1e-12:
                constant = error_norms[k+1] / (error_norms[k]**2)
                if np.isfinite(constant):
                    quadratic_constants.append(constant)
        
        if quadratic_constants:
            is_quadratic = (len(quadratic_constants) > 2 and 
                          np.std(quadratic_constants[-3:]) < np.mean(quadratic_constants[-3:]) * 0.5)
        else:
            is_quadratic = False
        
        convergence_analysis['quadratic_constants'] = quadratic_constants
        convergence_analysis['is_quadratic'] = is_quadratic
        
        # Store sequences for plotting
        convergence_analysis['iterates'] = iterates
        convergence_analysis['error_norms'] = error_norms
        convergence_analysis['function_values'] = function_values
        convergence_analysis['function_errors'] = function_errors
        convergence_analysis['gradient_norms'] = gradient_norms
        convergence_analysis['n_iterations'] = n_iters
        
        return convergence_analysis
    
    def gradient_descent(self, x0: np.ndarray, 
                        step_size: float = 0.01,
                        max_iters: int = 1000,
                        tolerance: float = 1e-8) -> List[np.ndarray]:
        """
        Run gradient descent algorithm.
        """
        x = np.array(x0)
        sequence = [x.copy()]
        
        for k in range(max_iters):
            grad = self.gradient(x)
            
            if np.linalg.norm(grad) < tolerance:
                break
            
            x = x - step_size * grad
            sequence.append(x.copy())
        
        return sequence
    
    def newton_method(self, x0: np.ndarray,
                     max_iters: int = 100,
                     tolerance: float = 1e-8) -> List[np.ndarray]:
        """
        Run Newton's method (requires Hessian).
        """
        if self.hessian is None:
            raise ValueError("Newton method requires Hessian function")
        
        x = np.array(x0)
        sequence = [x.copy()]
        
        for k in range(max_iters):
            grad = self.gradient(x)
            
            if np.linalg.norm(grad) < tolerance:
                break
            
            try:
                hess = self.hessian(x)
                # Solve Newton system: H * p = -∇f
                p = np.linalg.solve(hess, -grad)
                x = x + p
                sequence.append(x.copy())
            except np.linalg.LinAlgError:
                # Singular Hessian - fall back to gradient step
                x = x - 0.01 * grad
                sequence.append(x.copy())
        
        return sequence
    
    def bfgs_method(self, x0: np.ndarray,
                   max_iters: int = 1000,
                   tolerance: float = 1e-8) -> List[np.ndarray]:
        """
        Run BFGS quasi-Newton method.
        """
        x = np.array(x0)
        n = len(x)
        
        # Initialize inverse Hessian approximation
        B_inv = np.eye(n)
        
        sequence = [x.copy()]
        grad = self.gradient(x)
        
        for k in range(max_iters):
            if np.linalg.norm(grad) < tolerance:
                break
            
            # BFGS direction
            p = -np.dot(B_inv, grad)
            
            # Line search (simple backtracking)
            alpha = 1.0
            x_new = x + alpha * p
            
            # Simple Armijo condition
            while (self.objective(x_new) > 
                   self.objective(x) + 1e-4 * alpha * np.dot(grad, p)):
                alpha *= 0.5
                x_new = x + alpha * p
                if alpha < 1e-10:
                    break
            
            # Update
            s = x_new - x
            x = x_new
            grad_new = self.gradient(x)
            y = grad_new - grad
            
            # BFGS update of inverse Hessian
            rho = 1.0 / np.dot(y, s)
            if np.isfinite(rho) and rho > 1e-10:
                A1 = np.eye(n) - rho * np.outer(s, y)
                A2 = np.eye(n) - rho * np.outer(y, s)
                B_inv = np.dot(A1, np.dot(B_inv, A2)) + rho * np.outer(s, s)
            
            grad = grad_new
            sequence.append(x.copy())
        
        return sequence


class ConvergenceTheorems:
    """
    Implementation of key convergence theorems.
    """
    
    @staticmethod
    def strongly_convex_quadratic():
        """
        Strongly convex quadratic: f(x) = ½xᵀQx - bᵀx
        with condition number κ = λ_max/λ_min.
        """
        # Create well-conditioned quadratic
        eigenvals = np.array([1, 4])  # Condition number = 4
        Q = np.diag(eigenvals)
        b = np.array([2, 1])
        
        def objective(x):
            x = np.array(x)
            return 0.5 * np.dot(x, np.dot(Q, x)) - np.dot(b, x)
        
        def gradient(x):
            x = np.array(x)
            return np.dot(Q, x) - b
        
        def hessian(x):
            return Q
        
        # Analytical optimum: Qx* = b
        x_star = np.linalg.solve(Q, b)
        
        return ConvergenceAnalyzer(objective, gradient, hessian, x_star, 
                                 "Strongly Convex Quadratic")
    
    @staticmethod
    def ill_conditioned_quadratic():
        """
        Ill-conditioned quadratic with large condition number.
        """
        # Create ill-conditioned quadratic
        eigenvals = np.array([0.1, 10])  # Condition number = 100
        Q = np.diag(eigenvals)
        b = np.array([1, 1])
        
        def objective(x):
            x = np.array(x)
            return 0.5 * np.dot(x, np.dot(Q, x)) - np.dot(b, x)
        
        def gradient(x):
            x = np.array(x)
            return np.dot(Q, x) - b
        
        def hessian(x):
            return Q
        
        x_star = np.linalg.solve(Q, b)
        
        return ConvergenceAnalyzer(objective, gradient, hessian, x_star,
                                 "Ill-Conditioned Quadratic")
    
    @staticmethod
    def rosenbrock_function():
        """
        Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
        Classic test for optimization algorithms.
        """
        a, b = 1, 100  # Standard parameters
        
        def objective(x):
            x = np.array(x)
            return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
        
        def gradient(x):
            x = np.array(x)
            grad_x = -2*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2)
            grad_y = 2*b*(x[1] - x[0]**2)
            return np.array([grad_x, grad_y])
        
        def hessian(x):
            x = np.array(x)
            h11 = 2 + 12*b*x[0]**2 - 4*b*(x[1] - x[0]**2)
            h12 = h21 = -4*b*x[0]
            h22 = 2*b
            return np.array([[h11, h12], [h21, h22]])
        
        x_star = np.array([a, a**2])  # Global minimum at (1, 1)
        
        return ConvergenceAnalyzer(objective, gradient, hessian, x_star,
                                 "Rosenbrock Function")


def demonstrate_convergence_theory():
    """
    Comprehensive demonstration of convergence theory.
    """
    print("⚡ CONVERGENCE THEORY: Algorithm Analysis")
    print("=" * 50)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Example 1: Well-conditioned quadratic
    print("\n📊 EXAMPLE 1: Well-Conditioned Quadratic")
    print("-" * 45)
    
    problem1 = ConvergenceTheorems.strongly_convex_quadratic()
    
    # Run different algorithms
    x0 = np.array([3, 2])
    
    # Gradient descent
    gd_sequence = problem1.gradient_descent(x0, step_size=0.1, max_iters=50)
    gd_analysis = problem1.compute_convergence_rate(gd_sequence)
    
    # Newton method
    newton_sequence = problem1.newton_method(x0, max_iters=20)
    newton_analysis = problem1.compute_convergence_rate(newton_sequence)
    
    # BFGS
    bfgs_sequence = problem1.bfgs_method(x0, max_iters=50)
    bfgs_analysis = problem1.compute_convergence_rate(bfgs_sequence)
    
    print(f"Problem: {problem1.name}")
    print(f"True optimum: {problem1.true_optimum}")
    print(f"Condition number: {4}")
    
    print(f"\nGradient Descent:")
    print(f"  Iterations: {gd_analysis['n_iterations']}")
    print(f"  Linear convergent: {gd_analysis['is_linear_convergent']}")
    if gd_analysis['linear_rate']:
        print(f"  Linear rate: {gd_analysis['linear_rate']:.4f}")
    
    print(f"\nNewton Method:")
    print(f"  Iterations: {newton_analysis['n_iterations']}")
    print(f"  Quadratic convergent: {newton_analysis['is_quadratic']}")
    print(f"  Superlinear convergent: {newton_analysis['is_superlinear']}")
    
    print(f"\nBFGS Method:")
    print(f"  Iterations: {bfgs_analysis['n_iterations']}")
    print(f"  Superlinear convergent: {bfgs_analysis['is_superlinear']}")
    if bfgs_analysis['linear_rate']:
        print(f"  Linear rate: {bfgs_analysis['linear_rate']:.4f}")
    
    # Plot convergence comparison
    ax1 = plt.subplot(2, 3, 1)
    
    # Plot error norms
    iterations_gd = range(len(gd_analysis['error_norms']))
    iterations_newton = range(len(newton_analysis['error_norms']))
    iterations_bfgs = range(len(bfgs_analysis['error_norms']))
    
    ax1.semilogy(iterations_gd, gd_analysis['error_norms'], 'b-o', 
                label='Gradient Descent', linewidth=2, markersize=4)
    ax1.semilogy(iterations_newton, newton_analysis['error_norms'], 'r-s',
                label='Newton Method', linewidth=2, markersize=4)
    ax1.semilogy(iterations_bfgs, bfgs_analysis['error_norms'], 'g-^',
                label='BFGS', linewidth=2, markersize=4)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('||x_k - x*|| (log scale)')
    ax1.set_title('Convergence: Well-Conditioned Problem')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Example 2: Ill-conditioned quadratic
    print("\n📊 EXAMPLE 2: Ill-Conditioned Quadratic")
    print("-" * 45)
    
    problem2 = ConvergenceTheorems.ill_conditioned_quadratic()
    
    # Run algorithms on ill-conditioned problem
    gd_sequence_ill = problem2.gradient_descent(x0, step_size=0.05, max_iters=200)
    newton_sequence_ill = problem2.newton_method(x0, max_iters=20)
    bfgs_sequence_ill = problem2.bfgs_method(x0, max_iters=100)
    
    gd_analysis_ill = problem2.compute_convergence_rate(gd_sequence_ill)
    newton_analysis_ill = problem2.compute_convergence_rate(newton_sequence_ill)
    bfgs_analysis_ill = problem2.compute_convergence_rate(bfgs_sequence_ill)
    
    print(f"Problem: {problem2.name}")
    print(f"Condition number: {100}")
    
    print(f"\nGradient Descent:")
    print(f"  Iterations: {gd_analysis_ill['n_iterations']}")
    if gd_analysis_ill['linear_rate']:
        print(f"  Linear rate: {gd_analysis_ill['linear_rate']:.4f}")
    
    print(f"\nNewton Method:")
    print(f"  Iterations: {newton_analysis_ill['n_iterations']}")
    print(f"  Quadratic convergent: {newton_analysis_ill['is_quadratic']}")
    
    print(f"\nBFGS Method:")
    print(f"  Iterations: {bfgs_analysis_ill['n_iterations']}")
    print(f"  Superlinear convergent: {bfgs_analysis_ill['is_superlinear']}")
    
    # Plot ill-conditioned convergence
    ax2 = plt.subplot(2, 3, 2)
    
    iterations_gd_ill = range(len(gd_analysis_ill['error_norms']))
    iterations_newton_ill = range(len(newton_analysis_ill['error_norms']))
    iterations_bfgs_ill = range(len(bfgs_analysis_ill['error_norms']))
    
    ax2.semilogy(iterations_gd_ill, gd_analysis_ill['error_norms'], 'b-o',
                label='Gradient Descent', linewidth=2, markersize=3)
    ax2.semilogy(iterations_newton_ill, newton_analysis_ill['error_norms'], 'r-s',
                label='Newton Method', linewidth=2, markersize=4)
    ax2.semilogy(iterations_bfgs_ill, bfgs_analysis_ill['error_norms'], 'g-^',
                label='BFGS', linewidth=2, markersize=3)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('||x_k - x*|| (log scale)')
    ax2.set_title('Convergence: Ill-Conditioned Problem')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Convergence rate analysis
    ax3 = plt.subplot(2, 3, 3)
    
    # Show convergence rates for different algorithms
    algorithms = ['GD (well)', 'Newton (well)', 'BFGS (well)', 
                 'GD (ill)', 'Newton (ill)', 'BFGS (ill)']
    
    analyses = [gd_analysis, newton_analysis, bfgs_analysis,
               gd_analysis_ill, newton_analysis_ill, bfgs_analysis_ill]
    
    linear_rates = []
    colors = []
    
    for analysis in analyses:
        if analysis['linear_rate'] is not None:
            linear_rates.append(analysis['linear_rate'])
        else:
            linear_rates.append(0)  # Very fast convergence
        
        # Color based on convergence type
        if analysis['is_quadratic']:
            colors.append('gold')
        elif analysis['is_superlinear']:
            colors.append('lightgreen')
        elif analysis['is_linear_convergent']:
            colors.append('lightblue')
        else:
            colors.append('lightcoral')
    
    bars = ax3.bar(range(len(algorithms)), linear_rates, color=colors, alpha=0.7)
    
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Linear Convergence Rate')
    ax3.set_title('Convergence Rate Comparison')
    ax3.set_xticks(range(len(algorithms)))
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')
    
    # Add value labels
    for bar, rate in zip(bars, linear_rates):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add legend for colors
    ax3.text(0.02, 0.98, 'Colors:\n🟡 Quadratic\n🟢 Superlinear\n🔵 Linear\n🔴 Other',
            transform=ax3.transAxes, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax3.grid(True, alpha=0.3)
    
    # Rosenbrock function example
    print("\n📊 EXAMPLE 3: Rosenbrock Function")
    print("-" * 40)
    
    problem3 = ConvergenceTheorems.rosenbrock_function()
    
    # Starting point away from optimum
    x0_rosen = np.array([-2, 3])
    
    # Run algorithms (with more iterations for this difficult problem)
    gd_rosen = problem3.gradient_descent(x0_rosen, step_size=0.001, max_iters=5000)
    newton_rosen = problem3.newton_method(x0_rosen, max_iters=100)
    bfgs_rosen = problem3.bfgs_method(x0_rosen, max_iters=1000)
    
    gd_analysis_rosen = problem3.compute_convergence_rate(gd_rosen)
    newton_analysis_rosen = problem3.compute_convergence_rate(newton_rosen)
    bfgs_analysis_rosen = problem3.compute_convergence_rate(bfgs_rosen)
    
    print(f"Problem: {problem3.name}")
    print(f"Starting point: {x0_rosen}")
    print(f"True optimum: {problem3.true_optimum}")
    
    print(f"\nGradient Descent: {gd_analysis_rosen['n_iterations']} iterations")
    print(f"Newton Method: {newton_analysis_rosen['n_iterations']} iterations") 
    print(f"BFGS Method: {bfgs_analysis_rosen['n_iterations']} iterations")
    
    # Plot Rosenbrock convergence paths
    ax4 = plt.subplot(2, 3, 4)
    
    # Create contour plot of Rosenbrock function
    x_range = np.linspace(-2.5, 2.5, 100)
    y_range = np.linspace(-1, 4, 100) 
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100*(Y - X**2)**2
    
    # Use log scale for better visualization
    levels = np.logspace(0, 3, 20)
    contour = ax4.contour(X, Y, Z, levels=levels, alpha=0.6, colors='gray')
    
    # Plot convergence paths
    if len(gd_rosen) > 1:
        gd_x = [x[0] for x in gd_rosen[::50]]  # Subsample for clarity
        gd_y = [x[1] for x in gd_rosen[::50]]
        ax4.plot(gd_x, gd_y, 'b-o', linewidth=2, markersize=3, 
                label=f'GD ({len(gd_rosen)} iters)', alpha=0.7)
    
    if len(newton_rosen) > 1:
        newton_x = [x[0] for x in newton_rosen]
        newton_y = [x[1] for x in newton_rosen]
        ax4.plot(newton_x, newton_y, 'r-s', linewidth=2, markersize=4,
                label=f'Newton ({len(newton_rosen)} iters)')
    
    if len(bfgs_rosen) > 1:
        bfgs_x = [x[0] for x in bfgs_rosen]
        bfgs_y = [x[1] for x in bfgs_rosen]
        ax4.plot(bfgs_x, bfgs_y, 'g-^', linewidth=2, markersize=3,
                label=f'BFGS ({len(bfgs_rosen)} iters)', alpha=0.8)
    
    # Mark start and end points
    ax4.plot(x0_rosen[0], x0_rosen[1], 'ko', markersize=10, label='Start')
    ax4.plot(1, 1, 'k*', markersize=15, label='Optimum')
    
    ax4.set_xlim(-2.5, 2.5)
    ax4.set_ylim(-1, 4)
    ax4.set_xlabel('x₁')
    ax4.set_ylabel('x₂')
    ax4.set_title('Rosenbrock Function: Convergence Paths')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Function value convergence for Rosenbrock
    ax5 = plt.subplot(2, 3, 5)
    
    # Plot function value convergence
    if len(gd_analysis_rosen['function_errors']) > 1:
        ax5.semilogy(range(0, len(gd_rosen), 50), 
                    gd_analysis_rosen['function_errors'][::50], 
                    'b-o', linewidth=2, markersize=3, label='Gradient Descent')
    
    if len(newton_analysis_rosen['function_errors']) > 1:
        ax5.semilogy(range(len(newton_analysis_rosen['function_errors'])), 
                    newton_analysis_rosen['function_errors'],
                    'r-s', linewidth=2, markersize=4, label='Newton Method')
    
    if len(bfgs_analysis_rosen['function_errors']) > 1:
        ax5.semilogy(range(len(bfgs_analysis_rosen['function_errors'])),
                    bfgs_analysis_rosen['function_errors'],
                    'g-^', linewidth=2, markersize=3, label='BFGS')
    
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('|f(x_k) - f*| (log scale)')
    ax5.set_title('Rosenbrock: Function Value Convergence')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Theoretical convergence rates
    ax6 = plt.subplot(2, 3, 6)
    
    # Show theoretical vs observed convergence rates
    theoretical_data = {
        'Gradient Descent\n(well-conditioned)': {'theory': 0.6, 'observed': gd_analysis.get('linear_rate', 0)},
        'Gradient Descent\n(ill-conditioned)': {'theory': 0.98, 'observed': gd_analysis_ill.get('linear_rate', 0)},
        'Newton Method\n(quadratic)': {'theory': 0.0, 'observed': 0.0 if newton_analysis['is_quadratic'] else 0.5},
        'BFGS\n(superlinear)': {'theory': 0.1, 'observed': bfgs_analysis.get('linear_rate', 0)}
    }
    
    methods = list(theoretical_data.keys())
    theory_rates = [theoretical_data[m]['theory'] for m in methods]
    observed_rates = [theoretical_data[m]['observed'] if theoretical_data[m]['observed'] else 0 
                     for m in methods]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax6.bar(x_pos - width/2, theory_rates, width, 
                   label='Theoretical Rate', alpha=0.7, color='lightblue')
    bars2 = ax6.bar(x_pos + width/2, observed_rates, width,
                   label='Observed Rate', alpha=0.7, color='orange')
    
    ax6.set_ylabel('Linear Convergence Rate')
    ax6.set_title('Theoretical vs Observed Rates')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(methods, rotation=45, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def convergence_theorems_summary():
    """
    Summary of key convergence theorems and their implications.
    """
    print("\n📚 KEY CONVERGENCE THEOREMS")
    print("=" * 40)
    
    print("🎯 GRADIENT DESCENT CONVERGENCE:")
    print("Theorem (Strongly Convex Case):")
    print("  If f is L-smooth and μ-strongly convex, then")
    print("  ||x_{k+1} - x*|| ≤ ((L-μ)/(L+μ))||x_k - x*||")
    print("  Linear convergence rate: ρ = (κ-1)/(κ+1)")
    print("  where κ = L/μ is the condition number")
    
    print("\n🎯 NEWTON METHOD CONVERGENCE:")
    print("Theorem (Local Quadratic Convergence):")
    print("  If f is twice differentiable, ∇²f Lipschitz,")
    print("  and x_0 close enough to x*, then")
    print("  ||x_{k+1} - x*|| ≤ C||x_k - x*||²")
    print("  Quadratic convergence (very fast near optimum)")
    
    print("\n🎯 BFGS CONVERGENCE:")
    print("Theorem (Superlinear Convergence):")  
    print("  Under standard assumptions (LICQ, positive definite Hessian),")
    print("  BFGS achieves superlinear convergence:")
    print("  lim_{k→∞} ||x_{k+1} - x*|| / ||x_k - x*|| = 0")
    
    print("\n🎯 DESCENT LEMMA:")
    print("Lemma (Fundamental Tool):")
    print("  If f is L-smooth, then for any x, y:")
    print("  f(y) ≤ f(x) + ∇f(x)ᵀ(y-x) + (L/2)||y-x||²")
    print("  Essential for analyzing step sizes and convergence")
    
    print("\n🎯 ARMIJO-GOLDSTEIN CONDITIONS:")
    print("Line Search Convergence:")
    print("  Armijo: f(x + αd) ≤ f(x) + c₁α∇f(x)ᵀd")
    print("  Wolfe: ∇f(x + αd)ᵀd ≥ c₂∇f(x)ᵀd") 
    print("  Guarantees convergence with backtracking")
    
    print("\n⚡ PRACTICAL IMPLICATIONS:")
    print("1. CONDITIONING MATTERS:")
    print("   - Well-conditioned: All methods work well")
    print("   - Ill-conditioned: Newton/BFGS much better than GD")
    
    print("\n2. GLOBAL vs LOCAL:")
    print("   - GD: Global convergence, slow near optimum")
    print("   - Newton: Fast locally, may diverge globally")
    print("   - BFGS: Good compromise, superlinear + global")
    
    print("\n3. ALGORITHM SELECTION:")
    print("   - Smooth, well-conditioned: Gradient descent OK")
    print("   - Ill-conditioned: Use Newton or BFGS")
    print("   - Large-scale: Limited-memory BFGS (L-BFGS)")
    print("   - Non-smooth: Subgradient methods")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_convergence_theory()
    convergence_theorems_summary()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Convergence rate determines how fast algorithms reach the optimum")  
    print("- Linear: ||x_{k+1} - x*|| ≤ ρ||x_k - x*|| with 0 < ρ < 1")
    print("- Superlinear: Rate factor ρ_k → 0 as k → ∞")
    print("- Quadratic: ||x_{k+1} - x*|| ≤ C||x_k - x*||² (Newton's method)")
    print("- Condition number κ = L/μ determines convergence difficulty")
    print("- Well-conditioned problems: κ ≈ 1, fast convergence")
    print("- Ill-conditioned problems: κ >> 1, slow gradient descent")
    print("\nConvergence theory guides algorithm choice and performance analysis! 🚀")