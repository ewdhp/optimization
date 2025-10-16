"""
Convergence Analysis for Gradient Methods

This module provides theoretical and empirical convergence analysis for
gradient descent and related first-order optimization methods.

Key Convergence Results:

1. GRADIENT DESCENT ON CONVEX FUNCTIONS:
   - Sublinear convergence: O(1/k)
   - f(x_k) - f* ≤ ||x_0 - x*||² / (2αk)

2. GRADIENT DESCENT ON STRONGLY CONVEX FUNCTIONS:
   - Linear convergence: O(ρ^k)  
   - ||x_k - x*|| ≤ ρ^k ||x_0 - x*||
   - Rate: ρ = (κ-1)/(κ+1) where κ = L/μ

3. DESCENT LEMMA:
   - For L-smooth f:
   - f(y) ≤ f(x) + ∇f(x)^T(y-x) + (L/2)||y-x||²

4. POLYAK-ŁOJASIEWICZ CONDITION:
   - ||∇f(x)||² ≥ 2μ(f(x) - f*)
   - Implies linear convergence even for non-convex functions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class ConvergenceAnalysis:
    """
    Tools for analyzing convergence of gradient-based optimization.
    """
    
    def __init__(self,
                 objective: Callable[[np.ndarray], float],
                 gradient: Callable[[np.ndarray], np.ndarray],
                 hessian: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 true_minimum: Optional[np.ndarray] = None,
                 name: str = "Function"):
        """
        Initialize convergence analyzer.
        
        Args:
            objective: Objective function f(x)
            gradient: Gradient function ∇f(x)
            hessian: Optional Hessian ∇²f(x)
            true_minimum: Known optimal point x*
            name: Function name
        """
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.true_minimum = true_minimum
        self.name = name
    
    def estimate_lipschitz_constant(self,
                                   x_samples: List[np.ndarray],
                                   sample_gradients: bool = True) -> float:
        """
        Estimate Lipschitz constant L of gradient.
        
        L = sup ||∇f(x) - ∇f(y)|| / ||x - y||
        
        Args:
            x_samples: Sample points for estimation
            sample_gradients: Whether to sample gradients or use Hessian
            
        Returns:
            Estimated Lipschitz constant L
        """
        if not sample_gradients and self.hessian is not None:
            # Use eigenvalues of Hessian if available
            max_eigenvals = []
            for x in x_samples:
                H = self.hessian(x)
                eigenvals = np.linalg.eigvals(H)
                max_eigenvals.append(np.max(np.abs(eigenvals)))
            return np.max(max_eigenvals)
        
        # Estimate from gradient differences
        L_estimates = []
        for i, x in enumerate(x_samples):
            for j, y in enumerate(x_samples):
                if i != j:
                    grad_diff = np.linalg.norm(self.gradient(x) - self.gradient(y))
                    point_diff = np.linalg.norm(x - y)
                    if point_diff > 1e-10:
                        L_estimates.append(grad_diff / point_diff)
        
        return np.max(L_estimates) if L_estimates else 1.0
    
    def estimate_strong_convexity(self,
                                 x_samples: List[np.ndarray]) -> float:
        """
        Estimate strong convexity parameter μ.
        
        For strongly convex functions:
        f(y) ≥ f(x) + ∇f(x)^T(y-x) + (μ/2)||y-x||²
        
        Args:
            x_samples: Sample points
            
        Returns:
            Estimated strong convexity parameter μ
        """
        if self.hessian is not None:
            # Use minimum eigenvalue of Hessian
            min_eigenvals = []
            for x in x_samples:
                H = self.hessian(x)
                eigenvals = np.linalg.eigvals(H)
                min_eigenvals.append(np.min(np.real(eigenvals)))
            return max(0, np.min(min_eigenvals))
        
        # Estimate from function values
        mu_estimates = []
        for i, x in enumerate(x_samples):
            for j, y in enumerate(x_samples):
                if i != j:
                    f_x = self.objective(x)
                    f_y = self.objective(y)
                    grad_x = self.gradient(x)
                    diff = y - x
                    
                    # Rearrange strong convexity inequality
                    lhs = f_y - f_x - np.dot(grad_x, diff)
                    rhs = 0.5 * np.dot(diff, diff)
                    
                    if rhs > 1e-10:
                        mu_estimates.append(lhs / rhs)
        
        return max(0, np.min(mu_estimates)) if mu_estimates else 0.0
    
    def verify_descent_lemma(self,
                            x: np.ndarray,
                            y: np.ndarray,
                            L: float) -> Dict:
        """
        Verify descent lemma (fundamental smoothness inequality).
        
        For L-smooth functions:
        f(y) ≤ f(x) + ∇f(x)^T(y-x) + (L/2)||y-x||²
        
        Args:
            x: First point
            y: Second point  
            L: Lipschitz constant
            
        Returns:
            Verification results
        """
        x = np.array(x)
        y = np.array(y)
        
        f_x = self.objective(x)
        f_y = self.objective(y)
        grad_x = self.gradient(x)
        
        diff = y - x
        linear_approx = f_x + np.dot(grad_x, diff)
        quadratic_upper = linear_approx + 0.5 * L * np.dot(diff, diff)
        
        satisfied = f_y <= quadratic_upper + 1e-8
        violation = max(0, f_y - quadratic_upper)
        
        return {
            'satisfied': satisfied,
            'f_y': f_y,
            'linear_approx': linear_approx,
            'quadratic_upper': quadratic_upper,
            'violation': violation,
            'gap': quadratic_upper - f_y
        }
    
    def theoretical_convergence_rate(self,
                                    mu: float,
                                    L: float,
                                    method: str = 'gradient_descent') -> Dict:
        """
        Compute theoretical convergence rates.
        
        Args:
            mu: Strong convexity parameter
            L: Lipschitz constant
            method: Optimization method
            
        Returns:
            Theoretical convergence information
        """
        results = {'mu': mu, 'L': L}
        
        if mu > 0:
            # Strongly convex case
            kappa = L / mu
            results['condition_number'] = kappa
            results['strongly_convex'] = True
            
            if method == 'gradient_descent':
                # Linear convergence rate
                rho = (kappa - 1) / (kappa + 1)
                results['convergence_type'] = 'linear'
                results['linear_rate'] = rho
                
                # Optimal step size
                alpha_optimal = 2 / (L + mu)
                results['optimal_step_size'] = alpha_optimal
                
                # Iterations for ε-accuracy
                def iters_for_accuracy(epsilon):
                    if rho >= 1:
                        return np.inf
                    return int(np.ceil(np.log(epsilon) / np.log(rho)))
                
                results['iterations_to_1e-6'] = iters_for_accuracy(1e-6)
                results['iterations_to_1e-9'] = iters_for_accuracy(1e-9)
        else:
            # Convex but not strongly convex
            results['strongly_convex'] = False
            results['convergence_type'] = 'sublinear'
            
            if method == 'gradient_descent':
                # O(1/k) convergence
                results['rate_description'] = 'O(1/k)'
                
                # Step size
                alpha = 1 / L
                results['suggested_step_size'] = alpha
        
        return results
    
    def analyze_optimization_run(self,
                                history: Dict,
                                true_minimum: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze convergence from optimization history.
        
        Args:
            history: Optimization history with 'x', 'f', 'grad_norm'
            true_minimum: Known optimal point
            
        Returns:
            Convergence analysis results
        """
        if true_minimum is None:
            true_minimum = self.true_minimum
        
        iterates = history['x']
        f_values = history['f']
        grad_norms = history['grad_norm']
        
        n_iters = len(f_values)
        
        results = {
            'total_iterations': n_iters,
            'final_function_value': f_values[-1],
            'final_gradient_norm': grad_norms[-1]
        }
        
        # Analyze if we know true minimum
        if true_minimum is not None:
            true_minimum = np.array(true_minimum)
            f_star = self.objective(true_minimum)
            
            # Error norms
            error_norms = [np.linalg.norm(np.array(x) - true_minimum) 
                          for x in iterates]
            results['error_norms'] = error_norms
            results['final_error_norm'] = error_norms[-1]
            
            # Function value gaps
            f_gaps = [f - f_star for f in f_values]
            results['function_gaps'] = f_gaps
            results['final_function_gap'] = f_gaps[-1]
            
            # Estimate convergence rate
            if n_iters > 10:
                # Linear rate estimation from last iterations
                recent_errors = error_norms[-10:]
                recent_ratios = [recent_errors[i+1] / recent_errors[i] 
                               for i in range(len(recent_errors)-1)
                               if recent_errors[i] > 1e-15]
                
                if recent_ratios:
                    estimated_rate = np.mean(recent_ratios)
                    results['estimated_linear_rate'] = estimated_rate
                    results['appears_linear'] = 0.1 < estimated_rate < 0.99
                else:
                    results['estimated_linear_rate'] = None
                    results['appears_linear'] = False
        
        # Gradient norm decay
        if len(grad_norms) > 10:
            grad_ratio = grad_norms[-1] / grad_norms[0]
            results['gradient_reduction_factor'] = grad_ratio
        
        return results


def demonstrate_convergence_analysis():
    """
    Comprehensive demonstration of convergence analysis.
    """
    print("📊 CONVERGENCE ANALYSIS FOR GRADIENT METHODS")
    print("=" * 60)
    
    # Test Problem 1: Well-conditioned strongly convex quadratic
    print("\n🎯 PROBLEM 1: Well-Conditioned Quadratic")
    print("-" * 50)
    
    Q1 = np.array([[2, 0.5], [0.5, 2]])
    b1 = np.array([1, -1])
    x_star_1 = np.linalg.solve(Q1, b1)
    
    def quad1_obj(x):
        x = np.array(x)
        return 0.5 * np.dot(x, np.dot(Q1, x)) - np.dot(b1, x)
    
    def quad1_grad(x):
        x = np.array(x)
        return np.dot(Q1, x) - b1
    
    def quad1_hess(x):
        return Q1
    
    analyzer1 = ConvergenceAnalysis(quad1_obj, quad1_grad, quad1_hess, 
                                   x_star_1, "Well-Conditioned")
    
    # Estimate smoothness constants
    sample_points = [np.random.randn(2) for _ in range(20)]
    L1 = analyzer1.estimate_lipschitz_constant(sample_points)
    mu1 = analyzer1.estimate_strong_convexity(sample_points)
    
    print(f"Lipschitz constant L: {L1:.4f}")
    print(f"Strong convexity μ: {mu1:.4f}")
    print(f"Condition number κ = L/μ: {L1/mu1:.4f}")
    
    # Theoretical convergence rate
    theory1 = analyzer1.theoretical_convergence_rate(mu1, L1)
    print(f"\nTheoretical Analysis:")
    print(f"  Convergence type: {theory1['convergence_type']}")
    print(f"  Linear rate ρ: {theory1['linear_rate']:.6f}")
    print(f"  Optimal step size: {theory1['optimal_step_size']:.6f}")
    print(f"  Iterations to 10⁻⁶: {theory1['iterations_to_1e-6']}")
    
    # Run gradient descent
    from steepest_descent import SteepestDescent
    
    optimizer1 = SteepestDescent(quad1_obj, quad1_grad)
    x0 = np.array([5, 5])
    result1 = optimizer1.optimize(x0, step_size=theory1['optimal_step_size'],
                                 max_iters=100, tolerance=1e-9)
    
    # Analyze convergence
    analysis1 = analyzer1.analyze_optimization_run(result1['history'], x_star_1)
    print(f"\nActual Performance:")
    print(f"  Iterations: {analysis1['total_iterations']}")
    print(f"  Final error: {analysis1['final_error_norm']:.2e}")
    print(f"  Estimated rate: {analysis1.get('estimated_linear_rate', 'N/A'):.6f}")
    
    # Verify descent lemma
    x_test = np.array([1, 1])
    y_test = np.array([0.5, 0.5])
    descent_check = analyzer1.verify_descent_lemma(x_test, y_test, L1)
    print(f"\nDescent Lemma Verification:")
    print(f"  Satisfied: {descent_check['satisfied']}")
    print(f"  f(y): {descent_check['f_y']:.6f}")
    print(f"  Upper bound: {descent_check['quadratic_upper']:.6f}")
    print(f"  Gap: {descent_check['gap']:.6f}")
    
    # Test Problem 2: Ill-conditioned quadratic
    print("\n🎯 PROBLEM 2: Ill-Conditioned Quadratic")
    print("-" * 50)
    
    Q2 = np.array([[10, 0], [0, 0.1]])
    b2 = np.array([1, 1])
    x_star_2 = np.linalg.solve(Q2, b2)
    
    def quad2_obj(x):
        x = np.array(x)
        return 0.5 * np.dot(x, np.dot(Q2, x)) - np.dot(b2, x)
    
    def quad2_grad(x):
        x = np.array(x)
        return np.dot(Q2, x) - b2
    
    def quad2_hess(x):
        return Q2
    
    analyzer2 = ConvergenceAnalysis(quad2_obj, quad2_grad, quad2_hess,
                                   x_star_2, "Ill-Conditioned")
    
    L2 = analyzer2.estimate_lipschitz_constant(sample_points)
    mu2 = analyzer2.estimate_strong_convexity(sample_points)
    
    print(f"Lipschitz constant L: {L2:.4f}")
    print(f"Strong convexity μ: {mu2:.4f}")
    print(f"Condition number κ: {L2/mu2:.4f}")
    
    theory2 = analyzer2.theoretical_convergence_rate(mu2, L2)
    print(f"\nTheoretical Analysis:")
    print(f"  Linear rate ρ: {theory2['linear_rate']:.6f}")
    print(f"  Optimal step size: {theory2['optimal_step_size']:.6f}")
    print(f"  Iterations to 10⁻⁶: {theory2['iterations_to_1e-6']}")
    
    optimizer2 = SteepestDescent(quad2_obj, quad2_grad)
    result2 = optimizer2.optimize(x0, step_size=theory2['optimal_step_size'],
                                 max_iters=500, tolerance=1e-9)
    
    analysis2 = analyzer2.analyze_optimization_run(result2['history'], x_star_2)
    print(f"\nActual Performance:")
    print(f"  Iterations: {analysis2['total_iterations']}")
    print(f"  Final error: {analysis2['final_error_norm']:.2e}")
    
    # Visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Error convergence comparison
    ax1 = plt.subplot(2, 3, 1)
    
    ax1.semilogy(range(len(analysis1['error_norms'])), analysis1['error_norms'],
                'b-o', linewidth=2, markersize=4, label='Well-conditioned')
    ax1.semilogy(range(len(analysis2['error_norms'])), analysis2['error_norms'],
                'r-s', linewidth=2, markersize=3, label='Ill-conditioned')
    
    # Plot theoretical rates
    k1 = np.arange(len(analysis1['error_norms']))
    theoretical1 = analysis1['error_norms'][0] * (theory1['linear_rate'] ** k1)
    ax1.semilogy(k1, theoretical1, 'b--', linewidth=2, alpha=0.7,
                label='Theoretical (well-cond.)')
    
    k2 = np.arange(len(analysis2['error_norms']))
    theoretical2 = analysis2['error_norms'][0] * (theory2['linear_rate'] ** k2)
    ax1.semilogy(k2, theoretical2, 'r--', linewidth=2, alpha=0.7,
                label='Theoretical (ill-cond.)')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('||x_k - x*|| (log scale)')
    ax1.set_title('Error Convergence: Theory vs Practice')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Function value convergence
    ax2 = plt.subplot(2, 3, 2)
    
    ax2.semilogy(range(len(analysis1['function_gaps'])), 
                np.abs(analysis1['function_gaps']),
                'b-o', linewidth=2, markersize=4, label='Well-conditioned')
    ax2.semilogy(range(len(analysis2['function_gaps'])),
                np.abs(analysis2['function_gaps']),
                'r-s', linewidth=2, markersize=3, label='Ill-conditioned')
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('|f(x_k) - f*| (log scale)')
    ax2.set_title('Function Value Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gradient norm convergence
    ax3 = plt.subplot(2, 3, 3)
    
    ax3.semilogy(range(len(result1['history']['grad_norm'])),
                result1['history']['grad_norm'],
                'b-o', linewidth=2, markersize=4, label='Well-conditioned')
    ax3.semilogy(range(len(result2['history']['grad_norm'])),
                result2['history']['grad_norm'],
                'r-s', linewidth=2, markersize=3, label='Ill-conditioned')
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('||∇f(x_k)|| (log scale)')
    ax3.set_title('Gradient Norm Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Convergence rate vs condition number
    ax4 = plt.subplot(2, 3, 4)
    
    condition_numbers = np.logspace(0, 3, 50)
    convergence_rates = (condition_numbers - 1) / (condition_numbers + 1)
    iterations_to_1e6 = np.log(1e-6) / np.log(convergence_rates)
    
    ax4.semilogx(condition_numbers, convergence_rates, 'b-', linewidth=3)
    ax4.axvline(x=theory1['condition_number'], color='green', linestyle='--',
               label=f'Well-cond. (κ={theory1["condition_number"]:.1f})')
    ax4.axvline(x=theory2['condition_number'], color='red', linestyle='--',
               label=f'Ill-cond. (κ={theory2["condition_number"]:.1f})')
    
    ax4.set_xlabel('Condition Number κ')
    ax4.set_ylabel('Linear Convergence Rate ρ')
    ax4.set_title('Convergence Rate vs Conditioning')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Iterations needed vs condition number
    ax5 = plt.subplot(2, 3, 5)
    
    ax5.loglog(condition_numbers, -iterations_to_1e6, 'b-', linewidth=3)
    ax5.axvline(x=theory1['condition_number'], color='green', linestyle='--')
    ax5.axvline(x=theory2['condition_number'], color='red', linestyle='--')
    
    ax5.set_xlabel('Condition Number κ')
    ax5.set_ylabel('Iterations to 10⁻⁶ Accuracy')
    ax5.set_title('Iteration Complexity vs Conditioning')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Descent lemma visualization
    ax6 = plt.subplot(2, 3, 6)
    
    # Show descent lemma along a line
    x_base = np.array([1, 1])
    direction = -quad1_grad(x_base)
    direction = direction / np.linalg.norm(direction)
    
    alphas = np.linspace(0, 1, 50)
    f_actual = []
    f_linear = []
    f_quadratic = []
    
    f_base = quad1_obj(x_base)
    grad_base = quad1_grad(x_base)
    
    for alpha in alphas:
        y = x_base + alpha * direction
        
        f_actual.append(quad1_obj(y))
        f_linear.append(f_base + alpha * np.dot(grad_base, direction))
        f_quadratic.append(f_base + alpha * np.dot(grad_base, direction) + 
                          0.5 * L1 * alpha**2)
    
    ax6.plot(alphas, f_actual, 'b-', linewidth=3, label='f(x + αd)')
    ax6.plot(alphas, f_linear, 'g--', linewidth=2, label='Linear approx')
    ax6.plot(alphas, f_quadratic, 'r--', linewidth=2, label='Quadratic upper bound')
    
    ax6.fill_between(alphas, f_actual, f_quadratic, alpha=0.3, color='yellow',
                    label='Descent lemma gap')
    
    ax6.set_xlabel('Step Size α')
    ax6.set_ylabel('Function Value')
    ax6.set_title('Descent Lemma Visualization')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def convergence_theory_summary():
    """
    Summary of convergence theory for gradient methods.
    """
    print("\n📚 CONVERGENCE THEORY SUMMARY")
    print("=" * 60)
    
    print("🔑 KEY CONCEPTS:")
    print("\n1. SMOOTHNESS (L-smoothness):")
    print("   ||∇f(x) - ∇f(y)|| ≤ L||x - y||")
    print("   Equivalent to: f(y) ≤ f(x) + ∇f^T(y-x) + (L/2)||y-x||²")
    print("   L is the Lipschitz constant of the gradient")
    
    print("\n2. STRONG CONVEXITY (μ-strong convexity):")
    print("   f(y) ≥ f(x) + ∇f^T(y-x) + (μ/2)||y-x||²")
    print("   Implies unique global minimum")
    print("   μ is the strong convexity parameter")
    
    print("\n3. CONDITION NUMBER:")
    print("   κ = L/μ")
    print("   Measures problem difficulty")
    print("   κ ≈ 1: Easy problem, fast convergence")
    print("   κ >> 1: Hard problem, slow convergence")
    
    print("\n📊 CONVERGENCE RATES:")
    print("\n   CONVEX (not strongly convex):")
    print("   - Rate: O(1/k) sublinear")
    print("   - f(x_k) - f* ≤ R²/(2αk) where R = ||x_0 - x*||")
    print("   - Need ~1/ε iterations for ε-accuracy")
    
    print("\n   STRONGLY CONVEX:")
    print("   - Rate: O(ρ^k) linear, where ρ = (κ-1)/(κ+1)")
    print("   - ||x_k - x*|| ≤ ρ^k||x_0 - x*||")
    print("   - Need ~κ log(1/ε) iterations for ε-accuracy")
    
    print("\n💡 PRACTICAL IMPLICATIONS:")
    print("1. Step size selection:")
    print("   - α = 1/L for convex functions")
    print("   - α = 2/(L+μ) optimal for strongly convex")
    
    print("\n2. Problem conditioning:")
    print("   - Preconditioning can reduce κ")
    print("   - Second-order methods implicitly precondition")
    
    print("\n3. Early stopping:")
    print("   - Monitor ||∇f(x_k)|| for convex functions")
    print("   - Can estimate distance to optimum")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_convergence_analysis()
    convergence_theory_summary()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("- Convergence rate depends on L (smoothness) and μ (strong convexity)")
    print("- Condition number κ = L/μ determines convergence difficulty")
    print("- Well-conditioned: ρ ≈ 0, fast geometric convergence")
    print("- Ill-conditioned: ρ ≈ 1, very slow convergence")
    print("- Descent lemma provides fundamental bound on progress")
    print("- Optimal step size: α* = 2/(L+μ) for strongly convex functions")
    print("\nUnderstanding convergence theory guides algorithm selection! 🚀")
