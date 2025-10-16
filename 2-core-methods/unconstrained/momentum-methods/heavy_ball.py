"""
Heavy Ball Method - Momentum-Based Optimization
==============================================

The Heavy Ball method adds momentum to gradient descent, accelerating convergence
by accumulating velocity from previous gradient directions.

Key Concepts:
- Momentum accumulation
- Velocity-based updates
- Faster convergence in ravines
- Damping for stability

Author: Optimization Framework
Date: October 16, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class HeavyBallMethod:
    """
    Heavy Ball Method with Momentum
    
    Update rule:
        v_{k+1} = β·v_k - α·∇f(x_k)
        x_{k+1} = x_k + v_{k+1}
    
    where:
        - α: learning rate (step size)
        - β: momentum coefficient (0 ≤ β < 1)
        - v_k: velocity at iteration k
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9,
                 max_iter: int = 1000, tol: float = 1e-6):
        """
        Initialize Heavy Ball optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Step size (α)
        momentum : float
            Momentum coefficient (β), typically 0.9
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.tol = tol
        
    def optimize(self, f: Callable, grad_f: Callable, x0: np.ndarray,
                 callback: Optional[Callable] = None) -> Tuple[np.ndarray, List]:
        """
        Minimize function using Heavy Ball method.
        
        Parameters:
        -----------
        f : callable
            Objective function
        grad_f : callable
            Gradient of objective
        x0 : ndarray
            Initial point
        callback : callable, optional
            Called after each iteration
            
        Returns:
        --------
        x_opt : ndarray
            Optimal point
        history : list
            Optimization history
        """
        x = x0.copy()
        velocity = np.zeros_like(x)
        history = []
        
        for k in range(self.max_iter):
            # Compute gradient
            grad = grad_f(x)
            
            # Store history
            history.append({
                'iteration': k,
                'x': x.copy(),
                'f': f(x),
                'grad_norm': np.linalg.norm(grad),
                'velocity': velocity.copy()
            })
            
            # Check convergence
            if np.linalg.norm(grad) < self.tol:
                print(f"Converged in {k} iterations")
                break
            
            # Update velocity with momentum
            velocity = self.momentum * velocity - self.learning_rate * grad
            
            # Update position
            x = x + velocity
            
            if callback:
                callback(k, x, f(x))
        
        return x, history


class ClassicalMomentum:
    """
    Classical Momentum Method (Polyak's momentum)
    
    Update rule:
        v_{k+1} = β·v_k + ∇f(x_k)
        x_{k+1} = x_k - α·v_{k+1}
    
    Slightly different formulation from Heavy Ball
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9,
                 max_iter: int = 1000, tol: float = 1e-6):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.tol = tol
        
    def optimize(self, f: Callable, grad_f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, List]:
        """Minimize using classical momentum."""
        x = x0.copy()
        velocity = np.zeros_like(x)
        history = []
        
        for k in range(self.max_iter):
            grad = grad_f(x)
            
            history.append({
                'iteration': k,
                'x': x.copy(),
                'f': f(x),
                'grad_norm': np.linalg.norm(grad),
                'velocity': velocity.copy()
            })
            
            if np.linalg.norm(grad) < self.tol:
                break
            
            # Classical momentum update
            velocity = self.momentum * velocity + grad
            x = x - self.learning_rate * velocity
        
        return x, history


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def quadratic_bowl(x: np.ndarray) -> float:
    """Simple quadratic function."""
    return 0.5 * (x[0]**2 + x[1]**2)

def grad_quadratic_bowl(x: np.ndarray) -> np.ndarray:
    """Gradient of quadratic bowl."""
    return np.array([x[0], x[1]])


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function - classic test case."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_rosenbrock(x: np.ndarray) -> np.ndarray:
    """Gradient of Rosenbrock."""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])


def beale(x: np.ndarray) -> float:
    """Beale function - has narrow valley."""
    return ((1.5 - x[0] + x[0]*x[1])**2 + 
            (2.25 - x[0] + x[0]*x[1]**2)**2 + 
            (2.625 - x[0] + x[0]*x[1]**3)**2)

def grad_beale(x: np.ndarray) -> np.ndarray:
    """Gradient of Beale function."""
    t1 = 1.5 - x[0] + x[0]*x[1]
    t2 = 2.25 - x[0] + x[0]*x[1]**2
    t3 = 2.625 - x[0] + x[0]*x[1]**3
    
    dx = (2*t1*(-1 + x[1]) + 2*t2*(-1 + x[1]**2) + 2*t3*(-1 + x[1]**3))
    dy = (2*t1*x[0] + 4*t2*x[0]*x[1] + 6*t3*x[0]*x[1]**2)
    
    return np.array([dx, dy])


# ============================================================================
# VISUALIZATION
# ============================================================================

def compare_with_without_momentum():
    """Compare gradient descent with and without momentum."""
    print("=" * 70)
    print("COMPARISON: Gradient Descent vs Heavy Ball Method")
    print("=" * 70)
    
    # Test on Rosenbrock
    x0 = np.array([-1.0, 1.0])
    
    # Without momentum (pure GD)
    gd_results = []
    x = x0.copy()
    lr = 0.001
    for k in range(500):
        grad = grad_rosenbrock(x)
        gd_results.append({'x': x.copy(), 'f': rosenbrock(x)})
        x = x - lr * grad
    
    # With Heavy Ball momentum
    hb = HeavyBallMethod(learning_rate=0.001, momentum=0.9, max_iter=500)
    x_hb, hb_history = hb.optimize(rosenbrock, grad_rosenbrock, x0)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convergence plot
    ax = axes[0]
    gd_f_vals = [h['f'] for h in gd_results]
    hb_f_vals = [h['f'] for h in hb_history]
    
    ax.semilogy(gd_f_vals, 'b-', label='Gradient Descent', linewidth=2)
    ax.semilogy(hb_f_vals, 'r-', label='Heavy Ball', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('f(x) [log scale]', fontsize=12)
    ax.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Trajectory plot
    ax = axes[1]
    
    # Create contour plot
    x_range = np.linspace(-1.5, 1.5, 100)
    y_range = np.linspace(-0.5, 2.0, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[rosenbrock(np.array([x, y])) for x in x_range] for y in y_range])
    
    contour = ax.contour(X, Y, Z, levels=20, alpha=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Plot trajectories
    gd_x = np.array([h['x'] for h in gd_results])
    hb_x = np.array([h['x'] for h in hb_history])
    
    ax.plot(gd_x[:, 0], gd_x[:, 1], 'b.-', label='GD', markersize=3, linewidth=1.5)
    ax.plot(hb_x[:, 0], hb_x[:, 1], 'r.-', label='Heavy Ball', markersize=3, linewidth=1.5)
    ax.plot(1, 1, 'g*', markersize=20, label='Optimum')
    
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Optimization Trajectories', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heavy_ball_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: heavy_ball_comparison.png")
    plt.show()
    
    print(f"\nGradient Descent: {len(gd_results)} iterations, final f = {gd_f_vals[-1]:.6f}")
    print(f"Heavy Ball:       {len(hb_history)} iterations, final f = {hb_f_vals[-1]:.6f}")


def momentum_effect_visualization():
    """Visualize the effect of different momentum values."""
    print("\n" + "=" * 70)
    print("MOMENTUM COEFFICIENT ANALYSIS")
    print("=" * 70)
    
    x0 = np.array([0.5, 2.0])
    momentum_values = [0.0, 0.5, 0.9, 0.95, 0.99]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # Create contour background
    x_range = np.linspace(-0.5, 3.5, 100)
    y_range = np.linspace(-0.5, 3.5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[beale(np.array([x, y])) for x in x_range] for y in y_range])
    
    for idx, beta in enumerate(momentum_values):
        ax = axes[idx]
        
        # Run optimization
        hb = HeavyBallMethod(learning_rate=0.001, momentum=beta, max_iter=200)
        x_opt, history = hb.optimize(beale, grad_beale, x0)
        
        # Plot
        contour = ax.contour(X, Y, Z, levels=30, alpha=0.4)
        
        trajectory = np.array([h['x'] for h in history])
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', markersize=4, linewidth=2)
        ax.plot(x0[0], x0[1], 'go', markersize=10, label='Start')
        ax.plot(3, 0.5, 'r*', markersize=15, label='Optimum')
        
        ax.set_title(f'β = {beta:.2f} ({len(history)} iter)', fontsize=12, fontweight='bold')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        print(f"β = {beta:.2f}: {len(history)} iterations, final f = {history[-1]['f']:.6f}")
    
    # Remove extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('momentum_effect.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: momentum_effect.png")
    plt.show()


def velocity_field_visualization():
    """Visualize velocity accumulation during optimization."""
    print("\n" + "=" * 70)
    print("VELOCITY FIELD VISUALIZATION")
    print("=" * 70)
    
    x0 = np.array([-1.0, 1.0])
    
    hb = HeavyBallMethod(learning_rate=0.001, momentum=0.9, max_iter=300)
    x_opt, history = hb.optimize(rosenbrock, grad_rosenbrock, x0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Velocity magnitude over time
    ax = axes[0]
    velocities = [np.linalg.norm(h['velocity']) for h in history]
    iterations = [h['iteration'] for h in history]
    
    ax.plot(iterations, velocities, 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Velocity Magnitude', fontsize=12)
    ax.set_title('Velocity Accumulation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Velocity vectors on trajectory
    ax = axes[1]
    
    # Contour background
    x_range = np.linspace(-1.5, 1.5, 100)
    y_range = np.linspace(-0.5, 2.0, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[rosenbrock(np.array([x, y])) for x in x_range] for y in y_range])
    
    contour = ax.contour(X, Y, Z, levels=20, alpha=0.4)
    
    # Plot trajectory
    trajectory = np.array([h['x'] for h in history])
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, alpha=0.5)
    
    # Plot velocity vectors (every 10th)
    for i in range(0, len(history), 10):
        pos = history[i]['x']
        vel = history[i]['velocity']
        ax.arrow(pos[0], pos[1], vel[0]*50, vel[1]*50,
                head_width=0.05, head_length=0.05, fc='blue', ec='blue', alpha=0.7)
    
    ax.plot(1, 1, 'g*', markersize=20, label='Optimum')
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Velocity Vectors Along Trajectory', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('velocity_field.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: velocity_field.png")
    plt.show()


# ============================================================================
# THEORETICAL ANALYSIS
# ============================================================================

def convergence_rate_analysis():
    """Analyze convergence rate with different parameters."""
    print("\n" + "=" * 70)
    print("CONVERGENCE RATE ANALYSIS")
    print("=" * 70)
    
    # Test on strongly convex quadratic
    def strongly_convex_quad(x):
        Q = np.array([[10, 0], [0, 1]])  # Condition number = 10
        return 0.5 * x.T @ Q @ x
    
    def grad_strongly_convex(x):
        Q = np.array([[10, 0], [0, 1]])
        return Q @ x
    
    x0 = np.array([5.0, 5.0])
    
    # Theoretical optimal parameters
    # For quadratic with eigenvalues λ_min, λ_max:
    # α_opt = 4 / (sqrt(λ_max) + sqrt(λ_min))^2
    # β_opt = ((sqrt(κ) - 1) / (sqrt(κ) + 1))^2
    # where κ = λ_max / λ_min
    
    lambda_max = 10
    lambda_min = 1
    kappa = lambda_max / lambda_min
    
    alpha_opt = 4 / (np.sqrt(lambda_max) + np.sqrt(lambda_min))**2
    beta_opt = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1))**2
    
    print(f"\nTheoretical optimal parameters:")
    print(f"  Condition number κ = {kappa:.2f}")
    print(f"  Optimal α = {alpha_opt:.6f}")
    print(f"  Optimal β = {beta_opt:.6f}")
    
    # Test different configurations
    configs = [
        ('GD (no momentum)', 0.1, 0.0),
        ('Suboptimal momentum', 0.1, 0.5),
        ('Optimal parameters', alpha_opt, beta_opt),
        ('High momentum', 0.1, 0.9),
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, alpha, beta in configs:
        hb = HeavyBallMethod(learning_rate=alpha, momentum=beta, max_iter=100)
        x_opt, history = hb.optimize(strongly_convex_quad, grad_strongly_convex, x0)
        
        f_vals = [h['f'] for h in history]
        ax.semilogy(f_vals, linewidth=2, label=name)
        
        print(f"\n{name}:")
        print(f"  α = {alpha:.6f}, β = {beta:.6f}")
        print(f"  Iterations: {len(history)}")
        print(f"  Final f: {f_vals[-1]:.2e}")
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('f(x) [log scale]', fontsize=12)
    ax.set_title('Convergence Rate with Different Parameters', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_rate_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: convergence_rate_analysis.png")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("HEAVY BALL METHOD - MOMENTUM-BASED OPTIMIZATION")
    print("=" * 70)
    
    # Run demonstrations
    compare_with_without_momentum()
    momentum_effect_visualization()
    velocity_field_visualization()
    convergence_rate_analysis()
    
    print("\n" + "=" * 70)
    print("✓ ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("\nGenerated visualizations:")
    print("  1. heavy_ball_comparison.png - GD vs Heavy Ball")
    print("  2. momentum_effect.png - Effect of momentum coefficient")
    print("  3. velocity_field.png - Velocity accumulation")
    print("  4. convergence_rate_analysis.png - Theoretical analysis")
