"""
Nesterov Accelerated Gradient (NAG) - Advanced Momentum Method
============================================================

Nesterov's Accelerated Gradient provides optimal convergence rate O(1/k²) for
smooth convex functions, compared to O(1/k) for standard gradient descent.

Key Innovation: "Look-ahead" gradient evaluation
- Evaluate gradient at anticipated future position
- Better informed momentum updates
- Provably optimal convergence rate

References:
- Nesterov, Y. (1983). "A method for solving the convex programming problem 
  with convergence rate O(1/k²)"
- Sutskever et al. (2013). "On the importance of initialization and momentum 
  in deep learning"

Author: Optimization Framework
Date: October 16, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class NesterovAcceleratedGradient:
    """
    Nesterov's Accelerated Gradient (NAG) Method
    
    Original formulation:
        y_{k+1} = x_k - α·∇f(x_k)
        x_{k+1} = y_{k+1} + β_k·(y_{k+1} - y_k)
    
    where β_k = (k-1)/(k+2) for optimal convergence
    
    Momentum formulation (Sutskever et al.):
        v_{k+1} = μ·v_k - α·∇f(x_k + μ·v_k)  # "look-ahead"
        x_{k+1} = x_k + v_{k+1}
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9,
                 use_adaptive_momentum: bool = True, max_iter: int = 1000,
                 tol: float = 1e-6):
        """
        Initialize Nesterov optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Step size α
        momentum : float
            Momentum coefficient μ (if not adaptive)
        use_adaptive_momentum : bool
            Use optimal β_k = (k-1)/(k+2) schedule
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.use_adaptive_momentum = use_adaptive_momentum
        self.max_iter = max_iter
        self.tol = tol
        
    def optimize(self, f: Callable, grad_f: Callable, x0: np.ndarray,
                 callback: Optional[Callable] = None) -> Tuple[np.ndarray, List]:
        """
        Minimize function using Nesterov's accelerated gradient.
        
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
            # Adaptive momentum coefficient
            if self.use_adaptive_momentum:
                beta_k = (k - 1) / (k + 2) if k > 0 else 0
            else:
                beta_k = self.momentum
            
            # Look-ahead position
            x_lookahead = x + beta_k * velocity
            
            # Evaluate gradient at look-ahead position
            grad = grad_f(x_lookahead)
            
            # Store history
            history.append({
                'iteration': k,
                'x': x.copy(),
                'f': f(x),
                'grad_norm': np.linalg.norm(grad),
                'velocity': velocity.copy(),
                'beta': beta_k,
                'lookahead': x_lookahead.copy()
            })
            
            # Check convergence
            if np.linalg.norm(grad) < self.tol:
                print(f"Converged in {k} iterations")
                break
            
            # Update velocity and position
            velocity = beta_k * velocity - self.learning_rate * grad
            x = x + velocity
            
            if callback:
                callback(k, x, f(x))
        
        return x, history


class NAGVariant:
    """
    Nesterov Momentum Variant (for comparison)
    
    Alternative formulation that's easier to implement in practice:
        v_{k+1} = β·v_k - α·∇f(x_k)
        x_{k+1} = x_k + β·v_{k+1} - α·∇f(x_k)
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9,
                 max_iter: int = 1000, tol: float = 1e-6):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.tol = tol
        
    def optimize(self, f: Callable, grad_f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, List]:
        """Minimize using NAG variant."""
        x = x0.copy()
        velocity = np.zeros_like(x)
        history = []
        
        for k in range(self.max_iter):
            grad = grad_f(x)
            
            history.append({
                'iteration': k,
                'x': x.copy(),
                'f': f(x),
                'grad_norm': np.linalg.norm(grad)
            })
            
            if np.linalg.norm(grad) < self.tol:
                break
            
            # NAG variant update
            velocity = self.momentum * velocity - self.learning_rate * grad
            x = x + self.momentum * velocity - self.learning_rate * grad
        
        return x, history


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_rosenbrock(x: np.ndarray) -> np.ndarray:
    """Gradient of Rosenbrock."""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])


def ill_conditioned_quadratic(x: np.ndarray) -> float:
    """Ill-conditioned quadratic (κ = 100)."""
    Q = np.array([[100, 0], [0, 1]])
    return 0.5 * x.T @ Q @ x

def grad_ill_conditioned(x: np.ndarray) -> np.ndarray:
    """Gradient of ill-conditioned quadratic."""
    Q = np.array([[100, 0], [0, 1]])
    return Q @ x


def beale(x: np.ndarray) -> float:
    """Beale function."""
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

def compare_momentum_methods():
    """Compare standard momentum, NAG, and gradient descent."""
    print("=" * 70)
    print("COMPARISON: GD vs Momentum vs Nesterov")
    print("=" * 70)
    
    x0 = np.array([5.0, 5.0])
    max_iter = 100
    
    # Gradient Descent
    print("\n1. Running Gradient Descent...")
    gd_history = []
    x = x0.copy()
    lr = 0.01
    for k in range(max_iter):
        grad = grad_ill_conditioned(x)
        gd_history.append({'x': x.copy(), 'f': ill_conditioned_quadratic(x)})
        if np.linalg.norm(grad) < 1e-6:
            break
        x = x - lr * grad
    
    # Classical Momentum (Heavy Ball)
    print("2. Running Classical Momentum...")
    from heavy_ball import HeavyBallMethod
    hb = HeavyBallMethod(learning_rate=0.01, momentum=0.9, max_iter=max_iter)
    _, hb_history = hb.optimize(ill_conditioned_quadratic, grad_ill_conditioned, x0)
    
    # Nesterov Accelerated Gradient
    print("3. Running Nesterov Accelerated Gradient...")
    nag = NesterovAcceleratedGradient(learning_rate=0.01, momentum=0.9,
                                      use_adaptive_momentum=False, max_iter=max_iter)
    _, nag_history = nag.optimize(ill_conditioned_quadratic, grad_ill_conditioned, x0)
    
    # Nesterov with Adaptive Momentum
    print("4. Running NAG with Adaptive Momentum...")
    nag_adaptive = NesterovAcceleratedGradient(learning_rate=0.01,
                                               use_adaptive_momentum=True,
                                               max_iter=max_iter)
    _, nag_adaptive_history = nag_adaptive.optimize(ill_conditioned_quadratic,
                                                     grad_ill_conditioned, x0)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convergence plot
    ax = axes[0]
    gd_f = [h['f'] for h in gd_history]
    hb_f = [h['f'] for h in hb_history]
    nag_f = [h['f'] for h in nag_history]
    nag_adaptive_f = [h['f'] for h in nag_adaptive_history]
    
    ax.semilogy(gd_f, 'b-', label='Gradient Descent', linewidth=2)
    ax.semilogy(hb_f, 'g-', label='Heavy Ball', linewidth=2)
    ax.semilogy(nag_f, 'r-', label='NAG (fixed μ)', linewidth=2)
    ax.semilogy(nag_adaptive_f, 'm-', label='NAG (adaptive)', linewidth=2)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('f(x) [log scale]', fontsize=12)
    ax.set_title('Convergence Comparison (κ=100)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Trajectory plot
    ax = axes[1]
    
    # Contour
    x_range = np.linspace(-1, 6, 100)
    y_range = np.linspace(-1, 6, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[ill_conditioned_quadratic(np.array([x, y]))
                   for x in x_range] for y in y_range])
    
    contour = ax.contour(X, Y, Z, levels=30, alpha=0.3)
    
    # Plot trajectories
    gd_x = np.array([h['x'] for h in gd_history])
    hb_x = np.array([h['x'] for h in hb_history])
    nag_x = np.array([h['x'] for h in nag_history])
    nag_adaptive_x = np.array([h['x'] for h in nag_adaptive_history])
    
    ax.plot(gd_x[:, 0], gd_x[:, 1], 'b.-', label='GD', markersize=3, alpha=0.7)
    ax.plot(hb_x[:, 0], hb_x[:, 1], 'g.-', label='Heavy Ball', markersize=3, alpha=0.7)
    ax.plot(nag_x[:, 0], nag_x[:, 1], 'r.-', label='NAG (fixed)', markersize=3, alpha=0.7)
    ax.plot(nag_adaptive_x[:, 0], nag_adaptive_x[:, 1], 'm.-', 
            label='NAG (adaptive)', markersize=3)
    ax.plot(0, 0, 'k*', markersize=20, label='Optimum')
    
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Optimization Trajectories', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nesterov_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: nesterov_comparison.png")
    plt.show()
    
    print(f"\nResults:")
    print(f"  GD:              {len(gd_history):3d} iter, final f = {gd_f[-1]:.2e}")
    print(f"  Heavy Ball:      {len(hb_history):3d} iter, final f = {hb_f[-1]:.2e}")
    print(f"  NAG (fixed):     {len(nag_history):3d} iter, final f = {nag_f[-1]:.2e}")
    print(f"  NAG (adaptive):  {len(nag_adaptive_history):3d} iter, final f = {nag_adaptive_f[-1]:.2e}")


def lookahead_visualization():
    """Visualize the look-ahead mechanism of Nesterov."""
    print("\n" + "=" * 70)
    print("LOOK-AHEAD MECHANISM VISUALIZATION")
    print("=" * 70)
    
    x0 = np.array([4.0, 4.0])
    
    nag = NesterovAcceleratedGradient(learning_rate=0.02, momentum=0.9,
                                      use_adaptive_momentum=False, max_iter=50)
    x_opt, history = nag.optimize(ill_conditioned_quadratic, grad_ill_conditioned, x0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Contour
    x_range = np.linspace(-1, 5, 100)
    y_range = np.linspace(-1, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[ill_conditioned_quadratic(np.array([x, y]))
                   for x in x_range] for y in y_range])
    
    contour = ax.contour(X, Y, Z, levels=25, alpha=0.4)
    
    # Plot trajectory
    trajectory = np.array([h['x'] for h in history])
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2,
            label='Actual trajectory', alpha=0.7)
    
    # Show look-ahead positions every 5 iterations
    for i in range(0, min(30, len(history)), 5):
        x_curr = history[i]['x']
        x_look = history[i]['lookahead']
        
        # Draw arrow from current to look-ahead
        ax.arrow(x_curr[0], x_curr[1],
                x_look[0] - x_curr[0], x_look[1] - x_curr[1],
                head_width=0.15, head_length=0.15,
                fc='red', ec='red', alpha=0.6, linewidth=1.5)
        
        ax.plot(x_curr[0], x_curr[1], 'bo', markersize=8)
        ax.plot(x_look[0], x_look[1], 'ro', markersize=8)
        
        if i == 0:
            ax.text(x_curr[0], x_curr[1] + 0.3, 'Current', ha='center')
            ax.text(x_look[0], x_look[1] + 0.3, 'Look-ahead', ha='center')
    
    ax.plot(0, 0, 'g*', markersize=20, label='Optimum')
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Nesterov Look-Ahead Mechanism', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nesterov_lookahead.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: nesterov_lookahead.png")
    plt.show()


def adaptive_momentum_schedule():
    """Visualize the adaptive momentum schedule β_k = (k-1)/(k+2)."""
    print("\n" + "=" * 70)
    print("ADAPTIVE MOMENTUM SCHEDULE")
    print("=" * 70)
    
    iterations = np.arange(0, 100)
    beta_schedule = np.array([(k - 1) / (k + 2) if k > 0 else 0 
                              for k in iterations])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Momentum schedule
    ax = axes[0]
    ax.plot(iterations, beta_schedule, 'b-', linewidth=2)
    ax.axhline(y=0.9, color='r', linestyle='--', label='Fixed μ=0.9', linewidth=2)
    ax.set_xlabel('Iteration k', fontsize=12)
    ax.set_ylabel('Momentum β_k', fontsize=12)
    ax.set_title('Nesterov Adaptive Momentum Schedule\nβ_k = (k-1)/(k+2)',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add annotations
    ax.annotate('β₀ = 0', xy=(0, 0), xytext=(10, 0.1),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               fontsize=11)
    ax.annotate('β₁₀ = 0.75', xy=(10, beta_schedule[10]), xytext=(20, 0.6),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               fontsize=11)
    ax.annotate('β → 1 as k → ∞', xy=(80, beta_schedule[80]), xytext=(60, 0.7),
               arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
               fontsize=11)
    
    # Plot 2: Convergence with different schedules
    ax = axes[1]
    
    x0 = np.array([5.0, 5.0])
    
    # Fixed momentum
    nag_fixed = NesterovAcceleratedGradient(learning_rate=0.01, momentum=0.9,
                                            use_adaptive_momentum=False, max_iter=100)
    _, history_fixed = nag_fixed.optimize(ill_conditioned_quadratic,
                                          grad_ill_conditioned, x0)
    
    # Adaptive momentum
    nag_adaptive = NesterovAcceleratedGradient(learning_rate=0.01,
                                               use_adaptive_momentum=True,
                                               max_iter=100)
    _, history_adaptive = nag_adaptive.optimize(ill_conditioned_quadratic,
                                                 grad_ill_conditioned, x0)
    
    f_fixed = [h['f'] for h in history_fixed]
    f_adaptive = [h['f'] for h in history_adaptive]
    
    ax.semilogy(f_fixed, 'r-', label='Fixed μ=0.9', linewidth=2)
    ax.semilogy(f_adaptive, 'b-', label='Adaptive β_k', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('f(x) [log scale]', fontsize=12)
    ax.set_title('Impact of Adaptive vs Fixed Momentum', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_momentum.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: adaptive_momentum.png")
    plt.show()
    
    print(f"\nFixed momentum (μ=0.9):    {len(history_fixed)} iter, final f = {f_fixed[-1]:.2e}")
    print(f"Adaptive momentum:         {len(history_adaptive)} iter, final f = {f_adaptive[-1]:.2e}")


def convergence_rate_theory():
    """Demonstrate O(1/k²) convergence rate."""
    print("\n" + "=" * 70)
    print("THEORETICAL CONVERGENCE RATE O(1/k²)")
    print("=" * 70)
    
    x0 = np.array([10.0, 10.0])
    
    # Run NAG
    nag = NesterovAcceleratedGradient(learning_rate=0.01, use_adaptive_momentum=True,
                                      max_iter=200)
    x_opt, history = nag.optimize(ill_conditioned_quadratic, grad_ill_conditioned, x0)
    
    f_vals = np.array([h['f'] for h in history])
    iterations = np.arange(1, len(f_vals) + 1)
    
    # Theoretical O(1/k²) bound
    f_star = 0
    C = f_vals[0] - f_star
    theoretical_bound = C / (iterations ** 2)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(iterations, f_vals - f_star, 'b-', linewidth=2, label='NAG (actual)')
    ax.loglog(iterations, theoretical_bound, 'r--', linewidth=2, label='O(1/k²) bound')
    
    ax.set_xlabel('Iteration k', fontsize=12)
    ax.set_ylabel('f(x_k) - f*  [log scale]', fontsize=12)
    ax.set_title('Nesterov Convergence Rate: O(1/k²)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add slope reference
    ax.text(50, 1e-1, 'Slope = -2', fontsize=12, color='red')
    
    plt.tight_layout()
    plt.savefig('convergence_rate_theory.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: convergence_rate_theory.png")
    plt.show()
    
    print(f"\nNAG converged in {len(history)} iterations")
    print(f"Final f: {f_vals[-1]:.2e}")
    print(f"Theoretical complexity: O(1/k²)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NESTEROV ACCELERATED GRADIENT - OPTIMAL MOMENTUM")
    print("=" * 70)
    
    # Run demonstrations
    compare_momentum_methods()
    lookahead_visualization()
    adaptive_momentum_schedule()
    convergence_rate_theory()
    
    print("\n" + "=" * 70)
    print("✓ ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("\nGenerated visualizations:")
    print("  1. nesterov_comparison.png - Comparison with other methods")
    print("  2. nesterov_lookahead.png - Look-ahead mechanism")
    print("  3. adaptive_momentum.png - Adaptive momentum schedule")
    print("  4. convergence_rate_theory.png - O(1/k²) convergence")
