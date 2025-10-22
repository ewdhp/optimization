"""
Gradient Descent with Momentum
===============================

Momentum methods accelerate gradient descent by accumulating a velocity vector
in directions of persistent reduction in the objective. This helps overcome
local oscillations and accelerates convergence in ravines.

Key Features:
- Faster convergence than vanilla gradient descent
- Reduces oscillations in high-curvature directions
- Builds up speed in consistent gradient directions
- Two main variants: Classical Momentum and Nesterov Momentum

Classical (Heavy Ball) Momentum:
v_{k+1} = β v_k - α ∇f(x_k)
x_{k+1} = x_k + v_{k+1}

Nesterov Accelerated Gradient (NAG):
v_{k+1} = β v_k - α ∇f(x_k + β v_k)
x_{k+1} = x_k + v_{k+1}

where:
- β ∈ [0, 1) is the momentum coefficient (typically 0.9)
- α is the learning rate
- v is the velocity (momentum term)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class MomentumConfig:
    """Configuration for momentum-based methods"""
    momentum_type: str = 'classical'  # 'classical' or 'nesterov'
    beta: float = 0.9  # Momentum coefficient
    learning_rate: float = 0.01
    max_iter: int = 1000
    tol: float = 1e-6
    adaptive_lr: bool = False  # Whether to use adaptive learning rate


class MomentumGD:
    """
    Gradient Descent with Momentum (Classical and Nesterov variants).
    """
    
    def __init__(self, config: Optional[MomentumConfig] = None):
        """Initialize momentum optimizer."""
        self.config = config or MomentumConfig()
        self.history = []
        
    def optimize(self,
                 f: Callable[[np.ndarray], float],
                 grad_f: Callable[[np.ndarray], np.ndarray],
                 x0: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Minimize function f starting from x0.
        
        Args:
            f: Objective function
            grad_f: Gradient of objective function
            x0: Initial point
            
        Returns:
            Optimal point and optimization history
        """
        x = x0.copy()
        v = np.zeros_like(x)  # Initialize velocity
        alpha = self.config.learning_rate
        
        self.history = [{
            'x': x.copy(),
            'f': f(x),
            'grad_norm': np.linalg.norm(grad_f(x)),
            'velocity_norm': 0.0
        }]
        
        for k in range(self.config.max_iter):
            # Compute gradient (location depends on method)
            if self.config.momentum_type == 'nesterov':
                # Nesterov: look ahead
                grad = grad_f(x + self.config.beta * v)
            else:
                # Classical: current position
                grad = grad_f(x)
            
            grad_norm = np.linalg.norm(grad)
            
            # Check convergence
            if grad_norm < self.config.tol:
                print(f"Converged in {k} iterations")
                break
            
            # Adaptive learning rate (simple decay)
            if self.config.adaptive_lr:
                alpha = self.config.learning_rate / (1 + 0.01 * k)
            
            # Update velocity
            v = self.config.beta * v - alpha * grad
            
            # Update position
            x = x + v
            
            # Store history
            self.history.append({
                'x': x.copy(),
                'f': f(x),
                'grad_norm': grad_norm,
                'velocity_norm': np.linalg.norm(v),
                'learning_rate': alpha
            })
            
        return x, self.history


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])


def beale(x: np.ndarray) -> float:
    """Beale function - has ravines and valleys"""
    return ((1.5 - x[0] + x[0]*x[1])**2 +
            (2.25 - x[0] + x[0]*x[1]**2)**2 +
            (2.625 - x[0] + x[0]*x[1]**3)**2)


def beale_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of Beale function"""
    t1 = 1.5 - x[0] + x[0]*x[1]
    t2 = 2.25 - x[0] + x[0]*x[1]**2
    t3 = 2.625 - x[0] + x[0]*x[1]**3
    
    dx = 2*t1*(-1 + x[1]) + 2*t2*(-1 + x[1]**2) + 2*t3*(-1 + x[1]**3)
    dy = 2*t1*x[0] + 2*t2*(2*x[0]*x[1]) + 2*t3*(3*x[0]*x[1]**2)
    
    return np.array([dx, dy])


def visualize_momentum_effect(histories: dict,
                              f: Callable,
                              xlim: Tuple[float, float] = (-2, 2),
                              ylim: Tuple[float, float] = (-2, 2),
                              title: str = "Momentum Methods"):
    """Visualize momentum effects on optimization."""
    fig = plt.figure(figsize=(18, 5))
    
    # Plot 1: Trajectories with velocity vectors
    ax1 = plt.subplot(131)
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    ax1.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    
    colors = ['r', 'b', 'g', 'orange']
    for idx, (method, history) in enumerate(histories.items()):
        traj = np.array([h['x'] for h in history])
        ax1.plot(traj[:, 0], traj[:, 1],
                f'{colors[idx]}.-', linewidth=2, markersize=4,
                label=f'{method} ({len(history)} iter)', alpha=0.7)
        
        # Draw some velocity vectors
        step = max(1, len(history) // 10)
        for i in range(0, len(history)-1, step):
            if i + 1 < len(history):
                dx = history[i+1]['x'][0] - history[i]['x'][0]
                dy = history[i+1]['x'][1] - history[i]['x'][1]
                ax1.arrow(history[i]['x'][0], history[i]['x'][1],
                         dx*0.5, dy*0.5,
                         head_width=0.05, head_length=0.05,
                         fc=colors[idx], ec=colors[idx], alpha=0.3)
    
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title(f'{title}\nTrajectories with Velocity Vectors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Function value convergence
    ax2 = plt.subplot(132)
    for idx, (method, history) in enumerate(histories.items()):
        f_vals = [h['f'] for h in history]
        ax2.semilogy(range(len(f_vals)), f_vals,
                    f'{colors[idx]}-', linewidth=2, label=method)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Function Value (log scale)')
    ax2.set_title('Convergence Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Velocity magnitude
    ax3 = plt.subplot(133)
    for idx, (method, history) in enumerate(histories.items()):
        vel_norms = [h['velocity_norm'] for h in history]
        ax3.plot(range(len(vel_norms)), vel_norms,
                f'{colors[idx]}-', linewidth=2, label=method)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Velocity Magnitude')
    ax3.set_title('Momentum Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def example_momentum_comparison():
    """Compare vanilla GD, classical momentum, and Nesterov momentum"""
    print("=" * 70)
    print("Example 1: Momentum Comparison on Rosenbrock Function")
    print("=" * 70)
    print("Comparing vanilla GD, classical momentum, and Nesterov momentum")
    print()
    
    x0 = np.array([-1.0, 1.0])
    histories = {}
    
    # Vanilla gradient descent (momentum = 0)
    config_vanilla = MomentumConfig(
        momentum_type='classical',
        beta=0.0,
        learning_rate=0.001,
        max_iter=1000
    )
    vanilla = MomentumGD(config_vanilla)
    x_vanilla, history_vanilla = vanilla.optimize(rosenbrock, rosenbrock_grad, x0)
    histories['Vanilla GD (β=0)'] = history_vanilla
    
    print("Vanilla Gradient Descent:")
    print("-" * 40)
    print(f"Optimal point: {x_vanilla}")
    print(f"Optimal value: {rosenbrock(x_vanilla):.2e}")
    print(f"Iterations: {len(history_vanilla) - 1}")
    
    # Classical momentum
    config_classical = MomentumConfig(
        momentum_type='classical',
        beta=0.9,
        learning_rate=0.001,
        max_iter=1000
    )
    classical = MomentumGD(config_classical)
    x_classical, history_classical = classical.optimize(rosenbrock, rosenbrock_grad, x0)
    histories['Classical Momentum (β=0.9)'] = history_classical
    
    print("\nClassical Momentum:")
    print("-" * 40)
    print(f"Optimal point: {x_classical}")
    print(f"Optimal value: {rosenbrock(x_classical):.2e}")
    print(f"Iterations: {len(history_classical) - 1}")
    print(f"Speedup: {len(history_vanilla) / len(history_classical):.2f}x")
    
    # Nesterov momentum
    config_nesterov = MomentumConfig(
        momentum_type='nesterov',
        beta=0.9,
        learning_rate=0.001,
        max_iter=1000
    )
    nesterov = MomentumGD(config_nesterov)
    x_nesterov, history_nesterov = nesterov.optimize(rosenbrock, rosenbrock_grad, x0)
    histories['Nesterov Momentum (β=0.9)'] = history_nesterov
    
    print("\nNesterov Momentum:")
    print("-" * 40)
    print(f"Optimal point: {x_nesterov}")
    print(f"Optimal value: {rosenbrock(x_nesterov):.2e}")
    print(f"Iterations: {len(history_nesterov) - 1}")
    print(f"Speedup: {len(history_vanilla) / len(history_nesterov):.2f}x")
    
    visualize_momentum_effect(histories, rosenbrock,
                             xlim=(-1.5, 1.5), ylim=(-0.5, 1.5),
                             title="Momentum Methods on Rosenbrock")


def example_beta_sensitivity():
    """Analyze sensitivity to momentum coefficient β"""
    print("\n" + "=" * 70)
    print("Example 2: Momentum Coefficient Sensitivity")
    print("=" * 70)
    print("Testing different values of β on Beale function")
    print()
    
    x0 = np.array([1.0, 1.0])
    betas = [0.0, 0.5, 0.9, 0.95, 0.99]
    histories = {}
    
    for beta in betas:
        config = MomentumConfig(
            momentum_type='classical',
            beta=beta,
            learning_rate=0.001,
            max_iter=2000
        )
        optimizer = MomentumGD(config)
        x_opt, history = optimizer.optimize(beale, beale_grad, x0)
        histories[f'β={beta}'] = history
        
        print(f"β = {beta:.2f}: {len(history)-1:4d} iterations, "
              f"f* = {beale(x_opt):.2e}")
    
    visualize_momentum_effect(histories, beale,
                             xlim=(-0.5, 3.5), ylim=(-0.5, 3.5),
                             title="Effect of Momentum Coefficient β")


def example_ravine_problem():
    """Demonstrate momentum advantage in ravine-like landscapes"""
    print("\n" + "=" * 70)
    print("Example 3: Ravine Problem - Beale Function")
    print("=" * 70)
    print("Beale function has narrow valleys where momentum excels")
    print()
    
    x0 = np.array([3.0, 0.5])
    histories = {}
    
    # Try with and without momentum
    for use_momentum in [False, True]:
        config = MomentumConfig(
            momentum_type='nesterov' if use_momentum else 'classical',
            beta=0.9 if use_momentum else 0.0,
            learning_rate=0.005,
            max_iter=5000
        )
        optimizer = MomentumGD(config)
        x_opt, history = optimizer.optimize(beale, beale_grad, x0)
        
        method_name = 'With Nesterov Momentum' if use_momentum else 'Without Momentum'
        histories[method_name] = history
        
        print(f"\n{method_name}:")
        print("-" * 40)
        print(f"Optimal point: {x_opt}")
        print(f"Optimal value: {beale(x_opt):.2e}")
        print(f"Iterations: {len(history) - 1}")
    
    # Calculate improvement
    iter_no_momentum = len(histories['Without Momentum'])
    iter_with_momentum = len(histories['With Nesterov Momentum'])
    print(f"\nSpeedup with momentum: {iter_no_momentum / iter_with_momentum:.2f}x")
    
    visualize_momentum_effect(histories, beale,
                             xlim=(0.5, 3.5), ylim=(0, 1.5),
                             title="Momentum on Beale Function (Ravine)")


def example_adaptive_learning_rate():
    """Compare fixed vs adaptive learning rate with momentum"""
    print("\n" + "=" * 70)
    print("Example 4: Adaptive Learning Rate with Momentum")
    print("=" * 70)
    print("Comparing fixed vs time-decaying learning rate")
    print()
    
    x0 = np.array([-1.0, 1.0])
    histories = {}
    
    # Fixed learning rate
    config_fixed = MomentumConfig(
        momentum_type='nesterov',
        beta=0.9,
        learning_rate=0.001,
        max_iter=1000,
        adaptive_lr=False
    )
    fixed = MomentumGD(config_fixed)
    x_fixed, history_fixed = fixed.optimize(rosenbrock, rosenbrock_grad, x0)
    histories['Fixed LR'] = history_fixed
    
    # Adaptive learning rate
    config_adaptive = MomentumConfig(
        momentum_type='nesterov',
        beta=0.9,
        learning_rate=0.005,  # Start higher
        max_iter=1000,
        adaptive_lr=True
    )
    adaptive = MomentumGD(config_adaptive)
    x_adaptive, history_adaptive = adaptive.optimize(rosenbrock, rosenbrock_grad, x0)
    histories['Adaptive LR'] = history_adaptive
    
    print("Fixed Learning Rate:")
    print(f"  Iterations: {len(history_fixed) - 1}")
    print(f"  Final f(x): {history_fixed[-1]['f']:.2e}")
    
    print("\nAdaptive Learning Rate:")
    print(f"  Iterations: {len(history_adaptive) - 1}")
    print(f"  Final f(x): {history_adaptive[-1]['f']:.2e}")
    
    visualize_momentum_effect(histories, rosenbrock,
                             xlim=(-1.5, 1.5), ylim=(-0.5, 1.5),
                             title="Fixed vs Adaptive Learning Rate")


if __name__ == "__main__":
    # Run examples
    example_momentum_comparison()
    example_beta_sensitivity()
    example_ravine_problem()
    example_adaptive_learning_rate()
    
    print("\n" + "=" * 70)
    print("Key Observations:")
    print("=" * 70)
    print("1. Momentum significantly accelerates convergence")
    print("2. β = 0.9 is a good default choice")
    print("3. Nesterov momentum often outperforms classical momentum")
    print("4. Too high β (>0.95) can cause overshooting")
    print("5. Momentum is especially effective in ravines and valleys")
    print("6. Adaptive learning rate can improve convergence")
    print("7. Momentum builds up speed in consistent directions")
