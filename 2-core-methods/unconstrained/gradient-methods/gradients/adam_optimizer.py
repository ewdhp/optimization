"""
Adam Optimizer (Adaptive Moment Estimation)
============================================

Adam combines ideas from momentum and adaptive learning rates (RMSprop).
It maintains separate adaptive learning rates for each parameter based on
estimates of first and second moments of gradients.

Key Features:
- Combines momentum and adaptive learning rates
- Works well in practice with little tuning
- Computationally efficient
- Suitable for large-scale problems
- Most popular optimizer in deep learning

Algorithm:
1. Initialize: m_0 = 0, v_0 = 0
2. For t = 1, 2, 3, ...
   - g_t = ∇f(x_t)
   - m_t = β₁ m_{t-1} + (1-β₁) g_t          [First moment (momentum)]
   - v_t = β₂ v_{t-1} + (1-β₂) g_t²         [Second moment (RMSprop)]
   - m̂_t = m_t / (1 - β₁^t)                 [Bias correction]
   - v̂_t = v_t / (1 - β₂^t)                 [Bias correction]
   - x_t = x_{t-1} - α m̂_t / (√v̂_t + ε)

Default hyperparameters (work well in practice):
- α = 0.001 (learning rate)
- β₁ = 0.9 (exponential decay rate for 1st moment)
- β₂ = 0.999 (exponential decay rate for 2nd moment)
- ε = 1e-8 (small constant for numerical stability)

Variants:
- AdaMax: Variant using infinity norm
- AMSGrad: Fixes convergence issues in some cases
- Nadam: Adam + Nesterov momentum
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class AdamConfig:
    """Configuration for Adam optimizer and variants"""
    variant: str = 'adam'  # 'adam', 'adamax', 'amsgrad', 'nadam'
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_iter: int = 1000
    tol: float = 1e-6


class Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer and variants.
    """
    
    def __init__(self, config: Optional[AdamConfig] = None):
        """Initialize Adam optimizer."""
        self.config = config or AdamConfig()
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
        m = np.zeros_like(x)  # First moment vector
        v = np.zeros_like(x)  # Second moment vector
        v_hat_max = np.zeros_like(x)  # For AMSGrad (initialized for all cases)
        
        self.history = [{
            'x': x.copy(),
            'f': f(x),
            'grad_norm': np.linalg.norm(grad_f(x))
        }]
        
        for t in range(1, self.config.max_iter + 1):
            # Compute gradient
            grad = grad_f(x)
            grad_norm = np.linalg.norm(grad)
            
            # Check convergence
            if grad_norm < self.config.tol:
                print(f"Converged in {t-1} iterations")
                break
            
            if self.config.variant == 'adam':
                x_new, m, v = self._adam_step(x, grad, m, v, t)
            elif self.config.variant == 'adamax':
                x_new, m, v = self._adamax_step(x, grad, m, v, t)
            elif self.config.variant == 'amsgrad':
                x_new, m, v, v_hat_max = self._amsgrad_step(x, grad, m, v, v_hat_max, t)
            elif self.config.variant == 'nadam':
                x_new, m, v = self._nadam_step(x, grad, m, v, t)
            else:
                raise ValueError(f"Unknown variant: {self.config.variant}")
            
            # Store history
            self.history.append({
                'x': x_new.copy(),
                'f': f(x_new),
                'grad_norm': grad_norm,
                'learning_rate': self.config.learning_rate
            })
            
            x = x_new
            
        return x, self.history
    
    def _adam_step(self, x, grad, m, v, t):
        """Standard Adam update"""
        # Update biased first moment estimate
        m = self.config.beta1 * m + (1 - self.config.beta1) * grad
        
        # Update biased second raw moment estimate
        v = self.config.beta2 * v + (1 - self.config.beta2) * grad**2
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - self.config.beta1**t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - self.config.beta2**t)
        
        # Update parameters
        x_new = x - self.config.learning_rate * m_hat / (np.sqrt(v_hat) + self.config.epsilon)
        
        return x_new, m, v
    
    def _adamax_step(self, x, grad, m, v, t):
        """AdaMax update (uses infinity norm)"""
        # Update biased first moment estimate
        m = self.config.beta1 * m + (1 - self.config.beta1) * grad
        
        # Update the exponentially weighted infinity norm
        v = np.maximum(self.config.beta2 * v, np.abs(grad))
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - self.config.beta1**t)
        
        # Update parameters
        x_new = x - self.config.learning_rate * m_hat / (v + self.config.epsilon)
        
        return x_new, m, v
    
    def _amsgrad_step(self, x, grad, m, v, v_hat_max, t):
        """AMSGrad update (maintains max of v_hat)"""
        # Update biased first moment estimate
        m = self.config.beta1 * m + (1 - self.config.beta1) * grad
        
        # Update biased second raw moment estimate
        v = self.config.beta2 * v + (1 - self.config.beta2) * grad**2
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - self.config.beta2**t)
        
        # Maintain the maximum of all v_hat
        v_hat_max = np.maximum(v_hat_max, v_hat)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - self.config.beta1**t)
        
        # Update parameters using the maximum
        x_new = x - self.config.learning_rate * m_hat / (np.sqrt(v_hat_max) + self.config.epsilon)
        
        return x_new, m, v, v_hat_max
    
    def _nadam_step(self, x, grad, m, v, t):
        """Nadam update (Nesterov + Adam)"""
        # Update biased first moment estimate
        m = self.config.beta1 * m + (1 - self.config.beta1) * grad
        
        # Update biased second raw moment estimate
        v = self.config.beta2 * v + (1 - self.config.beta2) * grad**2
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - self.config.beta1**t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - self.config.beta2**t)
        
        # Nesterov momentum
        m_bar = self.config.beta1 * m_hat + (1 - self.config.beta1) * grad / (1 - self.config.beta1**t)
        
        # Update parameters
        x_new = x - self.config.learning_rate * m_bar / (np.sqrt(v_hat) + self.config.epsilon)
        
        return x_new, m, v


# Test functions
def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function (highly multimodal)"""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def rastrigin_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of Rastrigin function"""
    A = 10
    return 2 * x + 2 * A * np.pi * np.sin(2 * np.pi * x)


def sphere(x: np.ndarray) -> float:
    """Simple sphere function"""
    return np.sum(x**2)


def sphere_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of sphere function"""
    return 2 * x


def visualize_adam_variants(histories: dict,
                           f: Callable,
                           xlim: Tuple[float, float] = (-2, 2),
                           ylim: Tuple[float, float] = (-2, 2),
                           title: str = "Adam Variants"):
    """Visualize different Adam variants."""
    fig = plt.figure(figsize=(16, 5))
    
    # Plot 1: Contour with trajectories
    ax1 = plt.subplot(131)
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    ax1.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    
    colors = ['r', 'b', 'g', 'orange', 'm']
    for idx, (method, history) in enumerate(histories.items()):
        traj = np.array([h['x'] for h in history])
        ax1.plot(traj[:, 0], traj[:, 1],
                f'{colors[idx]}.-', linewidth=2, markersize=4,
                label=f'{method} ({len(history)} iter)', alpha=0.7)
    
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.set_title(f'{title}\nTrajectories')
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
    
    # Plot 3: Gradient norm
    ax3 = plt.subplot(133)
    for idx, (method, history) in enumerate(histories.items()):
        grad_norms = [h['grad_norm'] for h in history]
        ax3.semilogy(range(len(grad_norms)), grad_norms,
                    f'{colors[idx]}-', linewidth=2, label=method)
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient Norm (log scale)')
    ax3.set_title('Gradient Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def example_adam_variants():
    """Compare different Adam variants"""
    print("=" * 70)
    print("Example 1: Adam Variants on Rosenbrock Function")
    print("=" * 70)
    print("Comparing Adam, AdaMax, AMSGrad, and Nadam")
    print()
    
    x0 = np.array([-1.0, 1.0])
    variants = ['adam', 'adamax', 'amsgrad', 'nadam']
    histories = {}
    
    for variant in variants:
        config = AdamConfig(variant=variant, max_iter=500)
        optimizer = Adam(config)
        x_opt, history = optimizer.optimize(rosenbrock, rosenbrock_grad, x0)
        histories[variant.upper()] = history
        
        print(f"\n{variant.upper()}:")
        print("-" * 40)
        print(f"Optimal point: {x_opt}")
        print(f"Optimal value: {rosenbrock(x_opt):.2e}")
        print(f"Iterations: {len(history) - 1}")
        print(f"Final gradient norm: {history[-1]['grad_norm']:.2e}")
    
    visualize_adam_variants(histories, rosenbrock,
                           xlim=(-1.5, 1.5), ylim=(-0.5, 1.5),
                           title="Adam Variants on Rosenbrock")


def example_learning_rate_sensitivity():
    """Test sensitivity to learning rate"""
    print("\n" + "=" * 70)
    print("Example 2: Learning Rate Sensitivity")
    print("=" * 70)
    print("Testing different learning rates with Adam")
    print()
    
    x0 = np.array([-1.0, 1.0])
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    histories = {}
    
    for lr in learning_rates:
        config = AdamConfig(learning_rate=lr, max_iter=1000)
        optimizer = Adam(config)
        x_opt, history = optimizer.optimize(rosenbrock, rosenbrock_grad, x0)
        histories[f'α={lr}'] = history
        
        print(f"Learning rate α = {lr}: {len(history)-1} iterations, "
              f"f* = {rosenbrock(x_opt):.2e}")
    
    visualize_adam_variants(histories, rosenbrock,
                           xlim=(-1.5, 1.5), ylim=(-0.5, 1.5),
                           title="Learning Rate Sensitivity")


def example_multimodal_function():
    """Test on multimodal Rastrigin function"""
    print("\n" + "=" * 70)
    print("Example 3: Multimodal Function (Rastrigin)")
    print("=" * 70)
    print("Rastrigin function has many local minima")
    print()
    
    x0 = np.array([4.0, 4.0])
    
    config = AdamConfig(learning_rate=0.01, max_iter=1000)
    optimizer = Adam(config)
    x_opt, history = optimizer.optimize(rastrigin, rastrigin_grad, x0)
    
    print("Adam on Rastrigin:")
    print("-" * 40)
    print(f"Starting point: {x0}")
    print(f"Optimal point: {x_opt}")
    print(f"Optimal value: {rastrigin(x_opt):.2e}")
    print(f"Iterations: {len(history) - 1}")
    print(f"Global minimum at (0, 0) with f* = 0")
    
    # Visualize
    histories = {'Adam': history}
    visualize_adam_variants(histories, rastrigin,
                           xlim=(-5, 5), ylim=(-5, 5),
                           title="Adam on Rastrigin Function")


def compare_with_other_methods():
    """Compare Adam with other gradient methods"""
    print("\n" + "=" * 70)
    print("Example 4: Adam vs Other Methods")
    print("=" * 70)
    print("Comparing Adam with GD, Momentum, and BFGS")
    print()
    
    x0 = np.array([-1.0, 1.0])
    histories = {}
    
    # Adam
    adam_config = AdamConfig(learning_rate=0.01, max_iter=500)
    adam = Adam(adam_config)
    _, history_adam = adam.optimize(rosenbrock, rosenbrock_grad, x0)
    histories['Adam'] = history_adam
    
    # Try importing other methods
    try:
        from steepest_descent import SteepestDescent
        sd = SteepestDescent(step_size=0.001, max_iter=500, line_search='backtracking')
        _, history_sd = sd.optimize(rosenbrock, rosenbrock_grad, x0)
        histories['Steepest Descent'] = history_sd
    except ImportError:
        print("Note: Could not import steepest_descent")
    
    # Note: momentum_methods.py not yet implemented in this directory
    # Uncomment when momentum_methods.py is added to gradients/
    # try:
    #     from momentum_methods import MomentumGD, MomentumConfig
    #     mom_config = MomentumConfig(beta=0.9, learning_rate=0.001, max_iter=500)
    #     mom = MomentumGD(mom_config)
    #     _, history_mom = mom.optimize(rosenbrock, rosenbrock_grad, x0)
    #     histories['Momentum'] = history_mom
    # except ImportError:
    #     print("Note: Could not import momentum_methods")
    
    try:
        from quasi_newton import BFGS, QuasiNewtonConfig
        bfgs_config = QuasiNewtonConfig(max_iter=500)
        bfgs = BFGS(bfgs_config)
        _, history_bfgs = bfgs.optimize(rosenbrock, rosenbrock_grad, x0)
        histories['BFGS'] = history_bfgs
    except ImportError:
        print("Note: Could not import quasi_newton")
    
    # Print comparison
    print("\nIterations to convergence:")
    print("-" * 40)
    for method, history in histories.items():
        print(f"{method:20s}: {len(history)-1:4d} iterations")
    
    if len(histories) > 1:
        visualize_adam_variants(histories, rosenbrock,
                               xlim=(-1.5, 1.5), ylim=(-0.5, 1.5),
                               title="Method Comparison")


if __name__ == "__main__":
    # Run examples
    example_adam_variants()
    example_learning_rate_sensitivity()
    example_multimodal_function()
    
    # Uncomment to compare with other methods
    # compare_with_other_methods()
    
    print("\n" + "=" * 70)
    print("Key Observations:")
    print("=" * 70)
    print("1. Adam combines best of momentum and adaptive learning rates")
    print("2. Default hyperparameters (α=0.001, β₁=0.9, β₂=0.999) work well")
    print("3. Adam is robust to choice of learning rate")
    print("4. Nadam often converges slightly faster than Adam")
    print("5. AMSGrad provides theoretical convergence guarantees")
    print("6. Adam is the most popular optimizer in deep learning")
    print("7. Works well on noisy, sparse gradients")
    print("8. Memory efficient: O(n) storage")
