"""
Adaptive Learning Rate Methods - Adam, RMSprop, AdaGrad
======================================================

Modern adaptive optimization methods that adjust learning rates per parameter
based on gradient history. Essential for deep learning and non-convex optimization.

Methods Implemented:
- AdaGrad: Adaptive learning rates based on gradient accumulation
- RMSprop: Moving average of squared gradients
- Adam: Adaptive Moment Estimation (combines momentum + RMSprop)
- AdaMax: Variant of Adam based on infinity norm

References:
- Duchi et al. (2011). "Adaptive Subgradient Methods" (AdaGrad)
- Tieleman & Hinton (2012). "RMSprop" (unpublished)
- Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization"

Author: Optimization Framework
Date: October 16, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class AdaGrad:
    """
    Adaptive Gradient Algorithm (AdaGrad)
    
    Update rule:
        g_t = ∇f(x_t)
        G_t = G_{t-1} + g_t ⊙ g_t  (accumulate squared gradients)
        x_{t+1} = x_t - α / (√G_t + ε) ⊙ g_t
    
    Key Feature: Individual learning rates for each parameter
    Drawback: Learning rate decay may be too aggressive
    """
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8,
                 max_iter: int = 1000, tol: float = 1e-6):
        """
        Initialize AdaGrad optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Initial learning rate α
        epsilon : float
            Small constant for numerical stability
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        
    def optimize(self, f: Callable, grad_f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, List]:
        """Minimize function using AdaGrad."""
        x = x0.copy()
        G = np.zeros_like(x)  # Accumulated squared gradients
        history = []
        
        for k in range(self.max_iter):
            grad = grad_f(x)
            
            # Accumulate squared gradient
            G += grad ** 2
            
            # Adaptive learning rate
            adapted_lr = self.learning_rate / (np.sqrt(G) + self.epsilon)
            
            history.append({
                'iteration': k,
                'x': x.copy(),
                'f': f(x),
                'grad_norm': np.linalg.norm(grad),
                'adapted_lr': adapted_lr.copy(),
                'G': G.copy()
            })
            
            if np.linalg.norm(grad) < self.tol:
                print(f"AdaGrad converged in {k} iterations")
                break
            
            # Update
            x = x - adapted_lr * grad
        
        return x, history


class RMSprop:
    """
    Root Mean Square Propagation (RMSprop)
    
    Update rule:
        g_t = ∇f(x_t)
        E[g²]_t = ρ·E[g²]_{t-1} + (1-ρ)·g_t²  (moving average)
        x_{t+1} = x_t - α / (√E[g²]_t + ε) ⊙ g_t
    
    Key Feature: Uses moving average instead of full accumulation
    Advantage: Doesn't suffer from aggressive learning rate decay
    """
    
    def __init__(self, learning_rate: float = 0.01, decay_rate: float = 0.9,
                 epsilon: float = 1e-8, max_iter: int = 1000, tol: float = 1e-6):
        """
        Initialize RMSprop optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Step size α
        decay_rate : float
            Decay rate ρ for moving average (typically 0.9)
        epsilon : float
            Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        
    def optimize(self, f: Callable, grad_f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, List]:
        """Minimize function using RMSprop."""
        x = x0.copy()
        E_g2 = np.zeros_like(x)  # Moving average of squared gradients
        history = []
        
        for k in range(self.max_iter):
            grad = grad_f(x)
            
            # Update moving average of squared gradient
            E_g2 = self.decay_rate * E_g2 + (1 - self.decay_rate) * (grad ** 2)
            
            # Adaptive learning rate
            adapted_lr = self.learning_rate / (np.sqrt(E_g2) + self.epsilon)
            
            history.append({
                'iteration': k,
                'x': x.copy(),
                'f': f(x),
                'grad_norm': np.linalg.norm(grad),
                'adapted_lr': adapted_lr.copy(),
                'E_g2': E_g2.copy()
            })
            
            if np.linalg.norm(grad) < self.tol:
                print(f"RMSprop converged in {k} iterations")
                break
            
            # Update
            x = x - adapted_lr * grad
        
        return x, history


class Adam:
    """
    Adaptive Moment Estimation (Adam)
    
    Combines:
    - Momentum (first moment)
    - RMSprop (second moment)
    - Bias correction
    
    Update rule:
        g_t = ∇f(x_t)
        m_t = β₁·m_{t-1} + (1-β₁)·g_t           (first moment)
        v_t = β₂·v_{t-1} + (1-β₂)·g_t²          (second moment)
        m̂_t = m_t / (1 - β₁^t)                  (bias correction)
        v̂_t = v_t / (1 - β₂^t)                  (bias correction)
        x_{t+1} = x_t - α·m̂_t / (√v̂_t + ε)
    
    Most popular optimizer in deep learning
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8,
                 max_iter: int = 1000, tol: float = 1e-6):
        """
        Initialize Adam optimizer.
        
        Parameters:
        -----------
        learning_rate : float
            Step size α (default: 0.001)
        beta1 : float
            Exponential decay rate for first moment (typically 0.9)
        beta2 : float
            Exponential decay rate for second moment (typically 0.999)
        epsilon : float
            Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        
    def optimize(self, f: Callable, grad_f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, List]:
        """Minimize function using Adam."""
        x = x0.copy()
        m = np.zeros_like(x)  # First moment
        v = np.zeros_like(x)  # Second moment
        history = []
        
        for k in range(1, self.max_iter + 1):  # Note: k starts at 1 for bias correction
            grad = grad_f(x)
            
            # Update biased first and second moments
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** k)
            v_hat = v / (1 - self.beta2 ** k)
            
            # Adaptive learning rate
            adapted_lr = self.learning_rate / (np.sqrt(v_hat) + self.epsilon)
            
            history.append({
                'iteration': k - 1,
                'x': x.copy(),
                'f': f(x),
                'grad_norm': np.linalg.norm(grad),
                'adapted_lr': adapted_lr.copy(),
                'm': m.copy(),
                'v': v.copy(),
                'm_hat': m_hat.copy(),
                'v_hat': v_hat.copy()
            })
            
            if np.linalg.norm(grad) < self.tol:
                print(f"Adam converged in {k} iterations")
                break
            
            # Update
            x = x - adapted_lr * m_hat
        
        return x, history


class AdaMax:
    """
    AdaMax - Variant of Adam based on infinity norm
    
    More robust to outliers in gradients
    Uses infinity norm instead of L2 norm for second moment
    """
    
    def __init__(self, learning_rate: float = 0.002, beta1: float = 0.9,
                 beta2: float = 0.999, max_iter: int = 1000, tol: float = 1e-6):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_iter = max_iter
        self.tol = tol
        
    def optimize(self, f: Callable, grad_f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, List]:
        """Minimize function using AdaMax."""
        x = x0.copy()
        m = np.zeros_like(x)
        u = np.zeros_like(x)  # Exponentially weighted infinity norm
        history = []
        
        for k in range(1, self.max_iter + 1):
            grad = grad_f(x)
            
            # Update moments
            m = self.beta1 * m + (1 - self.beta1) * grad
            u = np.maximum(self.beta2 * u, np.abs(grad))
            
            # Bias correction for first moment
            m_hat = m / (1 - self.beta1 ** k)
            
            history.append({
                'iteration': k - 1,
                'x': x.copy(),
                'f': f(x),
                'grad_norm': np.linalg.norm(grad)
            })
            
            if np.linalg.norm(grad) < self.tol:
                print(f"AdaMax converged in {k} iterations")
                break
            
            # Update
            x = x - (self.learning_rate / (u + 1e-8)) * m_hat
        
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
    """Ill-conditioned quadratic."""
    Q = np.array([[100, 0], [0, 1]])
    return 0.5 * x.T @ Q @ x

def grad_ill_conditioned(x: np.ndarray) -> np.ndarray:
    """Gradient of ill-conditioned quadratic."""
    Q = np.array([[100, 0], [0, 1]])
    return Q @ x


def saddle_point(x: np.ndarray) -> float:
    """Function with saddle point."""
    return x[0]**2 - x[1]**2

def grad_saddle(x: np.ndarray) -> np.ndarray:
    """Gradient of saddle function."""
    return np.array([2*x[0], -2*x[1]])


# ============================================================================
# VISUALIZATION
# ============================================================================

def compare_adaptive_methods():
    """Compare all adaptive methods."""
    print("=" * 70)
    print("COMPARISON: Adaptive Learning Rate Methods")
    print("=" * 70)
    
    x0 = np.array([5.0, 5.0])
    max_iter = 150
    
    # Run all methods
    print("\nRunning optimizers...")
    
    adagrad = AdaGrad(learning_rate=1.0, max_iter=max_iter)
    _, adagrad_hist = adagrad.optimize(ill_conditioned_quadratic, grad_ill_conditioned, x0)
    
    rmsprop = RMSprop(learning_rate=0.1, max_iter=max_iter)
    _, rmsprop_hist = rmsprop.optimize(ill_conditioned_quadratic, grad_ill_conditioned, x0)
    
    adam = Adam(learning_rate=0.1, max_iter=max_iter)
    _, adam_hist = adam.optimize(ill_conditioned_quadratic, grad_ill_conditioned, x0)
    
    adamax = AdaMax(learning_rate=0.1, max_iter=max_iter)
    _, adamax_hist = adamax.optimize(ill_conditioned_quadratic, grad_ill_conditioned, x0)
    
    # Standard GD for comparison
    gd_hist = []
    x = x0.copy()
    for k in range(max_iter):
        gd_hist.append({'x': x.copy(), 'f': ill_conditioned_quadratic(x)})
        grad = grad_ill_conditioned(x)
        if np.linalg.norm(grad) < 1e-6:
            break
        x = x - 0.01 * grad
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convergence plot
    ax = axes[0]
    
    ax.semilogy([h['f'] for h in gd_hist], 'k--', label='GD (α=0.01)', linewidth=2, alpha=0.7)
    ax.semilogy([h['f'] for h in adagrad_hist], 'b-', label='AdaGrad', linewidth=2)
    ax.semilogy([h['f'] for h in rmsprop_hist], 'g-', label='RMSprop', linewidth=2)
    ax.semilogy([h['f'] for h in adam_hist], 'r-', label='Adam', linewidth=2)
    ax.semilogy([h['f'] for h in adamax_hist], 'm-', label='AdaMax', linewidth=2)
    
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
    
    # Plot trajectories (subsample for clarity)
    def plot_traj(hist, color, label):
        traj = np.array([h['x'] for h in hist[::5]])  # Every 5th point
        ax.plot(traj[:, 0], traj[:, 1], f'{color}.-', 
                label=label, markersize=3, linewidth=1.5, alpha=0.8)
    
    gd_traj = np.array([h['x'] for h in gd_hist[::5]])
    ax.plot(gd_traj[:, 0], gd_traj[:, 1], 'k.--', 
            label='GD', markersize=3, linewidth=1.5, alpha=0.5)
    plot_traj(adagrad_hist, 'b', 'AdaGrad')
    plot_traj(rmsprop_hist, 'g', 'RMSprop')
    plot_traj(adam_hist, 'r', 'Adam')
    plot_traj(adamax_hist, 'm', 'AdaMax')
    
    ax.plot(0, 0, 'k*', markersize=20, label='Optimum')
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Optimization Trajectories', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: adaptive_methods_comparison.png")
    plt.show()
    
    print(f"\nResults:")
    print(f"  GD:       {len(gd_hist):3d} iter, final f = {gd_hist[-1]['f']:.2e}")
    print(f"  AdaGrad:  {len(adagrad_hist):3d} iter, final f = {adagrad_hist[-1]['f']:.2e}")
    print(f"  RMSprop:  {len(rmsprop_hist):3d} iter, final f = {rmsprop_hist[-1]['f']:.2e}")
    print(f"  Adam:     {len(adam_hist):3d} iter, final f = {adam_hist[-1]['f']:.2e}")
    print(f"  AdaMax:   {len(adamax_hist):3d} iter, final f = {adamax_hist[-1]['f']:.2e}")


def learning_rate_adaptation():
    """Visualize how learning rates adapt over time."""
    print("\n" + "=" * 70)
    print("LEARNING RATE ADAPTATION ANALYSIS")
    print("=" * 70)
    
    x0 = np.array([5.0, 5.0])
    max_iter = 100
    
    # Run Adam and RMSprop
    adam = Adam(learning_rate=0.1, max_iter=max_iter)
    _, adam_hist = adam.optimize(ill_conditioned_quadratic, grad_ill_conditioned, x0)
    
    rmsprop = RMSprop(learning_rate=0.1, max_iter=max_iter)
    _, rmsprop_hist = rmsprop.optimize(ill_conditioned_quadratic, grad_ill_conditioned, x0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Adam learning rates
    ax = axes[0, 0]
    adam_lr_x1 = [h['adapted_lr'][0] for h in adam_hist]
    adam_lr_x2 = [h['adapted_lr'][1] for h in adam_hist]
    
    ax.plot(adam_lr_x1, 'b-', label='x₁ (fast direction)', linewidth=2)
    ax.plot(adam_lr_x2, 'r-', label='x₂ (slow direction)', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Adapted Learning Rate', fontsize=11)
    ax.set_title('Adam: Per-Parameter Learning Rates', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RMSprop learning rates
    ax = axes[0, 1]
    rms_lr_x1 = [h['adapted_lr'][0] for h in rmsprop_hist]
    rms_lr_x2 = [h['adapted_lr'][1] for h in rmsprop_hist]
    
    ax.plot(rms_lr_x1, 'b-', label='x₁ (fast direction)', linewidth=2)
    ax.plot(rms_lr_x2, 'r-', label='x₂ (slow direction)', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Adapted Learning Rate', fontsize=11)
    ax.set_title('RMSprop: Per-Parameter Learning Rates', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Adam first and second moments
    ax = axes[1, 0]
    adam_m_norm = [np.linalg.norm(h['m']) for h in adam_hist]
    adam_v_norm = [np.linalg.norm(h['v']) for h in adam_hist]
    
    ax.plot(adam_m_norm, 'b-', label='||m_t|| (first moment)', linewidth=2)
    ax.plot(adam_v_norm, 'r-', label='||v_t|| (second moment)', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Moment Magnitude', fontsize=11)
    ax.set_title('Adam: Moment Evolution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gradient norm comparison
    ax = axes[1, 1]
    adam_grad = [h['grad_norm'] for h in adam_hist]
    rms_grad = [h['grad_norm'] for h in rmsprop_hist]
    
    ax.semilogy(adam_grad, 'b-', label='Adam', linewidth=2)
    ax.semilogy(rms_grad, 'g-', label='RMSprop', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Gradient Norm [log scale]', fontsize=11)
    ax.set_title('Gradient Norm Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_rate_adaptation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: learning_rate_adaptation.png")
    plt.show()


def hyperparameter_sensitivity():
    """Analyze sensitivity to hyperparameters."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    x0 = np.array([5.0, 5.0])
    max_iter = 100
    
    # Test different learning rates for Adam
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Contour background
    x_range = np.linspace(-1, 6, 100)
    y_range = np.linspace(-1, 6, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[ill_conditioned_quadratic(np.array([x, y]))
                   for x in x_range] for y in y_range])
    
    for idx, lr in enumerate(learning_rates):
        ax = axes[idx]
        
        adam = Adam(learning_rate=lr, max_iter=max_iter)
        _, history = adam.optimize(ill_conditioned_quadratic, grad_ill_conditioned, x0)
        
        # Plot
        contour = ax.contour(X, Y, Z, levels=30, alpha=0.3)
        
        traj = np.array([h['x'] for h in history])
        ax.plot(traj[:, 0], traj[:, 1], 'r.-', markersize=3, linewidth=2)
        ax.plot(x0[0], x0[1], 'go', markersize=10, label='Start')
        ax.plot(0, 0, 'b*', markersize=15, label='Optimum')
        
        ax.set_title(f'α = {lr:.3f} ({len(history)} iter)', fontsize=12, fontweight='bold')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        print(f"α = {lr:.3f}: {len(history)} iterations, final f = {history[-1]['f']:.2e}")
    
    plt.tight_layout()
    plt.savefig('hyperparameter_sensitivity.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: hyperparameter_sensitivity.png")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ADAPTIVE LEARNING RATE METHODS")
    print("=" * 70)
    
    # Run demonstrations
    compare_adaptive_methods()
    learning_rate_adaptation()
    hyperparameter_sensitivity()
    
    print("\n" + "=" * 70)
    print("✓ ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("\nGenerated visualizations:")
    print("  1. adaptive_methods_comparison.png - All methods comparison")
    print("  2. learning_rate_adaptation.png - Learning rate evolution")
    print("  3. hyperparameter_sensitivity.png - Sensitivity analysis")
