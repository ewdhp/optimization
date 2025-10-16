"""
Active Set Method for Constrained Optimization
============================================

The active set method solves inequality constrained problems by iteratively
identifying which constraints are "active" (binding) at the optimum.

Problem:
    minimize f(x)
    subject to: A·x = b    (equality constraints)
                C·x ≥ d    (inequality constraints)

Key Idea:
- At each iteration, treat active inequalities as equalities
- Solve equality-constrained subproblem
- Update active set based on:
  * Lagrange multipliers (drop constraints with negative multipliers)
  * Constraint violations (add violated constraints)

References:
- Nocedal & Wright (2006). "Numerical Optimization", Chapter 16
- Gill, Murray & Wright (1981). "Practical Optimization"

Author: Optimization Framework
Date: October 16, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional, Set
from scipy.linalg import solve
import warnings
warnings.filterwarnings('ignore')


class ActiveSetMethod:
    """
    Active Set Method for Quadratic Programming
    
    Solves problems of the form:
        minimize   (1/2)·x^T·Q·x + c^T·x
        subject to: A·x = b    (equality)
                    C·x ≥ d    (inequality)
    
    Algorithm:
        1. Start with feasible point and initial active set
        2. Solve equality-constrained QP with active constraints
        3. If optimal for current active set:
           - Check Lagrange multipliers
           - If all non-negative: DONE
           - Otherwise: drop constraint with most negative multiplier
        4. If not optimal:
           - Compute search direction
           - Find step that respects all constraints
           - Add blocking constraint to active set
        5. Repeat
    """
    
    def __init__(self, max_iter: int = 100, tol: float = 1e-8):
        """
        Initialize Active Set Method.
        
        Parameters:
        -----------
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol
        
    def solve_equality_qp(self, Q: np.ndarray, c: np.ndarray,
                          A: np.ndarray, b: np.ndarray,
                          x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve equality-constrained QP:
            minimize   (1/2)·x^T·Q·x + c^T·x
            subject to: A·x = b
        
        Using KKT conditions:
            [ Q   A^T ] [ p  ]   [ -∇f(x) ]
            [ A    0  ] [ λ  ] = [  b-Ax  ]
        
        Returns:
        --------
        p : ndarray
            Search direction
        lam : ndarray
            Lagrange multipliers
        """
        n = len(x)
        m = len(b) if A.size > 0 else 0
        
        # Gradient at current point
        grad = Q @ x + c
        
        if m == 0:
            # Unconstrained case
            p = -np.linalg.solve(Q, grad)
            lam = np.array([])
        else:
            # Constrained case: solve KKT system
            KKT_matrix = np.block([
                [Q, A.T],
                [A, np.zeros((m, m))]
            ])
            
            rhs = np.concatenate([-grad, b - A @ x])
            
            try:
                solution = solve(KKT_matrix, rhs, assume_a='sym')
                p = solution[:n]
                lam = solution[n:]
            except np.linalg.LinAlgError:
                # Singular matrix - use pseudo-inverse
                p = -grad  # Fallback: steepest descent
                lam = np.zeros(m)
        
        return p, lam
    
    def find_blocking_constraint(self, x: np.ndarray, p: np.ndarray,
                                  C: np.ndarray, d: np.ndarray,
                                  active_set: Set[int]) -> Tuple[float, int]:
        """
        Find step size and blocking constraint.
        
        For each inactive inequality C_i·x ≥ d_i:
            Find α such that C_i·(x + α·p) = d_i
            α_i = (d_i - C_i·x) / (C_i·p)
        
        Returns:
        --------
        alpha : float
            Maximum step size
        blocking : int
            Index of blocking constraint (-1 if none)
        """
        alpha = 1.0  # Start with full step
        blocking = -1
        
        for i in range(len(d)):
            if i in active_set:
                continue
            
            Cp = C[i] @ p
            
            if Cp < -self.tol:  # Moving towards constraint
                alpha_i = (d[i] - C[i] @ x) / Cp
                
                if 0 <= alpha_i < alpha:
                    alpha = alpha_i
                    blocking = i
        
        return alpha, blocking
    
    def optimize(self, Q: np.ndarray, c: np.ndarray,
                 A: np.ndarray = None, b: np.ndarray = None,
                 C: np.ndarray = None, d: np.ndarray = None,
                 x0: np.ndarray = None) -> Tuple[np.ndarray, List]:
        """
        Solve QP using active set method.
        
        Parameters:
        -----------
        Q : ndarray (n×n)
            Hessian matrix (must be positive definite)
        c : ndarray (n,)
            Linear term
        A : ndarray (m_eq×n)
            Equality constraint matrix
        b : ndarray (m_eq,)
            Equality constraint RHS
        C : ndarray (m_ineq×n)
            Inequality constraint matrix (C·x ≥ d)
        d : ndarray (m_ineq,)
            Inequality constraint RHS
        x0 : ndarray (n,)
            Initial feasible point
            
        Returns:
        --------
        x_opt : ndarray
            Optimal solution
        history : list
            Optimization history
        """
        n = len(c)
        
        # Handle None inputs
        if A is None:
            A = np.zeros((0, n))
            b = np.array([])
        if C is None:
            C = np.zeros((0, n))
            d = np.array([])
        if x0 is None:
            x0 = np.zeros(n)
        
        x = x0.copy()
        active_set = set()  # Indices of active inequality constraints
        history = []
        
        # Identify initially active constraints
        for i in range(len(d)):
            if abs(C[i] @ x - d[i]) < self.tol:
                active_set.add(i)
        
        print(f"Starting active set method")
        print(f"Initial active set: {active_set}")
        
        for iteration in range(self.max_iter):
            # Build equality constraints: original equalities + active inequalities
            if len(active_set) > 0:
                active_indices = list(active_set)
                A_eq = np.vstack([A, C[active_indices]]) if A.size > 0 else C[active_indices]
                b_eq = np.concatenate([b, d[active_indices]]) if b.size > 0 else d[active_indices]
            else:
                A_eq = A
                b_eq = b
            
            # Solve equality-constrained QP
            p, lam_all = self.solve_equality_qp(Q, c, A_eq, b_eq, x)
            
            # Compute objective and gradient
            f = 0.5 * x.T @ Q @ x + c.T @ x
            grad = Q @ x + c
            
            history.append({
                'iteration': iteration,
                'x': x.copy(),
                'f': f,
                'grad_norm': np.linalg.norm(grad),
                'active_set': active_set.copy(),
                'n_active': len(active_set)
            })
            
            # Check if search direction is zero (stationary point)
            if np.linalg.norm(p) < self.tol:
                # Check Lagrange multipliers for active inequalities
                n_eq = len(b)
                lam_ineq = lam_all[n_eq:] if len(lam_all) > n_eq else np.array([])
                
                # Find most negative multiplier
                if len(lam_ineq) > 0:
                    min_idx = np.argmin(lam_ineq)
                    min_lam = lam_ineq[min_idx]
                    
                    if min_lam < -self.tol:
                        # Drop constraint with negative multiplier
                        active_indices = list(active_set)
                        constraint_to_drop = active_indices[min_idx]
                        active_set.remove(constraint_to_drop)
                        print(f"Iter {iteration}: Dropping constraint {constraint_to_drop} "
                              f"(λ = {min_lam:.6f})")
                        continue
                
                # All multipliers non-negative: optimal!
                print(f"\nOptimal solution found at iteration {iteration}")
                print(f"Active set: {active_set}")
                break
            
            # Find step size and blocking constraint
            alpha, blocking = self.find_blocking_constraint(x, p, C, d, active_set)
            
            # Update solution
            x = x + alpha * p
            
            # Add blocking constraint to active set
            if blocking >= 0 and alpha < 1.0:
                active_set.add(blocking)
                print(f"Iter {iteration}: Added constraint {blocking} to active set (α = {alpha:.6f})")
            else:
                print(f"Iter {iteration}: Full step (α = {alpha:.6f})")
        
        return x, history


# ============================================================================
# TEST PROBLEMS
# ============================================================================

def test_problem_simple():
    """
    Simple QP:
        minimize   (1/2)·(x₁² + x₂²) - x₁ - 2·x₂
        subject to: x₁ + x₂ ≤ 2
                    x₁ ≥ 0
                    x₂ ≥ 0
    
    Solution: x = [0, 2], f* = -4
    """
    Q = np.eye(2)
    c = np.array([-1.0, -2.0])
    
    # Inequality constraints: C·x ≥ d
    # x₁ + x₂ ≤ 2  →  -x₁ - x₂ ≥ -2
    # x₁ ≥ 0
    # x₂ ≥ 0
    C = np.array([
        [-1.0, -1.0],  # x₁ + x₂ ≤ 2
        [1.0, 0.0],    # x₁ ≥ 0
        [0.0, 1.0]     # x₂ ≥ 0
    ])
    d = np.array([-2.0, 0.0, 0.0])
    
    x0 = np.array([0.1, 0.1])
    
    return Q, c, None, None, C, d, x0


def test_problem_2d():
    """
    2D QP with box constraints:
        minimize   (x₁-1)² + (x₂-2)²
        subject to: 0 ≤ x₁ ≤ 2
                    0 ≤ x₂ ≤ 3
    
    Solution: x = [1, 2], f* = 0
    """
    Q = 2 * np.eye(2)
    c = np.array([-2.0, -4.0])
    
    # Box constraints
    C = np.array([
        [1.0, 0.0],   # x₁ ≥ 0
        [0.0, 1.0],   # x₂ ≥ 0
        [-1.0, 0.0],  # x₁ ≤ 2  →  -x₁ ≥ -2
        [0.0, -1.0]   # x₂ ≤ 3  →  -x₂ ≥ -3
    ])
    d = np.array([0.0, 0.0, -2.0, -3.0])
    
    x0 = np.array([0.5, 0.5])
    
    return Q, c, None, None, C, d, x0


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_active_set_progression():
    """Visualize how active set evolves."""
    print("=" * 70)
    print("ACTIVE SET METHOD PROGRESSION")
    print("=" * 70)
    
    Q, c, A, b, C, d, x0 = test_problem_simple()
    
    solver = ActiveSetMethod(max_iter=20)
    x_opt, history = solver.optimize(Q, c, A, b, C, d, x0)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Trajectory with constraints
    ax = axes[0]
    
    # Contour plot
    x1_range = np.linspace(-0.5, 2.5, 100)
    x2_range = np.linspace(-0.5, 2.5, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.zeros_like(X1)
    for i in range(len(x1_range)):
        for j in range(len(x2_range)):
            x = np.array([X1[j, i], X2[j, i]])
            Z[j, i] = 0.5 * x.T @ Q @ x + c.T @ x
    
    contour = ax.contour(X1, X2, Z, levels=20, alpha=0.5)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Draw constraints
    x1_line = np.linspace(0, 2.5, 100)
    
    # x₁ + x₂ ≤ 2
    x2_line = 2 - x1_line
    ax.plot(x1_line, x2_line, 'r-', linewidth=2, label='x₁+x₂≤2')
    ax.fill_between(x1_line, 0, np.minimum(x2_line, 2.5), alpha=0.1, color='green')
    
    # x₁ ≥ 0
    ax.axvline(x=0, color='b', linestyle='-', linewidth=2, alpha=0.5, label='x₁≥0')
    
    # x₂ ≥ 0
    ax.axhline(y=0, color='g', linestyle='-', linewidth=2, alpha=0.5, label='x₂≥0')
    
    # Plot trajectory
    traj = np.array([h['x'] for h in history])
    ax.plot(traj[:, 0], traj[:, 1], 'ko-', markersize=8, linewidth=2,
            label='Optimization path')
    
    # Color code by active set size
    for i, h in enumerate(history):
        n_active = h['n_active']
        color = ['blue', 'orange', 'red', 'purple'][min(n_active, 3)]
        ax.plot(h['x'][0], h['x'][1], 'o', color=color, markersize=10)
        ax.annotate(f"{i}", h['x'], fontsize=9, ha='center', va='bottom')
    
    ax.plot(0, 2, 'g*', markersize=20, label='Optimum')
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Active Set Method Trajectory', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([-0.5, 2.5])
    
    # Plot 2: Active set evolution
    ax = axes[1]
    
    iterations = [h['iteration'] for h in history]
    n_active = [h['n_active'] for h in history]
    f_vals = [h['f'] for h in history]
    
    ax2 = ax.twinx()
    
    # Number of active constraints
    ax.plot(iterations, n_active, 'bo-', linewidth=2, markersize=8,
            label='# Active constraints')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Number of Active Constraints', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    
    # Objective value
    ax2.plot(iterations, f_vals, 'r^-', linewidth=2, markersize=8,
             label='Objective f(x)')
    ax2.set_ylabel('Objective Value', fontsize=12, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.axhline(y=-4, color='g', linestyle='--', label='f* = -4')
    
    ax.set_title('Active Set Size and Objective', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('active_set_progression.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: active_set_progression.png")
    plt.show()
    
    print(f"\nFinal solution: x = {x_opt}")
    print(f"True solution:  x = [0, 2]")
    print(f"Final f(x) = {0.5 * x_opt.T @ Q @ x_opt + c.T @ x_opt:.6f}")


def compare_starting_points():
    """Compare convergence from different starting points."""
    print("\n" + "=" * 70)
    print("EFFECT OF STARTING POINT")
    print("=" * 70)
    
    Q, c, A, b, C, d, _ = test_problem_2d()
    
    starting_points = [
        (np.array([0.5, 0.5]), 'Interior'),
        (np.array([0.0, 0.0]), 'Corner (0,0)'),
        (np.array([2.0, 3.0]), 'Corner (2,3)'),
        (np.array([1.5, 2.5]), 'Near optimum')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Contour background
    x1_range = np.linspace(-0.5, 2.5, 100)
    x2_range = np.linspace(-0.5, 3.5, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.zeros_like(X1)
    for i in range(len(x1_range)):
        for j in range(len(x2_range)):
            x = np.array([X1[j, i], X2[j, i]])
            Z[j, i] = 0.5 * x.T @ Q @ x + c.T @ x
    
    for idx, (x0, label) in enumerate(starting_points):
        ax = axes[idx]
        
        solver = ActiveSetMethod(max_iter=20)
        x_opt, history = solver.optimize(Q, c, A, b, C, d, x0)
        
        # Plot
        contour = ax.contour(X1, X2, Z, levels=20, alpha=0.3)
        
        # Draw box constraints
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=2, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=3, color='r', linestyle='--', alpha=0.5)
        
        # Feasible region
        ax.add_patch(plt.Rectangle((0, 0), 2, 3, fill=True,
                                   alpha=0.1, color='green'))
        
        # Trajectory
        traj = np.array([h['x'] for h in history])
        ax.plot(traj[:, 0], traj[:, 1], 'bo-', markersize=6, linewidth=2)
        ax.plot(x0[0], x0[1], 'go', markersize=12, label='Start')
        ax.plot(1, 2, 'r*', markersize=15, label='Optimum')
        
        ax.set_title(f'Start: {label} ({len(history)} iter)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-0.5, 2.5])
        ax.set_ylim([-0.5, 3.5])
        
        print(f"{label}: {len(history)} iterations, final f = "
              f"{0.5 * x_opt.T @ Q @ x_opt + c.T @ x_opt:.6f}")
    
    plt.tight_layout()
    plt.savefig('active_set_starting_points.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: active_set_starting_points.png")
    plt.show()


def demonstrate_constraint_cycling():
    """Demonstrate potential for constraint cycling."""
    print("\n" + "=" * 70)
    print("CONSTRAINT IDENTIFICATION")
    print("=" * 70)
    
    Q, c, A, b, C, d, x0 = test_problem_simple()
    
    solver = ActiveSetMethod(max_iter=20)
    x_opt, history = solver.optimize(Q, c, A, b, C, d, x0)
    
    # Track which constraints are active
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iterations = range(len(history))
    
    # Create binary matrix: 1 if constraint i is active at iteration j
    n_constraints = len(d)
    active_matrix = np.zeros((n_constraints, len(history)))
    
    for j, h in enumerate(history):
        for i in h['active_set']:
            active_matrix[i, j] = 1
    
    # Plot as heatmap
    im = ax.imshow(active_matrix, aspect='auto', cmap='RdYlGn', 
                   interpolation='nearest')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Constraint Index', fontsize=12)
    ax.set_title('Active Constraint Identification', fontsize=14, fontweight='bold')
    ax.set_yticks(range(n_constraints))
    ax.set_yticklabels(['x₁+x₂≤2', 'x₁≥0', 'x₂≥0'])
    ax.set_xticks(range(len(history)))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Active (1) / Inactive (0)', fontsize=11)
    
    # Add grid
    ax.set_xticks(np.arange(len(history)) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_constraints) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('constraint_identification.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: constraint_identification.png")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ACTIVE SET METHOD FOR CONSTRAINED OPTIMIZATION")
    print("=" * 70)
    
    # Run demonstrations
    visualize_active_set_progression()
    compare_starting_points()
    demonstrate_constraint_cycling()
    
    print("\n" + "=" * 70)
    print("✓ ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("\nGenerated visualizations:")
    print("  1. active_set_progression.png - Method progression")
    print("  2. active_set_starting_points.png - Different starting points")
    print("  3. constraint_identification.png - Active constraint tracking")
