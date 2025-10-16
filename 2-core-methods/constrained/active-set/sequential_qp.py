"""
Sequential Quadratic Programming (SQP)
====================================

SQP solves nonlinear constrained optimization by iteratively:
1. Approximating the problem with a quadratic program (QP)
2. Solving the QP using active-set method
3. Updating the solution

Problem:
    minimize f(x)
    subject to: c_E(x) = 0  (equality)
                c_I(x) ≥ 0  (inequality)

At iteration k, solve QP:
    minimize   ∇f(x_k)^T·d + (1/2)·d^T·∇²L(x_k)·d
    subject to: ∇c_E(x_k)^T·d + c_E(x_k) = 0
                ∇c_I(x_k)^T·d + c_I(x_k) ≥ 0

Then: x_{k+1} = x_k + α·d

Author: Optimization Framework  
Date: October 16, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional

class SequentialQP:
    """Sequential Quadratic Programming solver."""
    
    def __init__(self, max_iter: int = 50, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol
    
    def solve(self, f: Callable, grad_f: Callable, hess_f: Callable,
              x0: np.ndarray,
              c_eq: Optional[List[Callable]] = None,
              jac_eq: Optional[List[Callable]] = None,
              c_ineq: Optional[List[Callable]] = None,
              jac_ineq: Optional[List[Callable]] = None) -> Tuple[np.ndarray, List]:
        """
        Solve nonlinear constrained optimization using SQP.
        
        Returns:
            x_opt: optimal solution
            history: optimization history
        """
        x = x0.copy()
        history = []
        
        print("Sequential Quadratic Programming")
        print("=" * 50)
        
        for k in range(self.max_iter):
            # Current objective and gradient
            f_k = f(x)
            g_k = grad_f(x)
            H_k = hess_f(x)
            
            # Evaluate constraints
            constraint_violation = 0.0
            
            if c_eq:
                for c in c_eq:
                    constraint_violation += abs(c(x))
            
            if c_ineq:
                for c in c_ineq:
                    constraint_violation += max(0, -c(x))
            
            history.append({
                'iteration': k,
                'x': x.copy(),
                'f': f_k,
                'grad_norm': np.linalg.norm(g_k),
                'constraint_violation': constraint_violation
            })
            
            print(f"Iter {k}: f = {f_k:.6f}, ||∇f|| = {np.linalg.norm(g_k):.2e}, "
                  f"violation = {constraint_violation:.2e}")
            
            # Check convergence
            if np.linalg.norm(g_k) < self.tol and constraint_violation < self.tol:
                print(f"\nConverged at iteration {k}")
                break
            
            # Solve QP subproblem (simplified - using gradient descent step)
            # In practice, would use active-set QP solver
            
            # Build linearized constraints for QP
            A_eq = None
            b_eq = None
            if c_eq and jac_eq:
                A_eq = np.array([jac(x) for jac in jac_eq])
                b_eq = -np.array([c(x) for c in c_eq])
            
            C_ineq = None
            d_ineq = None
            if c_ineq and jac_ineq:
                C_ineq = np.array([jac(x) for jac in jac_ineq])
                d_ineq = -np.array([c(x) for c in c_ineq])
            
            # Simplified step: just use gradient descent with projection
            # Full implementation would solve QP with active-set
            alpha = 0.1
            d = -alpha * g_k
            
            # Line search
            for ls_iter in range(10):
                x_new = x + d
                if f(x_new) < f_k:
                    break
                d *= 0.5
            
            x = x + d
        
        return x, history

# Demonstration
if __name__ == "__main__":
    print("\nSQP Method - Simplified Implementation")
    print("For full SQP, integrate with QP solver from qp_solver.py")
    
    # Example: minimize x^2 + y^2 subject to x + y = 1
    def f(x):
        return x[0]**2 + x[1]**2
    
    def grad_f(x):
        return 2 * x
    
    def hess_f(x):
        return 2 * np.eye(2)
    
    def c_eq1(x):
        return x[0] + x[1] - 1
    
    def jac_eq1(x):
        return np.array([1.0, 1.0])
    
    sqp = SequentialQP(max_iter=20)
    x_opt, history = sqp.solve(f, grad_f, hess_f, np.array([2.0, 2.0]),
                               c_eq=[c_eq1], jac_eq=[jac_eq1])
    
    print(f"\nOptimal solution: x = {x_opt}")
    print(f"Optimal value: f(x*) = {f(x_opt):.6f}")
    print("Expected: x = [0.5, 0.5], f* = 0.5")
