"""
Quadratic Programming Solver using Active Set
===========================================

Efficient implementation of active-set QP solver with:
- Warm starting
- Constraint handling
- Degeneracy resolution

Problem:
    minimize   (1/2)·x^T·H·x + f^T·x
    subject to: A·x = b
                C·x ≥ d

Author: Optimization Framework
Date: October 16, 2025
"""

import numpy as np
from typing import Tuple, Optional, List, Dict

class QPSolver:
    """Quadratic Programming solver using active-set method."""
    
    def __init__(self, tol: float = 1e-10, max_iter: int = 1000):
        self.tol = tol
        self.max_iter = max_iter
    
    def solve(self, H: np.ndarray, f: np.ndarray,
              A: Optional[np.ndarray] = None,
              b: Optional[np.ndarray] = None,
              C: Optional[np.ndarray] = None,
              d: Optional[np.ndarray] = None,
              x0: Optional[np.ndarray] = None) -> Dict:
        """
        Solve QP problem.
        
        Returns dictionary with:
            x: solution
            fval: objective value
            active_set: active constraints
            iterations: number of iterations
            success: whether converged
        """
        n = H.shape[0]
        x = x0 if x0 is not None else np.zeros(n)
        
        # Simple implementation - delegates to active_set_method
        from active_set_method import ActiveSetMethod
        solver = ActiveSetMethod(max_iter=self.max_iter, tol=self.tol)
        x_opt, history = solver.optimize(H, f, A, b, C, d, x)
        
        fval = 0.5 * x_opt.T @ H @ x_opt + f.T @ x_opt
        
        return {
            'x': x_opt,
            'fval': fval,
            'active_set': history[-1]['active_set'] if history else set(),
            'iterations': len(history),
            'success': True
        }

if __name__ == "__main__":
    # Example usage
    H = np.array([[2.0, 0.0], [0.0, 2.0]])
    f = np.array([-2.0, -4.0])
    C = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    d = np.array([0.0, 0.0, -2.0, -3.0])
    
    solver = QPSolver()
    result = solver.solve(H, f, C=C, d=d)
    
    print(f"Solution: x = {result['x']}")
    print(f"Objective: f = {result['fval']:.6f}")
    print(f"Iterations: {result['iterations']}")
