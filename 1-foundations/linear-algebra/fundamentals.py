"""
Linear Algebra Fundamentals for Optimization

This module implements core linear algebra concepts essential for optimization theory,
including vector norms, matrix operations, eigenvalue analysis, and positive definiteness.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import warnings

class LinearAlgebraOptimization:
    """Core linear algebra operations for optimization"""
    
    def __init__(self):
        """Initialize the linear algebra toolkit"""
        self.tolerance = 1e-12
        
    def vector_norms(self, x: np.ndarray, show_comparison: bool = True) -> dict:
        """
        Calculate different vector norms and their properties
        
        Args:
            x: Input vector
            show_comparison: Whether to display norm comparison
            
        Returns:
            Dictionary with norm values and properties
        """
        norms = {
            'l1': np.linalg.norm(x, 1),           # L1 norm (Manhattan)
            'l2': np.linalg.norm(x, 2),           # L2 norm (Euclidean)  
            'linf': np.linalg.norm(x, np.inf),    # L∞ norm (Maximum)
            'l2_squared': np.dot(x, x)            # Squared L2 norm
        }
        
        # Theoretical relationships
        n = len(x)
        relations = {
            'l2_leq_l1': norms['l2'] <= norms['l1'],
            'l1_leq_sqrt_n_l2': norms['l1'] <= np.sqrt(n) * norms['l2'],
            'linf_leq_l2': norms['linf'] <= norms['l2'],
            'l2_leq_sqrt_n_linf': norms['l2'] <= np.sqrt(n) * norms['linf']
        }
        
        if show_comparison:
            self._plot_norm_comparison(x, norms)
            
        return {'norms': norms, 'relationships': relations}
    
    def _plot_norm_comparison(self, x: np.ndarray, norms: dict):
        """Visualize different norms"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Norm values comparison
        norm_names = ['L1', 'L2', 'L∞']
        norm_values = [norms['l1'], norms['l2'], norms['linf']]
        
        bars = ax1.bar(norm_names, norm_values, alpha=0.7, 
                      color=['red', 'blue', 'green'])
        ax1.set_ylabel('Norm Value')
        ax1.set_title('Vector Norm Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, norm_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Unit balls visualization (2D case)
        if len(x) == 2:
            theta = np.linspace(0, 2*np.pi, 1000)
            
            # L2 ball: circle
            l2_x = np.cos(theta)
            l2_y = np.sin(theta)
            ax2.plot(l2_x, l2_y, 'b-', label='L2 ball', linewidth=2)
            
            # L1 ball: diamond
            l1_x = np.array([1, 0, -1, 0, 1])
            l1_y = np.array([0, 1, 0, -1, 0])
            ax2.plot(l1_x, l1_y, 'r-', label='L1 ball', linewidth=2)
            
            # L∞ ball: square
            linf_x = np.array([1, 1, -1, -1, 1])
            linf_y = np.array([1, -1, -1, 1, 1])
            ax2.plot(linf_x, linf_y, 'g-', label='L∞ ball', linewidth=2)
            
            # Plot the vector
            ax2.arrow(0, 0, x[0], x[1], head_width=0.05, head_length=0.05,
                     fc='black', ec='black', linewidth=2)
            ax2.text(x[0], x[1], f'  x=({x[0]:.2f}, {x[1]:.2f})', 
                    fontsize=10, fontweight='bold')
            
            ax2.set_xlim(-1.5, 1.5)
            ax2.set_ylim(-1.5, 1.5)
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_title('Unit Balls in Different Norms')
            ax2.set_xlabel('x₁')
            ax2.set_ylabel('x₂')
        
        plt.tight_layout()
        plt.show()
    
    def check_positive_definiteness(self, A: np.ndarray, method: str = 'eigenvalues') -> dict:
        """
        Check if matrix is positive definite using different methods
        
        Args:
            A: Symmetric matrix to test
            method: 'eigenvalues', 'cholesky', or 'sylvester'
            
        Returns:
            Dictionary with definiteness classification and details
        """
        if not np.allclose(A, A.T):
            warnings.warn("Matrix is not symmetric, results may be unreliable")
        
        results = {
            'positive_definite': False,
            'positive_semidefinite': False, 
            'negative_definite': False,
            'negative_semidefinite': False,
            'indefinite': False,
            'method_used': method,
            'details': {}
        }
        
        if method == 'eigenvalues':
            eigenvals = np.linalg.eigvals(A)
            eigenvals_real = np.real(eigenvals)  # Should be real for symmetric matrices
            
            results['details']['eigenvalues'] = eigenvals_real
            
            min_eig = np.min(eigenvals_real)
            max_eig = np.max(eigenvals_real)
            
            if min_eig > self.tolerance:
                results['positive_definite'] = True
            elif min_eig >= -self.tolerance:
                results['positive_semidefinite'] = True
            elif max_eig < -self.tolerance:
                results['negative_definite'] = True
            elif max_eig <= self.tolerance:
                results['negative_semidefinite'] = True
            else:
                results['indefinite'] = True
                
        elif method == 'cholesky':
            try:
                L = np.linalg.cholesky(A)
                results['positive_definite'] = True
                results['details']['cholesky_factor'] = L
            except np.linalg.LinAlgError:
                # Try to determine type using eigenvalues as fallback
                return self.check_positive_definiteness(A, 'eigenvalues')
                
        elif method == 'sylvester':
            # Sylvester's criterion: check leading principal minors
            n = A.shape[0]
            minors = []
            
            for k in range(1, n + 1):
                minor = np.linalg.det(A[:k, :k])
                minors.append(minor)
            
            results['details']['principal_minors'] = minors
            
            if all(minor > self.tolerance for minor in minors):
                results['positive_definite'] = True
            # Add more cases as needed
        
        return results
    
    def quadratic_form_analysis(self, A: np.ndarray, visualize: bool = True) -> dict:
        """
        Analyze quadratic form f(x) = x^T A x
        
        Args:
            A: Symmetric matrix defining quadratic form
            visualize: Whether to create visualization
            
        Returns:
            Analysis results including convexity properties
        """
        # Check definiteness
        definiteness = self.check_positive_definiteness(A)
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(A)
        
        # Condition number
        if np.min(eigenvals) > self.tolerance:
            condition_number = np.max(eigenvals) / np.min(eigenvals)
        else:
            condition_number = np.inf
        
        analysis = {
            'matrix': A,
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'condition_number': condition_number,
            'definiteness': definiteness,
            'convex': definiteness['positive_semidefinite'] or definiteness['positive_definite'],
            'strictly_convex': definiteness['positive_definite']
        }
        
        if visualize and A.shape[0] == 2:
            self._plot_quadratic_form_2d(analysis)
        
        return analysis
    
    def _plot_quadratic_form_2d(self, analysis: dict):
        """Visualize 2D quadratic form"""
        
        A = analysis['matrix']
        eigenvals = analysis['eigenvalues']
        eigenvecs = analysis['eigenvectors']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Create grid for contour plot
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate quadratic form
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j]])
                Z[i, j] = point.T @ A @ point
        
        # Contour plot
        contours = ax1.contour(X, Y, Z, levels=20)
        ax1.clabel(contours, inline=True, fontsize=8)
        
        # Plot eigenvectors
        for i in range(2):
            vec = eigenvecs[:, i] * np.sqrt(abs(eigenvals[i])) * 2
            ax1.arrow(0, 0, vec[0], vec[1], head_width=0.1, head_length=0.1,
                     fc=f'C{i}', ec=f'C{i}', linewidth=2,
                     label=f'λ{i+1}={eigenvals[i]:.3f}')
        
        ax1.set_title('Quadratic Form Contours')
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 3D surface
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis')
        ax2.set_title('Quadratic Form Surface')
        ax2.set_xlabel('x₁')
        ax2.set_ylabel('x₂')
        ax2.set_zlabel('f(x)')
        
        # Eigenvalue plot
        ax3.bar(range(1, len(eigenvals) + 1), eigenvals, alpha=0.7)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('Eigenvalues')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Eigenvalue')
        ax3.grid(True, alpha=0.3)
        
        # Classification
        definiteness = analysis['definiteness']
        classification = 'Indefinite'
        if definiteness['positive_definite']:
            classification = 'Positive Definite'
        elif definiteness['positive_semidefinite']:
            classification = 'Positive Semidefinite'
        elif definiteness['negative_definite']:
            classification = 'Negative Definite'
        elif definiteness['negative_semidefinite']:
            classification = 'Negative Semidefinite'
        
        ax4.text(0.5, 0.7, f'Classification: {classification}', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                transform=ax4.transAxes)
        ax4.text(0.5, 0.5, f'Condition Number: {analysis["condition_number"]:.2e}',
                ha='center', va='center', fontsize=12,
                transform=ax4.transAxes)
        ax4.text(0.5, 0.3, f'Convex: {analysis["convex"]}',
                ha='center', va='center', fontsize=12,
                transform=ax4.transAxes)
        ax4.text(0.5, 0.1, f'Strictly Convex: {analysis["strictly_convex"]}',
                ha='center', va='center', fontsize=12,
                transform=ax4.transAxes)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def solve_linear_system(self, A: np.ndarray, b: np.ndarray, 
                           method: str = 'auto') -> dict:
        """
        Solve linear system Ax = b using appropriate method
        
        Args:
            A: Coefficient matrix
            b: Right-hand side vector
            method: 'auto', 'lu', 'cholesky', 'qr', 'svd'
            
        Returns:
            Solution and method information
        """
        results = {
            'solution': None,
            'method_used': method,
            'condition_number': np.linalg.cond(A),
            'well_conditioned': False,
            'residual_norm': None
        }
        
        # Check condition number
        results['well_conditioned'] = results['condition_number'] < 1e12
        
        if method == 'auto':
            # Choose method based on matrix properties
            if np.allclose(A, A.T) and self.check_positive_definiteness(A)['positive_definite']:
                method = 'cholesky'
            elif A.shape[0] == A.shape[1]:
                method = 'lu'
            else:
                method = 'qr'
            results['method_used'] = method
        
        try:
            if method == 'cholesky':
                L = np.linalg.cholesky(A)
                y = np.linalg.solve(L, b)
                x = np.linalg.solve(L.T, y)
            elif method == 'lu':
                x = np.linalg.solve(A, b)
            elif method == 'qr':
                Q, R = np.linalg.qr(A)
                x = np.linalg.solve(R, Q.T @ b)
            elif method == 'svd':
                U, s, Vt = np.linalg.svd(A)
                x = Vt.T @ np.diag(1/s) @ U.T @ b
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results['solution'] = x
            results['residual_norm'] = np.linalg.norm(A @ x - b)
            
        except np.linalg.LinAlgError as e:
            results['error'] = str(e)
            
        return results

def demonstrate_cauchy_schwarz():
    """Demonstrate Cauchy-Schwarz inequality with examples"""
    
    print("=== Cauchy-Schwarz Inequality Demonstration ===\n")
    
    # Generate random vectors
    np.random.seed(42)
    examples = [
        np.array([1, 2, 3]),
        np.array([4, 5, 6]),
        np.random.randn(5),
        np.random.randn(5),
        np.array([1, 0, 0]),
        np.array([0, 1, 0])
    ]
    
    for i in range(0, len(examples), 2):
        u, v = examples[i], examples[i+1]
        
        # Calculate terms
        inner_product = np.abs(np.dot(u, v))
        norm_product = np.linalg.norm(u) * np.linalg.norm(v)
        
        print(f"Example {i//2 + 1}:")
        print(f"u = {u}")
        print(f"v = {v}")
        print(f"|u^T v| = {inner_product:.6f}")
        print(f"||u||₂ ||v||₂ = {norm_product:.6f}")
        print(f"Inequality satisfied: {inner_product <= norm_product + 1e-10}")
        print(f"Equality case: {np.allclose(inner_product, norm_product)}")
        print()

def main():
    """Demonstrate linear algebra concepts for optimization"""
    
    print("=== Linear Algebra for Optimization ===\n")
    
    la = LinearAlgebraOptimization()
    
    # 1. Vector norms demonstration
    print("1. Vector Norms Analysis")
    x = np.array([3, -4, 1, 2])
    norm_results = la.vector_norms(x)
    print(f"Vector: {x}")
    print(f"Norms: {norm_results['norms']}")
    print(f"Norm relationships verified: {all(norm_results['relationships'].values())}")
    print()
    
    # 2. Positive definiteness checking
    print("2. Positive Definiteness Analysis")
    
    # Positive definite matrix
    A_pd = np.array([[2, 1], [1, 3]])
    pd_results = la.check_positive_definiteness(A_pd)
    print(f"Matrix A = \n{A_pd}")
    print(f"Positive definite: {pd_results['positive_definite']}")
    print(f"Eigenvalues: {pd_results['details']['eigenvalues']}")
    print()
    
    # Indefinite matrix  
    A_indef = np.array([[1, 2], [2, 1]])
    indef_results = la.check_positive_definiteness(A_indef)
    print(f"Matrix B = \n{A_indef}")
    print(f"Indefinite: {indef_results['indefinite']}")
    print(f"Eigenvalues: {indef_results['details']['eigenvalues']}")
    print()
    
    # 3. Quadratic form analysis
    print("3. Quadratic Form Analysis")
    quad_analysis = la.quadratic_form_analysis(A_pd, visualize=True)
    print(f"Condition number: {quad_analysis['condition_number']:.2f}")
    print(f"Convex: {quad_analysis['convex']}")
    print(f"Strictly convex: {quad_analysis['strictly_convex']}")
    print()
    
    # 4. Linear system solving
    print("4. Linear System Solving")
    A_system = np.array([[4, 2], [2, 5]])
    b_system = np.array([6, 7])
    
    solution_results = la.solve_linear_system(A_system, b_system)
    print(f"System: Ax = b")
    print(f"A = \n{A_system}")
    print(f"b = {b_system}")
    print(f"Solution x = {solution_results['solution']}")
    print(f"Method used: {solution_results['method_used']}")
    print(f"Residual norm: {solution_results['residual_norm']:.2e}")
    print(f"Well-conditioned: {solution_results['well_conditioned']}")
    print()
    
    # 5. Cauchy-Schwarz demonstration
    demonstrate_cauchy_schwarz()
    
    print("=== Key Takeaways ===")
    print("• Vector norms provide distance measures and convergence criteria")
    print("• Positive definite matrices ensure convexity of quadratic functions")
    print("• Eigenvalues determine optimization landscape properties")
    print("• Condition numbers indicate numerical stability")
    print("• Efficient linear algebra is crucial for optimization algorithms")

if __name__ == "__main__":
    main()