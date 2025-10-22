# Gradient Methods for Unconstrained Optimization

This directory contains documentation and implementations for gradient-based optimization methods.

# Gradient Methods for Unconstrained Optimization

This directory contains documentation and implementations for gradient-based optimization methods.

## Directory Structure

### 📁 **[gradients/](gradients/)** - Core Implementations
Complete, production-ready implementations of gradient-based methods:
- **steepest_descent.py** - Fundamental gradient descent with line search
- **conjugate_gradient.py** - Conjugate gradient methods (FR, PR, HS, DY)
- **quasi_newton.py** - BFGS and L-BFGS quasi-Newton methods
- **adam_optimizer.py** - Adam and variants (AdaMax, AMSGrad, Nadam)
- **README.md** - Comprehensive guide with selection criteria

### 📁 **[examples/](examples/)** - Simple Examples
Illustrative examples for learning:
- **gradient_example_1.py** - 3D visualization on Rosenbrock function
- **gradient_example_2.py** - Simple 1D gradient descent

### 📁 **[theory/](theory/)** - Convergence Theory
Theoretical analysis tools:
- **convergence_analysis.py** - Comprehensive convergence analysis toolkit
  - Lipschitz constant estimation
  - Strong convexity analysis
  - Theoretical rate computation
  - Empirical validation

### 📁 **[line-search/](line-search/)** - Line Search Methods
Complete documentation on line search techniques:
- **README.md** - In-depth theory, algorithms, and examples
- **line_search_interactive.ipynb** - Interactive Jupyter notebook
- **line_search.py** - Implementation of various line search methods
- **QUICK_REFERENCE.md** - Cheat sheet
- **COMPLETE_TOOLKIT_GUIDE.md** - Visualization tools guide
- **VISUALIZATION_GUIDE.md** - Setup instructions

## Quick Start Guide

### For Learning Gradient Methods

**Start here:** Choose your path based on your goal:

#### 1. **Want to SEE how it works?**
→ Go to `examples/gradient_example_1.py`
- Beautiful 3D visualization
- Watch gradient descent in action
- Understand visually

#### 2. **Want to UNDERSTAND the theory?**
→ Go to `theory/convergence_analysis.py`
- Estimate convergence rates
- Verify theoretical properties
- Connect theory to practice

#### 3. **Want to USE in your project?**
→ Go to `gradients/` directory
- Production-ready implementations
- Well-tested and documented
- Multiple algorithms to choose from

#### 4. **Want to IMPLEMENT from scratch?**
→ Go to `line-search/README.md`
- Complete theoretical foundations
- Step-by-step algorithms
- Implementation guidance

---

### Quick Reference

**Choose an algorithm:**

```
Need simplicity? → Steepest Descent (gradients/steepest_descent.py)
Need speed on quadratics? → Conjugate Gradient (gradients/conjugate_gradient.py)
Need general fast method? → BFGS/L-BFGS (gradients/quasi_newton.py)
Need robustness? → Adam (gradients/adam_optimizer.py)
```

**Example usage:**

```python
# Import any method from gradients/
from gradients.steepest_descent import SteepestDescent

# Define your problem
def f(x):
    return x[0]**2 + 4*x[1]**2

def grad_f(x):
    return np.array([2*x[0], 8*x[1]])

# Optimize
optimizer = SteepestDescent(step_size=0.1, line_search='backtracking')
x_opt, history = optimizer.optimize(f, grad_f, x0=np.array([1.0, 1.0]))

print(f"Optimal point: {x_opt}")
print(f"Optimal value: {f(x_opt)}")
print(f"Iterations: {len(history)}")
```

---

## Topics Covered

### Core Gradient Methods (`gradients/`)
- **Steepest Descent** - Basic gradient descent with multiple line search strategies
- **Conjugate Gradient** - FR, PR, HS, DY variants for faster convergence
- **Quasi-Newton Methods** - BFGS and L-BFGS for superlinear convergence
- **Adaptive Methods** - Adam, AdaMax, AMSGrad, Nadam

### Line Search Theory (`line-search/`)
- Exact line search
- Inexact line search (Wolfe conditions, Goldstein conditions)
- Backtracking line search
- Convergence theory and guarantees

### Convergence Theory (`theory/`)
- Lipschitz constant estimation
- Strong convexity analysis
- Condition number and its impact
- Theoretical vs empirical convergence rates
- Descent lemma verification

### Visual Examples (`examples/`)
- 3D trajectory visualization
- Convergence behavior demonstration
- Interactive learning tools

## Prerequisites

```bash
# Activate your virtual environment
source ~/github/ewdhp/python/venv/bin/activate

# Install required packages
pip install numpy matplotlib scipy prettytable
```

### For Interactive Notebooks
```bash
pip install jupyter notebook ipywidgets
```

---

## Repository Structure Overview

```
gradient-methods/
├── README.md                    # This file
├── gradients/                   # Core implementations
│   ├── README.md               # Method selection guide
│   ├── steepest_descent.py
│   ├── conjugate_gradient.py
│   ├── quasi_newton.py
│   └── adam_optimizer.py
├── examples/                    # Learning examples
│   ├── README.md
│   ├── gradient_example_1.py   # 3D visualization
│   └── gradient_example_2.py   # Simple 1D example
├── theory/                      # Convergence analysis
│   ├── README.md
│   └── convergence_analysis.py
└── line-search/                 # Line search methods
    ├── README.md               # Complete theory
    ├── line_search.py
    ├── line_search_interactive.ipynb
    ├── QUICK_REFERENCE.md
    └── diagrams/
```

---

## Tools & Extensions

For the best learning experience, install these VS Code extensions:
- **Markdown Preview Enhanced** - Math equations and diagrams
- **Jupyter** - Interactive notebooks
- **Draw.io Integration** - Professional diagrams
- **Python** - Code highlighting and IntelliSense
- **Pylance** - Advanced Python language support

See [line-search/VISUALIZATION_GUIDE.md](line-search/VISUALIZATION_GUIDE.md) for setup instructions.

---

## Running Examples

Each directory has runnable examples:

```bash
# Activate virtual environment
source ~/github/ewdhp/python/venv/bin/activate

# Run simple examples
python examples/gradient_example_1.py
python examples/gradient_example_2.py

# Run convergence analysis
python theory/convergence_analysis.py

# Run comprehensive method examples
python gradients/steepest_descent.py
python gradients/conjugate_gradient.py
python gradients/quasi_newton.py
python gradients/adam_optimizer.py

# Interactive notebook
jupyter notebook line-search/line_search_interactive.ipynb
```

---

## Contributing

This is part of the EWDHP optimization repository. Feel free to:
- Report issues
- Suggest improvements
- Add new examples
- Enhance visualizations

## Related Topics

- [Momentum Methods](../momentum-methods/) - Coming soon
- [Newton Methods](../newton-methods/) - Coming soon
- [Optimality Conditions](../optimality-conditions/) - Coming soon

---

**Repository:** [ewdhp/optimization](https://github.com/ewdhp/optimization)  
**Path:** `2-core-methods/unconstrained/gradient-methods/`
