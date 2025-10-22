# Gradient Descent Examples

This directory contains simple, illustrative examples of gradient descent in action.

## Files

### gradient_example_1.py
**3D Visualization of Gradient Descent on Rosenbrock Function**

A beautiful animated 3D visualization showing:
- Rosenbrock function surface plot
- Gradient descent trajectory in 3D space
- Step-by-step convergence animation

**Features:**
- Uses `matplotlib` for 3D visualization
- Animated trajectory with `FuncAnimation`
- Shows the challenging "banana valley" of Rosenbrock function
- Demonstrates slow convergence in narrow valleys

**Run it:**
```bash
python gradient_example_1.py
```

**Expected Output:**
- 3D surface plot with contours
- Animated point moving along optimization trajectory
- Visual demonstration of gradient descent behavior

---

### gradient_example_2.py
**Simple 1D Gradient Descent Example**

A minimal example demonstrating:
- Basic gradient descent on a simple quadratic function
- Step-by-step iteration printout
- Convergence to minimum

**Features:**
- Easy to understand code
- Clear printout of each iteration
- Good starting point for learning

**Run it:**
```bash
python gradient_example_2.py
```

**Expected Output:**
```
Iteration 1: p = ..., loss = ...
Iteration 2: p = ..., loss = ...
...
Final result: p = ...
```

---

## Purpose

These examples are designed to:
1. **Visualize** how gradient descent works
2. **Demonstrate** convergence behavior
3. **Provide** simple starting points for learning
4. **Show** practical implementation patterns

## Prerequisites

```bash
pip install numpy matplotlib prettytable
```

## Related

- For **comprehensive implementations**, see `../gradients/`
- For **convergence theory**, see `../theory/`
- For **line search methods**, see `../line-search/`

---

**Note:** These are simplified examples for learning. For production use, see the implementations in `../gradients/`.
