import numpy as np
import sympy as sp
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = 'browser'  # Open in default web browser

# Define symbols
x, y = sp.symbols('x y')

# Define the function
f = -(x**2 + y**2) + 4

# Compute gradient and Hessian
grad_f = [sp.diff(f, var) for var in (x, y)]
H_f = sp.hessian(f, (x, y))

# Display symbolic results
print("f(x, y) =", f)
print("Gradient =", grad_f)
print("Hessian =")
sp.pprint(H_f)

# Critical points (solve grad = 0)
crit_points = sp.solve(grad_f, (x, y))
print("Critical points:", crit_points)

# Evaluate Hessian at critical points
if isinstance(crit_points, dict):
    # Single solution as dict
    x_val = crit_points.get(x, 0)
    y_val = crit_points.get(y, 0)
    H_val = H_f.subs({x: x_val, y: y_val})
    eigenvals = H_val.eigenvals()
    print(f"\nAt point {crit_points}:")
    sp.pprint(H_val)
    print("Eigenvalues:", list(eigenvals.keys()))
else:
    for point in crit_points:
        if isinstance(point, dict):
            x_val = point.get(x, 0)
            y_val = point.get(y, 0)
        else:
            x_val, y_val = point
        H_val = H_f.subs({x: x_val, y: y_val})
        eigenvals = H_val.eigenvals()
        print(f"\nAt point {point}:")
        sp.pprint(H_val)
        print("Eigenvalues:", list(eigenvals.keys()))

# Plot surface
X = np.linspace(-3, 3, 100)
Y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(X, Y)
Z = -(X**2 + Y**2) + 4

surface = go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.8)
critical_point = go.Scatter3d(x=[0], y=[0], z=[4], mode='markers',
                              marker=dict(size=6, color='red'), name='Critical Point (max)')
layout = go.Layout(title="Surface z = - (x² + y²) + 4",
                   scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'))
fig = go.Figure(data=[surface, critical_point], layout=layout)
fig.show()
