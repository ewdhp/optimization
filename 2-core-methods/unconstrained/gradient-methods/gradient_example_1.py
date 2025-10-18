import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from prettytable import PrettyTable

# ------------------------
# Parameters
# ------------------------
eta = 0.1          # learning rate
max_iter = 30      # max iterations
tol = 1e-6         # tolerance for convergence

# Initial point (far from minimum)
x, y = 3.5, -2.5

# Function and gradient
def f(x, y):
    return x**2 + y**2

def grad(x, y):
    return np.array([2*x, 2*y])

# ------------------------
# Gradient Descent
# ------------------------
trajectory = [(x, y, f(x, y))]

for i in range(max_iter):
    g = grad(x, y)
    x_new, y_new = x - eta*g[0], y - eta*g[1]
    if np.linalg.norm([x_new - x, y_new - y]) < tol:
        break
    x, y = x_new, y_new
    trajectory.append((x, y, f(x, y)))

# ------------------------
# Print formatted table
# ------------------------
table = PrettyTable(["Iter", "x", "y", "f(x,y)"])
for i, (xi, yi, fi) in enumerate(trajectory):
    table.add_row([i, round(xi, 5), round(yi, 5), round(fi, 6)])
print(table)

# ------------------------
# 3D Animation
# ------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Create mesh grid for surface
X = np.linspace(-4, 4, 100)
Y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(X, Y)
Z = f(X, Y)

# Plot the surface
ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

# Trajectory line
traj = np.array(trajectory)
point, = ax.plot([], [], [], 'ro', markersize=8)
line, = ax.plot([], [], [], 'r-', linewidth=2)

def init():
    point.set_data(np.array([]), np.array([]))
    point.set_3d_properties(np.array([]))
    line.set_data(np.array([]), np.array([]))
    line.set_3d_properties(np.array([]))
    return point, line

def update(frame):
    data = traj[:frame+1]
    line.set_data(data[:,0], data[:,1])
    line.set_3d_properties(data[:,2])
    point.set_data(data[frame,0], data[frame,1])
    point.set_3d_properties(data[frame,2])
    return point, line

ani = FuncAnimation(fig, update, frames=len(traj), init_func=init, interval=300, blit=True)
plt.show()
