import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

# ------------------------
# Loss function and gradient
# ------------------------
def loss(p):
    # Ideal prices: 5, 7, 3
    return (p[0] - 5)**2 + (p[1] - 7)**2 + (p[2] - 3)**2

def gradient(p):
    # Gradient of the loss function
    return np.array([2*(p[0] - 5), 2*(p[1] - 7), 2*(p[2] - 3)])

# ------------------------
# Gradient Descent Setup
# ------------------------
p = np.array([10.0, 2.0, 8.0])  # initial prices (far from optimum)
eta = 0.1                       # learning rate
tol = 1e-6
max_iter = 50

trajectory = [p.copy()]
loss_values = [loss(p)]

# ------------------------
# Iteration Loop
# ------------------------
for i in range(max_iter):
    grad = gradient(p)
    new_p = p - eta * grad
    if np.linalg.norm(new_p - p) < tol:
        break
    p = new_p
    trajectory.append(p.copy())
    loss_values.append(loss(p))

# ------------------------
# Display formatted table
# ------------------------
table = PrettyTable(["Iter", "p1", "p2", "p3", "Loss"])
for i, (pi, L) in enumerate(zip(trajectory, loss_values)):
    table.add_row([i, round(pi[0], 4), round(pi[1], 4), round(pi[2], 4), round(L, 6)])
print(table)

# ------------------------
# Plot convergence
# ------------------------
plt.figure(figsize=(8,5))
plt.plot(loss_values, 'o-', label='Loss Value')
plt.title("Gradient Descent Convergence on Product Prices")
plt.xlabel("Iteration")
plt.ylabel("Loss (Error)")
plt.grid(True)
plt.legend()
plt.show()

# Final result
print("\n✅ Final optimized prices:")
for i, val in enumerate(p, start=1):
    print(f"Product {i}: price ≈ {val:.4f}")
