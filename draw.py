import matplotlib.pyplot as plt
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Draw G1 (two unconnected triangles)
# First triangle
triangle1 = np.array([[0, 1], [-1, 0], [1, 0], [0, 1]])
ax.plot(triangle1[:, 0], triangle1[:, 1], 'b-', label='G1: Triangle 1')
ax.text(0, 1.1, 'v1', ha='center', color='orange', fontsize=10)
ax.text(-1.1, -0.1, '1', ha='center', fontsize=10)
ax.text(1.1, -0.1, '2', ha='center', fontsize=10)

# Second triangle
triangle2 = np.array([[3, 1], [2, 0], [4, 0], [3, 1]])
ax.plot(triangle2[:, 0], triangle2[:, 1], 'g-', label='G1: Triangle 2')
ax.text(3, 1.1, 'v2', ha='center', color='orange', fontsize=10)
ax.text(2, -0.3, '3', ha='center', fontsize=10)
ax.text(4, -0.3, '4', ha='center', fontsize=10)

# Draw G2 (hexagon)
hexagon = np.array([[6 + np.cos(i * np.pi / 3), 4 + np.sin(i * np.pi / 3)] for i in range(7)])
ax.plot(hexagon[:, 0], hexagon[:, 1], 'r-', label='G2: Hexagon')
for i in range(6):
    ax.text(hexagon[i, 0], hexagon[i, 1] + 0.2, str(i + 1), ha='center', fontsize=10)

# Formatting
ax.set_xlim(-2, 7)
ax.set_ylim(-1, 6)
ax.set_aspect('equal', adjustable='datalim')
ax.legend()
ax.axis('off')

# Save as vector file
plt.savefig("graph_example.svg", format="svg")
plt.show()
