import numpy as np
import matplotlib.pyplot as plt

# ReLU function
def relu(x):
    return np.maximum(0, x)

# GeLU function
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2) / 2 * (x + 0.044715 * x ** 3)))

# Create data for the x-axis
x = np.linspace(-3, 3, 1000)

# Calculate y-values for ReLU and GeLU functions
relu_y = relu(x)
gelu_y = gelu(x)

# Plot the ReLU function in blue
plt.plot(x, relu_y, color='blue', label='ReLU')

# Plot the GeLU function in orange
plt.plot(x, gelu_y, color='orange', label='GeLU')

# Highlight the superiority of GeLU
# plt.text(0.5, 0.2, "GeLU > ReLU", fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Add small ticks and labels for y-axis
yticks = np.arange(-1, 4, 0.5)
plt.gca().set_yticks(yticks)

plt.gca().set_yticklabels(yticks + 0.5)

# Remove grid
plt.grid(False)

# Add legend
plt.legend()

# Set axis labels and title
# plt.xlabel('x')
# plt.ylabel('y')
plt.title('ReLU and GeLU Activation Functions', fontsize=12)

# Show the plot
plt.show()
