import matplotlib.pyplot as plt
import numpy as np
import math

# Function to calculate endpoint coordinates given angle in degrees
def calculate_endpoint_coordinates(length, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    x = length * np.cos(angle_radians)
    y = length * np.sin(angle_radians)
    return x, y

# Coordinates for lines at 45 and 75 degrees
x1, y1 = calculate_endpoint_coordinates(1, 45)
x2, y2 = calculate_endpoint_coordinates(1, 75)

# Plotting
plt.figure(figsize=(6, 6))
plt.plot([0, x1], [0, y1], label='45 degrees')
plt.plot([0, x2], [0, y2], label='75 degrees')

# Marking the points
plt.scatter([x1, x2], [y1, y2], color='red')

# Adding labels and legend
plt.title('Lines at 45 and 75 degrees')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()

# Display the plot
plt.show()


def cosine_similarity(x, y):
    assert len(x) == len(y), "Dimension mismatch: Vectors must have the same length"

    dot_products = sum(xi * yi for xi, yi in zip(x, y))
    x_magnitude = math.sqrt(sum(xi ** 2 for xi in x))
    y_magnitude = math.sqrt(sum(yi ** 2 for yi in y))

    cosine_similarity = dot_products / (x_magnitude * y_magnitude) if x_magnitude * y_magnitude != 0 else 0

    return cosine_similarity

# calculate cosine similarity on [1, 1, 1, 0, 1, 0] and [1, 1, 1, 1, 0, 1]
x = [1, 1, 1, 0, 1, 0]
y = [1, 1, 1, 1, 0, 1]
cs = cosine_similarity(x, y)  # 0.75
print(cs)
