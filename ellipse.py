import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig

def fit_ellipse_to_points(points):
    x, y = points[:, 0], points[:, 1]
    x_mean, y_mean = np.mean(x), np.mean(y)
    x_centered, y_centered = x - x_mean, y - y_mean
    covariance_matrix = np.cov(x_centered, y_centered)
    eigenvalues, eigenvectors = eig(covariance_matrix)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    a, b = np.sqrt(2 * eigenvalues)
    angle = np.arctan2(*eigenvectors[:, 0][::-1])
    return x_mean, y_mean, a, b, np.degrees(angle)

def generate_ellipse_points(x_mean, y_mean, a, b, angle, num_points=100):
    angle_rad = np.radians(angle)
    theta = np.linspace(0, 2 * np.pi, num_points)
    x_ellipse = x_mean + a * np.cos(theta) * np.cos(angle_rad) - b * np.sin(theta) * np.sin(angle_rad)
    y_ellipse = y_mean + a * np.cos(theta) * np.sin(angle_rad) + b * np.sin(theta) * np.cos(angle_rad)
    return np.column_stack((x_ellipse, y_ellipse))

def reg_ellipse(polylines):
    x_mean, y_mean, a, b, angle = fit_ellipse_to_points(polylines)
    return generate_ellipse_points(x_mean, y_mean, a, b, angle)

def visualize_ellipse(polylines, ellipse_points):
    plt.figure(figsize=(8, 6))
    plt.plot(polylines[:, 0], polylines[:, 1], 'o-', label='Hand-drawn Ellipse')
    plt.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'r--', label='Fitted Regular Ellipse')
    plt.title('Comparison of Hand-drawn and Fitted Ellipse')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.axis('equal')
    plt.legend()
    plt.show()

# Example usage:
# polylines = np.random.rand(100, 2)  # Replace this with your actual data
# regular_ellipse = fit_and_generate_ellipse(polylines)
# visualize_ellipse(polylines, regular_ellipse)
