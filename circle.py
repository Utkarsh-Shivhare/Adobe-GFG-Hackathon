import numpy as np
import matplotlib.pyplot as plt

def calculate_centroid(points):
    """Calculate the centroid from a set of points."""
    return np.mean(points, axis=0)

def calculate_radii(points, centroid):
    """Calculate the radii from each point to the centroid."""
    return np.sqrt(np.sum((points - centroid) ** 2, axis=1))

def circle_score(radii):
    """Calculate how close the shape is to a perfect circle, returned as a percentage."""
    mean_radius = np.mean(radii)
    radius_variance = np.var(radii)
    return 100 * (1 - np.sqrt(radius_variance) / mean_radius)

def regularize_circle(points, centroid, desired_radius):
    """Adjust points to form a perfect circle with the given radius."""
    radii = calculate_radii(points, centroid)
    unit_vectors = (points - centroid) / np.expand_dims(radii, axis=1)
    return centroid + unit_vectors * desired_radius

def plot_circle(points, regularized_points):
    """Plot the original and regularized circle points."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(points[:, 0], points[:, 1], color='blue')
    ax[0].set_title('Original Points')
    ax[0].set_aspect('equal', adjustable='box')

    ax[1].scatter(regularized_points[:, 0], regularized_points[:, 1], color='red')
    ax[1].set_title('Regularized Circle')
    ax[1].set_aspect('equal', adjustable='box')

    plt.show()

# Sample Data: Points that approximately form a circle
# points = circle_point

def reg_circle(points):
    centroid = calculate_centroid(points)
    radii = calculate_radii(points, centroid)
    score = circle_score(radii)
    mean_radius = np.mean(radii)
    regularized_points = regularize_circle(points, centroid, mean_radius)
    return regularized_points

# print(f"Circle-ness Score: {score:.2f}%")
# plot_circle(points, regularized_points)
