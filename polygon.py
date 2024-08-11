import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
from math import cos, sin, pi

def create_regular_pentagon(center, radius):
    """ Generate coordinates for a regular pentagon given a center and radius. """
    return [(center[0] + radius * cos(2 * pi * k / 5), center[1] + radius * sin(2 * pi * k / 5)) for k in range(5)]

def fit_pentagon_to_points(points):
    """ Fit a regular pentagon to given points using least squares method. """
    center_estimate = np.mean(points, axis=0)
    radius_estimate = np.mean(np.sqrt(np.sum((points - center_estimate)**2, axis=1)))

    def error_function(x):
        center = x[:2]
        radius = x[2]
        pentagon = create_regular_pentagon(center, radius)
        distances = [min([np.linalg.norm(np.array(p) - np.array(point)) for p in pentagon]) for point in points]
        return sum(distances)

    result = minimize(error_function, x0=np.append(center_estimate, radius_estimate), method='L-BFGS-B')
    fitted_center = result.x[:2]
    fitted_radius = result.x[2]
    return create_regular_pentagon(fitted_center, fitted_radius)

def reg_pentagon(points):
    """Compute and visualize the regularization of points to a pentagon."""
    # Fit a regular pentagon to these points
    fitted_pentagon_points = fit_pentagon_to_points(points)
    fitted_pentagon_points.append(fitted_pentagon_points[0])  # Close the fitted pentagon

    # Plot the results
    polygon = Polygon(points)
    fitted_polygon = Polygon(fitted_pentagon_points)

    plt.figure()
    x, y = polygon.exterior.xy
    plt.plot(x, y, label='Original Polygon')

    x, y = fitted_polygon.exterior.xy
    plt.plot(x, y, label='Fitted Regular Pentagon', linestyle='--')
    plt.legend()
    plt.title('Polygon Regularization by Fitting')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()

    return np.array(fitted_pentagon_points)

# Example usage:
# points = np.array([[x1, y1], [x2, y2], ..., [xn, yn]])  # Define your points array
# regularized_points = regularize_to_pentagon(points)
