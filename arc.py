import numpy as np
from scipy.optimize import minimize

def fit_circle_to_arc(points):
    # Define the cost function for circle fitting
    def cost(params):
        x0, y0, r = params
        xi, yi = points[:, 0], points[:, 1]
        return np.sum((np.sqrt((xi - x0)**2 + (yi - y0)**2) - r)**2)
    
    # Initial guess for the center (x0, y0) and radius r
    x0, y0 = np.mean(points[:, 0]), np.mean(points[:, 1])
    r_initial = np.mean(np.sqrt((points[:, 0] - x0)**2 + (points[:, 1] - y0)**2))
    
    # Minimize the cost function
    result = minimize(cost, [x0, y0, r_initial])
    if not result.success:
        raise Exception("Circle fitting optimization failed: " + result.message)
    
    x0_opt, y0_opt, r_opt = result.x
    return x0_opt, y0_opt, r_opt

def generate_circle_polyline(x0, y0, r, num_points=360):
    # Generate points on the circle
    angles = np.linspace(0, 2 * np.pi, num_points)
    x_circle = x0 + r * np.cos(angles)
    y_circle = y0 + r * np.sin(angles)
    return np.column_stack((x_circle, y_circle))

# Example usage:
# Assume 'points' is your actual data for the arc
points = np.array([[np.cos(theta) * 5 + 10, np.sin(theta) * 5 + 10] for theta in np.linspace(-np.pi/4, np.pi/4, 100)])
x0, y0, r = fit_circle_to_arc(points)
circle_polyline = generate_circle_polyline(x0, y0, r)
print(circle_polyline)
