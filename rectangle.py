import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def reg_rectangle(polylines):
    # Fit these points to a polygon using a convex hull
    hull = ConvexHull(polylines)
    hull_points = polylines[hull.vertices]

    # Find the minimum bounding rectangle, which is aligned with the axes
    min_x = np.min(hull_points[:, 0])
    max_x = np.max(hull_points[:, 0])
    min_y = np.min(hull_points[:, 1])
    max_y = np.max(hull_points[:, 1])

    # Coordinates of the regularized rectangle
    rectangle = np.array([
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y],
        [min_x, min_y]  # Closing the rectangle
    ])

    # Rectangularity measure
    area_hull = hull.volume
    area_rectangle = (max_x - min_x) * (max_y - min_y)
    rectangularity = area_rectangle / area_hull

    # # Visualization
    # plt.figure(figsize=(12, 6))

    # # Plotting the original polygon
    # plt.subplot(1, 2, 1)
    # plt.plot(polylines[:, 0], polylines[:, 1], 'o-', label='Original')
    # plt.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.3, color='lightgrey')
    # plt.title('Original Hand-drawn Shape')
    # plt.legend()

    # # Plotting the regularized rectangle
    # plt.subplot(1, 2, 2)
    # plt.plot(rectangle[:, 0], rectangle[:, 1], 's-', label='Regularized Rectangle', color='red')
    # plt.title('Regularized Rectangle')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    return rectangle

# Example usage:
# polylines = np.array([[1, 1], [5, 1], [5, 4], [1, 4], [1, 1]])  # Example of a rectangle
# rectangle, rectangularity = regularize_rectangle(polylines)
# print(f"Rectangularity (0 to 1, higher is more rectangular): {rectangularity:.2f}")
