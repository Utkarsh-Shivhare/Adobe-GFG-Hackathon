import numpy as np

# Example input: a numpy array of points
points = np.array([
    [1, 2],
    [2, 2.1],
    [3, 1.9],
    [4, 2.05],
    [5, 2.1]
])
def reg_line(points):
    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Perform linear regression (fit a line)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    # Calculate line points at the minimum and maximum x values
    x_min, x_max = x.min(), x.max()
    y_min, y_max = m * x_min + c, m * x_max + c

    # Create a new array of points representing the polyline (line endpoints)
    line_polyline = np.array([
        [x_min, y_min],
        [x_max, y_max]
    ])
    return line_polyline

# Output the polyline numpy array
# print(line_polyline)

