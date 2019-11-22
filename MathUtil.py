import numpy as np

def point_interpolant_1d(val, min_val, max_val):
    return (float(val) - float(min_val)) / (float(max_val) - float(min_val))

def linear_interpolate_2d(start_point, end_point, interpolant):
    x1 = float(start_point[0])
    y1 = float(start_point[1])
    x2 = float(end_point[0])
    y2 = float(end_point[1])

    res_x = int(x1 + ((x2 - x1) * interpolant))
    res_y = int(y1 + ((y2 - y1) * interpolant))

    return (res_x, res_y)

def scale_matrix(matrix, min_val, max_val, global_min=None, global_max=None):
    if global_min is None:
        global_min = matrix.min()
    if global_max is None:
        global_max = matrix.max()
    float_matrix = matrix.astype(np.float64)
    float_matrix = (float_matrix - float(global_min))/(float(global_max) - float(global_min))
    float_matrix = (float_matrix * (max_val - min_val)) + min_val
    res_matrix = float_matrix.astype(np.uint8)

    return res_matrix

def inside_quadrilateral(point, bounds):
    pass
