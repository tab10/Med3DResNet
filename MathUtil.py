import numpy as np

def point_interpolant_1d(val, min_val, max_val):
    return (float(val) - float(min_val)) / (float(max_val) - float(min_val))

def linear_interpolate_1d(val1, val2, interpolant):
    float_val1 = float(val1)
    float_val2 = float(val2)

    return int(float_val1 + ((float_val2 - float_val1) * interpolant))

def linear_interpolate_2d(start_point, end_point, interpolant):
    res_x = linear_interpolate_1d(start_point[0], end_point[0], interpolant)
    res_y = linear_interpolate_1d(start_point[1], end_point[1], interpolant)

    return (res_x, res_y)

def sample_image_bilinear(image, x, y):
    if int(x) >= image.shape[1] or int(y) >= image.shape[0]:
        return 0
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + 1 if x1 <= image.shape[1] - 1 else x1
    y2 = y1 + 1 if y1 <= image.shape[0] - 1 else y1

    val_x1y1 = image[y1, x1]
    val_x2y1 = image[y1, x2]
    val_x2y2 = image[y2, x2]
    val_x1y2 = image[y2, x1]

    x_interpolant = point_interpolant_1d(x, x1, x2) if x1 != x2 else 0
    y_interpolant = point_interpolant_1d(y, y1, y2) if y1 != y2 else 0

    val_x1 = linear_interpolate_1d(val_x1y1, val_x1y2, y_interpolant)
    val_x2 = linear_interpolate_1d(val_x2y1, val_x2y2, y_interpolant)

    result = linear_interpolate_1d(val_x1, val_x2, x_interpolant)
    return result

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