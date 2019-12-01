import numpy as np
import math

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
    float_matrix = np.where(float_matrix < global_min, global_min, float_matrix)
    float_matrix = np.where(float_matrix > global_max, global_max, float_matrix)
    float_matrix = (float_matrix - float(global_min))/(float(global_max) - float(global_min))
    float_matrix = (float_matrix * (max_val - min_val)) + min_val
    res_matrix = float_matrix.astype(np.uint8)

    return res_matrix

def scale_bounds(bounds, scale_factor):
    center = center_of_bounds(bounds)
    results = [(0, 0)] * len(bounds)

    for i in range(0, len(bounds)):
        bound = bounds[i]
        direction = [float(bound[0]) - center[0], float(bound[1]) - center[1]]
        direction_scaled = [x * scale_factor for x in direction]
        results[i] = (int(center[0] + direction_scaled[0]), int(center[1] + direction_scaled[1]))

    return results

def center_of_bounds(bounds):
    x_values = [bound[0] for bound in bounds]
    y_values = [bound[1] for bound in bounds]

    x_mean = float(sum(x_values))/float(len(x_values))
    y_mean = float(sum(y_values))/float(len(y_values))

    return [x_mean, y_mean]

def second_min(array):
    min = array.min()
    temp_array = np.where(array == min, array.max(), array)
    return temp_array.min()
