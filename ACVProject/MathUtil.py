'''
MathUtil
Author: Luben Popov
This library provides useful custom math functions used throughout the annotation tool code.
'''

import numpy as np

# Gets the interpolant for a value between two other values (from 0.0 - 1.0 if the value is in between the min and max)
# val: The value in between the min and max values
# min_val: The minimum value
# max_val: The maximum value
# output: A floating point interpolant value
def point_interpolant_1d(val, min_val, max_val):
    return (float(val) - float(min_val)) / (float(max_val) - float(min_val))

# Interpolates between a minimum and maximum value by the given interpolant (0.0 = min value, 1.0 = max value)
# val1: The minimum value
# val2: The maximum value
# interpolant: The interpolant used to get the final result
# output: The interpolated floating point value
def linear_interpolate_1d(val1, val2, interpolant):
    float_val1 = float(val1)
    float_val2 = float(val2)

    return int(float_val1 + ((float_val2 - float_val1) * interpolant))

# Performs linear interpolation on two 2D points in (x, y) form by performing 1D linear interpolation on the x and y
# coordinates respectively
# start_point: The starting point
# end_point: The end point
# output: A 2D coordinate in (x, y) form representing the interpolated x and y values
def linear_interpolate_2d(start_point, end_point, interpolant):
    res_x = linear_interpolate_1d(start_point[0], end_point[0], interpolant)
    res_y = linear_interpolate_1d(start_point[1], end_point[1], interpolant)

    return (res_x, res_y)

# Takes a bilinear sample from a given 2D NumPy image array using the given x and y interpolant values
# image: The 2D NumPy array representing the image data
# x: The floating point x interpolant used to sample the image (should be from 0.0 to 1.0)
# y: The floating point y interpolant used to sample the image (should be from 0.0 to 1.0)
# output: The floating point pixel value sampled from the image
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

# Multiplies all elements in a 2D NumPy matrix by a calculated scalar such that its minimum value equals the given
# minimum value and its maximum value equals the given maximum value
# matrix: The 2D NumPy matrix to perform scaling on
# min_val: This will be the minimum value of the matrix after scaling
# max_val: This will be the maximum value of the matrix after scaling
# global_min: If this argument is not None, any values less than this value in the input matrix will be set to this
#             value
# global_max: If this argument is not None, any values greater than this value in the input matrix will be set to this
#             value
# output: A 2D NumPy matrix containing the scaled matrix data
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

# Scales four boundary points from the center of the points by the given scale factor
# bounds: A list of four boundary coordinates
# scale_factor: The scale factor used for scaling (1.0 will not change the input value)
# output: A list of the four scaled boundary coordinates in integer form
def scale_bounds(bounds, scale_factor):
    center = center_of_bounds(bounds)
    results = [(0, 0)] * len(bounds)

    for i in range(0, len(bounds)):
        bound = bounds[i]
        direction = [float(bound[0]) - center[0], float(bound[1]) - center[1]]
        direction_scaled = [x * scale_factor for x in direction]
        results[i] = (int(center[0] + direction_scaled[0]), int(center[1] + direction_scaled[1]))

    return results

# Gets the center of four boundary coordinates
# bounds: A list of four boundary coordinates
# output: A floating point coordinate representing the center point of the four input boundary coordinates
def center_of_bounds(bounds):
    x_values = [bound[0] for bound in bounds]
    y_values = [bound[1] for bound in bounds]

    x_mean = float(sum(x_values))/float(len(x_values))
    y_mean = float(sum(y_values))/float(len(y_values))

    return [x_mean, y_mean]

# Gets the second lowest value in a NumPy array
# array: A NumPy array of any dimensionality
# output: The second lowest value in the input array
def second_min(array):
    min = array.min()
    temp_array = np.where(array == min, array.max(), array)
    return temp_array.min()
