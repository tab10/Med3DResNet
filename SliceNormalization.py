import numpy as np
from matplotlib.path import Path
from skimage.transform import ProjectiveTransform
import pydicom
import MathUtil
from matplotlib.path import Path
import cv2

def normalize_slice_affine(slice_pixel_array, crop_bounds, landmark_bounds, desired_width, desired_height):
    mask = np.zeros(slice_pixel_array.shape)
    crop_corners = np.array([[crop_bounds[0], crop_bounds[1], crop_bounds[2], crop_bounds[3]]], dtype=np.int32)
    cv2.fillConvexPoly(mask, crop_corners, 1)
    result = np.where(mask == 1, slice_pixel_array, slice_pixel_array.min())
    result = result[landmark_bounds[1][0]:landmark_bounds[1][1] + 1,
                   landmark_bounds[0][0]:landmark_bounds[0][1] + 1]
    result = cv2.resize(result, (desired_height, desired_width), cv2.INTER_LINEAR)
    return result

def normalize_slice_projection(slice_pixel_array, crop_bounds, desired_width, desired_height):
    result = np.zeros((desired_height, desired_width))
    #Crop bounds must be converted from (x, y) points to (y, x) points
    source_bounds = np.asarray([[0, 0], [0, desired_width], [desired_height, desired_width], [desired_height, 0]])
    destination_bounds = np.asarray([[crop_bounds[0][1], crop_bounds[0][0]], [crop_bounds[1][1], crop_bounds[1][0]],
                               [crop_bounds[2][1], crop_bounds[2][0]], [crop_bounds[3][1], crop_bounds[3][0]]])
    projective_transform = ProjectiveTransform()
    if not projective_transform.estimate(source_bounds, destination_bounds):
        print("Cannot project from crop bounds to desired image dimensions")
    else:
        for x in range(0, desired_width):
            for y in range(0, desired_height):
                normalized_point = [y, x]
                transform = projective_transform(normalized_point)
                slice_point = transform[0]
                value = MathUtil.sample_image_bilinear(slice_pixel_array, slice_point[1], slice_point[0])
                result[y][x] = value

    return result

def slice_to_hu_pixels(slice):
    image = slice.pixel_array
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = slice.RescaleIntercept
    slope = slice.RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

