import numpy as np
from skimage.transform import ProjectiveTransform
import MathUtil
import cv2
from scipy.ndimage import zoom
import visualization

def normalize_cross_sectional_image_affine(cross_sectional_image, desired_width, desired_height,
                                           desired_depth):
    print(f"exporting normalized 3d image (affine) for {cross_sectional_image.patient_id}")
    num_crop_slices = cross_sectional_image.inferior_slice + 1 - cross_sectional_image.superior_slice
    slice_stack = np.zeros((num_crop_slices, cross_sectional_image.shape[0], cross_sectional_image.shape[1]),
                           np.int16)
    result = np.zeros((desired_depth, desired_height, desired_width), np.int16)
    landmark_bounds = cross_sectional_image.get_landmark_bounds()

    for slice_idx in range(cross_sectional_image.superior_slice, cross_sectional_image.inferior_slice + 1):
        slice = cross_sectional_image.get_slice(slice_idx)
        slice_pixels = slice_to_hu_pixels(slice)
        slice_stack[slice_idx - cross_sectional_image.superior_slice] = slice_pixels

    print("z resizing")
    z_resize = float(desired_depth) / float(num_crop_slices)
    slice_stack_resized = zoom(slice_stack, (z_resize, 1.0, 1.0))
    print("z resize done")

    print("normalizing slices")
    for depth in range(0, desired_depth):
        interpolant = MathUtil.point_interpolant_1d(depth, 0, desired_depth - 1)
        slice_pixels = slice_stack_resized[depth]
        slice_bounds = cross_sectional_image.get_slice_bounds_from_interpolant(interpolant)
        normalized_slice = normalize_slice_affine(slice_pixels, slice_bounds, landmark_bounds, desired_width,
                                                  desired_height)
        result[depth] = normalized_slice.astype(np.int16)
        print(f"{depth + 1}/{desired_depth}")
    print("slice normalization done")
    print("normalization done")
    return result

def normalize_cross_sectional_image_projected(cross_sectional_image, desired_width, desired_height,
                                              desired_depth):
    print(f"normalizing 3d image (projection) for {cross_sectional_image.patient_id}")
    num_crop_slices = cross_sectional_image.inferior_slice + 1 - cross_sectional_image.superior_slice
    result = np.zeros((num_crop_slices, desired_height, desired_width), np.int16)

    print("normalizing slices")
    for slice_idx in range(cross_sectional_image.superior_slice, cross_sectional_image.inferior_slice + 1):
        slice = cross_sectional_image.get_slice(slice_idx)
        slice_pixels = slice_to_hu_pixels(slice)
        slice_bounds = cross_sectional_image.get_slice_bounds(slice_idx)
        normalized_slice = normalize_slice_projection(slice_pixels, slice_bounds, desired_width, desired_height)
        result[slice_idx - cross_sectional_image.superior_slice] = normalized_slice.astype(np.int16)
        print(f"{slice_idx - cross_sectional_image.superior_slice + 1}/{num_crop_slices}")
    print("slice normalization done")

    print("z resizing")
    z_resize = float(desired_depth) / float(num_crop_slices)
    result = zoom(result, (z_resize, 1.0, 1.0))
    print("z resize done")
    print("normalization done")
    return result

def normalize_slice_affine(slice_pixel_array, crop_bounds, landmark_bounds, desired_width, desired_height):
    mask = np.zeros(slice_pixel_array.shape)
    crop_corners = np.array([[crop_bounds[0], crop_bounds[1], crop_bounds[2], crop_bounds[3]]], dtype=np.int32)
    cv2.fillConvexPoly(mask, crop_corners, 1)
    result = np.where(mask == 1, slice_pixel_array, slice_pixel_array.min())
    result = result[landmark_bounds[1][0]:landmark_bounds[1][1] + 1, landmark_bounds[0][0]:landmark_bounds[0][1] + 1]
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

def mask_3d_array(array):
    print("masking array")
    '''for i in range(len(test_labels)):
        test_image_temp = np.load(
            "%s/%s_images/%s_normalized_3d_%s.npy" % (path, flag, test_labels[i][0], flag))
        masked_test_image_temp = []
        if lungmask:
            for j in range(len(test_image_temp)):
                slice = np.squeeze(test_image_temp)[:][:][j]
                slice_mask = make_lungmask(slice)
                slice_masked = apply_lungmask(slice, slice_mask)
                masked_test_image_temp.append(slice_masked)
            masked_test_image_temp += offset
            MIN_HU, MAX_HU = _update_hu_range_(masked_test_image_temp, MIN_HU, MAX_HU)
            test_images.append(masked_test_image_temp)'''
    result = []
    #min_hu = max_hu = 0.0

    for i in range(len(array)):
        print(f"masking slice {i + 1}/{len(array)}")
        slice = np.squeeze(array)[:][:][i]
        slice_mask = visualization.make_lungmask(slice)
        masked_slice = visualization.apply_lungmask(slice, slice_mask)
        result.append(masked_slice)
        #min_hu, max_hu = visualization.update_hu_range(result, min_hu, max_hu)

    result = np.asarray(result)
    print("masking done")
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

