import SliceNormalization
import os
import numpy as np
import json
from scipy.ndimage import zoom
import cv2
import DataReader
import MathUtil

def export_annotations(dir_path, cross_sectional_image):
    file_name = os.path.join(dir_path, cross_sectional_image.patient_id + ".annotations")
    file = open(file_name, 'w')
    data = {}
    data['dir_path'] = cross_sectional_image.dir_path
    data['superior_slice'] = cross_sectional_image.superior_slice
    data['inferior_slice'] = cross_sectional_image.inferior_slice
    data['landmarks'] = cross_sectional_image.heart_landmarks.landmarks
    data['landmark_scale_factor'] = cross_sectional_image.landmark_scale_factor

    json.dump(data, file, indent=4)
    file.close()

def batch_export_annotations_to_3d_arrays(dir_path, desired_width, desired_height, desired_depth):
    dir_files = os.listdir(dir_path)

    for file_name in dir_files:
        file_ext = os.path.splitext(file_name)[1]
        file_path = os.path.join(dir_path, file_name)

        if file_ext == '.annotations':
            cross_sectional_image = DataReader.cross_sectional_image_from_annotation_file(file_path)

            export_normalized_slices_affine(dir_path, cross_sectional_image, desired_width, desired_height,
                                            desired_depth)
            export_normalized_slices_projected(dir_path, cross_sectional_image, desired_width, desired_height,
                                               desired_depth)

def export_normalized_slices_projected(dir_path, cross_sectional_image, desired_width, desired_height, desired_depth):
    print(f"exporting normalized 3d image (projection) for {cross_sectional_image.patient_id}")
    num_crop_slices = cross_sectional_image.inferior_slice + 1 - cross_sectional_image.superior_slice
    result = np.zeros((num_crop_slices, desired_height, desired_width), np.int16)

    print("normalizing slices")
    for slice_idx in range(cross_sectional_image.superior_slice, cross_sectional_image.inferior_slice + 1):
        slice = cross_sectional_image.get_slice(slice_idx)
        slice_pixels = SliceNormalization.slice_to_hu_pixels(slice)
        slice_bounds = cross_sectional_image.get_slice_bounds(slice_idx)
        normalized_slice = SliceNormalization.normalize_slice_projection(slice_pixels, slice_bounds, desired_width,
                                                                         desired_height)
        result[slice_idx - cross_sectional_image.superior_slice] = normalized_slice.astype(np.int16)
        print(f"{slice_idx - cross_sectional_image.superior_slice + 1}/{num_crop_slices}")
    print("slice normalization done")

    print("z resizing")
    z_resize = float(desired_depth)/float(num_crop_slices)
    result_z_resized = zoom(result, (z_resize, 1.0, 1.0))
    print("z resize done")

    file_name = os.path.join(dir_path, cross_sectional_image.patient_id + "_normalized_3d_projection.npy")
    np.save(file_name, result_z_resized)
    print("normalized 3d image exported")

def export_normalized_slices_affine(dir_path, cross_sectional_image, desired_width, desired_height, desired_depth):
    print(f"exporting normalized 3d image (affine) for {cross_sectional_image.patient_id}")
    num_crop_slices = cross_sectional_image.inferior_slice + 1 - cross_sectional_image.superior_slice
    slice_stack = np.zeros((num_crop_slices, cross_sectional_image.shape[0], cross_sectional_image.shape[1]),
                           np.int16)
    result = np.zeros((desired_depth, desired_height, desired_width), np.int16)
    landmark_bounds = cross_sectional_image.get_landmark_bounds()

    for slice_idx in range(cross_sectional_image.superior_slice, cross_sectional_image.inferior_slice + 1):
        slice = cross_sectional_image.get_slice(slice_idx)
        slice_pixels = SliceNormalization.slice_to_hu_pixels(slice)
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
        normalized_slice = SliceNormalization.normalize_slice_affine(slice_pixels, slice_bounds, landmark_bounds,
                                                                     desired_width, desired_height)
        result[depth] = normalized_slice.astype(np.int16)
        print(f"{depth + 1}/{desired_depth}")
    print("slice normalization done")

    file_name = os.path.join(dir_path, cross_sectional_image.patient_id + "_normalized_3d_affine.npy")
    np.save(file_name, result)
    print("normalized 3d image exported")
