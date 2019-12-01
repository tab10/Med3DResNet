from DICOMCrossSectionalImage import DICOMCrossSectionalImage
import SliceNormalization
import os
import numpy as np
import json
from scipy.ndimage import zoom
import cv2
import MathUtil

def export_annotations(dir_path, cross_sectional_image):
    file_name = os.path.join(dir_path, cross_sectional_image.patient_id + "_annotations.txt")
    file = open(file_name, 'w')
    data = {}
    data['slices'] = []

    for slice_idx in range(cross_sectional_image.superior_slice, cross_sectional_image.inferior_slice + 1):
        slice_bounds = cross_sectional_image.get_slice_bounds(slice_idx)
        file_name = os.path.basename(cross_sectional_image.get_slice(slice_idx).filename)
        pixel_data = cross_sectional_image.get_slice(slice_idx).pixel_array

        data['slices'].append({
            'slice_index': slice_idx,
            'file_name': file_name,
            'bounds': slice_bounds
        })

    json.dump(data, file, indent=4)
    file.close()

def export_normalized_slices_projected(dir_path, cross_sectional_image, desired_width, desired_height, desired_depth):
    print("exporting normalized 3d image (projection)")
    num_crop_slices = cross_sectional_image.inferior_slice + 1 - cross_sectional_image.superior_slice
    result = np.zeros((num_crop_slices, desired_height, desired_width), np.int16)

    for slice_idx in range(cross_sectional_image.superior_slice, cross_sectional_image.inferior_slice + 1):
        slice = cross_sectional_image.get_slice(slice_idx)
        slice_pixels = SliceNormalization.slice_to_hu_pixels(slice)
        slice_bounds = cross_sectional_image.get_slice_bounds(slice_idx)
        normalized_slice = SliceNormalization.normalize_slice_projection(slice_pixels, slice_bounds, desired_width,
                                                                         desired_height)
        result[slice_idx - cross_sectional_image.superior_slice] = normalized_slice.astype(np.int16)
        print(f"slice {slice_idx} normalization done")

    print("z resizing")
    z_resize = float(desired_depth)/float(num_crop_slices)
    result_z_resized = zoom(result, (z_resize, 1.0, 1.0))
    print("z resize done")

    file_name = os.path.join(dir_path, cross_sectional_image.patient_id + "_normalized_3d_projection.npy")
    np.save(file_name, result_z_resized)

def export_normalized_slices_affine(dir_path, cross_sectional_image, desired_width, desired_height, desired_depth):
    print("exporting normalized 3d image (affine)")
    num_crop_slices = cross_sectional_image.inferior_slice + 1 - cross_sectional_image.superior_slice
    result = np.zeros((num_crop_slices, desired_height, desired_width), np.int16)
    landmark_bounds = cross_sectional_image.heart_landmarks.get_bounds()

    for slice_idx in range(cross_sectional_image.superior_slice, cross_sectional_image.inferior_slice + 1):
        slice = cross_sectional_image.get_slice(slice_idx)
        slice_pixels = SliceNormalization.slice_to_hu_pixels(slice)
        slice_bounds = cross_sectional_image.get_slice_bounds(slice_idx)
        normalized_slice = SliceNormalization.normalize_slice_affine(slice_pixels, slice_bounds, landmark_bounds,
                                                                     desired_width, desired_height)
        result[slice_idx - cross_sectional_image.superior_slice] = normalized_slice.astype(np.int16)
        print(f"slice {slice_idx} normalization done")

    print("z resizing")
    z_resize = float(desired_depth) / float(num_crop_slices)
    result_z_resized = zoom(result, (z_resize, 1.0, 1.0))
    print("z resize done")

    file_name = os.path.join(dir_path, cross_sectional_image.patient_id + "_normalized_3d_affine.npy")
    np.save(file_name, result_z_resized)
