import DataNormalization
import os
import numpy as np
import json
import cv2
import DataReader

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

            affine_file_name = os.path.join(dir_path, cross_sectional_image.patient_id + "_normalized_3d_affine.npy")
            projection_file_name = os.path.join(dir_path,
                                                cross_sectional_image.patient_id + "_normalized_3d_projection.npy")

            affine_array = DataNormalization.normalize_cross_sectional_image_affine(dir_path, cross_sectional_image,
                                                                                    desired_width, desired_height,
                                                                                    desired_depth)
            projection_array = DataNormalization.normalize_cross_sectional_image_projected(dir_path,
                                                                                           cross_sectional_image,
                                                                                           desired_width,
                                                                                           desired_height,
                                                                                           desired_depth)

            np.save(affine_file_name, affine_array)
            np.save(projection_file_name, projection_array)

def batch_mask_3d_arrays(dir_path):
    dir_files = os.listdir(dir_path)

    for file_name in dir_files:
        file_ext = os.path.splitext(file_name)[1]
        file_path = os.path.join(dir_path, file_name)

        if file_ext == '.npy':
            array = np.load(file_path)
            if array.ndim == 3:
                masked_array = DataNormalization.mask_3d_array(array)

                file_name_nopath = os.path.splitext(file_name)[0]
                masked_file_name = os.path.join(dir_path, file_name_nopath + "_masked.npy")

                np.save(masked_file_name, masked_array)
