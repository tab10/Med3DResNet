'''
DataExport
Author: Luben Popov
This library handles exporting of annotation files and 3D arrays.
'''

import DataNormalization
import os
import numpy as np
import json
import cv2
import DataReader

# Exports a cross sectional image's annotation data to a directory
# dir_path: The absolute path of the target directory for exporting the annotation file
# cross_sectional_image: The DICOMCrossSectionalImage instance containing the annotation data to export
def export_annotations(dir_path, cross_sectional_image):
    # Despite the annotations being exported in basic .JSON format, we use a custom .annotations extension to make the
    # files more distinguishable
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

# Batch exports all annotation files in a directory to 3D array files stored in the same directory; for each
# annotation file, an affine normalized 3D array and a projection normalized 3D array will be exported
# dir_path: The absolute path of the target directory from which annotation files will be read and 3D array files
#           exported
# desired_width: The desired width (x-axis) of the exported 3D arrays
# desired_height: The desired height (y-axis) of the exported 3D arrays
# desired_depth: The desired depth (z-axis) of the exported 3D arrays
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

            affine_array = DataNormalization.normalize_cross_sectional_image_affine(cross_sectional_image,
                                                                                    desired_width, desired_height,
                                                                                    desired_depth)
            projection_array = DataNormalization.normalize_cross_sectional_image_projected(cross_sectional_image,
                                                                                           desired_width,
                                                                                           desired_height,
                                                                                           desired_depth)

            np.save(affine_file_name, affine_array)
            np.save(projection_file_name, projection_array)

# Batch exports all 3D array files in a directory to masked 3D array files stored in the same directory
# dir_path: The absolute path of the target directory from which 3D array files will be read and masked 3D array files
# exported
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
