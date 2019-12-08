import pydicom
import os
import json
from DICOMCrossSectionalImage import DICOMCrossSectionalImage

def read_3d_slices_from_dir(dir):
    dir_files = os.listdir(dir)
    slices = []

    for file_name in dir_files:
        file_ext = os.path.splitext(file_name)[1]
        file_path = os.path.join(dir, file_name)
        if file_ext == '.dcm' and file_name[0] != '.':
            dicom_file = pydicom.read_file(file_path)
            if hasattr(dicom_file, 'SliceLocation'):
                slices.append(dicom_file)

    if len(slices) > 0:
        slices = sorted(slices, reverse=True, key=lambda slice: slice.SliceLocation)

    return slices

def read_single_pixmap(path):
    dicom_file = pydicom.read_file(path)
    return dicom_file.pixel_array

def cross_sectional_image_from_annotation_file(file_path):
    json_file = open(file_path)
    json_data = json.load(json_file)
    slices_dir_path = json_data['dir_path']
    slices = read_3d_slices_from_dir(slices_dir_path)
    cross_sectional_image = DICOMCrossSectionalImage(slices_dir_path, slices)
    cross_sectional_image.superior_slice = json_data['superior_slice']
    cross_sectional_image.inferior_slice = json_data['inferior_slice']
    cross_sectional_image.heart_landmarks.landmarks = json_data['landmarks']
    cross_sectional_image.landmark_scale_factor = json_data['landmark_scale_factor']
    json_file.close()

    return cross_sectional_image
