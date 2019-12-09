'''
DataReader
Author: Luben Popov
This library handles reading file data for DICOM files and annotation files.
'''

import pydicom
import os
import json
from DICOMCrossSectionalImage import DICOMCrossSectionalImage

# Searches a directory for DICOM files and if they contain a SliceLocation attribute stores their data into a list
# sorted by slice location
# dir: The absolute path to the target directory to scan for DICOM files
# output: An array of PyDicom slice objects sorted by SliceLocation attribute value
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

# Reads the pixel array from a single DICOM file at a given path
# path: The absolute path to the DICOM file to read
# output: A 2D NumPy array containing the pixel array data for the DICOM file
def read_single_pixmap(path):
    dicom_file = pydicom.read_file(path)
    return dicom_file.pixel_array

# Reads a cross sectional image from an annotation file; requires the directory path referenced in the annotation file
# to exist locally with the proper DICOM files inside it, otherwise this will not work as intended.
# file_path: The absolute path to the annotation file to read
# output: A DICOMCrossSectionalImage object that reflects the data inside the annotation file
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
