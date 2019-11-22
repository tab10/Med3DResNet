import pydicom
import os

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
        slices = sorted(slices, reverse=True, key=lambda slice: slice.SliceLocation)

    return slices

