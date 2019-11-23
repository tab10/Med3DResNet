from DICOMCrossSectionalImage import DICOMCrossSectionalImage
import os
import numpy as np
import json

def export_annotations(file_path, cross_sectional_image):
    file_name = file_path + '.txt'
    file = open(file_name, 'w')
    data = {}
    data['slices'] = []

    for slice_idx in range(cross_sectional_image.superior_slice, cross_sectional_image.inferior_slice + 1):
        slice_bounds = cross_sectional_image.get_slice_bounds(slice_idx)
        file_name = os.path.basename(cross_sectional_image.get_slice(slice_idx).filename)
        pixel_data = cross_sectional_image.get_slice(slice_idx).pixel_array

        data['slices'].append({
            'file_name': file_name,
            'bounds': slice_bounds
        })

    json.dump(data, file, indent=4)
    file.close()
