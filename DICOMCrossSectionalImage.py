from HeartLandmarks import HeartLandmarks
import numpy as np
import MathUtil

class DICOMCrossSectionalImage:

    def __init__(self, dicom_slices):
        self.dicom_slices = dicom_slices

        self.patient_id = self.dicom_slices[0].__getattr__("PatientID")
        self.slice_shape = self.dicom_slices[0].pixel_array.shape
        self.slice_count = len(self.dicom_slices)
        self.shape = [self.slice_shape[0], self.slice_shape[1], self.slice_count]
        self.global_min = min(self.dicom_slices, key=lambda slice: slice.pixel_array.min()).pixel_array.min()
        self.global_max = max(self.dicom_slices, key=lambda slice: slice.pixel_array.max()).pixel_array.max()

        self.superior_slice = 0
        self.inferior_slice = self.slice_count - 1

        self.heart_landmarks = HeartLandmarks()
        self.set_default_landmarks()

    def set_default_landmarks(self):
        self.heart_landmarks.set_to_image_shape(self.slice_shape)

    def get_slice(self, slice_idx):
        return self.dicom_slices[slice_idx]

    def get_slice_bounds(self, slice_idx):
        bounds = [(0, 0)] * 4
        if self.superior_slice <= slice_idx <= self.inferior_slice:
            interp_factor = MathUtil.point_interpolant_1d(slice_idx, self.superior_slice, self.inferior_slice)
            bounds[0] = MathUtil.linear_interpolate_2d(self.heart_landmarks.landmarks[0],
                                                       self.heart_landmarks.landmarks[4], interp_factor)
            bounds[1] = MathUtil.linear_interpolate_2d(self.heart_landmarks.landmarks[1],
                                                       self.heart_landmarks.landmarks[5], interp_factor)
            bounds[2] = MathUtil.linear_interpolate_2d(self.heart_landmarks.landmarks[2],
                                                       self.heart_landmarks.landmarks[6], interp_factor)
            bounds[3] = MathUtil.linear_interpolate_2d(self.heart_landmarks.landmarks[3],
                                                       self.heart_landmarks.landmarks[7], interp_factor)

        return bounds

    def reverse_slices(self):
        self.dicom_slices.reverse()






