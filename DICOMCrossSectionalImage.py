'''
DICOMCrossSectionalImage
Author: Luben Popov
This important class stores all relevant data and annotations for an axial CT cross sectional image.
'''

from HeartLandmarks import HeartLandmarks
import MathUtil

class DICOMCrossSectionalImage:
    # dir_path: The absolute path to the directory containing the DICOM files used to construct the cross sectional
    #           image (only used to store the path in the object)
    # slices: An array containing the PyDicom slice objects used to construct the cross sectional image (assumes they
    #         are already sorted in correct order
    def __init__(self, dir_path, slices):
        ##### CT IMAGE DATA #####

        self.dir_path = dir_path
        self.dicom_slices = slices

        # The ID of the patient from whom the cross sectional image was scanned
        self.patient_id = self.dicom_slices[0].__getattr__("PatientID")

        # The shape of an individual slice in the cross sectional image in 2D NumPy form
        self.slice_shape = self.dicom_slices[0].pixel_array.shape

        # The shape the cross sectional image in (y, x, z) form; NOT NumPy shape format
        self.shape = [self.slice_shape[0], self.slice_shape[1], self.slice_count]

        # The number of slices in the cross sectional image
        self.slice_count = len(self.dicom_slices)

        # The second lowest pixel value of all slices in the cross sectional image (the lowest is padding and should
        # not be counted)
        self.global_min = MathUtil.second_min(
            min(self.dicom_slices, key=lambda slice: MathUtil.second_min(slice.pixel_array)).pixel_array)

        # The highest pixel value of all slices in the cross sectional image
        self.global_max = max(self.dicom_slices, key=lambda slice: slice.pixel_array.max()).pixel_array.max()

        ##### ANNOTATIONS #######

        # The superior crop boundary slice of the cross sectional image; when cropping, this will be the topmost slice
        # included in the cropped data
        self.superior_slice = 0

        # The inferior crop boundary slice of the cross sectional image; when cropping, this will be the bottommost
        # slice included in the cropped data
        self.inferior_slice = self.slice_count - 1

        # The scale factor used to scale XY crop boundaries for individual slices; when this is 1, the crop boundaries
        # at the superior and inferior slices will be exactly at the heart landmarks for those respective slices
        self.landmark_scale_factor = 1.6

        # The HeartLandmarks object that stores the eight annotated heart landmarks for the cross sectional image
        self.heart_landmarks = HeartLandmarks()

        self.set_default_landmarks()

    # Resets all heart landmarks to the outer XY boundaries of the cross sectional image
    def set_default_landmarks(self):
        self.heart_landmarks.set_to_image_shape(self.slice_shape)

    # Retrieves a slice from the cross sectional image at a given index
    # slice_idx: The index of the slice to retrieve
    # output: A PyDicom slice object representing the slice retrieved
    def get_slice(self, slice_idx):
        return self.dicom_slices[slice_idx]

    # Gets the XY crop boundaries for the slice at the given index; crop boundaries will be linearly interpolated from
    # the superior to the inferior slice
    # slice_idx: The index of the slice to retrieve crop boundaries for
    # output: A list representing the crop boundaries in the form (x, y) for each point in clockwise order starting from
    #         the top left point; this list will be all (0, 0) values if the index is less than the superior index or
    #         greater than the inferior index
    def get_slice_bounds(self, slice_idx):
        interpolant = -1.0
        if self.superior_slice <= slice_idx <= self.inferior_slice:
            interpolant = MathUtil.point_interpolant_1d(slice_idx, self.superior_slice, self.inferior_slice)
        return self.get_slice_bounds_from_interpolant(interpolant)

    # Gets the XY crop boundaries for the cross sectional image using an interpolant value
    # interpolant: The interpolant value used to get the crop boundaries (should be from 0.0 to 1.0)
    # output: A list representing the crop boundaries in the form (x, y) for each point in clockwise order starting from
    #         the top left point; this list will be all (0, 0) values if the interpolant is less than 0.0 or greater
    #         than 1.0
    def get_slice_bounds_from_interpolant(self, interpolant):
        bounds = [(0, 0)] * 4
        if 0.0 <= interpolant <= 1.0:
            scaled_superior_landmarks = self.heart_landmarks.get_scaled_superior(self.landmark_scale_factor)
            scaled_inferior_landmarks = self.heart_landmarks.get_scaled_inferior(self.landmark_scale_factor)

            bounds[0] = MathUtil.linear_interpolate_2d(scaled_superior_landmarks[0], scaled_inferior_landmarks[0],
                                                       interpolant)
            bounds[1] = MathUtil.linear_interpolate_2d(scaled_superior_landmarks[1], scaled_inferior_landmarks[1],
                                                       interpolant)
            bounds[2] = MathUtil.linear_interpolate_2d(scaled_superior_landmarks[2], scaled_inferior_landmarks[2],
                                                       interpolant)
            bounds[3] = MathUtil.linear_interpolate_2d(scaled_superior_landmarks[3], scaled_inferior_landmarks[3],
                                                       interpolant)

            bounds[0] = [max(bounds[0][0], 0), max(bounds[0][1], 0)]
            bounds[1] = [min(bounds[1][0], self.shape[1] - 1), max(bounds[1][1], 0)]
            bounds[2] = [min(bounds[2][0], self.shape[1] - 1), min(bounds[2][1], self.shape[0] - 1)]
            bounds[3] = [max(bounds[3][0], 0), min(bounds[3][1], self.shape[0] - 1)]

        return bounds

    # Gets the overall XY boundaries of the heart landmarks
    # output: A list representing the heart landmark boundaries in form [[x_min, x_max], [y_min, y_max]]
    def get_landmark_bounds(self):
        scaled_landmarks = self.get_slice_bounds(self.superior_slice) + self.get_slice_bounds(self.inferior_slice)
        x_values = [landmark[0] for landmark in scaled_landmarks]
        y_values = [landmark[1] for landmark in scaled_landmarks]

        return [[min(x_values), max(x_values)], [min(y_values), max(y_values)]]

    # Reverses the order of the slices in the cross sectional image
    def reverse_slices(self):
        self.dicom_slices.reverse()






