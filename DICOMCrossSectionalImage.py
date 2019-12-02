from HeartLandmarks import HeartLandmarks
import MathUtil

class DICOMCrossSectionalImage:

    def __init__(self, dir_path, slices):
        self.dir_path = dir_path
        self.dicom_slices = slices

        self.patient_id = self.dicom_slices[0].__getattr__("PatientID")
        self.slice_shape = self.dicom_slices[0].pixel_array.shape
        self.slice_count = len(self.dicom_slices)
        self.shape = [self.slice_shape[0], self.slice_shape[1], self.slice_count]
        self.global_min = MathUtil.second_min(
            min(self.dicom_slices, key=lambda slice: MathUtil.second_min(slice.pixel_array)).pixel_array)
        self.global_max = max(self.dicom_slices, key=lambda slice: slice.pixel_array.max()).pixel_array.max()

        self.superior_slice = 0
        self.inferior_slice = self.slice_count - 1
        self.landmark_scale_factor = 1.6

        self.heart_landmarks = HeartLandmarks()
        self.set_default_landmarks()

    def set_default_landmarks(self):
        self.heart_landmarks.set_to_image_shape(self.slice_shape)

    def get_slice(self, slice_idx):
        return self.dicom_slices[slice_idx]

    def get_slice_bounds(self, slice_idx):
        interpolant = -1.0
        if self.superior_slice <= slice_idx <= self.inferior_slice:
            interpolant = MathUtil.point_interpolant_1d(slice_idx, self.superior_slice, self.inferior_slice)
        return self.get_slice_bounds_from_interpolant(interpolant)


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

    def get_landmark_bounds(self):
        scaled_landmarks = self.get_slice_bounds(self.superior_slice) + self.get_slice_bounds(self.inferior_slice)
        x_values = [landmark[0] for landmark in scaled_landmarks]
        y_values = [landmark[1] for landmark in scaled_landmarks]

        return [[min(x_values), max(x_values)], [min(y_values), max(y_values)]]

    def reverse_slices(self):
        self.dicom_slices.reverse()






