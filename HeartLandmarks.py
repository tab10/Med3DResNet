from enum import Enum


class HeartLandmarks:

    def __init__(self):
        self.landmarks = [(0, 0)] * 8

    def set_to_image_shape(self, shape):
        self.landmarks[0] = self.landmarks[4] = (0, 0)
        self.landmarks[1] = self.landmarks[5] = (shape[1] - 1, 0)
        self.landmarks[2] = self.landmarks[6] = (0, shape[0] - 1)
        self.landmarks[3] = self.landmarks[7] = (shape[1] - 1, shape[0] - 1)

    def get_front_left_atrium(self):
        return self.landmarks[0]

    def set_front_left_atrium(self, coords):
        self.landmarks[0] = coords

    def get_front_right_atrium(self):
        return self.landmarks[1]

    def set_front_right_atrium(self, coords):
        self.landmarks[1] = coords

    def get_aorta(self):
        return self.landmarks[2]

    def set_aorta(self, coords):
        self.landmarks[2] = coords

    def get_back_right_atrium(self):
        return self.landmarks[3]

    def set_back_right_atrium(self, coords):
        self.landmarks[3] = coords

    def get_front_left_ventricle(self):
        return self.landmarks[4]

    def set_front_left_ventricle(self, coords):
        self.landmarks[4] = coords

    def get_front_right_ventricle(self):
        return self.landmarks[5]

    def set_front_right_ventricle(self, coords):
        self.landmarks[5] = coords

    def get_back_left_ventricle(self):
        return self.landmarks[6]

    def set_back_left_ventricle(self, coords):
        self.landmarks[6] = coords

    def get_back_right_ventricle(self):
        return self.landmarks[7]

    def set_back_right_ventricle(self, coords):
        self.landmarks[7] = coords
