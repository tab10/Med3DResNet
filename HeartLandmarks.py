from enum import Enum


class HeartLandmarks:

    def __init__(self):
        self.landmarks = [(0, 0)] * 8

    def set_to_image_shape(self, shape):
        self.landmarks[0] = self.landmarks[4] = (0, 0)
        self.landmarks[1] = self.landmarks[5] = (shape[1] - 1, 0)
        self.landmarks[2] = self.landmarks[6] = (shape[1] - 1, shape[0] - 1)
        self.landmarks[3] = self.landmarks[7] = (0, shape[0] - 1)
