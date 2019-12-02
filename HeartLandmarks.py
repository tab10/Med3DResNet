import MathUtil

class HeartLandmarks:

    def __init__(self):
        self.landmarks = [(0, 0)] * 8

    def set_to_image_shape(self, shape):
        self.landmarks[0] = self.landmarks[4] = [0, 0]
        self.landmarks[1] = self.landmarks[5] = [shape[1] - 1, 0]
        self.landmarks[2] = self.landmarks[6] = [shape[1] - 1, shape[0] - 1]
        self.landmarks[3] = self.landmarks[7] = [0, shape[0] - 1]

    def get_scaled_superior(self, scale_factor):
        bounds = self.landmarks[0:4]
        scaled_bounds = MathUtil.scale_bounds(bounds, scale_factor)
        return scaled_bounds

    def get_scaled_inferior(self, scale_factor):
        bounds = self.landmarks[4:8]
        scaled_bounds = MathUtil.scale_bounds(bounds, scale_factor)
        return scaled_bounds



