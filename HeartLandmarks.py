'''
HeartLandmarks
Author: Luben Popov
This class stores and scales heart landmarks used in CT image annotation.
'''

import MathUtil

class HeartLandmarks:

    def __init__(self):
        # The eight heart landmarks in (x, y) form; the first four are superior landmarks and the last four are inferior
        # landmarks, and each set of four landmarks is in clockwise order starting at the top left landmark
        self.landmarks = [(0, 0)] * 8

    # Sets the landmarks to the boundaries of a 2D image shape; each unique slice landmark will be its respective corner
    # of the image shape
    # shape: The 2D shape (in NumPy form) used to set the landmark positions
    def set_to_image_shape(self, shape):
        self.landmarks[0] = self.landmarks[4] = [0, 0]
        self.landmarks[1] = self.landmarks[5] = [shape[1] - 1, 0]
        self.landmarks[2] = self.landmarks[6] = [shape[1] - 1, shape[0] - 1]
        self.landmarks[3] = self.landmarks[7] = [0, shape[0] - 1]

    # Gets the four superior landmarks (in the order they are stored in the landmarks attribute) scaled from the center
    # of the four landmarks by the given scale factor
    # scale_factor: The scale factor used from scaling; 1.0 will provide no scaling
    # output: A list containing the four scaled superior landmarks in the order they are stored in self.landmarks
    def get_scaled_superior(self, scale_factor):
        bounds = self.landmarks[0:4]
        scaled_bounds = MathUtil.scale_bounds(bounds, scale_factor)
        return scaled_bounds

    # Gets the four inferior landmarks (in the order they are stored in the landmarks attribute) scaled from the center
    # of the four landmarks by the given scale factor
    # scale_factor: The scale factor used from scaling; 1.0 will provide no scaling
    # output: A list containing the four scaled inferior landmarks in the order they are stored in self.landmarks
    def get_scaled_inferior(self, scale_factor):
        bounds = self.landmarks[4:8]
        scaled_bounds = MathUtil.scale_bounds(bounds, scale_factor)
        return scaled_bounds



