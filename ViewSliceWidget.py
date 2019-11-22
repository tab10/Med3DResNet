from PyQt5.QtWidgets import QLabel
from PyQt5.Qt import *
import cv2
import numpy as np
import MathUtil

class ViewSliceWidget(QLabel):
    mouse_dragged = pyqtSignal()

    def __init__(self, parent=None):
        QLabel.__init__(self, parent=None)

        self.mouse_held = False
        self.mouse_x = 0
        self.mouse_y = 0

        default_pixmap = QPixmap(512, 512)
        default_pixmap.fill(Qt.black)
        self.setPixmap(default_pixmap)

        self.mousePressEvent = self.on_mouse_press_event
        self.mouseMoveEvent = self.on_mouse_drag_event
        self.mouseReleaseEvent = self.on_mouse_release_event

    def update_image(self, cross_sectional_image, slice_idx, landmark_idx):
        pixel_array = cross_sectional_image.get_slice(slice_idx).pixel_array
        heart_landmarks = cross_sectional_image.heart_landmarks
        slice_bounds = cross_sectional_image.get_slice_bounds(slice_idx)

        qt_image = self.pixel_array_to_qimage(pixel_array, cross_sectional_image.global_min,
                                              cross_sectional_image.global_max)
        qt_image = self.paint_landmarks_on_qimage(qt_image, heart_landmarks, slice_bounds, landmark_idx)

        pixmap = QPixmap.fromImage(qt_image)
        self.setPixmap(pixmap)

    def pixel_array_to_qimage(self, pixel_array, global_min, global_max):
        normalized_image = MathUtil.scale_matrix(pixel_array, 0, 256, global_min, global_max)
        normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2RGB)

        qt_image = QImage(normalized_image.data, normalized_image.shape[1], normalized_image.shape[0],
                          3 * normalized_image.shape[1], QImage.Format_RGB888)

        return qt_image

    def paint_landmarks_on_qimage(self, qt_image, heart_landmarks, slice_bounds, landmark_idx):
        painter = QPainter()
        painter.begin(qt_image)

        self.draw_bounds(slice_bounds, painter, Qt.red, 2)

        for idx in range(4, 8):
            landmark = heart_landmarks.landmarks[idx]
            self.draw_cross(landmark[0], landmark[1], painter, Qt.blue, 2, 5)

        for idx in range(0, 4):
            landmark = heart_landmarks.landmarks[idx]
            self.draw_cross(landmark[0], landmark[1], painter, Qt.green, 2, 5)

        if 0 <= landmark_idx < 4:
            landmark = heart_landmarks.landmarks[landmark_idx]
            self.draw_cross(landmark[0], landmark[1], painter, Qt.green, 2, 10)
        elif 4 <= landmark_idx < 8:
            landmark = heart_landmarks.landmarks[landmark_idx]
            self.draw_cross(landmark[0], landmark[1], painter, Qt.blue, 2, 10)

        painter.end()
        return qt_image

    def draw_cross(self, x, y, painter, color, thickness, side_length):
        old_pen = painter.pen()
        painter.setPen(QPen(color, thickness, Qt.SolidLine))

        painter.drawLine(x, y + side_length, x, y - side_length)
        painter.drawLine(x + side_length, y, x - side_length, y)

        painter.setPen(old_pen)

    def draw_bounds(self, bounds, painter, color, thickness):
        old_pen = painter.pen()
        painter.setPen(QPen(color, thickness, Qt.SolidLine))

        painter.drawLine(bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1])
        painter.drawLine(bounds[1][0], bounds[1][1], bounds[2][0], bounds[2][1])
        painter.drawLine(bounds[2][0], bounds[2][1], bounds[3][0], bounds[3][1])
        painter.drawLine(bounds[3][0], bounds[3][1], bounds[0][0], bounds[0][1])

        painter.setPen(old_pen)

    def on_mouse_press_event(self, event):
        if event.button() == 1:
            self.mouse_held = True

    def on_mouse_release_event(self, event):
        if event.button() == 1:
            self.mouse_held = False

    def on_mouse_drag_event(self, event):
        self.mouse_x = event.pos().x()
        self.mouse_y = event.pos().y()
        if self.mouse_held:
            self.mouse_dragged.emit()