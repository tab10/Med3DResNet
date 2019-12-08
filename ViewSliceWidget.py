from PyQt5.Qt import *
import cv2
import MathUtil

class ViewSliceWidget(QLabel):
    mouse_moved = pyqtSignal()
    mouse_dragged = pyqtSignal()
    mouse_scrolled = pyqtSignal(int)

    def __init__(self, parent=None):
        QLabel.__init__(self, parent=None)

        self.setCursor(Qt.BlankCursor)
        self.setMouseTracking(True)

        self.mouse_held = False
        self.mouse_x = 0
        self.mouse_y = 0

        default_pixmap = QPixmap(512, 512)
        default_pixmap.fill(Qt.black)
        self.image_data_pixmap = default_pixmap

        self.setPixmap(default_pixmap)

        self.mousePressEvent = self.on_mouse_press_event
        self.mouseReleaseEvent = self.on_mouse_release_event
        self.mouseMoveEvent = self.on_mouse_drag_event
        self.wheelEvent = self.on_mouse_wheel_event

    def update_image_data(self, cross_sectional_image, slice_idx, landmark_idx):
        pixel_array = cross_sectional_image.get_slice(slice_idx).pixel_array
        heart_landmarks = cross_sectional_image.heart_landmarks
        slice_bounds = cross_sectional_image.get_slice_bounds(slice_idx)

        qt_image = self.pixel_array_to_qimage(pixel_array, cross_sectional_image.global_min,
                                              cross_sectional_image.global_max)
        qt_image = self.paint_landmarks_on_qimage(qt_image, heart_landmarks, slice_bounds, landmark_idx)

        pixmap = QPixmap.fromImage(qt_image)
        self.image_data_pixmap = pixmap
        self.update_display()

    def update_display(self):
        qt_image = QImage(self.image_data_pixmap)

        qt_image = self.paint_target_on_qimage(qt_image)

        pixmap = QPixmap.fromImage(qt_image)
        self.setPixmap(pixmap)

    def pixel_array_to_qimage(self, pixel_array, global_min, global_max):
        normalized_image = MathUtil.scale_matrix(pixel_array, 0, 255, global_min, global_max)
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
            self.draw_cross(landmark[0], landmark[1], painter, Qt.blue, 1, 5)

        for idx in range(0, 4):
            landmark = heart_landmarks.landmarks[idx]
            self.draw_cross(landmark[0], landmark[1], painter, Qt.green, 1, 5)

        if 0 <= landmark_idx < 4:
            landmark = heart_landmarks.landmarks[landmark_idx]
            self.draw_cross(landmark[0], landmark[1], painter, Qt.green, 1, 20, True)
        elif 4 <= landmark_idx < 8:
            landmark = heart_landmarks.landmarks[landmark_idx]
            self.draw_cross(landmark[0], landmark[1], painter, Qt.blue, 1, 20, True)

        painter.end()
        return qt_image

    def paint_target_on_qimage(self, qt_image):
        painter = QPainter()
        painter.begin(qt_image)

        self.draw_target(self.mouse_x, self.mouse_y, qt_image.width(), qt_image.height(), painter, Qt.gray, 1)

        painter.end()
        return qt_image

    def draw_cross(self, x, y, painter, color, thickness, side_length, encircled=False):
        old_pen = painter.pen()
        painter.setPen(QPen(color, thickness, Qt.SolidLine))

        painter.drawLine(x, y + side_length, x, y - side_length)
        painter.drawLine(x + side_length, y, x - side_length, y)

        if encircled:
            painter.setPen(QPen(color, thickness + 1, Qt.SolidLine))
            painter.drawEllipse(x - side_length, y - side_length, side_length*2, side_length*2)

        painter.setPen(old_pen)

    def draw_bounds(self, bounds, painter, color, thickness):
        old_pen = painter.pen()
        painter.setPen(QPen(color, thickness, Qt.SolidLine))

        painter.drawLine(bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1])
        painter.drawLine(bounds[1][0], bounds[1][1], bounds[2][0], bounds[2][1])
        painter.drawLine(bounds[2][0], bounds[2][1], bounds[3][0], bounds[3][1])
        painter.drawLine(bounds[3][0], bounds[3][1], bounds[0][0], bounds[0][1])

        painter.setPen(old_pen)

    def draw_target(self, x, y, width, height, painter, color, thickness):
        old_pen = painter.pen()
        painter.setPen(QPen(color, thickness, Qt.SolidLine))

        painter.drawLine(x, 0, x, height)
        painter.drawLine(0, y, width, y)

        painter.setPen(old_pen)

    def clamp_mouse_coords(self):
        if self.mouse_x < 0:
            self.mouse_x = 0
        if self.mouse_x >= self.image_data_pixmap.width():
            self.mouse_x = self.image_data_pixmap.width() - 1

        if self.mouse_y < 0:
            self.mouse_y = 0
        if self.mouse_y >= self.image_data_pixmap.height():
            self.mouse_y = self.image_data_pixmap.height() - 1

    def on_mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_held = True
            self.mouse_dragged.emit()

    def on_mouse_release_event(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_held = False

    def on_mouse_drag_event(self, event):
        self.mouse_x = event.pos().x()
        self.mouse_y = event.pos().y()
        self.clamp_mouse_coords()

        self.mouse_moved.emit()
        self.update_display()
        if self.mouse_held:
            self.mouse_dragged.emit()

    def on_mouse_wheel_event(self, event):
        scroll_angle = event.angleDelta().y()
        scroll_factor = 0 if scroll_angle == 0 else (1 if scroll_angle > 0 else -1)
        self.mouse_scrolled.emit(scroll_factor)