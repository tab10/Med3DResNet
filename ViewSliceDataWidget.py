from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout, QFrame

class ViewSliceDataWidget(QWidget):

    def __init__(self, name="", default_value=0, min_val=0, max_val=100, parent=None):
        QWidget.__init__(self, parent=parent)

        self.file_name_label = QLabel("DICOM File:")
        self.pixel_value_label = QLabel("Pixel Value:")
        self.pixel_coords_label = QLabel("X: Y: ")

        self.setup_gui()

    def setup_gui(self):
        main_layout = QHBoxLayout(self)
        horizontal_layout = QHBoxLayout()

        horizontal_layout.addWidget(self.file_name_label)
        horizontal_layout.addStretch(1)
        horizontal_layout.addWidget(self.pixel_value_label)
        horizontal_layout.addStretch(1)
        horizontal_layout.addWidget(self.pixel_coords_label)

        frame = QFrame()
        frame.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        frame.setLayout(horizontal_layout)

        main_layout.addWidget(frame)


    def update_data(self, file_name, pixel_value, pixel_x, pixel_y):
        self.file_name_label.setText(f"DICOM File: {file_name}")
        self.pixel_value_label.setText(f"Pixel Value: {pixel_value}")
        self.pixel_coords_label.setText(f"X: {pixel_x} Y: {pixel_y}")
