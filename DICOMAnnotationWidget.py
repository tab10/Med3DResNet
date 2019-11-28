from PyQt5.Qt import *
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox, QFileDialog
from SpinnerDialComboWidget import SpinnerDialComboWidget
from XYSpinnerComboWidget import XYSpinnerComboWidget
from ViewSliceWidget import ViewSliceWidget
from ViewSliceDataWidget import ViewSliceDataWidget
from DICOMCrossSectionalImage import DICOMCrossSectionalImage
import DICOMSliceReader
import DataExport
import os

class DICOMAnnotationWidget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)

        self.cross_sectional_image = None
        self.view_slice = 0

        self.image_directory_button = QPushButton("Select Image Directory", self)
        self.image_directory_button.clicked.connect(self.on_image_directory_button_clicked)

        self.export_button = QPushButton("Export Annotations", self)
        self.export_button.clicked.connect(self.on_export_button_clicked)
        self.export_button.setEnabled(False)

        self.view_slice_adjuster = SpinnerDialComboWidget("View Slice", 0, 0, 0)
        self.view_slice_adjuster.value_changed.connect(self.on_view_slice_adjuster_changed)

        self.superior_slice_adjuster_label = QLabel("Superior Slice")
        self.superior_slice_adjuster = QSpinBox()
        self.superior_slice_adjuster.setMinimum(0)
        self.superior_slice_adjuster.setMaximum(0)
        self.superior_slice_adjuster.valueChanged.connect(self.on_superior_slice_adjuster_changed)
        self.view_to_superior_button = QPushButton("Set to view slice")
        self.view_to_superior_button.clicked.connect(self.on_view_to_superior_button_clicked)

        self.inferior_slice_adjuster_label = QLabel("Inferior Slice")
        self.inferior_slice_adjuster = QSpinBox()
        self.inferior_slice_adjuster.setMinimum(0)
        self.inferior_slice_adjuster.setMaximum(0)
        self.inferior_slice_adjuster.valueChanged.connect(self.on_inferior_slice_adjuster_changed)
        self.view_to_inferior_button = QPushButton("Set to view slice")
        self.view_to_inferior_button.clicked.connect(self.on_view_to_inferior_button_clicked)

        self.landmark_select_label = QLabel("Select a heart landmark to position")

        self.landmark_select_combo_box = QComboBox()
        self.landmark_select_combo_box.addItems(["Front right atrium (Superior X-Y+)",
                                                 "Front left atrium (Superior X+Y+)",
                                                 "Back left atrium (Superior X+Y-)",
                                                 "Aorta (Superior X-Y-)",
                                                 "Front right ventricle (Inferior X-Y+)",
                                                 "Front left ventricle (Inferior X+Y+)",
                                                 "Back left ventricle (Inferior X+Y-)",
                                                 "Back right ventricle (Inferior X-Y-)",
                                                 "None"])
        self.landmark_select_combo_box.setCurrentIndex(8)
        self.landmark_select_combo_box.currentIndexChanged.connect(self.on_landmark_selection_changed)

        self.landmark_position_adjuster = XYSpinnerComboWidget("Landmark position", (0, 0), (0, 0))
        self.landmark_position_adjuster.value_changed.connect(self.on_landmark_position_adjuster_changed)
        self.landmark_position_adjuster.setEnabled(False)

        self.reset_landmarks_button = QPushButton("Reset Landmarks")
        self.reset_landmarks_button.clicked.connect(self.on_reset_landmarks_button_clicked)

        self.reverse_slices_button = QPushButton("Reverse Slices")
        self.reverse_slices_button.clicked.connect(self.on_reverse_slices_button_clicked)

        self.view_slice_widget = ViewSliceWidget(self)
        self.view_slice_widget.mouse_dragged.connect(self.on_view_slice_widget_mouse_drag)
        self.view_slice_widget.mouse_moved.connect(self.on_view_slice_widget_mouse_move)
        self.view_slice_widget.mouse_scrolled.connect(self.on_view_slice_widget_mouse_scroll)

        self.view_slice_data_widget = ViewSliceDataWidget()
        self.view_slice_data_widget.setEnabled(False)

        self.update_view_slice_widget()

        self.setup_gui()

    def setup_gui(self):

        vertical_layout = QVBoxLayout(self)
        landmark_select_vertical_layout = QVBoxLayout()
        boundary_slice_vertical_layout = QVBoxLayout()
        tools_grid_layout = QGridLayout()

        boundary_slice_vertical_layout.addWidget(self.superior_slice_adjuster_label)
        boundary_slice_vertical_layout.addWidget(self.superior_slice_adjuster)
        boundary_slice_vertical_layout.addWidget(self.view_to_superior_button)
        boundary_slice_vertical_layout.addWidget(self.inferior_slice_adjuster_label)
        boundary_slice_vertical_layout.addWidget(self.inferior_slice_adjuster)
        boundary_slice_vertical_layout.addWidget(self.view_to_inferior_button)

        landmark_select_vertical_layout.addWidget(self.landmark_select_label)
        landmark_select_vertical_layout.addWidget(self.landmark_select_combo_box)

        tools_grid_layout.setColumnStretch(0, 1)
        tools_grid_layout.addWidget(self.image_directory_button, 0, 0)
        tools_grid_layout.setColumnStretch(1, 1)
        tools_grid_layout.addWidget(self.export_button, 0, 1)
        tools_grid_layout.addWidget(self.view_slice_adjuster, 1, 0)
        tools_grid_layout.addLayout(boundary_slice_vertical_layout, 1, 1)
        tools_grid_layout.addLayout(landmark_select_vertical_layout, 2, 0)
        tools_grid_layout.addWidget(self.landmark_position_adjuster, 2, 1)
        tools_grid_layout.addWidget(self.reset_landmarks_button, 3, 0)
        tools_grid_layout.addWidget(self.reverse_slices_button, 3, 1)

        vertical_layout.setAlignment(Qt.AlignCenter)
        vertical_layout.addLayout(tools_grid_layout)
        vertical_layout.addWidget(self.view_slice_widget)
        vertical_layout.addWidget(self.view_slice_data_widget)
        vertical_layout.addStretch(1)

    @pyqtSlot()
    def on_image_directory_button_clicked(self):
        dir_name = QFileDialog.getExistingDirectory(self, 'Select directory to read DICOM data', 'c:\\')
        if not dir_name:
            return

        dicom_slices = DICOMSliceReader.read_3d_slices_from_dir(dir_name)
        if len(dicom_slices) <= 1:
            self.showInvalidDirectoryMessageBox()
            return

        self.cross_sectional_image = DICOMCrossSectionalImage(dicom_slices)

        self.export_button.setEnabled(True)
        self.view_slice_data_widget.setEnabled(True)
        self.reset_controls()
        self.update_view_slice_widget()
        self.update_view_slice_data_widget()

    @pyqtSlot()
    def on_export_button_clicked(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select directory to create annotation file', 'c:\\')
        if not dir_path:
            return

        DataExport.export_annotations(dir_path, self.cross_sectional_image)

    @pyqtSlot()
    def on_view_slice_adjuster_changed(self):
        if self.cross_sectional_image is None:
            return

        self.view_slice = self.view_slice_adjuster.value

        self.update_view_slice_data_widget()

        self.update_view_slice_widget()

    @pyqtSlot()
    def on_superior_slice_adjuster_changed(self):
        if self.cross_sectional_image is None:
            return

        value = self.superior_slice_adjuster.value()
        self.cross_sectional_image.superior_slice = value

        self.update_view_slice_widget()

    @pyqtSlot()
    def on_inferior_slice_adjuster_changed(self):
        if self.cross_sectional_image is None:
            return

        value = self.inferior_slice_adjuster.value()
        self.cross_sectional_image.inferior_slice = value

        self.update_view_slice_widget()

    @pyqtSlot()
    def on_view_to_superior_button_clicked(self):
        if self.cross_sectional_image is None:
            return

        value = self.view_slice_adjuster.value
        if value >= self.cross_sectional_image.slice_count - 1:
            return

        self.inferior_slice_adjuster.setMinimum(value + 1)
        self.superior_slice_adjuster.setMaximum(self.inferior_slice_adjuster.value() - 1)
        self.superior_slice_adjuster.setValue(value)

    @pyqtSlot()
    def on_view_to_inferior_button_clicked(self):
        if self.cross_sectional_image is None:
            return

        value = self.view_slice_adjuster.value
        if value <= 0:
            return

        self.superior_slice_adjuster.setMaximum(value - 1)
        self.inferior_slice_adjuster.setMinimum(self.superior_slice_adjuster.value() + 1)
        self.inferior_slice_adjuster.setValue(value)

    @pyqtSlot(int)
    def on_landmark_selection_changed(self, index):
        if index != 8:
            self.landmark_position_adjuster.setEnabled(True)
        else:
            self.landmark_position_adjuster.setEnabled(False)

        if self.cross_sectional_image is None or index == 8:
            return

        self.landmark_position_adjuster.blockSignals(True)
        self.landmark_position_adjuster.set_coords(self.cross_sectional_image.heart_landmarks.landmarks[index])
        self.landmark_position_adjuster.blockSignals(False)

        self.update_view_slice_widget()

    @pyqtSlot()
    def on_landmark_position_adjuster_changed(self):
        index = self.landmark_select_combo_box.currentIndex()
        if self.cross_sectional_image is None or index == 8:
            return

        self.cross_sectional_image.heart_landmarks.landmarks[index] = (self.landmark_position_adjuster.cur_coords)

        self.update_view_slice_widget()

    @pyqtSlot()
    def on_view_slice_widget_mouse_drag(self):
        if self.cross_sectional_image is None or not self.landmark_position_adjuster.isEnabled():
            return

        x = self.view_slice_widget.mouse_x
        y = self.view_slice_widget.mouse_y

        self.landmark_position_adjuster.set_coords((x, y))

    @pyqtSlot()
    def on_view_slice_widget_mouse_move(self):
        if self.cross_sectional_image is None:
            return

        self.update_view_slice_data_widget()

    @pyqtSlot(int)
    def on_view_slice_widget_mouse_scroll(self, scroll_factor):
        if self.cross_sectional_image is None:
            return

        slice_idx = self.view_slice + scroll_factor
        if slice_idx < 0:
            slice_idx = 0
        if slice_idx >= self.cross_sectional_image.slice_count:
            slice_idx = self.cross_sectional_image.slice_count - 1

        self.view_slice_adjuster.set_value(slice_idx)

    @pyqtSlot()
    def on_reset_landmarks_button_clicked(self):
        self.cross_sectional_image.set_default_landmarks()
        self.update_view_slice_widget()
        self.update_view_slice_data_widget()

    @pyqtSlot()
    def on_reverse_slices_button_clicked(self):
        self.cross_sectional_image.reverse_slices()
        self.update_view_slice_widget()
        self.update_view_slice_data_widget()

    def showInvalidDirectoryMessageBox(self):
        message_box = QMessageBox()
        message_box.setWindowTitle("Invalid Directory")
        message_box.setText("Cannot open this directory; please select a directory with at least two valid DICOM "
                            "(.dcm) files.")
        message_box.exec_()
        pass

    def update_view_slice_widget(self):
        if self.cross_sectional_image is None:
            return

        self.view_slice_widget.update_image_data(self.cross_sectional_image, self.view_slice,
                                                 self.landmark_select_combo_box.currentIndex())

    def update_view_slice_data_widget(self):
        slice = self.cross_sectional_image.get_slice(self.view_slice)
        slice_file_name = os.path.basename(slice.filename)
        mouse_x = self.view_slice_widget.mouse_x
        mouse_y = self.view_slice_widget.mouse_y
        slice_pixel_value = slice.pixel_array[mouse_y, mouse_x]

        self.view_slice_data_widget.update_data(slice_file_name, slice_pixel_value, mouse_x, mouse_y)

    def reset_controls(self):
        self.view_slice_adjuster.blockSignals(True)
        self.superior_slice_adjuster.blockSignals(True)
        self.inferior_slice_adjuster.blockSignals(True)
        self.landmark_position_adjuster.blockSignals(True)

        max_slice = self.cross_sectional_image.slice_count - 1
        slice_shape = self.cross_sectional_image.slice_shape

        self.view_slice = 0

        self.superior_slice_adjuster.setMinimum(0)
        self.superior_slice_adjuster.setMaximum(max_slice - 1)
        self.superior_slice_adjuster.setValue(0)

        self.inferior_slice_adjuster.setMinimum(1)
        self.inferior_slice_adjuster.setMaximum(max_slice)
        self.inferior_slice_adjuster.setValue(max_slice)

        self.view_slice_adjuster.set_max(max_slice)
        self.view_slice_adjuster.set_value(0)

        self.landmark_position_adjuster.set_bounds((slice_shape[1], slice_shape[0]))
        self.landmark_position_adjuster.set_coords((0, 0))
        self.landmark_select_combo_box.setCurrentIndex(8)

        self.view_slice_adjuster.blockSignals(False)
        self.superior_slice_adjuster.blockSignals(False)
        self.inferior_slice_adjuster.blockSignals(False)
        self.landmark_position_adjuster.blockSignals(False)


