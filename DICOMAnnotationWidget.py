from PyQt5.Qt import *
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox, QFileDialog
from SpinnerDialComboWidget import SpinnerDialComboWidget
from XYSpinnerComboWidget import XYSpinnerComboWidget
from ViewSliceWidget import ViewSliceWidget
from DICOMCrossSectionalImage import DICOMCrossSectionalImage
import DataExport

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

        self.superior_slice_adjuster = SpinnerDialComboWidget("Superior Slice", 0, 0, 0)
        self.superior_slice_adjuster.value_changed.connect(self.on_superior_slice_adjuster_changed)

        self.inferior_slice_adjuster = SpinnerDialComboWidget("Inferior Slice", 0, 0, 0)
        self.inferior_slice_adjuster.value_changed.connect(self.on_inferior_slice_adjuster_changed)

        self.landmark_select_label = QLabel("Select a heart landmark to position")

        self.landmark_select_combo_box = QComboBox()
        self.landmark_select_combo_box.addItems(["Front left atrium (Superior)",
                                                 "Front right atrium (Superior)",
                                                 "Back right atrium (Superior)",
                                                 "Aorta (Superior)",
                                                 "Front left ventricle (Inferior)",
                                                 "Front right ventricle (Inferior)",
                                                 "Back right ventricle (Inferior)",
                                                 "Back left ventricle (Inferior)",
                                                 "None"])
        self.landmark_select_combo_box.setCurrentIndex(8)
        self.landmark_select_combo_box.currentIndexChanged.connect(self.on_landmark_selection_changed)

        self.landmark_position_adjuster = XYSpinnerComboWidget("Landmark position", (0, 0), (0, 0))
        self.landmark_position_adjuster.value_changed.connect(self.on_landmark_position_adjuster_changed)
        self.landmark_position_adjuster.setEnabled(False)

        self.view_slice_widget = ViewSliceWidget(self)
        self.view_slice_widget.mouse_dragged.connect(self.on_view_slice_widget_mouse_drag)

        self.update_instance_image()

        self.setup_gui()

    def setup_gui(self):

        vertical_layout = QVBoxLayout(self)
        dir_open_horizontal_layout = QHBoxLayout()
        landmark_select_vertical_layout = QVBoxLayout()
        tools_grid_layout = QGridLayout()

        dir_open_horizontal_layout.addWidget(self.image_directory_button)
        dir_open_horizontal_layout.addWidget(self.export_button)

        tools_grid_layout.addWidget(self.view_slice_adjuster, 0, 0)

        tools_grid_layout.addWidget(self.superior_slice_adjuster, 0, 1)
        tools_grid_layout.addWidget(self.inferior_slice_adjuster, 0, 2)

        landmark_select_vertical_layout.addWidget(self.landmark_select_label)
        landmark_select_vertical_layout.addWidget(self.landmark_select_combo_box)

        tools_grid_layout.addLayout(landmark_select_vertical_layout, 1, 0)

        tools_grid_layout.addWidget(self.landmark_position_adjuster, 1, 1)

        vertical_layout.setAlignment(Qt.AlignCenter)
        vertical_layout.addLayout(dir_open_horizontal_layout)
        vertical_layout.addLayout(tools_grid_layout)
        vertical_layout.addWidget(self.view_slice_widget)
        vertical_layout.addStretch(1)

    @pyqtSlot()
    def on_image_directory_button_clicked(self):
        dir_name = QFileDialog.getExistingDirectory(self, 'Select directory', 'c:\\')

        self.cross_sectional_image = DICOMCrossSectionalImage(dir_name)

        self.export_button.setEnabled(True)
        self.reset_controls()
        self.update_instance_image()

    @pyqtSlot()
    def on_export_button_clicked(self):
        file_path = QFileDialog.getSaveFileName(self, 'Save annotations', 'c:\\', 'TXT(.txt)')[0]
        print(file_path)
        DataExport.export_annotations(file_path, self.cross_sectional_image)

    @pyqtSlot()
    def on_view_slice_adjuster_changed(self):
        if self.cross_sectional_image is None:
            return

        self.view_slice = self.view_slice_adjuster.value
        self.update_instance_image()

    @pyqtSlot()
    def on_superior_slice_adjuster_changed(self):
        if self.cross_sectional_image is None:
            return

        self.cross_sectional_image.superior_slice = self.superior_slice_adjuster.value

    @pyqtSlot()
    def on_inferior_slice_adjuster_changed(self):
        if self.cross_sectional_image is None:
            return

        self.cross_sectional_image.inferior_slice = self.inferior_slice_adjuster.value

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

        self.update_instance_image()

    @pyqtSlot()
    def on_landmark_position_adjuster_changed(self):
        index = self.landmark_select_combo_box.currentIndex()
        if self.cross_sectional_image is None or index == 8:
            return

        self.cross_sectional_image.heart_landmarks.landmarks[index] = (self.landmark_position_adjuster.cur_coords)

        self.update_instance_image()

    @pyqtSlot()
    def on_view_slice_widget_mouse_drag(self):
        if self.cross_sectional_image is None or not self.landmark_position_adjuster.isEnabled():
            return

        x = self.view_slice_widget.mouse_x
        y = self.view_slice_widget.mouse_y

        self.landmark_position_adjuster.set_coords((x, y))

    def update_instance_image(self):
        if self.cross_sectional_image is None:
            return

        self.view_slice_widget.update_image(self.cross_sectional_image, self.view_slice,
                                                     self.landmark_select_combo_box.currentIndex())

    def reset_controls(self):
        self.view_slice_adjuster.blockSignals(True)
        self.superior_slice_adjuster.blockSignals(True)
        self.inferior_slice_adjuster.blockSignals(True)
        self.landmark_position_adjuster.blockSignals(True)

        max_slice = self.cross_sectional_image.slice_count - 1
        slice_shape = self.cross_sectional_image.slice_shape

        self.view_slice = 0

        self.superior_slice_adjuster.set_max(max_slice)
        self.superior_slice_adjuster.set_value(0)

        self.inferior_slice_adjuster.set_max(max_slice)
        self.inferior_slice_adjuster.set_value(max_slice)

        self.view_slice_adjuster.set_max(max_slice)
        self.view_slice_adjuster.set_value(0)

        self.landmark_position_adjuster.set_bounds((slice_shape[1], slice_shape[0]))
        self.landmark_position_adjuster.set_coords((0, 0))
        self.landmark_select_combo_box.setCurrentIndex(8)

        self.view_slice_adjuster.blockSignals(False)
        self.superior_slice_adjuster.blockSignals(False)
        self.inferior_slice_adjuster.blockSignals(False)
        self.landmark_position_adjuster.blockSignals(False)


