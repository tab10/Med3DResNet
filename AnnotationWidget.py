'''
AnnotationWidget
Author: Luben Popov
This class is the main PyQT widget that drives the annotation tool's logic.
'''

from PyQt5.Qt import *
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox, QFileDialog
from SpinnerDialComboWidget import SpinnerDialComboWidget
from XYSpinnerComboWidget import XYSpinnerComboWidget
from ViewSliceWidget import ViewSliceWidget
from ViewSliceDataWidget import ViewSliceDataWidget
from DICOMCrossSectionalImage import DICOMCrossSectionalImage
import DataReader
import DataExport
import os

class AnnotationWidget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)

        # The current cross sectional image being annotated
        self.cross_sectional_image = None

        # The current view slice index in the cross sectional image (the view slice is the one that is displayed in
        # the annotation tool GUI)
        self.view_slice = 0

        # Used to adjust the current view slice
        self.view_slice_adjuster = SpinnerDialComboWidget("View Slice", 0, 0, 0)
        self.view_slice_adjuster.value_changed.connect(self.on_view_slice_adjuster_changed)
        self.view_slice_adjuster.setToolTip("Adjust the currently displayed slice")

        self.superior_slice_adjuster_label = QLabel("Superior Slice (Main Pulmonary Artery Level)")

        # Used to adjust the superior crop boundary slice
        self.superior_slice_adjuster = QSpinBox()
        self.superior_slice_adjuster.setMinimum(0)
        self.superior_slice_adjuster.setMaximum(0)
        self.superior_slice_adjuster.valueChanged.connect(self.on_superior_slice_adjuster_changed)
        self.superior_slice_adjuster.setToolTip("The superior (top) crop boundary slice")

        # Used to set the superior crop boundary slice to the current view slice
        self.view_to_superior_button = QPushButton("Set to view slice")
        self.view_to_superior_button.clicked.connect(self.on_view_to_superior_button_clicked)
        self.view_to_superior_button.setToolTip("Set the superior (top) crop boundary slice to be the currently "
                                                "displayed slice")

        self.inferior_slice_adjuster_label = QLabel("Inferior Slice (Low Cardiac Level)")

        # Used to adjust the inferior crop boundary slice
        self.inferior_slice_adjuster = QSpinBox()
        self.inferior_slice_adjuster.setMinimum(0)
        self.inferior_slice_adjuster.setMaximum(0)
        self.inferior_slice_adjuster.valueChanged.connect(self.on_inferior_slice_adjuster_changed)
        self.inferior_slice_adjuster.setToolTip("The inferior (bottom) crop boundary slice")

        # Used to set the inferior crop boundary slice to the current view slice
        self.view_to_inferior_button = QPushButton("Set to view slice")
        self.view_to_inferior_button.clicked.connect(self.on_view_to_inferior_button_clicked)
        self.view_to_inferior_button.setToolTip("Set the inferior (bottom) crop boundary slice to be the currently "
                                                "displayed slice")

        self.slice_scale_label = QLabel("Boundary Slice Scale")

        # Used to adjust the slice crop boundary scale factor
        self.slice_scale_adjuster = QDoubleSpinBox()
        self.slice_scale_adjuster.setMinimum(0.0)
        self.slice_scale_adjuster.setSingleStep(0.1)
        self.slice_scale_adjuster.valueChanged.connect(self.on_slice_scale_adjuster_changed)
        self.slice_scale_adjuster.setToolTip("The scale factor used to scale crop boundaries for each slice")

        self.landmark_select_label = QLabel("Select a heart landmark to position")

        # Used to select a heart landmark to annotate
        self.landmark_select_combo_box = QComboBox()
        # The names here are purely for display purposes do not influence any other part of the code
        self.landmark_select_combo_box.addItems(["Ascending Aorta (Superior X-Y+)",
                                                 "Pulmonary Trunk (Superior X+Y+)",
                                                 "Descending Aorta (Superior X+Y-)",
                                                 "Superior Vena Cava (Superior X-Y-)",
                                                 "Right Ventricle (Inferior X-Y+)",
                                                 "Left Ventricle (Inferior X+Y+)",
                                                 "Descending Aorta (Inferior X+Y-)",
                                                 "Inferior Vena Cava (Inferior X-Y-)",
                                                 "None"])
        # Default heart landmark should be None
        self.landmark_select_combo_box.setCurrentIndex(8)
        self.landmark_select_combo_box.currentIndexChanged.connect(self.on_landmark_selection_changed)
        self.landmark_select_combo_box.setToolTip("Select the heart landmark to annotate")

        # Used to adjust the position of the currently selected heart landmark (if it is not None)
        self.landmark_position_adjuster = XYSpinnerComboWidget("Landmark position", (0, 0), (0, 0))
        self.landmark_position_adjuster.value_changed.connect(self.on_landmark_position_adjuster_changed)
        # Since the default landmark should be None, this widget should be disabled until a heart landmark is selected
        self.landmark_position_adjuster.setEnabled(False)
        self.landmark_position_adjuster.setToolTip("The current XY position of the selected heart landmark")

        # Used to display the view slice and provide secondary ways to adjust heart landmarks and adjust the view slice
        self.view_slice_widget = ViewSliceWidget(self)
        self.view_slice_widget.mouse_dragged.connect(self.on_view_slice_widget_mouse_drag)
        self.view_slice_widget.mouse_moved.connect(self.on_view_slice_widget_mouse_move)
        self.view_slice_widget.mouse_scrolled.connect(self.on_view_slice_widget_mouse_scroll)

        # Used to display data about the view slice
        self.view_slice_data_widget = ViewSliceDataWidget()
        self.view_slice_data_widget.setEnabled(False)

        self.update_view_slice_widget()

        self.setup_gui()

    # Sets up the positioning of the UI elements
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
        boundary_slice_vertical_layout.addWidget(self.slice_scale_label)
        boundary_slice_vertical_layout.addWidget(self.slice_scale_adjuster)

        landmark_select_vertical_layout.addWidget(self.landmark_select_label)
        landmark_select_vertical_layout.addWidget(self.landmark_select_combo_box)

        tools_grid_layout.setColumnStretch(0, 1)
        tools_grid_layout.addWidget(self.view_slice_adjuster, 0, 0)
        tools_grid_layout.setColumnStretch(1, 1)
        tools_grid_layout.addLayout(boundary_slice_vertical_layout, 0, 1)
        tools_grid_layout.addLayout(landmark_select_vertical_layout, 1, 0)
        tools_grid_layout.addWidget(self.landmark_position_adjuster, 1, 1)

        vertical_layout.setAlignment(Qt.AlignCenter)
        vertical_layout.addLayout(tools_grid_layout)
        vertical_layout.addWidget(self.view_slice_widget)
        vertical_layout.addWidget(self.view_slice_data_widget)
        vertical_layout.addStretch(1)

    # Callback for when the button to open a new image directory is clicked
    @pyqtSlot()
    def on_image_directory_button_clicked(self):
        dir_name = QFileDialog.getExistingDirectory(self, 'Select directory to read DICOM data', 'c:\\')
        if not dir_name:
            return

        dicom_slices = DataReader.read_3d_slices_from_dir(dir_name)
        if len(dicom_slices) <= 1:
            self.showInvalidDirectoryMessageBox()
            return

        self.cross_sectional_image = DICOMCrossSectionalImage(dir_name, dicom_slices)

        self.view_slice_data_widget.setEnabled(True)
        self.reset_controls()
        self.update_view_slice_widget()
        self.update_view_slice_data_widget()

    # Callback for when the button to export annotations is clicked
    @pyqtSlot()
    def on_export_annotations_button_clicked(self):
        if self.cross_sectional_image is None:
            return

        dir_path = QFileDialog.getExistingDirectory(self, 'Select directory to create annotation file', 'c:\\')
        if not dir_path:
            return

        DataExport.export_annotations(dir_path, self.cross_sectional_image)

    # Callback for when the button to import annotations is clicked
    @pyqtSlot()
    def on_import_annotations_button_clicked(self):
        file_path = QFileDialog.getOpenFileName(self, 'Select annotation file to import', 'c:\\')[0]
        self.cross_sectional_image = DataReader.cross_sectional_image_from_annotation_file(file_path)

        self.view_slice_data_widget.setEnabled(True)
        self.reset_controls()
        self.update_view_slice_widget()
        self.update_view_slice_data_widget()

    # Callback for when the button to batch export 3D arrays from annotations is clicked
    @pyqtSlot()
    def on_batch_export_3d_arrays_button_clicked(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select annotation file directory', 'c:\\')
        if not dir_path:
            return

        DataExport.batch_export_annotations_to_3d_arrays(dir_path, 256, 256, 256)

    # Callback for when the button to batch export masked 3D arrays from original 3D arrays is clicked
    @pyqtSlot()
    def on_batch_mask_3d_arrays_button_clicked(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select annotation file directory', 'c:\\')
        if not dir_path:
            return

        DataExport.batch_mask_3d_arrays(dir_path)

    # Callback from when the value of the widget that adjusts the current view slice is changed
    @pyqtSlot()
    def on_view_slice_adjuster_changed(self):
        if self.cross_sectional_image is None:
            return

        self.view_slice = self.view_slice_adjuster.value

        self.update_view_slice_data_widget()

        self.update_view_slice_widget()

    # Callback for when the value of the widget that adjusts the superior crop boundary slice is changed
    @pyqtSlot()
    def on_superior_slice_adjuster_changed(self):
        if self.cross_sectional_image is None:
            return

        value = self.superior_slice_adjuster.value()
        self.cross_sectional_image.superior_slice = value

        self.update_view_slice_widget()

    # Callback for when the value of the widget that adjusts the inferior crop boundary slice is changed
    @pyqtSlot()
    def on_inferior_slice_adjuster_changed(self):
        if self.cross_sectional_image is None:
            return

        value = self.inferior_slice_adjuster.value()
        self.cross_sectional_image.inferior_slice = value

        self.update_view_slice_widget()

    # Callback for when the button that sets the superior crop boundary slice to the current view slice is clicked
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

    # Callback for when the button that sets the inferior crop boundary slice to the current view slice is clicked
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

    # Callback for when the value of the widget that adjusts the slice crop boundary scale factor is changed
    @pyqtSlot()
    def on_slice_scale_adjuster_changed(self):
        if self.cross_sectional_image is None:
            return

        self.cross_sectional_image.landmark_scale_factor = self.slice_scale_adjuster.value()
        self.update_view_slice_widget()

    # Callback for when the selection in the combo box that chooses which heart landmark to position is changed
    # index: The new index of the selected item in the combo box
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

    # Callback for when the value of the widget that adjusts the current heart landmark position is changed
    @pyqtSlot()
    def on_landmark_position_adjuster_changed(self):
        index = self.landmark_select_combo_box.currentIndex()
        if self.cross_sectional_image is None or index == 8:
            return

        self.cross_sectional_image.heart_landmarks.landmarks[index] = (self.landmark_position_adjuster.cur_coords)

        self.update_view_slice_widget()

    # Callback for when the mouse is held and dragged over the widget that displays the view slice
    @pyqtSlot()
    def on_view_slice_widget_mouse_drag(self):
        if self.cross_sectional_image is None or not self.landmark_position_adjuster.isEnabled():
            return

        x = self.view_slice_widget.mouse_x
        y = self.view_slice_widget.mouse_y

        self.landmark_position_adjuster.set_coords((x, y))

    # Callback for when the mouse is moved over the widget that displays the view slice
    @pyqtSlot()
    def on_view_slice_widget_mouse_move(self):
        if self.cross_sectional_image is None:
            return

        self.update_view_slice_data_widget()

    # Callback for when the mouse is scrolled over the widget that displays the view slice
    # scroll_factor: The direction of the scrolling (1 if up, -1 if down)
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

    # Callback for when the button that resets the positions of all heart landmarks is clicked
    @pyqtSlot()
    def on_reset_landmarks_button_clicked(self):
        if self.cross_sectional_image is None:
            return

        self.cross_sectional_image.set_default_landmarks()
        self.update_view_slice_widget()
        self.update_view_slice_data_widget()

    # Callback for when the button that reverses the order of slices in the cross sectional image is clicked
    @pyqtSlot()
    def on_reverse_slices_button_clicked(self):
        if self.cross_sectional_image is None:
            return

        self.cross_sectional_image.reverse_slices()
        self.update_view_slice_widget()
        self.update_view_slice_data_widget()

    # Shows a message box indicating that an invalid directory has been selected
    def showInvalidDirectoryMessageBox(self):
        message_box = QMessageBox()
        message_box.setWindowTitle("Invalid Directory")
        message_box.setText("Cannot open this directory; please select a directory with at least two valid DICOM "
                            "(.dcm) files.")
        message_box.exec_()
        pass

    # Updates the widget that displays the view slice with the most recent view slice
    def update_view_slice_widget(self):
        if self.cross_sectional_image is None:
            return

        self.view_slice_widget.update_image_data(self.cross_sectional_image, self.view_slice,
                                                 self.landmark_select_combo_box.currentIndex())

    # Updates the widget that displays data about the view slice with the most recent data
    def update_view_slice_data_widget(self):
        slice = self.cross_sectional_image.get_slice(self.view_slice)
        slice_file_name = os.path.basename(slice.filename)
        mouse_x = self.view_slice_widget.mouse_x
        mouse_y = self.view_slice_widget.mouse_y
        slice_pixel_value = slice.pixel_array[mouse_y, mouse_x]

        self.view_slice_data_widget.update_data(slice_file_name, slice_pixel_value, mouse_x, mouse_y)

    # Resets all UI controls to reflect a new cross sectional image
    def reset_controls(self):
        # Block signals for all relevant widgets so that they don't trigger each other when their values are changed
        self.view_slice_adjuster.blockSignals(True)
        self.superior_slice_adjuster.blockSignals(True)
        self.inferior_slice_adjuster.blockSignals(True)
        self.slice_scale_adjuster.blockSignals(True)
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

        self.slice_scale_adjuster.setValue(self.cross_sectional_image.landmark_scale_factor)

        self.landmark_position_adjuster.set_bounds((slice_shape[1], slice_shape[0]))
        self.landmark_position_adjuster.set_coords((0, 0))
        self.landmark_select_combo_box.setCurrentIndex(8)

        # Unblock all signals after resetting controls
        self.view_slice_adjuster.blockSignals(False)
        self.superior_slice_adjuster.blockSignals(False)
        self.inferior_slice_adjuster.blockSignals(False)
        self.slice_scale_adjuster.blockSignals(False)
        self.landmark_position_adjuster.blockSignals(False)


