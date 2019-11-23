from PyQt5.QtWidgets import QLabel, QSpinBox, QDial, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.Qt import *

class XYSpinnerComboWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, name="", default_coords=(0, 0), bounds=(100,100), parent=None):
        QWidget.__init__(self, parent=parent)

        self.bounds = bounds
        self.cur_coords = default_coords

        self.title_label = QLabel(name)
        self.title_label.setAlignment(Qt.AlignHCenter)
        self.x_label = QLabel("X:")
        self.y_label = QLabel("Y:")

        self.x_spinner = QSpinBox(self)
        self.x_spinner.setMinimum(0)
        self.x_spinner.setMaximum(bounds[0])
        self.x_spinner.setValue(self.cur_coords[0])
        self.x_spinner.valueChanged.connect(self.on_value_changed)

        self.y_spinner = QSpinBox(self)
        self.y_spinner.setMinimum(0)
        self.y_spinner.setMaximum(bounds[1])
        self.y_spinner.setValue(self.cur_coords[1])
        self.y_spinner.valueChanged.connect(self.on_value_changed)

        self.setup_gui()

    def setup_gui(self):
        vertical_layout = QVBoxLayout(self)
        horizontal_layout = QHBoxLayout(self)

        horizontal_layout.addWidget(self.x_label)
        horizontal_layout.addWidget(self.x_spinner)
        horizontal_layout.addWidget(self.y_label)
        horizontal_layout.addWidget(self.y_spinner)

        vertical_layout.addWidget(self.title_label)
        vertical_layout.addLayout(horizontal_layout)

    @pyqtSlot()
    def on_value_changed(self):
        self.cur_coords = (self.x_spinner.value(), self.y_spinner.value())

        self.value_changed.emit()

    def set_bounds(self, bounds):
        if bounds[0] < 0 or bounds[1] < 0:
            pass

        self.x_spinner.blockSignals(True)
        self.y_spinner.blockSignals(True)

        self.x_spinner.setMaximum(bounds[0])
        self.y_spinner.setMaximum(bounds[1])

        self.x_spinner.blockSignals(False)
        self.y_spinner.blockSignals(False)

        self.value_changed.emit()

    def set_coords(self, coords):
        self.cur_coords = coords

        self.x_spinner.blockSignals(True)
        self.y_spinner.blockSignals(True)

        self.x_spinner.setValue(coords[0])
        self.y_spinner.setValue(coords[1])

        self.x_spinner.blockSignals(False)
        self.y_spinner.blockSignals(False)

        self.value_changed.emit()


