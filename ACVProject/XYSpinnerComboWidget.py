'''
SpinnerDialComboWidget
Author: Luben Popov
This custom QT widget combines two QSpinBox widgets to form an adjustment widget for XY coordinates. This widget only
supports coordinates where X >= 0 and Y >= 0.
'''

from PyQt5.Qt import *

class XYSpinnerComboWidget(QWidget):
    value_changed = pyqtSignal()

    # name: The string name that will be displayed on top of the widget
    # default_coords: the default coordinates that will be set in (x, y) format
    # bounds: the default maximum x and y values that will be set in (x, y) format
    def __init__(self, name="", default_coords=[0, 0], bounds=(100,100), parent=None):
        QWidget.__init__(self, parent=parent)

        # The maximum x and y values in (x, y) format
        self.bounds = bounds

        # The current x and y values in (x, y) format
        self.cur_coords = default_coords

        self.title_label = QLabel(name)
        self.title_label.setAlignment(Qt.AlignHCenter)
        self.x_label = QLabel("X:")
        self.y_label = QLabel("Y:")

        # Adjusts and displays the x value
        self.x_spinner = QSpinBox(self)
        self.x_spinner.setMinimum(0)
        self.x_spinner.setMaximum(bounds[0])
        self.x_spinner.setValue(self.cur_coords[0])
        self.x_spinner.valueChanged.connect(self.on_value_changed)

        # Adjusts and displays the y value
        self.y_spinner = QSpinBox(self)
        self.y_spinner.setMinimum(0)
        self.y_spinner.setMaximum(bounds[1])
        self.y_spinner.setValue(self.cur_coords[1])
        self.y_spinner.valueChanged.connect(self.on_value_changed)

        self.setup_gui()

    # Sets up the positioning of the UI elements
    def setup_gui(self):
        vertical_layout = QVBoxLayout(self)
        horizontal_layout = QHBoxLayout(self)

        horizontal_layout.addStretch(1)
        horizontal_layout.addWidget(self.x_label)
        horizontal_layout.addWidget(self.x_spinner)
        horizontal_layout.addWidget(self.y_label)
        horizontal_layout.addWidget(self.y_spinner)
        horizontal_layout.addStretch(1)

        vertical_layout.addWidget(self.title_label)
        vertical_layout.addLayout(horizontal_layout)

    # The callback for either the x or y value adjustment widgets being changed
    @pyqtSlot()
    def on_value_changed(self):
        self.cur_coords = [self.x_spinner.value(), self.y_spinner.value()]

        self.value_changed.emit()

    # Sets new maximum x and y values
    # bounds: The new maximum x and y values in (x, y) format
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

    # Sets new x and y values
    # coords: The new x and y values in (x, y) format
    def set_coords(self, coords):
        self.cur_coords = coords

        self.x_spinner.blockSignals(True)
        self.y_spinner.blockSignals(True)

        self.x_spinner.setValue(coords[0])
        self.y_spinner.setValue(coords[1])

        self.x_spinner.blockSignals(False)
        self.y_spinner.blockSignals(False)

        self.value_changed.emit()


