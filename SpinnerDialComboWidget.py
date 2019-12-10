'''
SpinnerDialComboWidget
Author: Luben Popov
This custom QT widget combines the QSpinBox and QDial into a single widget with the functionality of both.
'''

from PyQt5.QtWidgets import QLabel, QSpinBox, QDial, QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSlot, pyqtSignal

class SpinnerDialComboWidget(QWidget):
    value_changed = pyqtSignal()

    # name: The string name that will be displayed on top of the widget
    # default_value: The value that will be initially set as the widget's value
    # min_val: The minimum value that will be initially set
    # max_val: The maximum value that will be initially set
    def __init__(self, name="", default_value=0, min_val=0, max_val=100, parent=None):
        QWidget.__init__(self, parent=parent)

        # The minimum value that can be set
        self.min_val = min_val

        # The maximum value that can be set
        self.max_val = max_val

        # The widget's current value
        self.value = default_value

        self.title_label = QLabel(name)

        # The widget's dial
        self.dial = QDial(self)
        self.dial.setSingleStep(1)
        self.dial.setPageStep(1)
        self.dial.setMinimum(min_val)
        self.dial.setMaximum(max_val)
        self.dial.setValue(default_value)
        self.dial.valueChanged.connect(self.on_dial_changed)

        # The widget's spin box
        self.spinner = QSpinBox(self)
        self.spinner.setMinimum(min_val)
        self.spinner.setMaximum(max_val)
        self.spinner.setValue(default_value)
        self.spinner.valueChanged.connect(self.on_spinner_changed)

        self.setup_gui()

    # Sets up the positioning of the UI elements
    def setup_gui(self):
        vertical_layout = QVBoxLayout(self)

        vertical_layout.addStretch(1)
        vertical_layout.addWidget(self.title_label)
        vertical_layout.addWidget(self.spinner)
        vertical_layout.addWidget(self.dial)

    # The callback for when the dial is changes
    @pyqtSlot()
    def on_dial_changed(self):
        self.value = self.dial.value()

        self.spinner.blockSignals(True)

        self.spinner.setValue(self.dial.value())

        self.spinner.blockSignals(False)

        self.value_changed.emit()

    # The callback for when the spin box is changed
    @pyqtSlot()
    def on_spinner_changed(self):
        self.value = self.spinner.value()

        self.dial.blockSignals(True)

        self.dial.setValue(self.spinner.value())

        self.dial.blockSignals(False)

        self.value_changed.emit()

    # Sets the minimum value
    # new_min: The new minimum value to be set
    def set_min(self, new_min):
        if new_min > self.max_val:
            return

        self.min_val = new_min

        self.dial.blockSignals(True)
        self.spinner.blockSignals(True)

        self.spinner.setMinimum(new_min)
        self.dial.setMinimum(new_min)

        self.dial.blockSignals(False)
        self.spinner.blockSignals(False)

        self.value_changed.emit()

    # Sets the maximum value
    # new_max: The new maximum value to be set
    def set_max(self, new_max):
        if new_max < self.min_val:
            return

        self.max_val = new_max

        self.dial.blockSignals(True)
        self.spinner.blockSignals(True)

        self.spinner.setMaximum(new_max)
        self.dial.setMaximum(new_max)

        self.dial.blockSignals(False)
        self.spinner.blockSignals(False)

        self.value_changed.emit()

    # Sets the widget value
    # value: The value to be set
    def set_value(self, value):
        self.value = value

        self.dial.blockSignals(True)
        self.spinner.blockSignals(True)

        self.dial.setValue(value)
        self.spinner.setValue(value)

        self.dial.blockSignals(False)
        self.spinner.blockSignals(False)

        self.value_changed.emit()


