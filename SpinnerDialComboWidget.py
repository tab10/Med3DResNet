from PyQt5.QtWidgets import QLabel, QPushButton, QSpinBox, QDial, QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSlot, pyqtSignal

class SpinnerDialComboWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, name="", default_value=0, min_val=0, max_val=100, parent=None):
        QWidget.__init__(self, parent=parent)

        self.min_val = min_val
        self.max_val = max_val
        self.value = default_value

        self.title_label = QLabel(name)

        self.dial = QDial(self)
        self.dial.setSingleStep(1)
        self.dial.setPageStep(1)
        self.dial.setMinimum(min_val)
        self.dial.setMaximum(max_val)
        self.dial.setValue(default_value)
        self.dial.valueChanged.connect(self.on_dial_changed)

        self.spinner = QSpinBox(self)
        self.spinner.setMinimum(min_val)
        self.spinner.setMaximum(max_val)
        self.spinner.setValue(default_value)
        self.spinner.valueChanged.connect(self.on_spinner_changed)

        self.setup_gui()

    def setup_gui(self):
        vertical_layout = QVBoxLayout(self)

        vertical_layout.addStretch(1)
        vertical_layout.addWidget(self.title_label)
        vertical_layout.addWidget(self.spinner)
        vertical_layout.addWidget(self.dial)

    @pyqtSlot()
    def on_dial_changed(self):
        self.value = self.spinner.value()

        self.spinner.blockSignals(True)

        self.spinner.setValue(self.dial.value())

        self.spinner.blockSignals(False)

        self.value_changed.emit()

    @pyqtSlot()
    def on_spinner_changed(self):
        self.value = self.spinner.value()

        self.dial.blockSignals(True)

        self.dial.setValue(self.spinner.value())

        self.dial.blockSignals(False)

        self.value_changed.emit()

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


    def set_value(self, value):
        self.value = value

        self.dial.blockSignals(True)
        self.spinner.blockSignals(True)

        self.dial.setValue(value)
        self.spinner.setValue(value)

        self.dial.blockSignals(False)
        self.spinner.blockSignals(False)

        self.value_changed.emit()


