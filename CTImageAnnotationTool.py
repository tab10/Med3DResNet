'''
CTImageAnnotationTool
Author: Luben Popov
This serves as the entry point for the annotation tool, and should be run in order to open it.
'''

from PyQt5.QtWidgets import QApplication
from CTImageAnnotationToolMainWindow import CTImageAnnotationToolMainWindow


def main():
    app = QApplication([])
    app.setApplicationName("DICOM Image Annotation Tool")
    window = CTImageAnnotationToolMainWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()