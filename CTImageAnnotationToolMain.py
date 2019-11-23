#This file should be the entry point for the CT image annotation tool
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog
from DICOMAnnotationWidget import DICOMAnnotationWidget


def main():
    app = QApplication([])
    app.setApplicationName("DICOM Image Annotation Tool")
    window = DICOMAnnotationWidget()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()