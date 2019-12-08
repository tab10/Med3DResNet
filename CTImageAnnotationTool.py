#This file should be the entry point for the CT image annotation tool
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