#This file should be the entry point for the CT image annotation tool
from PyQt5.QtWidgets import QApplication
from AnnotationWidget import AnnotationWidget


def main():
    app = QApplication([])
    app.setApplicationName("DICOM Image Annotation Tool")
    window = AnnotationWidget()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()