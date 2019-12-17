'''
CTImageAnnotationToolMainWindow
Author: Luben Popov
This class is the QT MainWindow subclass that contains the AnnotationWidget for the annotation tool along with its menu
bars.
'''

from PyQt5.Qt import *
from ACVProject.AnnotationWidget import AnnotationWidget

class CTImageAnnotationToolMainWindow(QMainWindow):

    def __init__(self):
        super(CTImageAnnotationToolMainWindow, self).__init__()

        self.setWindowTitle("DICOM Image Annotation Tool")

        # The AnnotationWidget that drives most of the annotation tool's logic
        annotation_widget = AnnotationWidget(self)

        # Used to open a new directory of DICOM files for annotation
        select_image_directory_action = QAction("Select Image Directory", self)
        select_image_directory_action.triggered.connect(annotation_widget.on_image_directory_button_clicked)
        select_image_directory_action.setToolTip("Select a directory of DICOM images to annotate")

        # Used to import existing annotations
        import_annotations_action = QAction("Import Annotations", self)
        import_annotations_action.triggered.connect(annotation_widget.on_import_annotations_button_clicked)
        import_annotations_action.setToolTip("Select an existing annotation file to open its image data and modify")

        # Used to export current annotations
        export_annotations_action = QAction("Export Annotations", self)
        export_annotations_action.triggered.connect(annotation_widget.on_export_annotations_button_clicked)
        export_annotations_action.setToolTip("Export the current annotations")

        # Used to batch export 3D array files from a directory of annotation files
        batch_export_arrays_action = QAction("Batch Export Annotations to 3D Arrays", self)
        batch_export_arrays_action.triggered.connect(annotation_widget.on_batch_export_3d_arrays_button_clicked)
        batch_export_arrays_action.setToolTip("Select a directory of annotation files to export to normalized 3D "
                                              "arrays")
        # Used to batch export masked 3D array files from a directory of existing 3D array files
        batch_mask_arrays_action = QAction("Batch Mask 3D Arrays", self)
        batch_mask_arrays_action.triggered.connect(annotation_widget.on_batch_mask_3d_arrays_button_clicked)
        batch_mask_arrays_action.setToolTip("Select a directory of 3D array files to apply lung masking to")

        # Used to reset the positions of all heart landmarks
        reset_landmarks_action = QAction("Reset Landmarks", self)
        reset_landmarks_action.triggered.connect(annotation_widget.on_reset_landmarks_button_clicked)
        reset_landmarks_action.setToolTip("Reset the current annotated heart landmarks to their default positions")

        # Used to reverse the order of slices in the AnnotationWidget's current cross sectional image
        reverse_slices_action = QAction("Reverse Slices", self)
        reverse_slices_action.triggered.connect(annotation_widget.on_reverse_slices_button_clicked)
        reverse_slices_action.setToolTip("Reverse the order of slices in the currently open image data")

        # Register all menu bar actions into the window's menu bar
        self.statusBar()

        main_menu = self.menuBar()

        file_menu = main_menu.addMenu('&File')
        file_menu.setToolTipsVisible(True)
        file_menu.addAction(select_image_directory_action)
        file_menu.addAction(import_annotations_action)
        file_menu.addAction(export_annotations_action)
        file_menu.addAction(batch_export_arrays_action)
        file_menu.addAction(batch_mask_arrays_action)

        edit_menu = main_menu.addMenu('&Edit')
        edit_menu.setToolTipsVisible(True)
        edit_menu.addAction(reset_landmarks_action)
        edit_menu.addAction(reverse_slices_action)

        # Set the AnnotationWidget as the window's central widget
        self.setCentralWidget(annotation_widget)
