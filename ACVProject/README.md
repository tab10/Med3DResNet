# Cardiovascular risk computed via Deep Learning (DL) on thoracic CT scans (Med3DResNet)

### Timothy Burt, Luben Popov, Yuan Zi
#### COSC 7373 Adv. Computer Vision Fall 2019 Team 1
##### University of Houston

## Project Goal
+ Design, build, train, and test a deep learning pipeline for feature detection using thoracic CT scans of patients’ hearts
  + Obtain thoracic CT scans of patients
  + Design and implement an annotation tool to annotate crop landmarks for the CT scans
  + Implement a normalization pipeline to normalize the CT scans into 3D arrays
  + Design and implement a CNN that performs feature selection on the normalized arrays
 
### Project Pipeline
![image](ACVProject/images/project_pipeline.png)

## Setup/Evaluation
[See docs folder for a manual with usage and a tutorial.](ACVProject/documentation/SetupTutorial.pdf)

### Annotation GUI Tool
GUI tool to annotate DICOM images and batch export.

To run, type `python CTImageAnnotationTool.py`

[See docs folder for a manual with usage and a tutorial.](ACVProject/documentation/CTImageAnnotationToolManual.pdf)

### Med3DResNet Training/Testing CLI Tool
To Run, type `python CNN_main.py **kwargs`

[See docs folder for a manual with usage and a tutorial.](ACVProject/documentation/Med3DResNetManual.pdf)

## Contributions

#### Luben Popov
* About half the final presentation slides (3/8)
* Annotation Tool Manual/Tutorial
* Code
  * AnnotationWidget.py
  * CTImageAnnotationTool.py
  * CTImageAnnotationToolMainWindow.py
  * DataExport.py
  * DataNormalization.py (with Yuan's help)
  * DataReader.py
  * DICOMCrossSectionalImage.py
  * HeartLandmarks.py
  * MathUtil.py
  * SpinnerDialComboWidget.py
  * ViewSliceDataWidget.py
  * ViewSliceWidget.py
  * XYSpinnerComboWidget.py
  
#### Tim Burt
* About half the final presentation slides (4/8)
* Med3DResNet Manual/Tutorial
* Code
    * CNN_batch_run.py
    * CNN_main.py
    * CNN_ops.py
    * CNN_ResNet.py
    * CNN_utils.py
    * Visualization.py (includes most plots/videos)
    * README.md
    
#### Yuan Zi
* Some of the final presentation slides (1/8)
* Code
  * preprocess.py (some preprocessing images, mainly integrated into Luben's GUI)
  * Docker Container Setup and Build
  
  



