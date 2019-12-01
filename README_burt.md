# Prediction of atherosclerosis risk with unlabeled thoracic CT scans:  deep learning vs. Agatston method

## Timothy Burt, Luben Popov, Yuan Zi

### Description

### Annotation GUI
TODO: change annotation.txt to output annotation_ID*.txt with * the patient index (e.g. LIDC-0001 ->  1)

### Data preprocessing / normalization

Lipid-rich plaque: 47+-29 HU (range 18-76)
Fibrous (calcified) plaque: 86+-29 HU (range 57-115)
CT's with or without lumen-enhancing contrast show no statistical significance on these values.
(A Meta Analysis and Hierarchical Classification of HU-Based Atherosclerotic Plaque Characterization Criteria)
### 3D CNN Architecture
5 CNN classes:
0=Unknown
1=benign or non-malignant disease
2=malignant, primary lung cancer
3 = malignant metastatic

#### Algorithm
1. Feed 256x256 images with normalized intensity into  

#### Training/Validation
We will perform training on 200 CT scans.
80/20 training/testing split (160 train, 40 test)
### Results
TODO: mention correlation between lung cancer/smoking and heart disease as maybe bias
    how to discard patients under 40 years old (this test is almost always negative)
### Tutorial
TODO: build docker file
make youtube video

#### References
https://www.health.harvard.edu/heart-health/when-you-look-for-cancer-you-might-find-heart-disease
CT heart anatomy reference: https://www.youtube.com/watch?v=4pjkCFrcysk&t=216s
