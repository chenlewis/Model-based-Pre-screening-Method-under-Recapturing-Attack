# Model-based-Pre-screening-Method-under-Recapturing-Attack
<!-- Resource for "A Distortion Model-based Pre-screening Method for Document Image Tampering Localization under Recapturing Attack" -->
This is the implementation of the method proposed in "A Distortion Model-based Pre-screening Method for Document Image Tampering Localization under Recapturing Attack" with pytorch(1.10, gpu version). The associated datasets are available upon request.

## File Description
`correlate.py`

Calculate the correlation coefficient of the CMYK color channels between the image to be tested and the reference image.
    
`svm.m`

Training and testing of the SVM classifier.

## How to reproduce
1. Run `correlate.py` to get the correlation coefficients of CMYK color channels between the image block to be measured and the reference image. It will generate the corresponding CSV file.
2. Import the data of the CSV file into the `svm.m` for training and testing of the SVM classifier.

## Citation
If you use our code or dataset, please cite:
```
@article{CHEN2022108666,
title = {A distortion model-based pre-screening method for document image tampering localization under recapturing attack},
journal = {Signal Processing},
volume = {200},
pages = {108666},
year = {2022},
issn = {0165-1684},
doi = {https://doi.org/10.1016/j.sigpro.2022.108666},
url = {https://www.sciencedirect.com/science/article/pii/S0165168422002055},
author = {Changsheng Chen and Lin Zhao and Jiabin Yan and Haodong Li}
}
```
