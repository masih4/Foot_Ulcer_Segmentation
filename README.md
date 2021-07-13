# Foot_Ulcer_Segmentation
Contains the prediction code for Foot Ulcer Segmentation Challenge at MICCAI 2021

[![MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/shunk031/chainer-skin-lesion-detector/blob/master/LICENSE)
![](https://img.shields.io/badge/keras-tensorflow-blue.svg)


## Directory structure
```
.
├── code
│   ├── test.py (main run file)
│   ├── gpu_setting.py
│   ├── metric.py
│   └── params_test.py (all configs shoud be set here)
│
├── saved_models
│   ├── linknet(linknet models should be downloaded from googledrive and placed in this folder)
│   └── unet (unet models should be downloaded from googledrive and placed in this folder)
│
├── test images (place all test images here (size 512x512 pixels))
│
└── results
    ├── temp
    └── final (final results will be saved here)
 
```

## How to use
1- download the saved modles from Google Drive: https://drive.google.com/drive/folders/1lprFVD--hzFLglXpe8di0CkqSXRK2DFO?usp=sharing

2- put the test images inside the `test_images` folder

3- set up the Docker enviroment

4- run the following command
```
$ python test.py 
```
5- final results will be saved inside `results/final` folder

## Reference (dataset and benchmark)
Wang, C., Anisuzzaman, D.M., Williamson, V. et al. Fully automatic wound segmentation with deep convolutional neural networks. Sci Rep 10, 21897 (2020). https://doi.org/10.1038/s41598-020-78799-w
