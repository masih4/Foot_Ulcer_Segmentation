# Foot_Ulcer_Segmentation
Contains the prediction code for Foot Ulcer Segmentation Challenge at MICCAI 2021

[![MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/shunk031/chainer-skin-lesion-detector/blob/master/LICENSE)
![](https://img.shields.io/badge/keras-tensorflow-blue.svg)

## Method



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

## Results
To derive the results, we used the Medetec foot ulcer dataset [1] for pre-training. Then we used the training set of the MICCAI 2021 Foot Ulcer Segmentation Challenge dataset [2] (810 images) as the training and validation set. The reported results in the following table are based on the test set of the Foot Ulcer Segmentation dataset (200 images).

Segmentation results on the Kumar dataset:
| Model                             | Image base Dice (%)  | Dataset based Dice (%)  | Dataset based IOU (%)      |
| --------------------------------  |:--------------------:|:-----------------------:|:--------------------------:|
| VGG16  [1]                        |         -            |   81.03                 |   -                        |
| SegNet [1]                        |         -            |   85.05                 |   -                        |
| U-Net [1]                         |         -            |   90.15                 |   -                        |
| Mask-RCNN  [1]                    |         -            |   90.20                 |   -                        |
| MobileNetV2 [1]                   |         -            |   90.30                 |   -                        |
| MobileNetV2 + pp [1]              |         -            |   90.47                 |   -                        |
| EfficientNet2 U-Net (this work)   |         84.09        |   91.90                 |  85.01                     |
| EfficientNet1 LinkNet (this work) |         83.93        |   92.09                 |  85.35                     |
| Ensemble U-Net LinkNet (this work)|         84.42        |   92.07                 |  85.51                     |



## Reference (dataset and benchmark)
[1] Thomas, S. Stock pictures of wounds. Medetec Wound Database (2020). http://www.medetec.co.uk/files/medetec-image-databases.html

[2] Wang, C., Anisuzzaman, D.M., Williamson, V. et al. Fully automatic wound segmentation with deep convolutional neural networks. Sci Rep 10, 21897 (2020). https://doi.org/10.1038/s41598-020-78799-w
