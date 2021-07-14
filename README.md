# Foot_Ulcer_Segmentation
Contains the prediction code for Foot Ulcer Segmentation Challenge at MICCAI 2021

[![MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/shunk031/chainer-skin-lesion-detector/blob/master/LICENSE)
![](https://img.shields.io/badge/keras-tensorflow-blue.svg)

## Method
![Project Image](https://github.com/masih4/Foot_Ulcer_Segmentation/blob/main/git_image/method.png)


## Directory structure
```
.
├── code
│   ├── test.py (main run file)
│   ├── gpu_setting.py
│   ├── metric.py
│   └── params_test.py (all configs are here)
│
├── saved_models
│   ├── linknet(LinkNet models should be downloaded from Google drive and placed in this folder)
│   └── unet (U-Net models should be downloaded from Google drive and placed in this folder)
│
├── test images (place all test images here (size 512x512 pixels))
│
└── results
    ├── temp
    └── final (final results will be saved here)
 
```

## How to use
1- download the saved models from Google Drive: https://drive.google.com/drive/folders/1lprFVD--hzFLglXpe8di0CkqSXRK2DFO?usp=sharing

2- put the test images inside the `test_images` folder (already included 200 test images from: https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge/test/images but you could add/remove images)

3- set up the Docker environment
```
docker build -f Dockerfile -t FUSeg2021_AmirrezaMahbod_MedicalUniversityofVienna .
docker run -v /home/masih/Desktop/wound_docker/results/:/src/results/ -ti FUSeg2021_AmirrezaMahbod_MedicalUniversityofVienna /bin/bash
```
note: you need to chenge `/home/masih/Desktop/wound_docker/results/` to the path that you want to save the results on your local system

4- run the following commands inside the container:
```
$ cd src/code
$ python3 test.py 
```
5- final results will be saved inside `results/final` folder

## Results
To derive the results in the following table, we used the Medetec foot ulcer dataset [1] for pre-training. Then we used the training set of the MICCAI 2021 Foot Ulcer Segmentation Challenge dataset [2] (810 images) as the training set. The reported results in the following table are based on the validation set of the Foot Ulcer Segmentation dataset (200 images). For the challenge submssion, we used the entire 1010 images of the train and validation set to train our models. 

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
