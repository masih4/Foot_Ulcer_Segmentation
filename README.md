[![MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://github.com/shunk031/chainer-skin-lesion-detector/blob/master/LICENSE)
![](https://img.shields.io/badge/keras-tensorflow-blue.svg)

# Foot_Ulcer_Segmentation
Contains the prediction codes for our submission to the Foot Ulcer Segmentation Challenge at MICCAI 2021 that placed us in the **1st rank** in the challenge legacy leaderboard.

# Citation
If you find the contents of this repository useful or use the provided codes, please cite our arXiv preprint.
https://arxiv.org/abs/2109.01408

BibTex entry:
```
@article{UlcerSegMahod21,
author = "Mahbod, Amirreza and Ecker, Rupert and Ellinger, Isabella",
journal = "arXiv preprint arXiv:2109.01408",
title = "Automatic Foot Ulcer segmentation Using an Ensemble of Convolutional Neural Networks",
year = "2021"
}
```


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
1- download the saved models from Google Drive: https://drive.google.com/drive/folders/1lprFVD--hzFLglXpe8di0CkqSXRK2DFO?usp=sharing and place them in `saved_models` folder

2- put the test images inside the `test_images` folder (already included 200 test images from: https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge/test/images but you could add/remove images)

3- build the Docker environment
```
docker build -f Dockerfile -t fuseg2021_amirreza_mahbod_medicaluniversityofvienna .
```
or download the built image from:https://drive.google.com/file/d/1K4j9gXKzmLfLAe-LjAhRLIi1Vz-_ghkp/view?usp=sharing
and extract the `.tar` file.
```
docker load --input fuseg2021_amirreza_mahbod_medicaluniversityofvienna.tar
```
4- run the container
```
docker run --gpus all -v /home/masih/Desktop/wound_docker/results/:/src/results/ -ti fuseg2021_amirreza_mahbod_medicaluniversityofvienna /bin/bash
```
note: you need to chenge `/home/masih/Desktop/wound_docker/results/` to the path that you want to save the results on your local system

5- run the following commands inside the container:
```
$ cd src/code
$ python3 test.py 
```
6- final results will be saved inside `results/final` folder

## Results
To derive the results in the following table, we used the Medetec foot ulcer dataset [1] for pre-training. Then we used the training set of the MICCAI 2021 Foot Ulcer Segmentation Challenge dataset [2] (810 images) as the training set. The reported results in the following table are based on the validation set of the Foot Ulcer Segmentation dataset (200 images). For the challenge submssion, we used the entire 1010 images of the train and validation set to train our models. 

| Model                             | Image-based Dice (%) | Precision (%)           | Recall (%)              | Dataset-based IOU (%)      | Dataset-based Dice (%)     |
| --------------------------------  |:--------------------:|:-----------------------:|:-----------------------:|:--------------------------:|:--------------------------:|
| VGG16  [2]                        |         -            |       83.91             |   78.35                 |    -                       | 81.03                      |
| SegNet [2]                        |         -            |       83.66             |   86.49                 |    -                       | 85.05                      |
| U-Net [2]                         |         -            |       89.04             |   91.29                 |    -                       | 90.15                      |
| Mask-RCNN  [2]                    |         -            |       94.30             |   86.40                 |    -                       | 90.20                      |
| MobileNetV2 [2]                   |         -            |       90.86             |   89.76                 |    -                       | 90.30                      |
| MobileNetV2 + pp [2]              |         -            |       91.01             |   89.97                 |    -                       | 90.47                      |
| EfficientNet1 LinkNet (this work) |         83.93        |       92.88             |   91.33                 |    85.35                   | 92.09                      |
| EfficientNet2 U-Net (this work)   |         84.09        |       92.23             |   91.57                 |    85.01                   | 91.90                      |
| Ensemble U-Net LinkNet (this work)|         84.42        |       92.68             |   91.80                 |    85.51                   | 92.07                      |

## Contact
Amirreza Mahbod (Medical University of Vienna) 

Email: amirreza.mahbod@meduniwien.ac.at



## Reference (dataset and benchmark)
[1] Thomas, S. Stock pictures of wounds. Medetec Wound Database (2020). http://www.medetec.co.uk/files/medetec-image-databases.html

[2] Wang, C., Anisuzzaman, D.M., Williamson, V. et al. Fully automatic wound segmentation with deep convolutional neural networks. Sci Rep 10, 21897 (2020). https://doi.org/10.1038/s41598-020-78799-w
