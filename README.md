# Foot_Ulcer_Segmentation
Contains the code for Foot Ulcer Segmentation Challenge at MICCAI 2021


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


## Reference (dataset and benchmark)
Wang, C., Anisuzzaman, D.M., Williamson, V. et al. Fully automatic wound segmentation with deep convolutional neural networks. Sci Rep 10, 21897 (2020). https://doi.org/10.1038/s41598-020-78799-w
