# ZEROBACK
Object Detection and ROS Control

## 1. Prepare the Dataset
### 1.1 Download Object Datasets
TODO - URL GOOGLE DRIVE (OUR LABELLED DATASET)
### 1.2 Project Structure
The directory structures like below.
```
.$(ROOT)
├─dataset
│  ├─.labeling_data_rt
│  │  ├─labeling_ignore_data
│  │  │  └─annotations
│  │  └─labeling_non_ignore_data
│  │      └─annotations
│  └─annotations
├─ros
│  └─launch
└─src
    ├─driving
    │  ├─control
    │  └─lane_detection
    ├─object_detection
    └─training
        ├─dataloader
        │  └─__pycache__
        ├─datasets
        │  ├─Annotations
        │  ├─ImageSets
        │  └─JPEGImages
        ├─demo
        │  └─__pycache__
        ├─eval
        │  └─__pycache__
        ├─model
        │  └─__pycache__
        ├─output
        │  └─Evaluation
        │      ├─AP
        │      │  ├─crosswalk
        │      │  ├─left
        │      │  ├─right
        │      │  ├─stop
        │      │  └─uturn
        │      ├─F1
        │      │  ├─crosswalk
        │      │  ├─left
        │      │  ├─right
        │      │  ├─stop
        │      │  └─uturn
        │      ├─Precision
        │      │  ├─crosswalk
        │      │  ├─left
        │      │  ├─right
        │      │  ├─stop
        │      │  └─uturn
        │      └─Recall
        │          ├─crosswalk
        │          ├─left
        │          ├─right
        │          ├─stop
        │          └─uturn
        ├─pretrain
        ├─train
        │  └─__pycache__
        ├─util
        │  └─__pycache__
        └─utils
```

### 1.3 Unzip And Move Dataset
Unzip the files and move all images to project dataset folder like this,
```bash
$ mv YOUR_DATASET_LOCATION/data_object_image_2/training/image_2 \ YOUR_PROJECT_ROOT_DIR/dataset/training/image_sets

$ mv YOUR_DATASET_LOCATION/data_object_image_2/testing/image_2 \  YOUR_PROJECT_ROOT_DIR/dataset/testing/image_sets
```

### 1.4 Test DataLoader
TODO -TWEAK HERE
```bash
$ python src/main.py --mode train --cfg yolov3_kitti.cfg
```

## Image Augmentation
**Image Augmentation** is converting images into a new, much larger amount of images slighty altered.
This model uses [imgaug](https://www.github.com/aleju/imgaug) opensource library. To download the library,
```bash
pip install imgaug
```
