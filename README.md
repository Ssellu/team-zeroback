# ZEROBACK
Object Detection and ROS Control


## 1. Training Dataset
single stage object detection Yolov3.

This is made with Pytorch.


<img src=https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-24_at_12.52.19_PM_awcwYBa.png width=416>

----------------------------

## 1.1.  Install Required Framework And Libraries

### 1.1.1 For Windows

#### Use Anaconda

1. Download Anaconda : https://www.anaconda.com/products/individual#windows
2. In terminal,
   ```bash
   $ conda create --name ${environment_name} python=3.8

   $ activate ${environment_name}

   $ git clone https://github.com/2damin/yolov3-pytorch.git
   ```

### 1.1.2. For Linux

#### Use docker

We recommend **nvidia NGC docker image**. [Download](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
```bash
$ docker pull nvcr.io/nvidia/pytorch:xx.xx-py3

# Run and check "nvidia-smi" and "nvcc --version"
$ docker run --gpus all -it --rm -v local_dir:container_dir -p 8888:8888 nvcr.io/nvidia/pytorch:xx.xx-py3

$ git clone git@github.com:Ssellu/team-zeroback.git
```



## 1.2. Dependencies

```bash
$ pip install -r requirements.txt
```
```
python >= 3.6

Numpy

torch >= 1.9

torchvision >= 0.10

tensorboard

tensorboardX

torchsummary

pynvml

imgaug

onnx

onnxruntime
```

-------------------

## 1.3. Run

To train,

```bash
# single gpu
$ python main.py --mode train --cfg ./yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}

# multi gpu
python main.py --mode train --cfg ./yolov3.cfg --gpus 0 1 2 3 --checkpoint ${saved_checkpoint_path}
```

To evaluate,

```{r, engine='bash', count_lines}
python main.py --mode eval --cfg ./yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

To test,

```{r, engine='bash', count_lines}
$ python main.py --mode demo --cfg ./yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

For converting torch to onnx,

target tensorrt version > 7
```{r, engine='bash', count_lines}
python main.py --mode onnx --cfg ./cfg/yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

target tensorrt version is 5.x

1. **ONNX_EXPORT = True** in 'model/yolov3.py'

   tensorrt(v5.x) is not support upsample scale factor, so you have to change upsample layer not using scale factor.

```{r, engine='bash', count_lines}
python main.py --mode onnx --cfg ./cfg/yolov3.cfg --gpus 0 --checkpoint ${saved_checkpoint_path}
```

### option

--mode : train/eval/demo.

--cfg : the path of model.cfg.

--gpu : if you use GPU, set 1. If you use CPU, set 0.

--checkpoint (optional) : the path of saved model checkpoint. Use it when you want to load the previous train, or you want to test(evaluate) the model.

--pretrained (optional) : the path of darknet pretrained weights. Use it when you want to fine-tuning the model.



## 1.4. Visualize Training Graph

Using Tensorboard,

```{r, engine='bash', count_lines}
tensorboard --logdir=./output --port 8888
```

-------------------------



## 2. Download Object Datasets
TODO - URL GOOGLE DRIVE (OUR LABELLED DATASET)

## 3. Project Overview
### 3.1 Project Structure
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