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


### 3.2 Project Role

팀원: 임준
담당: 차선인식, 제어

## 4. Project Review
### 가. 개요
저의 미션은 차선인식과 함께 차선인식 노드에서 구한 차선인식 정보를 메시지에 담아 제어 노드에 전달하는 것입니다. 뿐만 아니라 제어노드에서 최적의 제어값(속도, 조향)을 찾아 모터 노드로 메시지를 보내는 역할을 하였습니다.

차선 인식의 알고리즘은 다음과 같습니다. 학습했던 houghlinesp 함수를 이용해 hough line을 만들어주고 roi 상에서 4개의 좌표의 평균인 값을 차선의 좌표값으로 설정하였습니다. 좌측, 우측 차선 좌표값의 center 좌표 값과 자이카 자체의 center값의 차이 값을 구하여 좌방향/우방향을 결정할 수 있습니다. 차선 인식 부분에서 차선 정보(lpos, rpos)를 제어 노드에 넘겨줌으로써 임무를 완료하는 것입니다. 
제어의 알고리즘은 다음과 같습니다. 차선인식 노드로부터 받은 메시지를 활용해 적절한 조향값과 speed값을 설정하여 필요하다면 적절한 제어기법(moving avg, PID)을 활용하는 임무입니다.
 
### 나. 진행 중 문제와 해결방법
1. 차선 인식:
 1) 노이즈가 심해서 차선이 아닌 지면의 스크래치를 차선으로 오인식하는 문제
  -> blur , erode 기법을 통해 해결하였습니다. (blur를 한 번 적용한 것과 세 번 적용한 것에 노이즈 제거에 대한 차이가 있기에, 세 번 적용하였고 노이즈 문제는 해결했지만, 조그마한 noise 또한 제거 목적으로 erode를 적용하였습니다)


이미지 전처리 전


![image](https://user-images.githubusercontent.com/76178551/216765415-25868638-0ee7-48e8-ab19-7a570829c0d4.png)


이미지 전처리 후 (noise 제거)


![image](https://user-images.githubusercontent.com/76178551/216765417-442dc1f9-23d0-408d-896c-d2185c2d83fc.png)


 2) 차선과 색이 동일한 정지선을 차선으로 오인식하는 문제. 

y = ax + b에서 기울기 a와  y절편 b가 0이 아닌 모든 라인들에 대해서 left line과 right라인 즉 차선 라인 좌/우를 구분하였습니다. 이때의 문제는 정지선 또한 차선으로 인식하게 만드는 문제가 발생하였습니다.

정지선의 특성은 기울기가 차선에 비해 낮다는 것입니다. 기울기가 약 -0.2 ~ 0.2 정도 가지고 있으며 적용했던 hough line함수를 통해서 나온 직선의 기울기가 -0.2 ~ 0.2 라 하면 left/right line리스트에 정보를 넣지 않게 하여 문제를 해결하였습니다.

 

기울기 변화 전


![image](https://user-images.githubusercontent.com/76178551/216765427-77468f43-d75c-4b23-91cd-a1a0e0b73166.png)


기울기 변화 후


![image](https://user-images.githubusercontent.com/76178551/216765435-de25ff3e-aa56-406d-aacb-7afa700a8416.png)



미해결

 3) 차선 인식 상에서의 문제
왼쪽으로 가는 곡선 구간에서 차선이 아닌 벽의 하단 부분 검정색 영향을 받아 차선을 잡는 line이 튀는 경향이 있었습니다. roi영역을 줄이면 특정 부분은 개선이 될 수 있겠지만, 전반적으로  houghlinesP함수를 사용해 sample 좌표값들의 개수도 줄어들어 차선을 제대로 인식하는 확률 또한 낮아지게 됩니다. 즉 차선 인식의 안전성에 영향이 가기에, 자이카를 이용해 여러 테스트를 하여 문제가 없으면 그대로 가고, 문제가 있다면 roi 영역을 줄여보는 테스트를 할 예정입니다.
-> 잠깐 차선 인식이 조금 벗어나기에 문제가 없다면 그대로 갈 예정입니다.

ROI Width 크기 변경 전/후 (1256번째 이미지 기준)

![image](https://user-images.githubusercontent.com/76178551/216765441-312610d8-3ab6-4ca6-bfc5-2b121f651bda.png)



 4) 갈림길 영역

미해결
ㄱ) 다른 구간의 차선을 인식하는 문제.

![image](https://user-images.githubusercontent.com/76178551/216765471-7e27c00c-3165-4a98-a458-cf6fe7966623.png)


  ㄴ) 표지판의 검정 영역을 인식하는 문제

![image](https://user-images.githubusercontent.com/76178551/216765477-28b6630b-6faa-405f-9c3e-e2b0e9d66a1c.png)


-> 상기에 언급한 벽 검정 부분을 인식하는 문제의 해결방안처럼 ROI 영역을 줄이거나 또는 파이썬 기능인 nonZero함수를 통해 기준이 되는 검정색의 값의 개수가 몇 개 이상이면 차선으로 인식하는 방법으로 해결해 볼 수 있을 것 같습니다. 

 nonZero 함수 이용 시 x값이 0부터가 아닌 Width/2부터 시작을 해야 위에 표지판 검정 부분을 오인식하는 문제가 없을 것입니다. 다만 전 구간의 상황을 감안해 적용해볼 예정입니다.

추가로, 검정선 차선이지만 어떤 경우 전등에 의해 연하게 색 값이 연하게 나오는 경우 차선 인식을 못하는 것 같습니다. 이에 임계값을 통한 이진화 작업을 검토하여 테스트할 수도 있을 것 같습니다.  

  ㄷ) 하나의 차선만 인식하는 문제 ( 이 때는 인식 못한 차선을 Width -1 값으로 보내기로 했습니다.)

![image](https://user-images.githubusercontent.com/76178551/216765498-81f7362e-1fdf-49e3-9ecb-76db9772ae98.png)


  ㄹ) 두 차선을 모두 인식하지 못하는 문제, 쭉 가다가 가야할 차선이 아닌 다른 차선으로 인식하면? 차선을 벗어 날 수 있습니다.

![image](https://user-images.githubusercontent.com/76178551/216765491-00aba63f-10f9-4d41-809b-37d29574bc63.png)


이 상황에서 어떻게 해야할지 고민중에 있습니다. 이전의 우측 곡선 주행시의 차선 정보를 적용하는 방법이 있을 수 있으며, 알고리즘을 적용 및 향후 기회가 되면 테스트 할 예정 입니다.


 5) 차선을 벗어나서 주행
자이카를 차선 인식과 제어 알고리즘 통해서 테스트했을 때, 우측 차선을 침범하여 주행하였으며, 차선 인식을 제대로 하며 주행하고 있는지 cv2.imshow 함수로 확인을 못하였습니다. 다만, 샘플 영상 몇몇 개를 통해 차선은 제대로 잡는 것을 확인했습니다. 그럼에도 불구하고 자이카 상황과 당시의 환경을 고려하여, 문제가 순수 차선 인식에서 일어나는 건지 제어단에 넘겨주는 조향 값에서 일어나는지 테스트하며 확인해 보아야 합니다.

2. 제어:
미 해결.
 1) PID: 지식이 부족하여 P, I, D에 대한 최적의 parameter 값을 찾지 못하였습니다.
 2) Moving Average: 곡선 주행 시 조향 값이 둔해지는 경향이 있어 실제 적용하지는 못했습니다. 그러나, Moving Average는 주행 시 안정감을 부여하기에 주행 시 필요한 존재라 생각합니다. 개선을 위해서 기존에 설정한 샘플 개수 50보다 적은 샘플 개수 값을 적용해 볼 필요가 있을 것 같습니다. 또한 속도를 높여가면서 주행 시 둔해지는 조향을 개선할 수도 있을 것 같습니다.
 3)  차선을 벗어나서 주행하는 부분이 제어의 문제라면 angle값을 조정해야 할 것입니다.

### 다.

회고: 자이카가 없을 때 USB Cam으로부터 실세계의 이미지를 받아오는 환경을 구성하는 것과 샘플 영상을 통해 이미지를 받아와서 여러 노드를 구성하여 launch파일로 돌려가며 알고리즘을 테스트 한 경험은 ros, linux, 이미지 처리, 파이썬에 대해 좀 더 알아간 유의미한 시간이었던 것 같습니다. 비록 우측 차선을 벗어나며 주행을 했지만, 전날 밤 찜질방에서 샘플 영상을 갖고 이전에 차선 인식 문제가 된 부분들을 해결 및 개선했던 경험은 지금도 짜릿한 경험으로 남아있습니다.

주어진 미션에 대해 회고하자면, 운에 따라 차선을 인식하여 처음에 잘 주행했지만, 환경이 바뀜에 따라, 차선 인식이 어려워지기 시작했습니다. 제 역할을 제대로 하지 못한 것 같아 팀원들에게도 미안한 감정을 가지고 있습니다. 기회가 되면, 완벽한 알고리즘으로 가게끔 노력하여 이전보다 더 나은 자율주행 자이카를 탄생시키고자 합니다.

 
