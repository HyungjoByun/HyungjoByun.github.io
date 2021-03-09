---
title: "TorchVision을 사용한 Faster R-CNN(3)"
categories:
 - MachineLearning
tags:
 - Object Detection
 - Deep Learning
 - Faster RCNN

toc: true
toc_sticky: true
---
## 이전 게시물

1편: <https://hyungjobyun.github.io/machinelearning/FasterRCNN1/>  
2편: <https://hyungjobyun.github.io/machinelearning/FasterRCNN2/>  

## 모델 평가
이번 포스팅에서는 이전까지 만들었던 Faster R-CNN의 성능을 평가 해 보도록 하겠습니다.

## 평가 환경
**Model**

논문에서는 다양한 환경에서 평가를 하였지만 저는 그 중 아래와 같은 환경에서 평가하였습니다.

* Dataset: PASCAL VOC 2007 test set  
* RPN proposal: 300  
* RPN anchors: scale – {128,256,512}, aspect ratio – {0.5, 1.0, 2.0} (학습과 동일)  
* Box detections: 300  
* Box NMS: 0.7, 0.5  
* Box Score: 0.6, 0.3, 0  
* Image Size: 학습때와 같이 짧은 쪽을 600pixel로 aspect ratio를 유지하도록 설정(논문과 동일)  
> This model performs well when trained and tested using single-scale images and thus benefits running speed.-Faster RCNN-

**방법**

PASCAL VOC 평가 방법으로 측정했습니다. precision에 따른 recall 변화를 측정하여 Average Precision을 구한 후 항목별로 평균을 구해 mean Average Precision, 즉 mAP를 측정하였습니다.  
아래 링크의 Github를 이용했습니다.  
<https://github.com/Cartucho/mAP#create-the-ground-truth-files>

## 코드

### 기본 설정
```python
from google.colab import drive
drive.mount('/content/drive')
```

구글 드라이브에 연결하는 코드입니다.

```python
!git clone https://github.com/Cartucho/mAP.git
!pip install imgaug --upgrade
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree
import torchvision
import cv2
from imgaug import augmenters as iaa

if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')
```
필요한 라이브러리를 import합니다. 첫 줄은 mAP를 측정하는 github자료를 clone하기 위한 코드입니다. 나머지는 이전 게시물을 참고해 주세요.

```python
if not os.path.isfile("/content/voc_test_2007_tar"):
  !cp "/content/drive/MyDrive/Faster_RCNN/VOCtest_06-Nov-2007.tar" "/content/voc_test_2007_tar" 
  ! tar -xf "/content/voc_test_2007_tar"
```
Dataset을 복사합니다. PASCAL VOC 2007을 사용하였습니다.
```python
def xml_parser(xml_path):
  xml_path = xml_path


  xml = open(xml_path, "r")
  tree = Et.parse(xml)
  root = tree.getroot()

  size = root.find("size")

  file_name = root.find("filename").text

  object_name = []
  bbox = []

  objects = root.findall("object")
  for _object in objects:
      name = _object.find("name").text
      object_name.append(name)
      bndbox = _object.find("bndbox")
      one_bbox = []
      xmin = bndbox.find("xmin").text
      one_bbox.append(int(float(xmin)))
      ymin = bndbox.find("ymin").text
      one_bbox.append(int(float(ymin)))


      xmax = bndbox.find("xmax").text
      one_bbox.append(int(float(xmax)))
      ymax = bndbox.find("ymax").text
      one_bbox.append(int(float(ymax)))
      

      bbox.append(one_bbox)
  
  return file_name, object_name, bbox
```
Xml데이터를 읽는 코드입니다.

```python
def makeBox(voc_im,bbox,objects):
  image = voc_im.copy()
  for i in range(len(objects)):
    cv2.rectangle(image,(int(bbox[i][0]),int(bbox[i][1])),(int(bbox[i][2]),int(bbox[i][3])),color = (0,255,0),thickness = 1)
    cv2.putText(image, objects[i], (int(bbox[i][0]), int(bbox[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2) # 크기, 색, 굵기
  return image
```
이미지에 Box와 사물 이름을 그리기 위한 코드입니다.

```python
xml_list = os.listdir("/content/VOCdevkit/VOC2007/Annotations/")
xml_list.sort()
print(len(xml_list))
label_set = set()

for i in range(len(xml_list)):
  xml_path = "/content/VOCdevkit/VOC2007/Annotations/"+str(xml_list[i])
  file_name, object_name, bbox = xml_parser(xml_path)
  for name in object_name:
    label_set.add(name)

label_set = sorted(list(label_set))

label_dic = {}

label_dic['background'] = 0

for i, key in enumerate(label_set):
  label_dic[key] = (i+1)

print(label_dic)
```
라벨을 만드는 코드입니다. Torchvision의 FasterRCNN자체적으로 배경라벨 0번은 제거해 주지만 label_dic에 추가해 주었습니다.

### mAP 관련 함수
```python
def make_info_txt(file_name, object_name, bbox, mode, scores = None):
  if mode == 'gt':
    with open("/content/mAP/input/ground-truth/{}.txt".format(file_name[:-4]),"w") as f:
      for i in range(len(object_name)):
        f.write("{} ".format(object_name[i])+" ".join(map(str,map(int,bbox[i])))+"\n")
  
  if mode == 'rt':
    assert scores != None
    with open("/content/mAP/input/detection-results/{}.txt".format(file_name[:-4]),"w") as f:
      for i in range(len(object_name)):
        f.write("{} ".format(object_name[i])+"{} ".format(scores[i])+" ".join(map(str,map(int,bbox[i])))+"\n")
```
mAP측정을 위한 코드 사용을 위해 Ground truth와 Predict각각의 box와 사물이 적힌 파일을 만드는 코드입니다. 자세한 형식은 mAP라이브러리 Github를 방문하시길 바랍니다.
```python
def evaluation(xml_list, new = False):
  if new == True:
    print("Clear Directory")
    filelist = [ f for f in os.listdir("/content/mAP/input/ground-truth")]
    for f in filelist:
      os.remove(os.path.join("/content/mAP/input/ground-truth", f)) 
    
    filelist = [ f for f in os.listdir("/content/mAP/input/detection-results")]
    for f in filelist:
      os.remove(os.path.join("/content/mAP/input/detection-results", f)) 

    print("Evaluating")
    model.eval()
    for i in range(len(xml_list)):
      xml_path = "/content/VOCdevkit/VOC2007/Annotations/"+str(xml_list[i])
      file_name, object_name, bbox = xml_parser(xml_path)
      image_path = "/content/VOCdevkit/VOC2007/JPEGImages/"+str(file_name)

      test_image = Image.open(image_path).convert("RGB")
      test_image = np.array(test_image)
      resize = iaa.Resize({"shorter-side": 600, "longer-side": "keep-aspect-ratio"})
      to_tensor = torchvision.transforms.ToTensor()
      
      test_image,bbox = resize(image = test_image,bounding_boxes = np.array([bbox]))
      make_info_txt(file_name, object_name, bbox.squeeze(0),mode='gt')

      
      test_image = to_tensor(test_image).unsqueeze(0)
      predictions = model(test_image.to(device))
      object_name = []
      boxes = predictions[0]['boxes']
      labels = predictions[0]['labels']
      scores = predictions[0]['scores']
      for lb in labels:
        object_name.append([k for k, v in label_dic.items() if v == lb][0])
      
      make_info_txt(file_name, object_name, boxes, mode='rt', scores = scores)
  
  print("Result")
  f = os.popen("python /content/mAP/main.py --no-animation ") #openCV작동 안해서 --no-animation
  print(f.read())
  
```
실제적으로 mAP평가 코드를 실행하는 함수입니다. new = True이면 이전에 평가한 파일을 지우고 새로 만들도록 했습니다. 평가 과정에서는 make_info_txt를 사용하고 이미지는 s=600으로 고정합니다.
다만 일반 PC환경에서는 main.py를 실행하면 되지만 Google colab환경에서는 새로운 윈도우를 열지 못하므로 --no-animatioin옵션을 넣어주고 출력 결과만 보는 os.popen을 사용합니다.

### Model과 실행
```python
backbone = torchvision.models.vgg16(pretrained=True).features[:-1]
backbone_out = 512
backbone.out_channels = backbone_out

anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))

resolution = 7
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=resolution, sampling_ratio=2)

box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(in_channels= backbone_out*(resolution**2),representation_size=4096) 
box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(4096,21) #21개 class

model = torchvision.models.detection.FasterRCNN(backbone, num_classes=None,
                   min_size = 600, max_size = 1000,
                   rpn_anchor_generator=anchor_generator,
                   rpn_pre_nms_top_n_train = 6000, rpn_pre_nms_top_n_test = 6000,
                   rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
                   rpn_nms_thresh=0.7,rpn_fg_iou_thresh=0.7,  rpn_bg_iou_thresh=0.3,
                   rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                   box_roi_pool=roi_pooler, box_head = box_head, box_predictor = box_predictor,
                   box_score_thresh=0.6, box_nms_thresh=0.7,box_detections_per_img=300,
                   box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                   box_batch_size_per_image=128, box_positive_fraction=0.25
                 )

model.to(device)
model.load_state_dict(torch.load("/content/drive/MyDrive/Faster_RCNN/Epoch39.pth",map_location=device)) #epoch지정
```
모델을 불러옵니다. 이후 box_score_thresh, box_nms_thresh, box_detections_per_img 세가지 항목을 바꿔가며 측정합니다.

```python
evaluation(xml_list,new=True) #파일 만들어야하면 True, 아니면 False
```
측정을 실행하는 코드입니다.

## 실험

1편에서 설명했듯 RPN이 처음에 사물 탐지를 잘 하지 못할것을 예상하여 논문에 나온 반복 횟수의 2배보다 약간 작은 40epoch를 학습 시켰습니다.
결과는 최고 63.69%가 나왔고 학습을 반복 할수록 성능이 향상되는 추세여서 다음 실험에서는 100epoch를 시도해 보았습니다.

100epoch를 실험 한 결과 60epoch일 때 최고 60.59%가 측정되었고 이후 반복에서는 성능이 저하되었습니다. Overfitting이 의심되어 40epoch와
100epoch의 중간이고 최고 성능을 낸 60epoch와 유사한 70epoch로 다시 실험을 했습니다. 70epoch를 학습시킨 결과 40epoch만큼 학습 되었을 때 62.77%로 최고 성능을 보이고 이후 Overfitting되었습니다.

마지막으로 전체적으로는 40epoch를 학습 시키되 RPN의 예측 능력을 향상시킨 상태에서 학습을 시작하면 어떻게 될 지 알아봤습니다. 우선 backbone과 RPN만 parameter를 학습 되도록 하고
6epoch를 학습시켰습니다. 이후 Fast RCNN부분의 backbone을 제외한 부분을 4epoch학습시켰습니다. 마지막으로 전체를 30epoch학습 시켰습니다.
RPN과 Fast RCNN부분을 따로 학습 시킬 때는 논문의 learning rate를 사용했고 이후 30epoch는 기존 방식대로 cosine decay를 하며 학습 시켰습니다.
결과는 63.85%로 처음부터 전체를 학습 시킨 경우보다 높았습니다. 

## 평가 방법
model의 box_score_thresh와 box_nms_thresh를 바꿔가며 결과를 출력하도록 합니다. box_score_thresh는 논문의 셈플 이미지가
0.6을 사용하였으므로 0.6, 0.3, 0을 사용하여 측정하였고 box_nms_thresh는 학습 때 0.7을 사용하였으므로 0.7과 일반적인 값인 0.5로
측정 하였습니다. box_score_thresh = 0이고, box_nms_thresh = 0.5일 때 일반적으로 가장 mAP가 높게 나왔습니다. 그래서 해당 값으로 가장
mAP가 높은 Epoch를 찾은 후 해당 Epoch의 state_dict를 불러와 다른 box_score_thresh와 box_nms_thresh값을 적용하여 측정했습니다.

## 평가 결과

![평가 결과](\assets\images\result1.jpg)  
*<font size = 3pt>표1 [평가 결과] <br> 가장 높은 mAP가 측정된 state_dict를 불러와 측정한 값 </font>*

box_score_thresh가 작을수록 mAP가 증가합니다. 이유는 precision보다 recall의 값이 더 증가하기 때문일 것이라고 생각합니다.
box_nms_thresh는 0.7보다 0.5가 더 성능이 좋았습니다. 왜냐하면 mAP측정 방식에서 같은 물체에 대해 중복된 box가 있으면 가장 IoU가 높은것을 제외한
나머지는 잘못된 예측으로 계산하기 때문입니다. 하지만 box_nms_thresh 값을 작게 하면 서로 겹쳐있는 다른 물체에 대한 검출이 어려워집니다.
추가로 box_nms_thresh = 0.3일 때는 0.5일 때에 비해 큰 성능 향상이 없거나 오히려 나빠졌습니다.

Epoch별 mAP변화는 다음과 같습니다.

![Epoch별 결과](\assets\images\result2.png)  
*<font size = 3pt>표2 [Epoch별 결과] <br> 전체 Epoch를 다르게 할 때 Epoch별 결과 </font>*

RPN과 Fast RCNN 일부를 먼저 학습시키고 이후 동시에 학습시킨 결과는 box_score_thresh = 0, box_nms_thresh = 0.5일 때 63.85%로
측정 값 중 가장 높았습니다. 따라서 논문처럼 RPN과 Fast RCNN 전체를 각각의 모델로 훈련 시키면서 backbone을 동기화 한다면 더 많은 성능 향상이 가능할 것으로 예상됩니다.

## 논문과 차이 분석
![논문 결과](\assets\images\paper_result.jpg)  
*<font size = 3pt>그림1 [논문 결과] <br> source> Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks </font>*

논문에서 Alternative training으로 학습시키고 Pascal Voc 2007로 테스트한 결과는 그림1과 같습니다. 형광팬으로 표시된 부분이 저의 결과와
비교해 볼 만한 부분입니다. 저는 학습을 Pascal Voc 2012로만 하였기 때문에 정확한 비교는 어렵지만 수치만 비교하면 저의 결과가 6%정도 낮다고 볼 수 있습니다.
Alternative training을 하지 않은 점, 2007 dataset이 없이 학습을 한 점, torchvision라이브러리가
논문과 다를 수 있는점이 차이의 원인이라고 생각합니다.

## 결론

Faster RCNN은 Fast RCNN에서 Selective Search알고리즘이 물체를 제안하는 기능을 RPN이 하도록 하여 속도와 성능을 향상시켰습니다.  
논문에서는 Alternative training을 하였지만 Approximate joint training도 가능하다고 하여 저는 후자를 선택했습니다.  
mAP평가 결과는 40Epoch를 학습 시켰을 때 63.69%가 나왔고 본 학습 전 RPN과 Fast RCNN을 각각 학습시킨 후 전체를 학습하면 약간의 성능 향상이 있었습니다.(63.85%)  
학습 때 Pascal Voc 2007 trainset을 포함시키고 논문처럼 Alternative training을 하면 논문과 더 근접한 결과가 나올 것이라고 예상합니다.

## 실행 결과
![NMS 0.7](\assets\images\Voc_result7.png)  
논문과 같이 score threshold 0.6, NMS threshold 0.7을 사용했을 때 결과로 아래에 있는 이미지일수록 안좋은 결과입니다. NMS threshold가
0.7이므로 IoU가 0.7이상일 때 같은 물체로 인식합니다. 하나의 물체에 대해 box를 그리는 정밀도가 떨어져 여러 box가 겹치는 경우가 확인됩니다.

![NMS 0.7](\assets\images\Voc_result5.png) 
NMS threshold 0.5를 사용했을 때 결과입니다. 마찬가지로 아래에 있는 이미지 일수록 안좋은 결과입니다. NMS threshold가 0.7 일 때 보다는 깔끔한
결과가 나오지만 성능이 향상된것이 아닌 표시의 차이입니다. 만약 0.5보다 더 겹쳐있는 서로 다른 물체가 있으면 감지하지 못합니다.

## 후기

Faster RCNN논문을 읽을 때 세부적인 내용을 앞선 두 논문 RCNN과 Fast RCNN을 참고해야 해서 이해에 어려움이 있었는데 직접 만들어 보니 정리가 되었습니다.
다만 논문에서 box의 위치와 크기를 표시한 형식이나 Loss function 같이 세부적인것을 직접 구현하지 않아 아쉬움이 남습니다. 다음에 기회가 되면 직접 코드를
작성해 봐야 겠습니다.  
Object detection분야는 처음 해본 프로젝트라 새로운 것을 많이 익힐 수 있었습니다. 객관적인 성능면에서 부족하지만 어느 정도 정확하게 사물을 감지해 내어
개인적으로 만족스럽습니다.다음 프로젝트로 YOLO와 같은 one stage detector를
만들어 보거나 Object Segmentation에 관련해서 알아보고 싶습니다. 아니면 주제를 바꿔 자연어 처리 모델을 알아보는것도 흥미로울 것 같습니다.
긴 글 관심가지고 읽어주셔서 감사합니다. 오류나 질문 있으면 자유롭게 댓글 달아주세요.

## References
Faster R-CNN논문: <https://arxiv.org/abs/1506.01497>

Fast R-CNN논문: <https://arxiv.org/abs/1504.08083>  

R-CNN논문: <https://arxiv.org/abs/1311.2524>  

참고한 블로그: <https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439>  


