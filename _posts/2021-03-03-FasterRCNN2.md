---
title: "TorchVision을 사용한 Faster R-CNN(2)"
categories:
 - MachineLearning
tags:
 - Object Detection
 - Deep Learning
 - Faster RCNN

toc: true
toc_sticky: true
---
## 이전 게시물 point
이전 게시물: <https://hyungjobyun.github.io/machinelearning/FasterRCNN1/>
* backbone: VGG16
* Dataset: Pascal Voc

**rpn**
* VGG16 마지막 차원: 512
* anchor box size: (128, 256, 512)
* anchor box aspect ratio: (0.5, 1.0, 2.0)
* 이미지 사이즈: 짧은 부분을 600픽셀로 고정, 긴 부분은 최대 1000픽셀
* NMS이전 anchor box 개수: 6000
* NMS threshold: 0.7
* NMS이후 anchor box 개수: 2000(train), 300(test)
* parameter 초기화: 평균0, 표준편차 0.01의 가우스 분포

**fast rcnn**
* VGG16의 마지막 pooling layer 제거
* RoI pooling layer: MultiScaleRoIAlign
* RoI pooling size: 7x7
* RoI head: 두개의 FC layer (inputsize x 4096),(4096x4096)
* Class 개수: 21개 (20개 class + 배경)
* Batch size: 128
* foreground threshold: 0.5
* background threshold: 0.5
* positive fraction: 0.25
* parameter 초기화: classification 평균 0, 표준편차 0.01 / bounding-box regression 평균 0, 표준편차 0.001 가우스 분포

**loss & optimizer**
* RPN lambda: 10
* FastRCNN lambda: 1
* Optimizer: SGD
* Learning rate: 0.001
* momentum: 0.9
* weight decay: 0.0005

**training**
* Approximate joint training
* Learning rate decay: 최고 0.001, 최저 0.00001인 cosine learning rate decay
* Epoch길이: 15k 이미지
* Epoch: 다양한 시도
* 이미지 horizontal flip

## Python 코드

### 환경

Python언어로 작성하였고 Pytorch와 기타 라이브러리를 이용했습니다. 학습은 Google colab을 이용했습니다.

### Dataset준비 & 라이브러리 import

```python
from google.colab import drive
drive.mount('/content/drive')
```  
구글 드라이브를 연결하는 코드입니다.

```python
!pip install imgaug --upgrade     #image augmentation을 위해 필요
import torch      #pytorch
import torch.nn as nn     #pytorch network
from torch.utils.data import Dataset, DataLoader      #pytorch dataset
from torch.utils.tensorboard import SummaryWriter     #tensorboard
import torchvision      #torchvision
import torch.optim as optim     #pytorch optimizer
import numpy as np      #numpy
import matplotlib.pyplot as plt     #matplotlib(이미지 표시를 위해 필요)
from collections import OrderedDict     #python라이브러리 (라벨 dictionary를 만들 때 필요)
import os     #os
import xml.etree.ElementTree as Et      #Pascal xml을 읽어올 때 필요
from xml.etree.ElementTree import Element, ElementTree
import cv2      #opencv (box 그리기를 할 때 필요)
from PIL import Image     #PILLOW (이미지 읽기)
import time     #time
import imgaug as ia     #imgaug
from imgaug import augmenters as iaa
from torchvision import transforms      #torchvision transform
#GPU연결
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')
print(device)
```

필요한 라이브러리를 import합니다. 대부분 google colab에 이미 제공된 라이브러리이지만 작성 당시 imgaug는 필요한 버전과 달라서 업그레이드했습니다.

```python
if not os.path.isfile("/content/voc_train_val2012_tar"):
  !cp "/content/drive/MyDrive/Faster_RCNN/VOCtrainval_11-May-2012.tar" "/content/voc_train_val2012_tar" 
  ! tar -xf "/content/voc_train_val2012_tar"
```
Google drive에 있는 dataset tar파일을 google colab런타임 저장소에 복사하여 압축을 푸는 코드입니다. cp는 복사, tar -xf는 압축 풀기입니다.  
코드를 실행하기 위해서는 사전에 dataset을 google drive에 저장해 두어야 하고 적절한 경로로 수정 해 주어야 합니다.
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
Pascal dataset의 xml파일에서 파일명, 사물이름, 박스 위치를 가져오는 함수입니다. 박스 좌표는 [x_min, y_min, x_max, y_max]순서입니다. 논문과 형식에 차이가 있지만 torchvision라이브러리 사용을 위해 수정했습니다.  
Xml parser의 자세한 내용은 아래 링크를 참고하세요.
<https://deepbaksuvision.github.io/Modu_ObjectDetection/posts/02_01_PASCAL_VOC.html>

```python
def makeBox(voc_im,bbox,objects):
  image = voc_im.copy()
  for i in range(len(objects)):
    cv2.rectangle(image,(int(bbox[i][0]),int(bbox[i][1])),(int(bbox[i][2]),int(bbox[i][3])),color = (0,255,0),thickness = 1)
    cv2.putText(image, objects[i], (int(bbox[i][0]), int(bbox[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2) # 크기, 색, 굵기
  return image
```
이미지에 box를 그리는 함수입니다. 이미지는 numpy 형식으로 입력 받고 bbox는 각 box당 좌표가 필요하므로 [x_min, y_min, x_max, y_max]순서의 2차원 리스트로 입력 받습니다. objects는 사물의 이름이 있는 1차원 리스트입니다.

```python
xml_list = os.listdir("/content/VOCdevkit/VOC2012/Annotations")
xml_list.sort()

label_set = set()

for i in range(len(xml_list)):
  xml_path = "/content/VOCdevkit/VOC2012/Annotations/"+str(xml_list[i])
  file_name, object_name, bbox = xml_parser(xml_path)
  for name in object_name:
    label_set.add(name)

label_set = sorted(list(label_set))

label_dic = {}
for i, key in enumerate(label_set):
  label_dic[key] = (i+1)
print(label_dic)
```
모든 xml파일을 열어보며 존재하는 라벨 집합을 만드는 코드입니다. 1번부터 알파벳 순으로 정렬됩니다.  
결과:
```python
{'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}
```

```python
class Pascal_Voc(Dataset):

  def __init__(self,xml_list,len_data):

    self.xml_list = xml_list
    self.len_data = len_data
    self.to_tensor = transforms.ToTensor()
    self.flip = iaa.Fliplr(0.5)
    self.resize = iaa.Resize({"shorter-side": 600, "longer-side": "keep-aspect-ratio"})

  def __len__(self):
    return self.len_data

  def __getitem__(self, idx):

    xml_path = "/content/VOCdevkit/VOC2012/Annotations/"+str(xml_list[idx])

    file_name, object_name, bbox = xml_parser(xml_path)
    image_path = "/content/VOCdevkit/VOC2012/JPEGImages/"+str(file_name)
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)

    image, bbox = self.flip(image = image, bounding_boxes = np.array([bbox]))
    image, bbox = self.resize(image = image,bounding_boxes = bbox)
    bbox = bbox.squeeze(0).tolist()
    image = self.to_tensor(image)

    targets = []
    d = {}
    d['boxes'] = torch.tensor(bbox,device=device)
    d['labels'] = torch.tensor([label_dic[x] for x in object_name],dtype=torch.int64,device = device)
    targets.append(d)

    return image, targets
```
데이터셋을 만드는 코드입니다. imgaug를 통해 0.5확률의 flip을 하고 이미지의 짧은 부분을 600픽셀로 만듭니다. imgaug의 입력으로 numpy형식, [H, W, C]형태가 필요함을 유의해야합니다.  
bounding box는 squeeze(0)으로 자동적으로 생성되는 batch에 해당하는 차원을 없애고 tensor로 변환해줍니다.  
torchvision 라이브러리를 이용하기 위해서 target은 dictionary형태로 전달해야 합니다. class label의 형식을 int로 명시하지 않았더니 오류가 발생하여 설정을 추가해 주었습니다.

### 모델 만들기

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
                   box_score_thresh=0.05, box_nms_thresh=0.7,box_detections_per_img=300,
                   box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                   box_batch_size_per_image=128, box_positive_fraction=0.25
                 )
#roi head 있으면 num_class = None으로 함

for param in model.rpn.parameters():
  torch.nn.init.normal_(param,mean = 0.0, std=0.01)

for name, param in model.roi_heads.named_parameters():
  if "bbox_pred" in name:
    torch.nn.init.normal_(param,mean = 0.0, std=0.001)
  elif "weight" in name:
    torch.nn.init.normal_(param,mean = 0.0, std=0.01)
  if "bias" in name:
    torch.nn.init.zeros_(param)
```

모델을 만드는 코드입니다. backbone으로 VGG16을 사용하며 마지막 max pooling층은 제거해 줍니다. Faster RCNN을 사용하기 위해서는 fully connected layer를 만들기 위해 최종 backbone output채널이 512임을 알려주어야 합니다.  
이후 anchor generator, roi pooler, box head, box predictor를 각각 만들어 줍니다.  
box head는 Fast RCNN에서 처음 두 FC layer에 해당하는 층이고 box predictor는 예측을 하는 FC layer입니다.  

모델은 torchvision.models.detection에 있는 FasterRCNN을 사용합니다. 입력해야 하는 값을 **point** 에 있는 값들을 참고하여 입력하면 됩니다. default로 None인 항목 중 **point**에 없는 항목은 그대로 두어도 논문과 같거나 큰 영향이 없는 값들입니다.  
마지막으로 weight와 bias를 초기화합니다. 참고로 box_score_thresh와 box_nms_thresh는 예측때 필요한 값이므로 일단 임의의 값을 입력합니다.

### Training
```python
writer = SummaryWriter("/content/drive/MyDrive/Faster_RCNN/runs/Faster_RCNN")
%load_ext tensorboard
%tensorboard --logdir="/content/drive/MyDrive/Faster_RCNN/runs"
```
Tensorboard를 사용하기 위한 코드입니다. SummaryWriter에 runs폴더가 생길 곳과 정보를 저장할 하위 경로를 지정해 줍니다.  
%load_ext를 사용하면 Google colab셀의 출력으로 tensorboard를 사용할 수 있습니다.

```python
def Total_Loss(loss):
  loss_objectness = loss['loss_objectness']
  loss_rpn_box_reg = loss['loss_rpn_box_reg']
  loss_classifier = loss['loss_classifier']
  loss_box_reg = loss['loss_box_reg']

  rpn_total = loss_objectness + 10*loss_rpn_box_reg
  fast_rcnn_total = loss_classifier + 1*loss_box_reg

  total_loss = rpn_total + fast_rcnn_total

  return total_loss
```
loss를 계산하는 함수입니다. Approximate joint training을 할 것이므로 rpn_box_reg에 λ=10, box_reg에 λ=1을 곱해서 모두 더합니다.

```python
total_epoch = 40

len_data = 15000
term = 1000

loss_sum = 0

model.to(device)

optimizer = torch.optim.SGD(params = model.parameters(),lr = 0.001, momentum = 0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total_epoch,eta_min=0.00001)

try:
  check_point = torch.load("/content/drive/My Drive/Faster_RCNN/Check_point.pth") 
  start_epoch = check_point['epoch']
  start_idx = check_point['iter']
  model.load_state_dict(check_point['state_dict'])
  optimizer.load_state_dict(check_point['optimizer'])
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,total_epoch,eta_min=0.00001,last_epoch = start_epoch)
  scheduler.load_state_dict(check_point['scheduler'])

  if start_idx == len_data: 
    start_idx = 0
    start_epoch = start_epoch + 1

except:
  print("check point load error!")
  start_epoch = 0
  start_idx = 0

print("start_epoch = {} , start_idx = {}".format(start_epoch,start_idx))

print("Training Start")
model.train()
start = time.time()

for epoch in range(start_epoch,total_epoch):
  
  writer.add_scalar('Learning Rate',scheduler.get_last_lr()[0], epoch)

  dataset = Pascal_Voc(xml_list[:len_data],len_data - start_idx)
  dataloader = DataLoader(dataset,shuffle=True)

  for i, (image,targets)in enumerate(dataloader,start_idx):

    optimizer.zero_grad()

    targets[0]['boxes'].squeeze_(0)
    targets[0]['labels'].squeeze_(0)
    
    loss = model(image.to(device),targets)
    total_loss = Total_Loss(loss)
    loss_sum += total_loss

    if (i+1) % term == 0:
      end = time.time()
      print("Epoch {} | Iter {} | Loss: {} | Duration: {} min".format(epoch,(i+1),(loss_sum/term).item(),int((end-start)/60)))
      writer.add_scalar('Training Loss',loss_sum / term, epoch * len_data + i)
      
      state = {
        'epoch': epoch,
        'iter' : i+1,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
      }
      torch.save(state,"/content/drive/My Drive/Faster_RCNN/Check_point.pth")
     
      loss_sum = 0
      start = time.time()
    
    total_loss.backward()
    optimizer.step()

  start_idx = 0
  scheduler.step() 

  state = {
      'epoch': epoch,
      'iter' : i+1,
      'state_dict': model.state_dict(),
      'optimizer' : optimizer.state_dict(),
      'scheduler': scheduler.state_dict()
    }
  torch.save(state,"/content/drive/My Drive/Faster_RCNN/Check_point.pth")

  if (epoch+1) % 10 == 0: 
    torch.save(model.state_dict(),"/content/drive/My Drive/Faster_RCNN/Epoch{}.pth".format(epoch))
```

실제로 Training을 하는 코드입니다. total_epoch는 총 epoch수, len_data는 한 epoch당 이미지 개수, term은 상황을 출력하고 임시 저장을 할 단위입니다.  
optimizer와 scheduler도 만들어 줍니다. 이전에 Training하던것을 이어서 할 경우 scheduler에 last_epoch를 수정하고 시작 위치를 바꾸는 등 약간의 변화를 줍니다(try부분).  
반복문 내부는 일반적인 딥러닝 방법과 동일합니다. 다만 Dataloader에서 target이 나올 때 차원이 하나씩 커져서 나오기 때문에 squeeze를 해 주었습니다.
```python
targets[0]['boxes'].squeeze_(0)
targets[0]['labels'].squeeze_(0)
```
그리고 term마다 Check point를 저장했고 10epoch마다 모델의 parameter만 따로 저장했습니다.

이제 전체를 실행시키면 학습이 진행 됩니다. 결과는 다음 게시물에 포스팅 하겠습니다. 오류나 질문 있으면 자유롭게 댓글 달아주세요.

전체 코드: <https://github.com/HyungjoByun/Projects/blob/main/Faster%20RCNN/FasterRCNN_Train.ipynb>

## References
Faster R-CNN논문: <https://arxiv.org/abs/1506.01497>

Fast R-CNN논문: <https://arxiv.org/abs/1504.08083>  

R-CNN논문: <https://arxiv.org/abs/1311.2524>  

참고한 블로그: <https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439>  



