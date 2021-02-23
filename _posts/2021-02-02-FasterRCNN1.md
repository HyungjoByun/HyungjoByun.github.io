---
title: "TorchVision을 사용한 Faster R-CNN(1)"
categories:
 - MachineLearning
tags:
 - Object Detection
 - Deep Learning
 - Faster RCNN

toc: true
toc_sticky: true
---

## 개요
 Faster RCNN은 사물 인식에 사용되는 RCNN을 개선하여 만들어진 모델입니다. CNN이 사물 분류에서 좋은 성능을 내자 사물인식(detection)에도 적용하여 RCNN이 탄생했습니다. 
 
 RCNN은 원본 이미지에서 기존의 사물 탐지 알고리즘인 Selective Search로 사물을 탐색한 후 CNN으로 해당 사물이 무엇인지 판단합니다. 사물이 탐지된 지역(Region)에 CNN을 사용하므로 Regions with CNN features 즉 RCNN이란 이름이 붙었습니다. 
 
 하지만 사물이 있다고 예상되는 지역마다 CNN을 적용 하므로 시간이 오래 걸렸습니다. 따라서 CNN backbone은 한 이미지당 한 번 통과시키고 이를 통과해서 나온 Feature map에 Selective Search를 적용하는 Fast RCNN으로 속도가 향상되었습니다. 
 
 Faster RCNN은 Selective Search대신 RPN (Region Proposal Network)을 사용하여 속도를 더욱 향상시키는 동시에 사물 탐지를 네트워크의 역할로 바꾸게 되었습니다. 추가로 Faster RCNN은 사물 탐지와 사물 판별, 두 과정으로 이루어져 있어 Two stage detector로 분류됩니다.
 
 
 이번 글에서는 논문의 내용을 분석하고 Pytorch, TorchVision등 라이브러리를 이용하여 Faster RCNN을 최대한 논문과 유사하게 만들어보는 과정을 다루겠습니다. 참고로 논문에서는 여러 backbone 네트워크와 dataset을 사용해 실험하였는데 이 글에서는 별도의 언급이 없다면 backbone으로 VGG16, dataset으로 PASCAL VOC를 사용한 기준으로 설명하겠습니다.
 
 **point**
 * backbone: VGG16
 * dataset: Pascal Voc

## 전체 구조

![FasterRCNN architecture](\assets\images\FasterRCNN_architecture.jpg)  
*<font size = 3pt>그림 1 [Faster RCNN구조] <br> source> Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks </font>*

그림1은 논문에 있는 그림입니다. 먼저 전체 이미지는 Conv layers를 통과합니다. Conv layers는 backbone 또는 feature extractor라고도 합니다. 
논문에서는 ZFNet, VGG16등의 네트워크를 conv layers로 사용했습니다.

이미지가 conv layers를 통과하면 feature map이 나오게 됩니다. RPN은 feature map을 바탕으로 사물의 위치를 표시하는 bounding box와 해당 box의 물체가 사물일 확률을 출력합니다. 자세한 내용은 RPN항목에서 다루겠습니다. 
RPN이 제시한 영역을 RoI(Region of Interest)라고 합니다. RoI가 구해지면 이후 과정은 Fast RCNN과 동일합니다. 그래서 Faster RCNN전체를 구현하기 위해서는 Fast RCNN논문도 참고해야 합니다. 자세한 내용은 Fast RCNN항목에서 다루겠습니다.

## RPN
![RPN구조](\assets\images\FasterRCNN_RPN.jpg)  
*<font size = 3pt>그림 2 [RPN구조] <br> source> Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks </font>*

RPN은 feature map을 입력으로 받는 n x n conv layer와 이후 1 x 1 conv layer로 구현된 두 종류의 fully connected layer로 이루어져 있습니다. n은 기본적으로 3으로 하였고 backbone이 VGG16인 경우 첫번째 n x n conv layer의 출력은 512채널이 됩니다.  

Feature map의 픽셀당 RoI후보가 되는 anchor box를 생성합니다. 한 픽셀당 생성하는 anchor box의 사이즈는 128, 256, 512이고 각각의 사이즈마다 가로: 세로의 비율이 1:1, 1:2, 2:1입니다.  
따라서 픽셀당 9개의 anchor box가 생성됩니다. 결론적으로 그림2의 k는 (feature map의 가로길이 X 세로길이 X 9) 가 됩니다. cls layer의 k에 2가 곱해지는 이유는 각 anchor box 마다 사물일 경우와 아닐 경우가 있기 때문이고 reg layer의 4는 anchor box의 x좌표, y좌표, 높이, 넓이를 의미합니다. 

**point**
 * VGG16 마지막 차원: 512
 * anchor box size: (128, 256, 512)
 * anchor box aspect ratio: (0.5, 1.0, 2.0)

## Fast RCNN
![Fast RCNN구조](\assets\images\FasterRCNN_FastRCNN.jpg)  
*<font size = 3pt>그림 3 [Fast RCNN구조] <br> source> Fast R-CNN(Ross Girshick, Microsoft Research) </font>*

> First, the last max pooling layer is replaced by a RoI
pooling layer that is configured by setting H and W to be compatible with the net’s first fully connected layer (e.g., H = W = 7 for VGG16).  
Second, the network’s last fully connected layer and softmax (which were trained for 1000-way ImageNet classification) are replaced with the two sibling layers described earlier (a fully connected layer and softmax over K+1 categories and category-specific bounding-box regressors). Third, the network is modified to take two data inputs: a
list of images and a list of RoIs in those images. -Fast RCNN-

Fast RCNN의 backbone은 VGG16에서 마지막 Max Pooling Layer를 RoI pooling layer로 교체한 것입니다. 
참고로 RPN이 입력으로 받는 feature map또한 RoI pooling layer이전까지 VGG16의 출력입니다. RPN에서 RoI를 출력해 주면 해당 영역을 이용해서 RoI Pooling을 합니다.  
RoI Pooling에 대한 자세한 내용은 아래 링크를 참고 해 주시기 바랍니다.
<https://towardsdatascience.com/understanding-region-of-interest-part-2-roi-align-and-roi-warp-f795196fc193>

다만 논문에서는 RoI max pooling을 사용했지만 저는 torchvision 라이브러리 사용을 위해 MultiScaleRoIAlign을 사용했습니다. 이 때 출력 사이즈는 논문에 나온 것과 같이 7X7을 유지했습니다.

> Layer fc6 is fully connected to pool5. To compute features, it multiplies a 4096×9216 weight matrix by the pool5 feature map -RCNN-  
Layer fc7 is the final layer of the network. It is implemented by multiplying the features computed by fc6 by a 4096 × 4096 weight matrix -RCNN-

Fast RCNN은 출력으로 사물의 class와 박스 위치(Bounding box 또는 bbox)를 출력합니다. 이를 위해 RoI pooling layer의 출력이 두개의 Fully connected layer를 거치고 class와 bbox를 출력하기 위한 각각의 FC layer로 입력됩니다.  
RCNN논문에서 RoI pooling layer 이후 FC layer의 weight은 inputsize x 4096과 4096 x 4096의 형태입니다. class score 출력의 FC layer는 Pascal Voc 데이터 셋을 사용하므로 4096 x 21 (20개의 class + 배경)이고 bounding box 출력의 FC layer는 4096 x 21 x 4가 됩니다.

**point**
 * VGG16의 마지막 pooling layer 제거
 * RoI pooling layer: MultiScaleRoIAlign
 * RoI pooling size: 7x7
 * RoI head: 두개의 FC layer (inputsize x 4096),(4096x4096)
 * Class 개수: 21개 (20개 class + 배경)

## Training RPN

>All single-scale experiments use s = 600 pixels; s maybe 
less than 600 for some images as we cap the longest image side at 1000 pixels and maintain the image’s aspect ratio. -Fast RCNN-  
>We re-scale the images such that their shorter side is s = 600 pixels -Faster RCNN-

Faster RCNN과 Fast RCNN논문은 이미지의 스케일을 작은 쪽이 600픽셀이 되도록 비율에 맞게 조정합니다. 
이 때 긴 부분은 최대 1000픽셀을 넘지 않도록 하였습니다. PASCAL 이미지들은 평균 384 × 473 픽셀이라 대부분 확대됩니다.

결국 RPN에 입력되는 이미지가 600 x 1000이라고 하면 VGG16을 통과하고 나서 feature map의 사이즈는 대략 40 x 60이 됩니다. 따라서 총 anchor box는 40 x 60 x 9 개가 생성되지만 이미지의 범위를 벗어나는 anchor box는 무시합니다. 
그렇지 않으면 Training중 발산하기 때문입니다. 따라서 범위를 벗어나는 anchor box를 무시하면 약 6000개의 anchor box가 생성된다고 합니다. 

각 anchor box는 정답(ground truth) bounding box와 겹치는 정도(IoU)에 따라 사물인지 배경인지 라벨이 붙게 됩니다. 
논문에서는 IoU가 0.7이상이면 사물, 0.3 이하면 배경으로 설정하였습니다. anchor box와 그에 맞는 라벨이 정해졌으니 학습이 가능합니다. 전체 anchor box중 256개를 minibatch로 뽑아 학습하는데 사물을 포함하는것과 하지 않는 것의 비율을 1:1로 합니다.

>Some RPN proposals highly overlap with eachother. 
To reduce redundancy, we adopt non-maximum suppression (NMS) on the proposal regions based on their cls scores. 
We fix the IoU threshold for NMS at 0.7, which leaves us about 2000 proposal regions per image. As we will show, NMS does not harm the ultimate detection accuracy, but substantially reduces the number of proposals. 
After NMS, we use the top-N ranked proposal regions for detection. In the following, we train Fast R-CNN using 2000 RPN proposals, but evaluate different numbers of proposals at test-time. – Faster RCNN-

RPN을 학습할 때 필요한 정보는 끝났지만 Fast RCNN predictor 부분을 학습시키기 위해 RPN에 설정해 줄 정보가 더 필요합니다. anchor box가 6000개가 나오는데 이것을 전부 RoI로 사용하면 비효율적입니다. 따라서 IoU가 0.7이상 서로 겹치는 anchor box는 삭제하는 Non Maximum Suppression(NMS)을 합니다.

이후 남아있는 anchor box중 점수가 높은 N개를 뽑아 Fast RCNN predictor에 전달합니다. 
논문에서는 여러 N을 선택하여 실험했는데 저는 그중에서 학습은 2000개, 테스트는 300개인 경우로 하였습니다.마지막으로 훈련 전 RPN의 parameter들은 mean: 0, std: 0.01의 가우스분포로 초기화 시킵니다.

**point**
 * 이미지 사이즈: 짧은 부분을 600픽셀로 고정, 긴 부분은 최대 1000픽셀
 * NMS이전 anchor box 개수: 6000
 * NMS threshold: 0.7
 * NMS이후 anchor box 개수: 2000(train), 300(test)
 * parameter 초기화: 평균0, 표준편차 0.01의 가우스 분포

## Training Fast RCNN

> We use mini-batches of size R = 128, sampling 64 RoIs from each image. As in [9], we take 25% of the RoIs from object proposals that have intersection over union (IoU) overlap with a groundtruth bounding box of at least 0.5. -Fast RCNN-

RPN에서 2000개의 RoI를 제시하면 Fast RCNN은 이 중에서 128개로 mini-batch를 만듭니다. 
논문은 2개의 이미지에서 총 128개의 box를 고르지만 저는 이미지 한 장씩 훈련시킬 것이므로 이미지 하나에서 128개를 골랐습니다. 이 때 실제 box와 0.5이상 겹치는 RoI이면 해당 사물로 라벨을 붙이고 그 이하로 겹치면 배경으로 라벨을 붙입니다.
mini-batch에서 사물로 라벨이 붙은 box의 비율은 0.25입니다. 

> The fully connected layers used for softmax classification and bounding-box regression are initialized from zero-mean Gaussian distributions with standard deviations 0.01 and 0.001, respectively. Biases are initialized to 0. – Fast RCNN-

논문처럼 classification 과 bounding-box regression 층의 weight을 각각 평균 0, 표준편차 0.01 그리고 평균 0, 표준편차 0.001을 따르는 가우스 분포로 초기화 했습니다. 그리고 bias는 0으로 초기화 했습니다.

**point**
 * Batch size: 128
 * foreground threshold: 0.5
 * background threshold: 0.5
 * positive fraction: 0.25
 * parameter 초기화: classification 평균 0, 표준편차 0.01 / bounding-box regression 평균 0, 표준편차 0.001 가우스 분포

## Loss Function and Optimizer

![RPN Loss](\assets\images\FasterRCNN_RPNLoss.jpg)  
*<font size = 3pt>그림 4 [RPN Loss] <br> source> Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks </font>*

RPN의 Loss function은 그림 4와 같습니다. object인지 판단하는 loss L_cls와 box의 위치를 판단하는 loss L_reg의 평균을 더해 구합니다. 이 때 p*는 사물이 있는 box만 더함을 의미하고 λ는 두 loss의 비율을 맞추기 위해 곱합니다. 논문에서 주로 사용한 값은 10입니다. 그리고 L_reg는 smooth L1 loss를 사용했습니다.

![FastRCNN Loss](\assets\images\FasterRCNN_FastRCNNLoss.jpg)  
*<font size = 3pt>그림 5 [RPN Loss] <br> source> Fast R-CNN(Ross Girshick, Microsoft Research) </font>*

Fast RCNN의 Loss function은 그림5와 같습니다. RPN과 유사한 형식입니다. 다만 λ값을 1로 하였습니다.

> We use a learning rate of 0.001 for 60k mini-batches, and 0.0001 for the next 20k mini-batches on the PASCAL VOC dataset. We use a momentum of 0.9 and a weight decay of 0.0005 -Faster RCNN-  
> All layers use a per-layer learning rate of 1 for weights and 2 for biases and a global learning rate of 0.001. When training on VOC07 or VOC12 trainval we run SGD for 30k mini-batch iterations, and then lower the learning rate to 0.0001 and train for another 10k iterations. -Fast RCNN-

Optimizer는 SGD optimizer를 사용하였고 초기 Learning rate 0.001, momentum 0.9, weight decay 0.0005로 설정했습니다. Fast RCNN에서는 weight과 biases에서 다른 learning rate를 사용하였는데 저는 전체 learning rate만 사용했습니다.

**point**
 * RPN lambda: 10
 * FastRCNN lambda: 1
 * Optimizer: SGD
 * Learning rate: 0.001
 * momentum: 0.9
 * weight decay: 0.0005

## Dataset

논문에서는 여러 dataset으로 학습과 평가를 했는데 저는 Pascal Voc 2012 trainval dataset으로 학습을 시키고 2007 testset으로 평가를 했습니다. 

## Training

> (i)Alternating training. In this solution, we first train
RPN, and use the proposals to train Fast R-CNN. The network tuned by Fast R-CNN is then used to initialize RPN, and this process is iterated. This is the solution that is used in all experiments in this paper.  
>(ii) Approximate joint training ··· backward propagated signals from both the RPN loss and the Fast R-CNN loss are combined. 
This solution is easy to implement. But this solution ignores the derivative w.r.t. the proposal boxes’ coordinates that are also network responses, so is approximate. ···  -Faster RCNN-


논문에서 Training절차는 Alternating training으로, 4단계로 이루어져 있습니다. **RPN학습-> RPN의 제안을 이용한 Fast RCNN학습-> Fast RCNN의 backbone이 동기화된 RPN 학습-> 전체 학습**의 순서입니다.

그러나 위 방법은 과정이 복잡하고 모든 loss를 더해서 한 번에 학습시키는 Approximate joint training도 가능하다고 하여 저는 Approximate joint training을 선택했습니다. 다만 RPN이 초반에 정확한 사물 위치를 제시하지 못하기 때문에 학습 때 더 많은 반복이 필요합니다.

논문에서 RPN의 경우 60k의 반복 후 learning rate를 0.0001로 낮춰 20k를 더 학습하고 Fast RCNN은 Pascal Voc 데이터셋에서 학습할 때 기준으로 총 100k반복을 하고 40k마다 0.1배 learning rate decay를 하였습니다. 

그러나 Approximate joint training을 한다면 초반에 RPN이 정확한 RoI를 제시하지 못하기 때문에 위 과정을 단순히 두 번 반복하는 것 보다 더 많은 학습시간이 필요하다고 생각했습니다. Alternating training을 했을 때 총 반복 횟수를 단순 계산하면 360k입니다. 저는 Voc 2012 trainval 약 17,000장 중 15,000장을 사용하여 한 epoch를 15k로 지정하였습니다. 그래서 최소한 24epoch를 학습시켜야 합니다. 처음 시도에서Alternating training이 아닌 것을 고려하여 2배에 약간 못 미치는 40epoch (600k iter)를 학습시켰습니다. 


전체 반복 횟수가 달라지면 논문에서 사용한 learning rate가 효율적이지 않을 수 있습니다. 따라서 초기 learning rate는 같게 하고 최종적으로 처음의 0.01배의 learning rate가 되도록 cosine learning rate decay를 사용했습니다. 결과적으로 최고 63.69mAP가 측정되었지만 성능 향상의 가능성이 남아있어 같은 방법으로 다양한 epoch를 시도해 봤습니다. 자세한 결과는 3편에 적겠습니다. 추가로 이미지는 논문처럼 0.5의 확률로 horizontal flip했습니다.

**point**
 * Approximate joint training
 * Learning rate decay: 최고 0.001, 최저 0.00001인 cosine learning rate decay
 * Epoch길이: 15k 이미지
 * Epoch: 다양한 시도
 * 이미지 horizontal flip

1편의 내용은 여기까지 입니다. 다음 포스트에는 현재까지 나온 정보를 이용하여 작성한 코드를 설명하겠습니다. 오류나 질문 있으면 자유롭게 댓글 달아주세요.

## 참고 문헌
Faster R-CNN논문: <https://arxiv.org/abs/1506.01497>

Fast R-CNN논문: <https://arxiv.org/abs/1504.08083>  

R-CNN논문: <https://arxiv.org/abs/1311.2524>  

참고한 블로그: <https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439>  


