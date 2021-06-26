---
title: "Convolution 실습"
categories:
 - MachineLearning
tags:
 - Convolution matrix
 - Deep Learning
 - CNN

toc: true
toc_sticky: true
---

## 동기
Convolution연산에 대해 자세히 알아보던 중 filter가 행렬로 바뀌는 과정을 실습해 보고 싶어 C언어로 convolution층 하나를 만들어 보았습니다. 
Padding과 dilation이 없는 가장 기본적인 연산입니다.

## 출력의 크기  
Convolution연산의 출력 크기는 다음과 같은 공식으로 구할 수 있습니다.  
![output_size](\assets\images\conv_mat\output_size.JPG)  
이 때 I = 입력 행렬 크기, K = 필터 크기, P = padding, S = stride 입니다.  

## 전체 과정

먼저 input 행렬이 vector형태로 바뀝니다. 이를 flatten vector라 하겠습니다.  
다음은 filter를 기반으로 convolution연산을 하는 방식대로 convolution matrix가 생성됩니다. 
마지막으로 convolution matrix와 flatten vector를 행렬 곱셈을 하여 결과 vector를 출력합니다. 이 때 결과를 다시 행렬 형태로 바꿔 줍니다.
예시로 사용할 행렬은 그림1과 같습니다.
Input: ![Pic1_1](\assets\images\conv_mat\pic1_1.JPG)  
Filter: ![Pic1_2](\assets\images\conv_mat\pic1_2.JPG)  
*<font size = 3pt>그림 1 </font>*

```c
#define HI 5
#define WI 5
#define HF 3
#define WF 3
#define STRIDE 1

int input[HI][WI] =
{ {1,2,3,4,5},
	{6,5,4,3,2},
	{3,4,5,6,7},
	{8,7,6,5,4},
	{5,4,3,2,1}
};

int filter[HF][WF] =
{
	{1,3,2},
	{3,4,1},
	{2,3,1}
};
```

여기서 HI,WI는 input의 행 수와 열 수를 의미합니다. HF, WF는 filter의 행 수와 열 수를 의미합니다.  
각 과정을 이야기하기 앞서 이후 배열, 횟수 등에서 숫자를 샐 때는 0번째를 가장 처음으로 가정하고 예시로 나오는 결과는 stride = 1일 때 결과입니다.

## Flatten

Input 행렬에서 행 순서대로 하나의 벡터로 이어주면 flatten vector가 됩니다. 
```c
int k = 0;

	for (int i = 0; i < HI; i++) {
		for (int j = 0; j < WI; j++) {
			flatten_vec[k] = input[i][j];
			k++;
		}
	}
```

결과: [1 2 3 4 5 6 5 4 3 2 3 4 5 6 7 8 7 6 5 4 5 4 3 2 1]

## Convolution matrix
Convolution matrix에서 n번째 행은 filter가 n번 움직였을 때 계산 결과입니다. 
예를 들어 예시로 사용하는 filter에 해당하는 convolution matrix를 만들었을 때, 0번째(가장 처음) 연산에 해당하는 0번째 행은 다음과 같습니다.

[1 3 2 0 0 3 4 1 0 0 2 3 1 0 0 0 0 0 0 0 0 0 0 0 0]

위의 0번째 행을 flatten vector와 행렬 곱을 하면 0번째 계산 결과인 78이 나오는 것을 알 수 있습니다.

Convolution matrix를 만들기 위한 방법으로 제가 사용한 방법은 다음과 같습니다. 
1. 그림2와 같이 input위에 n번째 convolution을 위한 filter를 위치시킵니다. 그림2의 경우 n=1입니다.  
![Pic2](\assets\images\conv_mat\pic2.JPG)  
*<font size = 3pt>그림 2 </font>*

2. input행렬의 0행 0열부터 차례대로 지나가며 filter와 겹치지 않는 영역이면 0을, 겹치는 영역이면 해당 부분의 filter값을 convolution matrix에 적어 넣습니다. 
n번째 convolution이므로 n행에 적는 것입니다.  
그림2의 경우 1번째 행에 [0 1 3 2 0 0 3 4 1 0 0 2 3 1 0 0 0 0 0 0 0 0 0 0 0] 이 입력되게 됩니다.

현재 filter가 어느 위치에 있는지 알기 위해 filter의 0행 0열의 위치를 구합니다. filter는 input을 가로로
![output_size](\assets\images\conv_mat\output_size.JPG)  
만큼 이동합니다. 따라서 filter의 0행 0열의 위치는 n번째 이동일 경우 아래와 같이 구할 수 있습니다.  
```c
row = STRIDE * (n / (1 + ((WI - WF) / STRIDE)));
col = STRIDE * (n % (1 + ((WI - WF) / STRIDE)));
```  
이제 input위에서 각 위치가 filter와 겹치는지 판단하고 겹친다면 해당 위치의 filter의 값을 convolution matrix에 저장합니다.  
```c
for (int i = 0; i < WI; i++) {
    for (int j = 0; j < HI; j++) {
        if ((row <= i) && (i < row + HF) && (col <= j) && (j < col + WF))
            conv_mat[n][WI * i + j] = filter[i - row][j - col];
    }
}
```  
if문은 input위에서 이동하며 보는 index가 filter의 범위에 있는지 판단하는 기능을 합니다. 그리고 해당 위치가 filter의 어느 index인지 알아보기 위해 i-row, j-col로 평행 이동을 합니다. 
해당하는 값을 Convolution matrix에 넣는데 n번째 이동이므로 n행이고, 몇 번째 반복을 했는지 알기 위해 WI*i+j를 계산하여 알맞은 열에 값을 저장합니다.

결과:  
![Conv_mat](\assets\images\conv_mat\Conv_mat.JPG)  

## Result

Convolution결과를 출력하기 위해 행렬 연산을 한 후 출력 형태를 계산하는 공식에 맞춰 모양을 만들어 줍니다.  
```c
for (int i=0; i<(1 +((HI - HF) / STRIDE))*(1 +((WI - WF) / STRIDE)); i++){
    int tmp = 0;
    for (int j = 0; j < HI * WI; j++) {
        tmp += a[i][j] * b[j];
    }
    printf("%d ", tmp);
    if ((i + 1) % (1 + ((HI - HF) / STRIDE)) == 0)
        printf("\n");
}
```  
## 전체 출력  
![Final_print](\assets\images\conv_mat\Final_print.JPG)

## 전체 코드  
```c
#include <stdio.h>

#define HI 5
#define WI 5
#define HF 3
#define WF 3
#define STRIDE 1

int input[HI][WI] =
{ {1,2,3,4,5},
	{6,5,4,3,2},
	{3,4,5,6,7},
	{8,7,6,5,4},
	{5,4,3,2,1}
};

int filter[HF][WF] =
{
	{1,3,2},
	{3,4,1},
	{2,3,1}
};

int flatten_vec[HI * WI] = {};
int conv_mat[(1 + ((HI - HF) / STRIDE)) * (1 + ((WI - WF) / STRIDE))][HI * WI] = {};

void flatten(int (*input)[WI], int* flatten) {
	int k = 0;

	for (int i = 0; i < HI; i++) {
		for (int j = 0; j < WI; j++) {
			flatten_vec[k] = input[i][j];
			k++;
		}
	}

	printf("flatten\n");
	for (int i = 0; i < HI * WI; i++) {
		printf("%d ", flatten_vec[i]);
	}
	printf("\n\n");
}

void to_conv_mat(int(*filter)[WF]) {
	int row = 0;
	int col = 0;

	for (int n = 0; n < (1 + ((HI - HF) / STRIDE)) * (1 + ((WI - WF) / STRIDE)); n++) {
		row = STRIDE * (n / (1 + ((WI - WF) / STRIDE)));
		col = STRIDE * (n % (1 + ((WI - WF) / STRIDE)));
		for (int i = 0; i < WI; i++) {
			for (int j = 0; j < HI; j++) {
				if ((row <= i) && (i < row + HF) && (col <= j) && (j < col + WF))
					conv_mat[n][WI * i + j] = filter[i - row][j - col];
			}
		}
	}

	printf("Convolution Matrix\n");
	for (int i = 0; i < (1 + ((HI - HF) / STRIDE)) * (1 + ((WI - WF) / STRIDE)); i++) {
		for (int j = 0; j < HI * WI; j++)
			printf("%d ", conv_mat[i][j]);
		printf("\n");
	}
	printf("\n");
}

void result(int(*a)[HI * WI], int b[]) {
	printf("result\n");
	for (int i = 0; i < (1 + ((HI - HF) / STRIDE)) * (1 + ((WI - WF) / STRIDE)); i++) {
		int tmp = 0;
		for (int j = 0; j < HI * WI; j++) {
			tmp += a[i][j] * b[j];
		}
		printf("%d ", tmp);
		if ((i + 1) % (1 + ((HI - HF) / STRIDE)) == 0)
			printf("\n");
	}

}


int main(void) {
	
	flatten(input, flatten_vec);

	to_conv_mat(filter);

	result(conv_mat, flatten_vec);

}
```