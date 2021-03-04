# LeNet-5
### Gradient-based learning applied to document recognition

Yann Lecun et al 1998년 개발한 CNN 알고리즘

![image](https://user-images.githubusercontent.com/72767245/109143674-a2d25500-77a3-11eb-83bf-07486df82cac.png)

> input - c1 - s2 - c3 - s4 - c5 - F6 - output

c1 ~ F6 layers의 활성화 함수는 **tanh** 

- 1) C1 레이어: 입력 영상(32x32 사이즈의 이미지)을 6개의 5x5 필터와 컨볼루션 연산
  - 28 x 28 x 6
```
parameters: (가중치 * 입력 맵 개수 + 바이어스) * 특성맵 갯수 = (5x5x1 + 1) x 6 = 156개
```

- 2) S2 레이어: 6장의 28x28 특성맵에 대해 서브 샘플링을 진행(stride = 2 : subsampling)
  - average pooling 사용
    - 논문에서는 평균을 낸 후 한 개의 훈련 가능한 가중치(trainable weight)를 곱해주고 또 한 개의 훈련 가능한 bias를 더해준다
    - 그 값이 시그모이드 함수를 통해 활성화 됨
  - 14 x 14 x 6
```
parameters: (가중치 + 바이어스) x 특성맵 갯수 = (1 + 1) x 6 = 12개
```

- 3) C3 레이어: 6장의 14x14 특성맵에 컨볼루션 연산을 수행해서 16장의 10x10 특성맵을 산출
  - 6개의 모든 feature map이 16개의 필터 처리하는 것이 아니라 선택적으로 입력 영상을 선택
    - 연산량의 크기를 줄임, 처음 convolution 으로부터 얻은 6개의 low-level feature가 서로 다른 조합으로 섞이며 global feature로 나타나기를 기대하기 때문
  - 10 x 10 x 16
```
parameters:

첫번째그룹=> (가중치*입력맵개수+바이어스)*특성맵 개수 = (5*5*3 + 1)*6 = 456

두번째그룹=> (가중치*입력맵개수+바이어스)*특성맵 개수 = (5*5*4 + 1)*6 = 606

세번째그룹=> (가중치*입력맵개수+바이어스)*특성맵 개수 = (5*5*4 + 1)*3 = 303

네번째그룹=> (가중치*입력맵개수+바이어스)*특성맵 개수 = (5*5*6 + 1)*1 = 151

456 + 606 + 303 + 151 = 1516

```
- 4) S4 레이어: 16장의 10 x 10 특성맵에 대해서 서브 샘플링을 진행해 16장의 5x5 특성맵으로 축소
  - 16 x 5 x 5
```
parameters: (가중치 + 바이어스)*특성맵개수 = (1 + 1)*16 = 32
```

- 5) C5 레이어: 16장의 5x5 특성맵을 120개 5x5x16 사이즈의 필터와 컨볼루션
  - 120 x 1 x 1 특성맵
```
parameters: (가중치 x 입력맵개수 + 바이어스) x 특성맵 개수 = (5 x 5 x 16 + 1) x 120 = 48120
```

- 6) F6 레이어: 84개의 유닛을 가진 Feed Forward Network
  - 84
```
parameters: (입력개수 + 바이어스)*출력개수 = (120 + 1) x 84 = 10164
```

- 7) 아웃풋 레이어
  - 10 유닛

# AlexNet

---

Reference  
https://bskyvision.com/418  
https://my-coding-footprints.tistory.com/97

