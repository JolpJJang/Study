# AlexNet

2012년에 개최된 **ILSVRC(ImageNet Large Scale Visual Recognition Challenge) 대회**에서 우승을 차지한 CNN 구조

- 2개의 GPU로 병렬 연산을 수행하기 위해서 **병렬적인 구조**로 설계 되었다.
- AlexNet은 **8개의 레이어**로 되어있다.
  - 5개의 convolution layers와 3개의 fully-connected layers로 구성
  - FC layer는 1000개의 category로 분류하기 위해 활성화 함수로 softmax 함수를 사용

![image](https://user-images.githubusercontent.com/72767245/109422544-52ffc200-7a1f-11eb-986c-c4387062705d.png)

---

### AlexNet의 구조
![image](https://user-images.githubusercontent.com/72767245/109422613-a5d97980-7a1f-11eb-923f-aeb9f556633f.png)

1) **input**: 227 x 227 x 3 (224 아님)
2) **1 Conv2D**:
  - 96개의 11 x 11 x 3 사이즈의 필터 커널로 입력 영상을 컨볼루션함
  - stride = 4 (zero-padding은 사용하지 않음)
  - 결과적으로 55 x 55 x 96 feature map 산출
  - ReLU 함수로 활성화함
    - 3 x 3 overlapping max pooling이 stride 2로 시행됨
    - 결과 27 x 27 x 96 특성맵 갖게 됨
      - 수렴 속도를 높이기 위해 local response normalization 시행
      - local response normalization은 특성맵의 차원을 변화시키지 않으므로, 특성맵의 크기는 27 x 27 x 96 유지
3) **2 Conv2D**:
  - 256개의 5 x 5 x 48 커널을 사용하여 전 단계의 특성맵을 컨볼루션함
  - stride = 1, zero-padding은 2로 설정
  - 27x27x256 특성맵을 얻음
  - ReLU로 활성화
    - 3 x 3 overlapping max pooling이 stride 2로 시행됨
    - 결과 13x13x256 특성맵을 얻게 됨
      - local response normalization이 시행
      - 특성맵의 크기는 13x13x256 유지
4) **3 Conv2D**:
  - 284개의 3x3x256 커널을 사용하여 전 단계의 특성맵을 컨볼루션해준다
  - stride와 zero-padding 모두 1로 설정
  - 따라서 13x13x384 특성맵을 얻음
  - ReLU 함수로 활성화
5) **4 Conv2D**:
  - 384개의 3x3x192커널을 사용해서 전 단계의 특성맵을 컨볼루션해준다
  - stride와 zero-padding을 모두 1로 설정
  - 13x13x384특성맵
  - ReLU함수로 활성화

6) **5 Conv2D**:
  - 256개의 3x3x192 커널을 사용해서 전 단계의 특성맵을 컨볼루션해줌
  - stride와 zero-padding을 모두 1로 설정
  - 13x13x256 특성맵을 얻음
  - ReLU 활성화함수
  - 3x3 overlapping max pooling을 stride 2로 시행
  - 결과 6x6x256 특성맵을 얻음
7) **6 FC**:
  - 6x6x256 특성맵을 flatten 해줘서 6x6x256 = 9216차원의 벡터로 만들어줌
  - 6번째 레이어 4096개의 뉴런과 fully connected 함
  - ReLU함수로 활성화
8) **7 FC**:
  - 4096개의 뉴런으로 구성
  - 전 단계의 4096개의 뉴런과 fully connected 되어있음
  - ReLU함수로 활성화
9) **8 FC**:
  - 1000개의 뉴런으로 구성
  - softmax함수를 적용

### AlexNet 특징
- ReLU함수
- dropout
- overlapping pooling
- local response normalization(LRN)
- data augmentation
