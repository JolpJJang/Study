# 3장(박희지)

# 3-1) 선형 회귀(= 수치 예측)

## 선형 회귀는 기울기와 절편을 찾아줌

- 머신러닝은 x,y가 주어질 때, 기울기(가중치)와 절편을 구함.

<img src="./images_3/untitled 0.png" width="50%" height="50%" />

- 방정식 = 모델
    - 방정식을 세우면, 새로운 데이터에 대한 예측 가능
    - 아래 그림에서는 3번째 방정식이 적합한 모델

<img src="./images_3/untitled 1.png" width="50%" height="50%" />

## 예시) 당뇨병 환자

In [1]:

```python
// 예제 데이터셋을 포함한 라이브러리 = 사이킷런
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()  // Bunch 클래스
```

```python
// data: 입력, target: 타깃
// (442,)은 튜플, 쉼표 빼면 안됨
/ 샘플: 442개, 특성: 10개
print(diabetes.data.shape, diabetes.target.shape)  // (442, 10) (442,)
```

<img src="./images_3/untitled 2.png" width="50%" height="50%" />

샘플과 특성

### 입력 데이터와 타깃 데이터 자세히 보기

```python
diabetes.data[0:3]  // 슬라이싱을 이용한 행 데이터 출력(0,1,2,3)
// 도메인 지식을 좀 알면 머신러닝에 도움이 됨.
```

`array([[ 0.03807591,  0.05068012,  0.06169621,  0.02187235, -0.0442235 ,
        -0.03482076, -0.04340085, -0.00259226,  0.01990842, -0.01764613],
       [-0.00188202, -0.04464164, -0.05147406, -0.02632783, -0.00844872,
        -0.01916334,  0.07441156, -0.03949338, -0.06832974, -0.09220405],
       [ 0.08529891,  0.05068012,  0.04445121, -0.00567061, -0.04559945,
        -0.03419447, -0.03235593, -0.00259226,  0.00286377, -0.02593034]])`

```python
diabetes.target[:3]  // 슬라이싱을 이용한 타깃 데이터 출력(0,1,2)
```

`array([151.,  75., 141.])`

### 당뇨병 환자 데이터 시각화하기

```python
import matplotlib.pyplot as plt
//[:, 2] : 모든 행, 3번째 특성 선택
plt.scatter(diabetes.data[:, 2], diabetes.target)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

<img src="./images_3/untitled 3.png" width="50%" height="50%" />

위 코드 시각화 결과

### x,y 정의

```python
x = diabetes.data[:, 2]
y = diabetes.target
```

## 3-2) 경사하강법

- 경사하강법으로 회귀식 찾기

<img src="./images_3/untitled 4.png" width="50%" height="50%" />

2번째 직선이 가장 적합한데 이를 경사하강법으로 찾을 수 있음.

### 타깃과 예측값

- 두번째 식에서 x는 계수
- 최적의 가중치와 절편을 찾아 예측값 찾기

<img src="./images_3/untitled 5.png" width="50%" height="50%" />

### 훈련 데이터에 맞는 w와 b를 찾는 방법

1. 무작위로 w,b 정함 (무작위 모델 생성)
2. x에서 샘플 하나를 선택하여 y^ 계산 (무작위 모델 예측)
3. y^과 선택한 샘플의 진짜 y를 비교(예측값과 실제값 비교, 틀릴 확률 99%)
4. y^가 y와 더 가까워지도록 w,b를 조정(모델 조정)
5. 모든 샘플을 처리할 때까지 2~4 반복

### 실제 훈련 데이터에 맞는 w,b 찾기

- 임의의 값으로 시작

```python
w = 1.0
b = 1.0
```

- 첫번째 샘플(당뇨병 3번째 특성)에 대한 예측 만들기

```python
y_hat = x[0] * w + b
print(y_hat)
```

`1.0616962065186886`

- 첫번째 샘플 실제 타깃

```python
print(y[0])
```

**`151.0`**

### w를 조절해 차이값 줄이기

- 경사를 0.1만큼 증가시키니 이전보다 타깃에 가까워짐
    - 나쁘지 않은 선택? ㅎ

```python
w_inc = w + 0.1
y_hat_inc = w_inc * x[0] + b
print(y_hat_inc)
```

1.0678658271705574

### 얼만큼 증가했는지 변화율로 알아보자

- 식으로 확인해보니 변화율 = x[0]
- 이걸보고 변화율이 양수인지 음수인지에 따라 w 조절 가능할까?

<img src="./images_3/untitled 6.png" width="50%" height="50%" />

- 코드

```python
w_rate = (y_hat_inc - y_hat) / (w_inc - w)
print(w_rate)
```

1.0678658271705574(변화율)

### 변화율 부호에 따라 가중치를 업데이트 하는 방법

- 변화율 양수 → w가 증가하면 y^ 증가
- 변화율 음수 → w가 감소하면 y^ 증가
- '새로운 w' = 기존 w + 변화율
- 이 문제에서는 w가 양수이고, w가 커질수록 예측값이 증가함.
    - 임의로 변화율을 더해주면서 w 키우기

<img src="./images_3/untitled 7.png" width="50%" height="50%" />

```python
w_new = w + w_rate
print(w_new)
```

### 변화율로 절편 업데이트하기

- 마찬가지로 새로운 w에 대한 b도 바꿀 수 있음.
- 식으로 인해 b에 대한 변화율은 항상 1
    - → '새로운 b' = 기존 b + 1

<img src="./images_3/untitled 8.png" width="50%" height="50%" />

### 이 방식의 문제점

- 예측값이 실제에 한참 못 미칠 경우, 더 큰 폭으로 수정 X
- 예측값이 실제보다 커지면 예측값을 감소 못함

<<해결법>>

- 실제값과 예측값의 차이가 크면 가중치와 절편을 그에 비례하게 바꾸기 → 빠르게 솔루션에 수렴
- 예측값이 실제보다 크면 가중치, 절편 감소시키기 → 예측값과 실제 능동적으로 대처

### 오차 역전파로 가중치와 절편 업데이트

- 오차와 변화율을 곱하여 가중치 업데이트

```python
err = y[0] - y_hat
w_new = w + w_rate * err
b_new = b + 1 * err
print(w_new, b_new)

```

### 두번째 샘플을 사용하여 w,b 계산

```python
y_hat = x[1] * w_new + b_new
err = y[1] - y_hat
w_rate = x[1]  // 두번째 샘플 변화율 = 샘플값 그 자체
w_new = w_new + w_rate * err
b_new = b_new + 1 * err
print(w_new, b_new)
```

### 전체 샘플 반복하여 가중치 절편 조정

- 위 방법으로 구한 w,b로 시각화

```python
for x_i, y_i in zip(x, y):
    y_hat = x_i * w + b
    err = y_i - y_hat
    w_rate = x_i
    w = w + w_rate * err
    b = b + 1 * err
print(w, b)  // 87.8654539985689 99.40935564531424

for x_i, y_i in zip(x, y):
    y_hat = x_i * w + b
    err = y_i - y_hat
    w_rate = x_i
    w = w + w_rate * err
    b = b + 1 * err
print(w, b)
```

<img src="./images_3/untitled 9.png" width="50%" height="50%" />

## 여러 에포크 반복

- 100번 돌림

```python
for i in range(1, 100):
    for x_i, y_i in zip(x, y):
        y_hat = x_i * w + b
        err = y_i - y_hat
        w_rate = x_i
        w = w + w_rate * err
        b = b + 1 * err
print(w, b)  // 913.5973364345905 123.39414383177204

plt.scatter(x, y)
pt1 = (-0.1, -0.1 * w + b)
pt2 = (0.15, 0.15 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

<img src="./images_3/untitled 10.png" width="50%" height="50%" />

### 모델로 예측

- 데이터 0.18 넣어서 예측

```python
x_new = 0.18
y_pred = x_new * w + b
print(y_pred)  // 287.8416643899983

plt.scatter(x, y)
plt.scatter(x_new, y_pred)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

<img src="./images_3/untitled 11.png" width="50%" height="50%" />

## 3-3) 손실함수와 경사 하강법의 관계

- 손실 함수는 예측한 값과 실제 타깃값의 차이를 측정함.
- 손실 함수의 차이를 줄이는 방법으로 경사 하강법 사용
- 대표적인 회귀, 분류 등에는 널리 사용하는 손실 함수가 있음
- 복잡한 다른 문제에서는 자신만의 손실 함수를 정의하여 사용

### 회귀의 손실 함수

- **제곱 오차(Squared error)**
- (실제값-예측값)^2
- 제곱을 하는 이유는 차이가 많이 날수록 가중치를 부가해 줄 수 있어서

<img src="./images_3/untitled 12.png" width="50%" height="50%" />

### 손실 함수의 기울기를 찾기 위해 미분

- 경사하강법은 손실함수의 기울기가 작은 쪽(0으로 수렴하는)으로 이동하는 알고리즘
- 제곱오차에 대해 미분하여 얻은 식으로
    - 그레이디언트 = -(y-y^)x

<img src="./images_3/untitled 13.png" width="50%" height="50%" />

### 미분 결과를 가중치에서 빼면 손실 함수의 낮은 쪽으로 이동

- 앞서 직관으로 계산한 오차 역전파가 제곱 오차를 미분한 것과 결과 같음

<img src="./images_3/untitled 14.png" width="50%" height="50%" />

## 절편에 대해 미분하고 업데이트하기

- 절편 역시 미분하면 1이 나오고, 새로운 절편은 1*차이값

<img src="./images_3/untitled 15.png" width="50%" height="50%" />

## 3-4) 선형 회귀로 뉴런 만들기

### 뉴런 클래스

```python
class Neuron:
    
    def __init__(self):
        self.w = 1.0     # 가중치를 초기화
        self.b = 1.0     # 절편을 초기화
    
    def forpass(self, x):
        y_hat = x * self.w + self.b       # 직선 방정식을 계산
        return y_hat
    
    def backprop(self, x, err):
        w_grad = x * err    # 가중치에 대한 그래디언트를 계산
        b_grad = 1 * err    # 절편에 대한 그래디언트를 계산
        return w_grad, b_grad

    def fit(self, x, y, epochs=100):
        for i in range(epochs):           # 에포크만큼 반복
            for x_i, y_i in zip(x, y):    # 모든 샘플에 대해 반복
                y_hat = self.forpass(x_i) # 정방향 계산
                err = -(y_i - y_hat)      # 오차 계산
                w_grad, b_grad = self.backprop(x_i, err)  # 역방향 계산
                self.w -= w_grad          # 가중치 업데이트
                self.b -= b_grad          # 절편 업데이트
```

### 역방향 계산 원리

<img src="./images_3/untitled 16.png" width="50%" height="50%" />