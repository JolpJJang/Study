# 이진 분류(Binary Classification)

> 분류하는 뉴런 만들기

- 이진 분류: 두개(True|False) 중 하나 고르기
- 다중 분류: 여러 개중 하나 고르기

## 4-1 초기 인공지능 알고리즘과 로지스틱 회귀

### 다층 퍼셉트론

- 다층 퍼셉트론은 거의 신경망 알고리즘 중 최초
  - 사이킷런에서도 다층 퍼셉트론 클래스 구현되어 있음.
- z의 값이 y^은 0보다 크면 1(양성클래스)
- z의 값이 0 또는 0보다 작으면 y^은 -1(음성클래스)
- 이를 그래프로 나타내면 계단 모양으로 생겨서 **계단함수(Step function)**이라고 부름.
- 오차역전파처럼 y^에서 먼저 출발하여 가중치와 절편 조정하는 알고리즘.

<img src="https://user-images.githubusercontent.com/47033052/105458293-9aca5580-5ccb-11eb-94d5-8ffcd9fe0c69.png" alt="image" style="zoom:50%;" />



### 여러 개의 특성을 표현하는 방법

- 3장과 다르게 가중치와 입력값을 묶어서

<img src="https://user-images.githubusercontent.com/47033052/105461579-c996fa80-5cd0-11eb-9f14-5a434df1cbf3.png" alt="image" style="zoom:50%;" />



### 아달린(Adaline)

- z에서 시작하여 역방향 계산

![image](https://user-images.githubusercontent.com/47033052/105462036-6bb6e280-5cd1-11eb-80f4-d2296ba5bb2f.png)

### 로지스틱 회귀

- 아달린처럼 중간에서 시작하지만, z가 아닌 활성화 함수(시그모이드)를 거친 a값 사용
- 시그모이드로 a를 0~1 범위의 값으로 만듦으로써 판단하기 쉬워짐

![image](https://user-images.githubusercontent.com/47033052/105462479-057e8f80-5cd2-11eb-937c-4fc795d9f1d1.png)



## 4-2 시그모이드 함수로 확률을 만듦

### 시그모이드 함수

- 오즈 비: p가 성공 확률일 때, 0부터 무한대까지 값이 바뀌는 확률
- 로짓 함수: 마찬가지로 p가 0~1일때, 값이 마이너스무한대에서 양수무한대로까지 바뀜
  - 이 값의 범위를 z로 생각 가능

![image](https://user-images.githubusercontent.com/47033052/105463104-eaf8e600-5cd2-11eb-8f51-45a712f1535b.png)

```python
# 오즈 비
probs = np.arange(0, 1, 0.01)
odds = [p/(1-p) for p in probs]
plt.plot(probs, odds)
plt.xlabel('p')
plt.ylabel('p/(1-p)')
plt.show()

# 로짓 함수
probs  = np.arange(0.001, 0.999, 0.001)
logit = [np.log(p/(1-p)) for p in probs]
plt.plot(probs, logit)
plt.xlabel('p')
plt.ylabel('log(p/(1-p))')
plt.show()

# 시그모이드 함수
zs = np.arange(-10., 10., 0.1)
gs = [1/(1+np.exp(-z)) for z in zs]
plt.plot(zs, gs)
plt.xlabel('z')
plt.ylabel('1/(1+e^-z)')
plt.show()
```



### 로지스틱 함수

- 로짓 함수를 확률 p에 대해 정리한 것
- 값의 범위가 무한대에서 0~1로 변경 -> 시그모이드 함수

![image](https://user-images.githubusercontent.com/47033052/105463383-55aa2180-5cd3-11eb-8d6c-9f0390c0486e.png)



### 로지스틱 회귀 중간 정리

- a를 기준 오차역전파 역방향 계산

![image](https://user-images.githubusercontent.com/47033052/105463586-a0c43480-5cd3-11eb-9850-dd77844fc8a6.png)



## 4-3 로지스틱 손실 함수를 경사 하강법에 적용

- 분류는 정확도는 선형 회귀와 달리 미분 가능한 함수가 아님.
- 대신 이진 크로스 엔트로피(binary cross entropy) 또는 로지스틱 손실 함수를 사용 
- 아래 사진은 로지스틱 손실 함수식
  - asms 0~1사이의 값이므로 y가 1인 경우(양성)에는 L이 1에 가까울수록  작은 값
  - y가 0인 경우(음성)에는 L이 0에 가까울수록  작은 값

![image](https://user-images.githubusercontent.com/47033052/105464573-25fc1900-5cd5-11eb-92b4-14b5f2a28db3.png)



### 로지스틱 손실 함수 미분

- 제곱 오차와 로지스틱 손실 함수의 미분 결과가 같음!
- 같은 경사하강법 적용 가능

![image](https://user-images.githubusercontent.com/47033052/105464743-5c399880-5cd5-11eb-84de-4ddde883cdc7.png)



### 미분의 연쇄 법칙

- 도함수의 원리를 이용해서 로지스틱 가중치와 절편에 대한 미분 가능
  - L을 w에 대해 미분하려면, L을 a에 대해, a를 z에 대해, z를 w에 대해 미분하기
- 역전파 되는듯한 모습

![image](https://user-images.githubusercontent.com/47033052/105465458-47a9d000-5cd6-11eb-9f6f-c61fa40e0777.png)

- L을 a에 대해 미분

![image](https://user-images.githubusercontent.com/47033052/105465933-eb937b80-5cd6-11eb-9a46-5b58a7aad6f4.png)

- a를 z에 대해 미분
  - 1/{1+e^-z}는 시그모이드 함수니까 a로 바꿀 수 있음

![image](https://user-images.githubusercontent.com/47033052/105466334-6a88b400-5cd7-11eb-8259-6c4ca4404188.png)

- z를 w에 대해 미분

![image](https://user-images.githubusercontent.com/47033052/105467020-6f9a3300-5cd8-11eb-8147-e4bce3a012a4.png)

- 세 미분식 곱하기(L을 w에 대한 미분값)
- 경사하강법을 따라서 w_new = w + (y-a)x

![image](https://user-images.githubusercontent.com/47033052/105467187-ad975700-5cd8-11eb-871c-4502e0d6faec.png)



### 절편에 대한 도함수

- 위 방법과 마찬가지로 구하면
- b_new = b + (y-a)*1

![image](https://user-images.githubusercontent.com/47033052/105468353-3ebafd80-5cda-11eb-90e1-8d8875b8ad24.png)



## 4-4 데이터셋 이용해 분류하는 뉴런 만들기

- 데이터셋은 양성/악성 종양 이진 분류
- 3장과 마찬가지로 사이킷런 라이브러리 제공 데이터셋 사용(load_breast_cancer)

``` python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

print(cancer.data.shape, cancer.target.shape)
// (569, 30) (569,)
```

### 박스 플롯 그려서 데이터 파악

- 중간값은 평균이 아님.

``` python
plt.boxplot(cancer.data)
plt.xlabel('feature')
plt.ylabel('value')
plt.show()
```

![image](https://user-images.githubusercontent.com/47033052/105479577-f6a2d780-5ce7-11eb-9cf1-05ed3aa4fa5a.png) <img src="https://user-images.githubusercontent.com/47033052/105480481-256d7d80-5ce9-11eb-8939-3b89afcea9b3.png" alt="image" style="zoom: 33%;" />

### 타깃 데이터 확인 및 훈련 데이터 준비

``` python
# 타깃 데이터 확인
np.unique(cancer.target, return_counts=True)
# (array([0, 1]), array([212, 357]))
# 음성(정상): 212개, 양성(악성): 357개
    
# 훈련 데이터
x = cancer.data
y = cancer.target
```



## 4-5 로지스틱 회귀 모델 생성

- 일반화 성능을 평가하기 위해 훈련set 테스트set 나누기

![image](https://user-images.githubusercontent.com/47033052/105480334-f22aee80-5ce8-11eb-9a13-8eef7f14572f.png)

### 훈련 dataset을 훈련 dataset과 테스트 dataset으로 나누는 규칙

- 테스트set보다 훈련set이 더 많아야 함.
- 훈련 dataset을 나누기 전, 양성/음성 클래스가 어느 한 쪽에 몰리지 않도록 골고루 섞어야 함.
- 코드

``` python
from sklearn.model_selection import train_test_split

# 75:25(기본값) split
# stratify: 양성/음성 훈련이랑 테스트 비율 동일하게 
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

# 분할 결과 확인(비율 비슷)
print(x_train.shape, x_test.shape)
# (455, 30) (114, 30)

np.unique(y_train, return_counts=True)
# (array([0, 1]), array([170, 285]))
np.unique(y_test, return_counts=True)
```



### 로지스틱 뉴런 구현

- 클래스

``` python
class LogisticNeuron:
    
    def __init__(self):
        self.w = None		# 특성의 개수가 유동적이라 미리 초기화 안함.
        self.b = None

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b  # 곱한 원소들 한번에 더하는 원소별 연산
        return z

    def backprop(self, x, err):
        w_grad = x * err    # 가중치에 대한 그래디언트
        b_grad = 1 * err    # 절편에 대한 그래디언트
        return w_grad, b_grad

    # 시그모이드
    def activation(self, z):
        z = np.clip(z, -100, None) # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z))  # 시그모이드 계산
        return a
        
    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])      # 가중치를 초기화(열 개수 size의 1 원소 가진 배열)
        self.b = 0                        # 절편을 초기화
        for i in range(epochs):           # epochs만큼 반복
            for x_i, y_i in zip(x, y):    # 모든 샘플에 대해 반복
                z = self.forpass(x_i)     # 정방향 계산
                a = self.activation(z)    # 활성화 함수 적용
                err = -(y_i - a)          # 오차 계산
                w_grad, b_grad = self.backprop(x_i, err) # 역방향 계산
                self.w -= w_grad          # 가중치 업데이트
                self.b -= b_grad          # 절편 업데이트
    
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]    # 정방향 계산
        a = self.activation(np.array(z))        # 활성화 함수 적용
        return a > 0.5
```

- 모델 훈련 및 결과 확인

``` python
neuron = LogisticNeuron()
neuron.fit(x_train, y_train)

np.mean(neuron.predict(x_test) == y_test)
# 0.8245614035087719
```

 

## 4-6 로지스틱 회귀 뉴런으로 단일층 신경망 만듦

- 로지스틱 회귀 = 가장 작은 신경망 단위
- 로지스틱이 쌓여서 신경망 알고리즘을 구현함
- 은닉층이 하나 이상이면 딥러닝, 심층 신경망 
- 은닉층이 없으면, 단일 신경망 = 로지스틱 회귀
- 이 단원에서는 활성화 함수가 시그모이드 함수
- 보통 입력층은 입력 데이터이며 라이브러리에 입력층 클래스가 존재하더라도 따로 알고리즘이 있지 않음.

![image](https://user-images.githubusercontent.com/47033052/105484869-636da000-5cef-11eb-8f5f-52a592097605.png)



### 손실 함수 결과값 추가한 단일층 신경망

- 에포크마다 랜덤으로 훈련 샘플을 섞어 fit

``` python
class SingleLayer:
    
    def __init__(self):
        self.w = None
        self.b = None
        self.losses = []

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b  # 직선 방정식을 계산합니다
        return z

    def backprop(self, x, err):
        w_grad = x * err    # 가중치에 대한 그래디언트를 계산합니다
        b_grad = 1 * err    # 절편에 대한 그래디언트를 계산합니다
        return w_grad, b_grad

    def activation(self, z):
        z = np.clip(z, -100, None) # 안전한 np.exp() 계산을 위해
        a = 1 / (1 + np.exp(-z))  # 시그모이드 계산
        return a
        
    def fit(self, x, y, epochs=100):
        self.w = np.ones(x.shape[1])               # 가중치를 초기화
        self.b = 0                                 # 절편을 초기화
        for i in range(epochs):                    # epochs만큼 반복
            loss = 0
            # 인덱스를 랜덤
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:                      # 모든 샘플에 대해 반복
                z = self.forpass(x[i])             # 정방향 계산
                a = self.activation(z)             # 활성화 함수 적용
                err = -(y[i] - a)                  # 오차 계산
                w_grad, b_grad = self.backprop(x[i], err) # 역방향 계산
                self.w -= w_grad                   # 가중치 업데이트
                self.b -= b_grad                   # 절편 업데이트
                # 안전한 로그 계산을 위해 클리핑한 후 손실을 누적
                a = np.clip(a, 1e-10, 1-1e-10)
                loss += -(y[i]*np.log(a)+(1-y[i])*np.log(1-a))
            # 에포크마다 평균 손실(로지스틱 손실)을 저장
            self.losses.append(loss/len(y))
    
    def predict(self, x):
        z = [self.forpass(x_i) for x_i in x]     # 정방향 계산
        return np.array(z) > 0                   # 스텝 함수 적용
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)
```

### 모델 훈련 결과 확인

``` python
layer = SingleLayer()
layer.fit(x_train, y_train)
layer.score(x_test, y_test)
# 0.9298245614035088
```

``` python
plt.plot(layer.losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```

![image](https://user-images.githubusercontent.com/47033052/105485535-64eb9800-5cf0-11eb-9975-88924f3795e6.png)



### 여러가지 경사 하강법

- 확률적 경사 하강법: 무작위로 샘플을 뽑아 여러 방향으로 흐르게 함으로써 그레이디언트 계산
- 배치 경사 하강법: 전체 샘플 모두 뽑아 한번에 그레이디언트 계산
- 데이터셋이 너무 클때는 배치가 좋지만, 확률적 경사 하강법은 데이터를 하나씩 꺼내쓰기 때문에 빠르게 최적의 가중치 다다를 수 있음.
- 이를 절충한 미니 배치 하강법 사용함.
  - 미니 배치 하강법: 전체 샘플 중 몇개를 무작위로 선택해서 그레이디언트 계산

![image](https://user-images.githubusercontent.com/47033052/105486130-5651b080-5cf1-11eb-8648-14be72e279f0.png)



## 4-7 사이킷런으로 로지스틱 회귀 수행

- `loss='log'`: 손실함수는 로지스틱 손실함수로

``` python
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(loss='log', max_iter=100, tol=1e-3, random_state=42)
sgd.fit(x_train, y_train)
sgd.score(x_test, y_test)
# 0.8333333333333334

sgd.predict(x_test[0:10])
# array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
```