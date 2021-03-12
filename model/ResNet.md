## ILSVRC에서 우승 차지 : ResNet

# Deep Residual Learning for Image Recognition

- GoogLeNet(2014) : 22개의 층
- ResNet : 152개의 층


---

- 기존의 방식으로 망을 무조건 깊게 한다고 성능이 모두 좋아짐을 아닌 것을 확인함
- **Residual Block** 탄생

---

## Residual Block
- 입력값을 출력값에 더해줄 수 있도록 shortcut 하나를 만들어 줌
- H(x) = F(x) + x (이것을 최소로 하는 것이 목표, F(x)를 0으로 가깝게 만들기)
- H(x)-x: 잔차(residual)
- 잔차를 최소로 해주는 것 ResNet


---

망을 깊게 할 수록 좋은 성능을 냄을 알 수 있다   
ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
