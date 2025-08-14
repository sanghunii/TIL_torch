# Torch DeepLearning Model 전체적인 구조와 이해 코드 (학습용 코드)
"""
<< ref >>
https://rueki.tistory.com/91
"""

import os
import torch 
from torch import nn
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
"""
torchvision.datasets : MNIST등 Test Data set을 지원
torchvision.transforms : Give Data Transform tools
"""


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
GPU에서 계산을 수행하고 싶을 때마다 계상에 필요한 모든 데이터를 GPU장치가 접근 가능한 메모리로 옮겨야 한다.
기본적으로 새로운 tensor는 CPU에 생성된다. 따라서 tensor를 GPU에 생성하고 싶을 때는 
device선택인자를 반드시 명시해줘야 한다. 

1. GPU에 tensor를 바로 생성 
2. CPU에 tensor를 생성하고 GPU로 옮김.
"""


"""
신경망은 데이터에 대한 연산을 수행하는 계층/모듈로 구성되어 있다. 
torch.nn 네임스페이스는 신경망을 구성하는데 필요한 모든 구성 요소를 제공한다.
PyTorch의 모든 모듈은 nn.Module의 하위 클래스이다.
신경망은 다른 모듈로 구성된 모듈이다. 
이러한 중첩된 구조는 복잡한 아키텍처를 쉽게 구축하고 관리할 수 있다. 
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
    

    def forward(self, x):
        """
        nn.Moduel을 상속받은 모든 클래스는 forward메서드를 오버라이딩 해야 한다.
        이때 입력데이터에 대한 연산들을 구현한다. 
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

# NeuralNetwork의 인스턴스를 생성하고 이를 device로 이동한 뒤, 구조(structure)를 출력
"""GPU를 사용하려면 계산상 필요한 모든 데이터가 GPU로 전달되어야 한다."""
model = NeuralNetwork().to(device=device) #  Move tensor on GPU after gerate tensor on CPU
print(f' << show model structure >> {model}\n\n') 


X = torch.rand(1,28,28, device=device) 
logits = model(X) #model에 입력을 전달하면 2-dim tensor를 반환, {dim=0 : 원시 예측 클래스, dim=1 : 원시 예측 클래스에 대한 수치 값}
pred_probab = nn.Softmax(dim=1)(logits) #이때 nn.Softmax는 callble object



# Model Layer - 28*28 image 3개짜리 minibatch를 이용한 model layer에 대한 고찰
input_image = torch.rand(3,28,28)
print(input_image.size())

## Flatten Layer
flatten = nn.Flatten() #평탄화작업 : [3,28,28] -> [784, 3]
flat_image = flatten(input_image)
print(flat_image.size())

## Linear Layer ; apply linaer transformation 
layer1 = nn.Linear(in_features=28*28, out_features=20)  # 28*28값의 input을 20으로 선형변환
hidden1 = layer1(flat_image)
print(f'{flat_image.size()}를 {hidden1.size()}로 선형변환\n\n')


## Activation Function Layer ; 네트워크에 비선형성을 추가해 줌으로서 Model이 복잡한 비선형 문제에서도 좋은 성능을 낼 수 있게 해준다. 
### 추가로 hidden Layer만 여러겹 쌓는 것은 그냥 1층짜리 Network와 다를바가 없다. (합성함수로 묶으면 그냥 하나의 Linear Function이다.)
print(f"Before ReLU : {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU : {hidden1}\n\n")


# Sequential Model
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10)       #nn.Linear(input_size, output_size)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)


# nn.Softmax ; Neural Network의 마지막 선형 계층은 softmax layer이다. (binary classification문제에서는 sigmoid 함수이다.)
softmax = nn.Softmax(dim=1) # dim=1은 열방향(왼쪽에서 오른쪽)으로 계산한다는 뜻 
pred_probab = softmax(logits)


# Model Parameter  (그닥 안중요한듯)
## 주요 메서드 -> 1. parameters(), 2. named_parameters()
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size : {param.size()} | Values : {param[:2]}\n")