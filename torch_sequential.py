## Ref : https://velog.io/@kim_haesol/Pytorch-MNIST-활용해보기
## Theme : MNIST Dataset을 이용한 torch CNN Model 예제 코드
## torch.nn.functional package사용

""" 
nn.Model VS nn.functional

nn.Model은 이전의 방식이고 nn.functional은 이후의 방식이다.
nn.Model은 stateful approach고
nn.functional은 stateless 방식이다. 

본 연구에서는 DQN을 구현할 때 torch를 이용해서 NerualNet(Q-Network)를 구성할 것이다.
이때 backpropagation
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# device 설정 ; cuda가 가능하면 cuda 아니면 cpu
is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')


# setting parametersa
batch_size = 50
learning_rate = 0.0001
epoch_num = 15


# Bring MNIST Dataset 
train_data = datasets.MNIST(root = './data',
                            train = True,
                            download= True,
                            transform = transforms.ToTensor())
test_data = datasets.MNIST(root = './data',
                           train = False,
                           transform=transforms.ToTensor())


# Check data 
print('number of train data : ', len(train_data))
print('number of test data : ', len(test_data))


""" 첫번째 MNIST DATA 그려보기
image, label = train_data[0]
plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title(f'label : {label}')
plt.show()"""



# Consist Minibatch
## torch에서는 Dataloader를 이용해서 Minibatch를 만들고 모델에 전달하고 하는 process를 구성한다.
train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_data,
                         batch_size=batch_size,
                         shuffle=True)
first_batch = train_loader.__iter__().__next__() # 해당 코드로 알 수 있듯이 DataLoader는 iterator를 반환한다.
""" torch의 DataLoader는 iterator인가? generator인가?
-> iterator인듯하다. 왜 그런지는 고민해 보자.
"""


print('{:15s} | {:<25s} | {}'.format('name', 'type', 'size'))
print('{:15s} | {:<25s} | {}'.format('Num of Batch','', len(train_loader)))
print('{:15s} | {:<25s} | {}'.format('first_batch', str(type(first_batch)), len(first_batch)))
print('{:15s} | {:<25s} | {}'.format('first_batch[0]', str(type(first_batch[0])), first_batch[0].shape))
print('{:15s} | {:<25s} | {}'.format('first_batch[1]', str(type(first_batch[1])), first_batch[1].shape))




#Build CNN Model 

"""nn.Conv2d 부가설명
self.conv1 = nn.Conv2d(1, 32, 3, 1, padding='same')은 아래와 같은 코드이다.
self.conv1 = nn.Conv2d(
                in_channels=1
                out_channels = 32,
                stride = 1,
                padding = "same")

padding = "same"  => output size가 input size와 동일하게 만든다. 

또한 torch에서는 입력 이미지의 픽셀크기 (input data의 크기 혹은 모양)을 명시하지 않아도 된다. 
torch nn.Conv2d에서는 입력데이터의 hegiht와 width는 동적으로 처리되며 
입력 텐서의 채널 수(inchannels)와 필터의 속성 (out_channels, kernel_size, stride, padding)만 정의하면 된다.
"""

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding='same')
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding='same')
        self.droput = nn.Dropout2d(0.25) #전체 뉴런의 25%를 끈다. 
        # (입력 뉴런, 출력 뉴런)
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)
        
    def forward(self, x): #forward propagation
        x = self.conv1(x)      # convolusion layer
        x = F.relu(x)          # activation function
        x = F.max_pool2d(x, 2) # pooling layer
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.droput(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1) 
        """softmax대신 log_softmax를 사용하는 이유
        - log_softmax는 softmax함수에 log를 취한 형태이다.
        - 출력 값이 확률(0~1)대신 로그 확률(음수 값)으로 변환된다.
        - 수치적 안정성을 높이고, 특히 손실함수 계산에서 효율적이다. 
        """

        return output 

model = CNN().to(device=device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss() # Lossfunctoin




# Train Model 
model.train()  ## Setting model training model
i = 1
for epoch in range(epoch_num):
    for data, target in train_loader:  ## iterator는 for문으로 순회할 수 있다!!
        data = data.to(device)
        target = target.to(device)

        # Gradient초기화
        optimizer.zero_grad() 

        # Forward propagation
        output = model(data)
        loss = criterion(output, target)
        
        # Back propagation
        loss.backward()  ## Compute gradient 
        optimizer.step()
        
        if i % 1000 == 0: ## model 1000번 돌아갈 때 마다.
            print("Train Step : {}\tLoss : {:3f}".format(i, loss.item()))
        i += 1


# Evaluate Model 
model.eval() ## eval mode에서는 drop out & batch normalization등 추론시에 불필요한 layer들을 다 꺼버린다 또한 back propagation또한 수행하지 않는다. 
correct = 0
for data, target in test_loader:
    data = data.to(device)
    target = target.to(device)
    output = model(data) ## forward propagation
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()  ## 맞춘 갯수
print('Test set Accuracy : {:.2f}%'.format(100. * correct / len(test_loader.dataset)))