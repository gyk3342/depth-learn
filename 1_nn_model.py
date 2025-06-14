import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import time


start = time.time()

#处理数据
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
train_dataloader = DataLoader(training_data , batch_size= batch_size)
test_dataloader = DataLoader(test_data , batch_size=batch_size)

for x , y in test_dataloader:
    print(f"shape of x [N,C,H,W] : {x.shape}")
    print(f"shape of y:{y.shape} {y.dtype}")
    break


#创建模型
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"using {device} device")

class NeuralNetwork(nn.Module): #神经网络
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #铺平为一维(28*28 -> 784)
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(28*28, 512), #线性层  In 728/Out 512
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self,x):
        x = self.flatten(x) #展平
        logits = self.linear_relu_stack(x) #神经网络
        return logits

model = NeuralNetwork().to(device)


#优化模型
loss_fn = nn.CrossEntropyLoss() #损失函数
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3) #优化器 SGD 学习率0.001 

def train(dataloader , model , loss_fn , optimezer):
    size = len(dataloader.dataset) #训练次数
    model.train()
    for batch, (x,y) in enumerate(dataloader): #遍历批次 x图片 y标签
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)
        
        loss.backward() #自动导数 计算梯度
        optimezer.step() #更新优化参数
        optimezer.zero_grad() #梯度为零

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"{batch}: loss {loss:>7f} [{current:>5d}/{size:>5d}]")



#评估模型
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() #评估模式
    test_loss , correct = 0 , 0
    with torch.no_grad():
        for x , y in dataloader:
            x , y = x.to(device) , y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred , y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /=size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")

model.load_state_dict(torch.load("model.pth" , weights_only=True))


#测试模型
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x , y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    x = x.unsqueeze(0)
    pred = model(x)
    predicted , actual = classes[pred[0].argmax(0)] , classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')



end = time.time()


print(f"train: {end - start:.2f} sec")