import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets , transforms

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
print(f"using device is {device}")

#nn神经网络
class NeuralNetwork(nn.Module): #nn.Module to device
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear__relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    
    def forward(self , x):
        x = self.flatten(x)
        logits = self.linear__relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


#test1
X = torch.rand(1, 28, 28, device=device) #构建28*28的tensor
logits = model(X)
pred_probab = nn.Softmax(dim = 1)(logits) #归一化
y_pred = pred_probab.argmax(1) #选出最大概率的
# print(pred_probab)
# print(f"Predicted class: {y_pred}")


#test2
input_image = torch.rand(3, 28, 28)

flatten = nn.Flatten()
flat_image = flatten(input_image)

layer1 = nn.Linear(28*28, 20)
hidden1 = layer1(flat_image)

hidden2 = nn.ReLU()(hidden1)

# print(input_image.size())
# print(flat_image.size())
# print(hidden1.size())
# print(hidden1)
# print(hidden2)

test_model = nn.Sequential(
    flatten ,
    layer1 ,
    nn.ReLU() ,
    nn.Linear(20, 10)
)
test_image = torch.rand(3, 28, 28)
logits = test_model(test_image)
# print(logits)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

