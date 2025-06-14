import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

x = torch.rand(5,3)
# print(x)

import torch
# print(torch.cuda.is_available())   # True 表示可用，False 表示不可用
# print(torch.cuda.device_count())   # 可用GPU的数量

import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.__version__)


print("GPU数量:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}:", torch.cuda.get_device_name(i))


import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.plot([1,2,3], [1,4,9])
plt.title("Test")
plt.show()
input("等待窗口关闭...")

