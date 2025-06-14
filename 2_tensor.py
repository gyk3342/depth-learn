import torch
from torchvision import datasets
import numpy as np

data = [[1,2],[3,4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(f"Epoch {0}\n-------------------------------")

x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data , dtype=torch.float)

print(x_ones)
print(x_rand)

print(f"Epoch {1}\n-------------------------------")

shape = (2,3,) #张量的维度数 参数多少表示多少维度
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

print(f"Epoch {2}\n-------------------------------")

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

print(f"Epoch {3}\n-------------------------------")

if torch.cuda.is_available():
    tensor = tensor.to("cuda")
else:
    tensor = tensor.to("cpu")
print(f"using {tensor.device} device")

print(f"Epoch {4}\n-------------------------------")

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

print(f"Epoch {5}\n-------------------------------")
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
# print(tensor)
# print(y2)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)
# print(tensor)
# print(z2)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))








