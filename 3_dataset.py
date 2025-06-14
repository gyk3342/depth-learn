import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import time 


training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor() 
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8,8))
cols , rows = 3 , 3
for i in range(1 , cols*rows + 1):
    sample_id = torch.randint(len(training_data), size=(1,)).item()
    img , label = training_data[sample_id]
    figure.add_subplot(rows , cols , i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze() , cmap="gray")

# plt.savefig("check_output.png")
try:
    # plt.show()
    plt.close()
except Exception as e:
    print("plt.show()异常：", e)

print(f"Epoch {1}\n-------------------------------")

import os
import pandas as pd
from torchvision.io import decode_image

start = time.time()

class CustomImageDataset(Dataset):
    def __init__(self , annotations_file , img_dir , transform = None , target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir #文件路径
        self.transform = transform #图像转换
        self.target_transform = target_transform #标签转换
    
    def __len__(self):
        return len(self.img_labels) #样本数量
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir , self.img_labels.iloc[index , 0])
        image = decode_image(img_path) #图像解码
        label = self.img_labels.iloc[index , 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image , label
    

from torch.utils.data import DataLoader

gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"using {device} device")

# train_dataloader = DataLoader(training_data , batch_size=64 , shuffle=True)
# test_dataloader = DataLoader(test_data , batch_size=64 , shuffle=True)

train_dataloader = DataLoader(training_data , batch_size=64 , shuffle=False)
test_dataloader = DataLoader(test_data , batch_size=64 , shuffle=False)

figure = plt.figure(figsize=(3,3))
cols , rows = 8 , 8
for i in range(1 , cols*rows + 1):
    train_features , train_labels = next(iter(train_dataloader)) #批量调度getitem
    # print(f"Feature batch shape: {train_features.size()}") #[64,1,28,28]
    # print(f"Labels batch shape: {train_labels.size()}") #[64]
    img = train_features[i-1].squeeze() #压缩1维通道 [28*28]
    label = train_labels[i-1]
    figure.add_subplot(rows , cols , i)
    plt.title(f"{i} : Tensor{label}")
    plt.axis("off")
    plt.imshow(img.squeeze() , cmap="gray")

try:
    plt.show()
    plt.close()
except Exception as e:
    print("plt.show()异常：", e)

# train_features , train_labels = next(iter(train_dataloader)) #批量调度getitem
# # print(f"Feature batch shape: {train_features.size()}") #[64,1,28,28]
# # print(f"Labels batch shape: {train_labels.size()}") #[64]
# img = train_features[4].squeeze() #压缩1维通道 [28*28]
# label = train_labels[4]

# end = time.time()
# print(f"train: {end - start:.2f} sec")

# plt.title({label})
# plt.imshow(img , cmap = "gray")
# plt.show()
# print(f"Label: {label}")


#检查第零项
# for i in range(20):
#     img, label = training_data[i]
#     print(f"label{i}: {label}")





