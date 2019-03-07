import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms

use_gpu = torch.cuda.is_available()


# 中间层特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    # 自己修改forward函数
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


extract_list = ["conv1", "maxpool", "layer1", "avgpool", "fc"]
img_path = "./1_00001.jpg"
saved_path = "./1_00001.txt"
resnet = models.resnet50(pretrained=True)
# print(resnet) 可以打印看模型结构

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()]
)

img = Image.open(img_path)
img = transform(img)

x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)

if use_gpu:
    x = x.cuda()
    resnet = resnet.cuda()

extract_result = FeatureExtractor(resnet, extract_list)
print(extract_result(x)[4])  # [0]:conv1  [1]:maxpool  [2]:layer1  [3]:avgpool  [4]:fc

