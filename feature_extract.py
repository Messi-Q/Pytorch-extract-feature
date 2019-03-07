import torch
import torchvision
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import models, transforms


# 中间层特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            print(name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=torchvision.transforms.ToTensor(),
                                        download=False)
train_loader = data.DataLoader(train_data, batch_size=64, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)) / 255
# test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)).cuda() / 255
test_y = test_data.test_labels
# test_y = test_data.test_labels.cuda()

vgg16 = models.vgg16(pretrained=True)
print(vgg16)
