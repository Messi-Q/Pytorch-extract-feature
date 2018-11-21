import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, glob

data_dir = './test'  # train
features_dir = './DenseNet_features_test'  # DenseNet_features_train

print(models.vgg16().children)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        densnet = models.densenet121(pretrained=True)
        self.feature = densnet.features
        self.classifier = nn.Sequential(*list(densnet.classifier.children())[:-1])
        pretrained_dict = densnet.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)

    def forward(self, x):
        output = self.feature(x)
        avg = nn.AvgPool2d(7, stride=1)
        output = avg(output)
        return output


model = Encoder()
model = model.cuda()


def extractor(img_path, saved_path, net, use_gpu):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )

    img = Image.open(img_path)
    img = transform(img)
    print(img.shape)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    print(x.shape)

    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()
    print(y.shape)
    np.savetxt(saved_path, y, delimiter=',')


if __name__ == '__main__':
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

    files_list = []
    x = os.walk(data_dir)
    for path, d, filelist in x:
        for filename in filelist:
            file_glob = os.path.join(path, filename)
            files_list.extend(glob.glob(file_glob))

    print(files_list)

    use_gpu = torch.cuda.is_available()

    for x_path in files_list:
        # print("x_path" + x_path)
        file_name = x_path.split('/')[-1]
        fx_path = os.path.join(features_dir, file_name + '.txt')
        # print(fx_path)
        extractor(x_path, fx_path, model, use_gpu)
