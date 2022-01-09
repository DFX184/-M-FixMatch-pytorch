from torchvision import models
import torch.nn as nn
import torch
def build_WideResNet(pretrain = False,num_classes = 10):
    model = models.wide_resnet50_2(pretrain)
    model.fc = nn.Linear(2048,num_classes)
    print(model)
    return model

def build_resnet18(pretrain=False,num_classes = 11):
    model = models.resnet18(pretrain)
    model.fc = nn.Linear(512,num_classes)
    return model

def build_resnet50(pretrain=False,num_classes = 5):
    model = models.resnext50_32x4d(pretrain)
    model.fc = nn.Linear(2048,num_classes)
    return model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3,1,2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(3),
            nn.Conv2d(16, 32, 3,1,2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128 * 1 * 1, 11)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        return self.fc(x)
    
##################### Mini-DenseNet #####################


class Bottleneck(nn.Module):
    def __init__(self, InChannel, GrowthRate):
        super(Bottleneck, self).__init__()
        self.Layer = nn.Sequential(
            nn.BatchNorm2d(InChannel),
            nn.ReLU(),
            nn.Conv2d(InChannel, GrowthRate*4, 1, bias=False),
            nn.BatchNorm2d(GrowthRate*4),
            nn.ReLU(),
            nn.Conv2d(GrowthRate*4, GrowthRate,
                      3, 1, 1, bias=False)
        )
    def forward(self, x):
        out = self.Layer(x)
        return torch.cat((x, out), dim=1)


class DenseBlock(nn.Module):
    def __init__(self, InChannel, GrowthRate, num=4):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(int(num)):
            layer.append(Bottleneck(InChannel, GrowthRate))
            InChannel += GrowthRate
        self.Block = nn.Sequential(*layer)

    def forward(self, x):
        return self.Block(x)


class Classifier(nn.Module):
    def __init__(self, InChannel, numclass):
        super(Classifier, self).__init__()
        self.module = nn.Sequential(
            DenseBlock(InChannel, 32),
            nn.MaxPool2d(2),  # 16
            nn.Dropout2d(0.2),
            DenseBlock(InChannel+32*4, 32),
            nn.MaxPool2d(2),  # 8
            nn.Dropout2d(0.2),
            DenseBlock((InChannel+32*4)+32*4, 32),
            nn.MaxPool2d(2),  # 4
            DenseBlock(((InChannel+32*4)+32*4)+32*4, 32),
            nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((1, 1))

        )
        self.L = nn.Linear((((InChannel+32*4)+32*4)+32*4)+32*4, numclass)

    def forward(self, x):
        x = self.module(x)
        x = x.view(x.size()[0], -1)
        return self.L(x)

