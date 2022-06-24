import torch
from torch import nn
import torchvision.models as models


def ResNet(name, GOP=3, pretrained=True):
    if name== 'resnet18':
        model = models.resnet18(pretrained=False)
        if pretrained:
            model.load_state_dict(torch.load('./resnet_model/resnet18-5c106cde.pth'))
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, 1)
        if GOP != 3:
            model.conv1 = nn.Conv2d(GOP, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if name == 'resnet34':
        model = models.resnet34(pretrained=False)
        if pretrained:
            model.load_state_dict(torch.load('./resnet_model/resnet34-b627a593.pth'))
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, 1)
        if GOP != 3:
            model.conv1 = nn.Conv2d(GOP, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if name == 'resnet50':
        model = models.resnet50(pretrained=False)
        if pretrained:
            model.load_state_dict(torch.load('./resnet_model/resnet50-19c8e357.pth'))
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, 1)
        if GOP != 3:
            model.conv1 = nn.Conv2d(GOP, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model

def CatResNet(name, GOP=3, pretrained=True):
    if name== 'resnet18':
        model = models.resnet18(pretrained=False)
        if pretrained:
            model.load_state_dict(torch.load('./resnet_model/resnet18-5c106cde.pth'))
        fc_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        model.conv1 = nn.Conv2d(GOP * 2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if name == 'resnet34':
        model = models.resnet34(pretrained=False)
        if pretrained:
            model.load_state_dict(torch.load('./resnet_model/resnet34-b627a593.pth'))
        fc_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        model.conv1 = nn.Conv2d(GOP * 2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if name == 'resnet50':
        model = models.resnet50(pretrained=False)
        if pretrained:
            model.load_state_dict(torch.load('./resnet_model/resnet50-19c8e357.pth'))
        fc_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        model.conv1 = nn.Conv2d(GOP * 2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model