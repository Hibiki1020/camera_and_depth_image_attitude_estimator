from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as nn_functional
from collections import OrderedDict

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 norm_layer=None, bn_eps=1e-5, bn_momentum=0.1,
                 downsample=None, inplace=True):
        super(Bottleneck, self).__init__()
        #print(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes, eps=bn_eps, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(planes * self.expansion, eps=bn_eps,
                              momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        # first path
        x1 = x[0]
        residual1 = x1

        out1 = self.conv1(x1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu(out1)

        out1 = self.conv3(out1)
        out1 = self.bn3(out1)

        if self.downsample is not None:
            residual1 = self.downsample(x1)

        out1 += residual1
        out1 = self.relu_inplace(out1)

        return [out1]

class ResNet(nn.Module):
    def __init__(self, block, layers, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 bn_momentum=0.1, deep_stem=False, stem_width=32, inplace=True):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNet, self).__init__()

        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1,
                          padding=1,
                          bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)

        self.bn1 = norm_layer(64, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(1,block, norm_layer, 64, layers[0],
                                       inplace,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(2,block, norm_layer, 128, layers[1],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(3,block, norm_layer, 256, layers[2],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(4,block, norm_layer, 512, layers[3],
                                       inplace, stride=2,
                                       bn_eps=bn_eps, bn_momentum=bn_momentum)

    def _make_layer(self, number, block, norm_layer, planes, blocks, inplace=True,
                    stride=1, bn_eps=1e-5, bn_momentum=0.1):
        
        downsample = None
        

        if stride != 1 or self.inplanes != planes * block.expansion:
            if number==1:
                downsample = nn.Sequential(
                    nn.Conv2d(64, 256,
                            kernel_size=1, stride=stride, bias=False),
                    norm_layer(256, eps=bn_eps,
                           momentum=bn_momentum),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    norm_layer(planes * block.expansion, eps=bn_eps,
                           momentum=bn_momentum),
                )

        layers = []
        #print(self.inplanes)
        #print(planes)
        if number==1:
            layers.append(block( planes, planes, stride, norm_layer, bn_eps, bn_momentum, downsample, inplace))
        else:
            layers.append(block(self.inplanes, planes, stride, norm_layer, bn_eps, bn_momentum, downsample, inplace))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if number==1:
                layers.append(block(self.inplanes, planes, norm_layer=norm_layer, bn_eps=bn_eps, bn_momentum=bn_momentum, inplace=inplace))
            else:
                layers.append(block(self.inplanes, planes, norm_layer=norm_layer, bn_eps=bn_eps, bn_momentum=bn_momentum, inplace=inplace))

        return nn.Sequential(*layers)

    def forward(self, input):
        x1 = self.conv1(input)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x = x1

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

def load_model(model, model_file, is_restore=False):
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file)

        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(raw_state_dict, strict=False)

    return model


def resnet50(pretrained_model=None, **kwargs):
    resnet_mono = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    resnet_depth = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained_model is not None:
        resnet_mono = load_model(resnet_mono, pretrained_model)
        resnet_depth = load_model(resnet_depth, pretrained_model)

    return resnet_mono, resnet_depth
