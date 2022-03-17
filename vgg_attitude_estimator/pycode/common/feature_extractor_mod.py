from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as nn_functional

class VGG16:
    def __init__(self, pretrained_model):
        self.vgg_mono = models.vgg16()
        self.vgg_depth = models.vgg16()

        self.vgg_mono.load_state_dict(torch.load(pretrained_model))
        self.vgg_depth.load_state_dict(torch.load(pretrained_model))

    def forward(self, mono, depth):
        x1 = self.vgg_mono(mono)
        x2 = self.vgg_depth(depth)
        x3 = torch.cat((x1, x2), 1)

        return x3

    def getParamValueList(self):
        list_vgg_mono_param_value = []
        list_vgg_depth_param_value = []

        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "vgg_mono" in param_name:
                list_vgg_mono_param_value.append(param_value)
            if "vgg_depth" in param_name:
                list_vgg_depth_param_value.append(param_value)

        return list_vgg_mono_param_value, list_vgg_depth_param_value
