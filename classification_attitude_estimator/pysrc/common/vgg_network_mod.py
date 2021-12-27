from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as nn_functional

class Network(nn.Module):
    def __init__(self, resize, dim_fc_out, dropout_rate, use_pretrained_vgg=True):
        super(Network, self).__init__()

        vgg = models.vgg16(pretrained=use_pretrained_vgg)
        self.cnn_feature = vgg.features

        self.dim_fc_in = 512 * 7 * 7
        self.dim_fc_out = dim_fc_out

        self.roll_fc = nn.Sequential(
            nn.Linear(self.dim_fc_in, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 150, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 100, self.dim_fc_out),
            nn.Softmax(dim=1)
        )

        self.pitch_fc = nn.Sequential(
            nn.Linear(self.dim_fc_in, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 150, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 100, self.dim_fc_out),
            nn.Softmax(dim=1)
        )

        self.initializeWeights()#no need?
    
    def initializeWeights(self):
        for m in self.roll_fc.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

        for m in self.pitch_fc.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
    def getParamValueList(self):
        list_cnn_param_value = []
        list_roll_fc_param_value = []
        list_pitch_fc_param_value = []
        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "cnn" in param_name:
                list_cnn_param_value.append(param_value)
            if "roll_fc" in param_name:
                list_roll_fc_param_value.append(param_value)
            if "pitch_fc" in param_name:
                list_pitch_fc_param_value.append(param_value)
            
        return list_cnn_param_value, list_roll_fc_param_value, list_pitch_fc_param_value

    def forward(self, x):
        feature = self.cnn_feature(x)
        feature = torch.flatten(feature, 1)

        roll = self.roll_fc(feature)
        pitch = self.pitch_fc(feature)

        logged_roll = nn_functional.log_softmax(roll, dim=1)
        logged_pitch = nn_functional.log_softmax(pitch, dim=1)

        torch.set_printoptions(edgeitems=10000)

        return logged_roll, logged_pitch, roll, pitch