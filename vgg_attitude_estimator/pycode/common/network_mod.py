import sys
sys.dont_write_bytecode = True

from common import feature_extractor_mod
from common import classification_fc_layer

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as nn_functional

class Network(nn.Module):
    def __init__(self, model, dim_fc_out, norm_layer, pretrained_model):
        super(Network, self).__init__()
        self.dim_fc_out = dim_fc_out

        self.feature_extractor = feature_extractor_mod.VGG16(pretrained_model)
        self.fully_connected = classification_fc_layer.ClassificationType("vgg", dim_fc_out, 0.1)

    def forward(self, mono, depth):
        feature_map = self.feature_extractor(mono, depth)
        fc_input = torch.flatten(feature_map, 1)

        roll = self.fully_connected.roll_fc(fc_input)
        pitch = self.fully_connected.pitch_fc(fc_input)

        logged_roll = nn_functional.log_softmax(roll, dim=1)
        logged_pitch = nn_functional.log_softmax(pitch, dim=1)

        torch.set_printoptions(edgeitems=10000)

        return logged_roll, logged_pitch, roll, pitch

    def getParamValueList(self):
        list_vgg_mono_param_value = []
        list_vgg_depth_param_value = []
        list_roll_fc_param_value = []
        list_pitch_fc_param_value = []

        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "vgg_mono" in param_name:
                list_vgg_mono_param_value.append(param_value)
            if "vgg_depth" in param_name:
                list_vgg_depth_param_value.append(param_value)
            if "roll_fc" in param_name:
                list_roll_fc_param_value.append(param_value)
            if "pitch_fc" in param_name:
                list_pitch_fc_param_value.append(param_value)

        return list_vgg_mono_param_value, list_vgg_depth_param_value, list_roll_fc_param_value, list_pitch_fc_param_value