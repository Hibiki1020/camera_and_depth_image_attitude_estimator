import feature_extractor
import net_util
import classification_fc_layer
import num_regression_fc_layer

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as nn_functional

class Network(nn.Module):
    def __init__(self, dim_fc_out):
        self.dim_fc_out = dim_fc_out
        self.feature_extractor = feature_extractor.DualResNet()
        self.fully_connected = classification_fc_layer.ClassificationType()

    def forward(self, mono, depth):
        blocks, merges = self.feature_extractor(mono, depth)

        roll = self.fully_connected.roll_fc(merges[3])
        pitch = self.fully_connected.pitch_fc(merges[3])

        logged_roll = nn_functional.log_softmax(roll, dim=1)
        logged_pitch = nn_functional.log_softmax(pitch, dim=1)

        torch.set_printoptions(edgeitems=10000)

        return logged_roll, logged_pitch, roll, pitch
