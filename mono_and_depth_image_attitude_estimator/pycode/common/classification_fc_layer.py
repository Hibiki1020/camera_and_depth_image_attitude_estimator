import functools
import torch
import torch.nn as nn
import torch.nn.functional as nn_functional

class ClassificationType(nn.Module):
    def __init__(self, dim_fc_out, dropout_rate):
        super(ClassificationType, self).__init__()

        self.dim_fc_in = 100352
        self.dim_fc_out = dim_fc_out
        self.dropout_rate = dropout_rate

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
        list_roll_fc_param_value = []
        list_pitch_fc_param_value = []
        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "roll_fc" in param_name:
                list_roll_fc_param_value.append(param_value)
            if "pitch_fc" in param_name:
                list_pitch_fc_param_value.append(param_value)

    def forward(self, feature):
        roll = self.roll_fc(feature)
        pitch = self.pitch_fc(feature)

        logged_roll = nn_functional.log_softmax(roll, dim=1)
        logged_pitch = nn_functional.log_softmax(pitch, dim=1)

        torch.set_printoptions(edgeitems=10000)

        return logged_roll, logged_pitch, roll, pitch


