import torch
from torchvision import models
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, dropout_rate):
        super(Network, self).__init__()

        dim_fc_in = 512
        self.fc = nn.Sequential(
            nn.Linear(dim_fc_in, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(100, 18),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(18, 2)
        )

    def initializeWeights(self):
        for m in self.fc.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def getParamValueList(self):
        list_fc_param_value = []
        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "fc" in param_name:
                list_fc_param_value.append(param_value)
            
        return list_fc_param_value

    def forward(self, x):
        x = self.fc(x)
        l2norm = torch.norm(x[:, :2].clone(), p=2, dim=1, keepdim=True)
        x[:, :2] = torch.div(x[:, :2].clone(), l2norm)
        return x