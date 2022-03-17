#!/bin/bash

cur_dir = %CD%

mkdir -p /home/weights
cd /home/weights
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
wget https://download.pytorch.org/models/resnet34-b627a593.pth
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
wget https://download.pytorch.org/models/resnet101-63fe2227.pth
wget https://download.pytorch.org/models/resnet152-394f9c45.pth
#wget https://download.pytorch.org/models/vgg16-397923af.pth
cd ${cur_dir}