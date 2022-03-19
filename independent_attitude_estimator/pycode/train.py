from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import random

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

import sys
from common import dataset_mod
from common import make_datalist_mod
from common import data_transform_mod

from common import network_mod

import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./train.py")

    parser.add_argument(
        '--train_cfg', '-c',
        type=str,
        required=True,
        help='Training configuration file'
    )

    FLAGS, unparsed = parser.parse_known_args()

    #Load yaml file
    try:
        print("Opening train config file %s", FLAGS.train_cfg)
        CFG = yaml.safe_load(open(FLAGS.train_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening train config file %s", FLAGS.train_cfg)
        quit()

    save_top_path = CFG["save_top_path"]
    yaml_path = save_top_path + "/train_config.yaml"
    weights_path = CFG["save_top_path"] + CFG["weights_path"]
    log_path = CFG["save_top_path"] + CFG["log_path"]
    graph_path = CFG["save_top_path"] + CFG["graph_path"]
    csv_name = CFG["csv_name"]
    index_csv_path = CFG["index_csv_path"]
    multiGPU = int(CFG["multiGPU"])

    model = CFG["model"]
    pretrained_model = CFG["pretrained_model"]

    train_sequences = CFG["train"]
    valid_sequences = CFG["valid"]

    dim_fc_out = int(CFG["hyperparameter"]["dim_fc_out"])
    resize = int(CFG["hyperparameter"]["resize"])
    mean_element = float(CFG["hyperparameter"]["mean_element"])
    std_element = float(CFG["hyperparameter"]["std_element"])
    original_size = int(CFG["hyperparameter"]["original_size"])
    batch_size = int(CFG["hyperparameter"]["batch_size"])
    num_epochs = int(CFG["hyperparameter"]["num_epochs"])
    optimizer_name = str(CFG["hyperparameter"]["optimizer_name"])
    lr_resnet = float(CFG["hyperparameter"]["lr_resnet"])
    lr_roll_fc = float(CFG["hyperparameter"]["lr_roll_fc"])
    lr_pitch_fc = float(CFG["hyperparameter"]["lr_pitch_fc"])
    weight_decay = float(CFG["hyperparameter"]["weight_decay"])
    alpha = float(CFG["hyperparameter"]["alpha"])

    shutil.copy(FLAGS.train_cfg, yaml_path)

    '''
    try:
        print("Copy files to %s for further reference." % log_path)
        copyfile(FLAGS.train_cfg, "/train_config.yaml")
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting....")
        quit()
    '''
    print("Load Training Dataset")
    train_dataset = dataset_mod.ClassOriginalDataset(
        data_list = make_datalist_mod.makeMultiDataList(train_sequences, csv_name),
        transform = data_transform_mod.DataTransform(
            resize,
            mean_element,
            std_element,
            original_size
        ),
        phase = "train",
        index_dict_path = index_csv_path,
        dim_fc_out = dim_fc_out
    )

    print("Load Valid Dataset")
    valid_dataset = dataset_mod.ClassOriginalDataset(
        data_list = make_datalist_mod.makeMultiDataList(valid_sequences, csv_name),
        transform = data_transform_mod.DataTransform(
            resize,
            mean_element,
            std_element,
            original_size
        ),
        phase = "valid",
        index_dict_path = index_csv_path,
        dim_fc_out = dim_fc_out
    )

    print("Load ", model)
    net = network_mod.Network(model, dim_fc_out, norm_layer=nn.BatchNorm2d,pretrained_model=pretrained_model)
    print(net)

    '''
    trainer = Trainer(
        save_top_path, 
        weights_path,
        log_path,
        graph_path,
        csv_name,
        index_csv_path,
        multiGPU,
        model,
        pretrained_model,
        train_sequences,
        valid_sequences,
        dim_fc_out,
        resize,
        mean_element,
        std_element,
        original_size,
        batch_size,
        num_epochs,
        optimizer_name,
        lr_resnet,
        lr_roll_fc,
        lr_pitch_fc,
        weight_decay,
        alpha,
        train_dataset,
        valid_dataset,
        net
    )

    trainer.train()
    '''