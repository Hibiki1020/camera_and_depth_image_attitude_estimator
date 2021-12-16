from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime

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

from tensorboardX import SummaryWriter

import sys
from common import dataset_mod
from common import make_datalist_mod
from common import data_transform_mod

from common import network_mod

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
    weights_path = CFG["save_top_path"] + CFG["weights_path"]
    log_path = CFG["save_top_path"] + CFG["log_path"]
    graph_path = CFG["save_top_path"] + CFG["graph_path"]
    csv_name = CFG["csv_name"]
    index_csv_path = CFG["index_csv_path"]
    multiGPU = int(CFG["multiGPU"])

    pretrained_model = CFG["pretrained_model"]

    train_sequences = CFG["train"]
    valid_sequences = CFG["valid"]

    dim_fc_out = int(CFG["hyperparameter"]["dim_fc_out"])
    resize = CFG["hyperparameter"]["resize"]
    mean_element = CFG["hyperparameter"]["mean_element"]
    std_element = CFG["hyperparameter"]["std_element"]
    original_size = CFG["hyperparameter"]["original_size"]

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

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 16,
        shuffle=True
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = 16,
        shuffle=True
    )

    dataloaders_dict = {"train":train_dataloader, "valid":valid_dataloader}

    net = network_mod.Network(dim_fc_out, norm_layer=nn.BatchNorm2d,pretrained_model=pretrained_model)
