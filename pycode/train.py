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

class Trainer:
    def __init__(self,
            save_top_path, 
            weights_path,
            log_path,
            graph_path,
            csv_name,
            index_csv_path,
            multiGPU,
            pretrained_model,
            train_sequences,
            valid_sequences,
            dim_fc_out,
            resize,
            mean_element,
            std_element,
            original_size,
            batch_size,
            train_dataset,
            valid_dataset,
            net
        ):

        self.save_top_path = save_top_path
        self.weights_path = weights_path
        self.log_path = log_path
        self.graph_path = graph_path
        self.csv_name = csv_name
        self.index_csv_path = index_csv_path
        self.multiGPU = multiGPU
        self.pretrained_model = pretrained_model
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.dim_fc_out = dim_fc_out
        self.resize = resize
        self.mean_element = mean_element
        self.std_element = std_element
        self.original_size = original_size
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.net = net
        
        if self.multiGPU == 0:
            self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.setRandomCondition()
        self.dataloaders_dict = self.getDataloaders(train_dataset, valid_dataset, batch_size)
        self.net = self.getNetwork(net)

    def setRandomCondition(self, keep_reproducibility=False, seed=123456789):
        if keep_reproducibility:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def getDataloaders(self, train_dataset, valid_dataset, batch_size):
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

        return dataloaders_dict

    def getNetwork(self, net):
        print(net)
        net = net.to(self.device)
        if self.multiGPU == 1 and self.device == 'cuda':
            net = nn.DataParallel(net)
            cudnn.benchmark = True
            print("Training on multiGPU device")
        
        return net

    def train(self):
        print("Start Training")


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
    batch_size = CFG["hyperparameter"]["batch_size"]

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

    net = network_mod.Network(dim_fc_out, norm_layer=nn.BatchNorm2d,pretrained_model=pretrained_model)

    trainer = Trainer(
        save_top_path, 
        weights_path,
        log_path,
        graph_path,
        csv_name,
        index_csv_path,
        multiGPU,
        pretrained_model,
        train_sequences,
        valid_sequences,
        dim_fc_out,
        resize,
        mean_element,
        std_element,
        original_size,
        batch_size,
        train_dataset,
        valid_dataset,
        net
    )

    trainer.train()
