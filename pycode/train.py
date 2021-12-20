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
        self.num_epochs = num_epochs
        self.lr_resnet = lr_resnet
        self.lr_roll_fc = lr_roll_fc
        self.lr_pitch_fc = lr_pitch_fc
        self.weight_decay = weight_decay
        self.alpha = alpha
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
        self.optimizer = self.getOptimizer(optimizer_name, lr_resnet, lr_roll_fc, lr_pitch_fc, weight_decay)

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
            shuffle=True,
            num_workers = 2,
            pin_memory =True
        )

        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size = 16,
            shuffle=True,
            num_workers = 2,
            pin_memory = True
        )

        dataloaders_dict = {"train":train_dataloader, "valid":valid_dataloader}

        return dataloaders_dict

    def getNetwork(self, net):
        print("Loading Network")
        #print(net)
        net = net.to(self.device)
        if self.multiGPU == 1 and self.device == 'cuda':
            net = nn.DataParallel(net)
            cudnn.benchmark = True
            print("Training on multiGPU device")
        else:
            cudnn.benchmark = True
            print("Training on single GPU device")
        
        return net
    
    def getOptimizer(self, optimizer_name, lr_resnet, lr_roll_fc, lr_pitch_fc, weight_decay):
        if self.multiGPU == 1 and self.device == 'cuda':
            list_resnet_param_value, list_roll_fc_param_value, list_pitch_fc_param_value = self.net.module.getParamValueList()
        elif self.multiGPU == 0:
            list_resnet_param_value, list_roll_fc_param_value, list_pitch_fc_param_value = self.net.getParamValueList()

        if optimizer_name == "SGD":
            optimizer = optim.SGD([
                {"params": list_resnet_param_value, "lr": lr_resnet},
                {"params": list_roll_fc_param_value, "lr": lr_roll_fc},
                {"params": list_pitch_fc_param_value, "lr": lr_pitch_fc}
            ], momentum=0.9, 
            weight_decay=self.weight_decay)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam([
                {"params": list_resnet_param_value, "lr": lr_resnet},
                {"params": list_roll_fc_param_value, "lr": lr_roll_fc},
                {"params": list_pitch_fc_param_value, "lr": lr_pitch_fc}
            ], weight_decay=self.weight_decay)

        print("optimizer: {}".format(optimizer_name))
        return optimizer

    def saveParam(self):
        save_path = self.weights_path + "weights.pth"
        torch.save(self.net.state_dict(), save_path)
        print("Saved Weight") 

    def train(self):
        print("Start Training")

        start_clock = time.time()

        #Loss record
        writer = SummaryWriter(log_dir=self.log_path+"log")

        record_train_loss = []
        record_valid_loss = []

        for epoch in range(self.num_epochs):
            print("----------------")
            print("Epoch {}/{}".format(epoch+1, self.num_epochs))

            for phase in ["train", "valid"]:
                if phase == "train":
                    self.net.train() #Change Training Mode
                elif phase == "valid":
                    self.net.eval()
                
                #Data Load
                epoch_loss = 0.0

                for mono_input, depth_input, label_roll, label_pitch in tqdm(self.dataloaders_dict[phase]):
                    mono_input = mono_input.to(self.device)
                    depth_input = depth_input.to(self.device)
                    label_roll = label_roll.to(self.device)
                    label_pitch = label_pitch.to(self.device)

                    #print(mono_input.size())

                    #Reset Gradient
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase=="train"):
                        logged_roll_inf, logged_pitch_inf, roll_inf, pitch_inf = self.net(mono_input, depth_input)

                        roll_loss = torch.mean(torch.sum(-label_roll*logged_roll_inf, 1))
                        pitch_loss = torch.mean(torch.sum(-label_pitch*logged_pitch_inf, 1))

                        torch.set_printoptions(edgeitems=1000000)

                        if self.device == 'cpu':
                            l2norm = torch.tensor(0., requires_grad = True).cpu()
                        else:
                            l2norm = torch.tensor(0., requires_grad = True).cuda()

                        for w in self.net.parameters():
                            l2norm = l2norm + torch.norm(w)**2
                        
                        total_loss = roll_loss + pitch_loss + self.alpha*l2norm

                        if phase == "train":
                            total_loss.backward()
                            self.optimizer.step()

                        epoch_loss += total_loss.item() * mono_input.size(0) * depth_input.size(0)

                epoch_loss = epoch_loss/len(self.dataloaders_dict[phase].dataset)
                print("{} Loss: {:.4f}".format(phase, epoch_loss))

                if phase == "train":
                    record_train_loss.append(epoch_loss)
                    writer.add_scalar("Loss/Train", epoch_loss, epoch)
                else:
                    record_valid_loss.append(epoch_loss)
                    writer.add_scalar("Loss/Valid", epoch_loss, epoch)

            if record_train_loss and record_valid_loss:
                writer.add_scalars("Loss/train_and_val", {"train": record_train_loss[-1], "val": record_valid_loss[-1]}, epoch)

        writer.close()
        self.saveParam()




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
