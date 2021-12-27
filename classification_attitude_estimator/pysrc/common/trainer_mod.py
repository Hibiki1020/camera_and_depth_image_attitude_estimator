from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import random

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

class Trainer:
    def __init__(self,
    method_name,
    train_dataset,
    valid_dataset,
    net,
    criterion,
    optimizer_name,
    loss_function,
    lr_cnn,
    lr_roll_fc,
    lr_pitch_fc,
    weight_decay,
    batch_size,
    num_epochs,
    weights_path,
    log_path,
    graph_path,
    multiGPU,
    alpha,
    clip_limit):

        self.weights_path = weights_path
        self.log_path = log_path
        self.graph_path = graph_path
        self.multiGPU = multiGPU
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.clip_limit = clip_limit
        self.loss_function = loss_function

        self.setRandomCondition()
        
        if self.multiGPU == 0:
            self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("Training Device: ", self.device)

        self.dataloaders_dict = self.getDataloader(train_dataset, valid_dataset, batch_size)
        self.net = self.getSetNetwork(net)
        self.criterion = criterion
        self.optimizer = self.getOptimizer(optimizer_name, lr_cnn, lr_roll_fc, lr_pitch_fc)
        self.num_epochs = num_epochs
        self.str_hyperparameter = self.getStrHyperparameter(method_name, train_dataset, optimizer_name, lr_cnn, lr_roll_fc, lr_pitch_fc,batch_size)

    def setRandomCondition(self, keep_reproducibility=False): #Random Training Environment
        #Refer https://nuka137.hatenablog.com/entry/2020/09/01/080038
        if keep_reproducibility:
            torch.manual_seed(19981020)
            np.random.seed(19981020)
            random.seed(19981020)
            torch.backends.cudnn.deterministic = True #https://qiita.com/chat-flip/items/c2e983b7f30ef10b91f6
            torch.backends.cudnn.benchmark = False
        
    def getDataloader(self, train_dataset, valid_dataset, batch_size):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True
        )

        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size = batch_size,
            shuffle = False
        )
        dataloaders_dict = {"train":train_dataloader, "valid":valid_dataloader}
        return dataloaders_dict
    
    def getSetNetwork(self, net):
        print(net)

        net = net.to(self.device) #Send to GPU

        if self.multiGPU == 1 and self.device == 'cuda':
            net = nn.DataParallel(net) #make parallel
            cudnn.benchmark = True

        return net

    def getOptimizer(self, optimizer_name, lr_cnn, lr_roll_fc, lr_pitch_fc):

        if self.multiGPU == 1 and self.device == 'cuda':
            list_cnn_param_value, list_roll_fc_param_value, list_pitch_fc_param_value = self.net.module.getParamValueList()
        elif self.multiGPU == 0:
            list_cnn_param_value, list_roll_fc_param_value, list_pitch_fc_param_value = self.net.getParamValueList()

        if optimizer_name == "SGD":
            optimizer = optim.SGD([
                {"params": list_cnn_param_value, "lr": lr_cnn},
                {"params": list_roll_fc_param_value, "lr": lr_roll_fc},
                {"params": list_pitch_fc_param_value, "lr": lr_pitch_fc}
            ], momentum=0.9, 
            weight_decay=self.weight_decay)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam([
                {"params": list_cnn_param_value, "lr": lr_cnn},
                {"params": list_roll_fc_param_value, "lr": lr_roll_fc},
                {"params": list_pitch_fc_param_value, "lr": lr_pitch_fc}
            ], weight_decay=self.weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW([
                {"params": list_cnn_param_value, "lr": lr_cnn},
                {"params": list_roll_fc_param_value, "lr": lr_roll_fc},
                {"params": list_pitch_fc_param_value, "lr": lr_pitch_fc}
            ], weight_decay=self.weight_decay)

        print("optimizer")
        print(optimizer)
        return optimizer
    
    def getStrHyperparameter(self, method_name, dataset, optimizer_name, lr_cnn, lr_roll_fc, lr_pitch_fc, batch_size):
        str_hyperparameter = method_name \
            + str(len(self.dataloaders_dict["train"].dataset)) + "train" \
            + str(len(self.dataloaders_dict["valid"].dataset)) + "valid" \
            + str(dataset.transform.mean) + "mean" \
            + str(dataset.transform.std) + "std" \
            + optimizer_name \
            + str(lr_cnn) + "lrcnn" \
            + str(lr_roll_fc) + "lrrollfc" \
            + str(lr_pitch_fc) + "lrpitchfc" \
            + str(batch_size) + "batch" \
            + str(self.num_epochs) + "epoch"
        print("str_hyperparameter = ", str_hyperparameter)
        return str_hyperparameter
    
    def computeLoss(self, outputs, labels):
        loss = self.criterion(outputs, labels)
        return loss
    
    def saveParam(self):
        save_path = self.weights_path + self.str_hyperparameter + ".pth"
        torch.save(self.net.state_dict(), save_path)
        print("Saved: ", save_path)

    def saveGraph(self, record_loss_train, record_loss_val):
        graph = plt.figure()
        plt.plot(range(len(record_loss_train)), record_loss_train, label="Training")
        plt.plot(range(len(record_loss_val)), record_loss_val, label="Validation")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss [m^2/s^4]")
        plt.title("loss: train=" + str(record_loss_train[-1]) + ", val=" + str(record_loss_val[-1]))
        graph.savefig(self.graph_path + self.str_hyperparameter + ".jpg")
        plt.show()

    def train(self):
        #Time
        start_clock = time.time()

        #Loss Record
        writer = SummaryWriter(logdir=self.log_path + self.str_hyperparameter)
        
        record_loss_train = []
        record_loss_val = []

        for epoch in range(self.num_epochs):
            print("----------------------")
            print("Epoch {}/{}".format(epoch+1, self.num_epochs))
            for phase in ["train", "valid"]:
                if phase == "train":
                    self.net.train()
                else:
                    self.net.eval()
                

                ##data load
                epoch_loss = 0.0
                for inputs, label_roll, label_pitch in tqdm(self.dataloaders_dict[phase]):
                    inputs = inputs.to(self.device)

                    #print(inputs)

                    label_roll = label_roll.to(self.device)
                    label_pitch = label_pitch.to(self.device)

                    #reset gradient
                    self.optimizer.zero_grad()

                    #Compute gradient
                    with torch.set_grad_enabled(phase == "train"):
                        logged_roll_inf, logged_pitch_inf, roll_inf, pitch_inf = self.net(inputs)

                        if self.loss_function == "CrossEntropyLoss":
                            roll_loss = torch.mean(torch.sum(-label_roll * logged_roll_inf, 1))
                            pitch_loss = torch.mean(torch.sum(-label_pitch * logged_pitch_inf, 1))
                        elif self.loss_function == "MSELoss":
                            roll_loss = self.computeLoss(roll_inf, label_roll)
                            pitch_loss = self.computeLoss(pitch_inf, label_pitch)
                    
                        torch.set_printoptions(edgeitems=10000)

                        if self.device == 'cpu':
                            l2norm = torch.tensor(0., requires_grad = True).cpu()
                        else: #device == gpu
                            l2norm = torch.tensor(0., requires_grad = True).cuda()
                        
                        for w in self.net.parameters():
                            l2norm = l2norm + torch.norm(w)**2

                        total_loss = roll_loss + pitch_loss + self.alpha*l2norm

                        if phase == "train":
                            total_loss.backward()

                            self.optimizer.step()    #update param depending on current .grad
                        
                        epoch_loss += total_loss.item() * inputs.size(0)

                epoch_loss = epoch_loss / len(self.dataloaders_dict[phase].dataset)
                print("{} Loss: {:.4f}".format(phase, epoch_loss))

                #record
                if phase == "train":
                    record_loss_train.append(epoch_loss)
                    writer.add_scalar("Loss/train", epoch_loss, epoch)
                else:
                    record_loss_val.append(epoch_loss)
                    writer.add_scalar("Loss/val", epoch_loss, epoch)
                    
            if record_loss_train and record_loss_val:
                writer.add_scalars("Loss/train_and_val", {"train": record_loss_train[-1], "val": record_loss_val[-1]}, epoch)
                
        writer.close()
        ## save
        self.saveParam()
        self.saveGraph(record_loss_train, record_loss_val)
        ## training time
        mins = (time.time() - start_clock) // 60
        secs = (time.time() - start_clock) % 60
        print ("training_time: ", mins, " [min] ", secs, " [sec]")
