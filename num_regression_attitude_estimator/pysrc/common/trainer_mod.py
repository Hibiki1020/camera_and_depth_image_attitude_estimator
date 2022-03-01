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
    lr_cnn,
    lr_fc,
    weight_decay,
    batch_size,
    num_epochs,
    save_top_path,
    multiGPU):

        self.multiGPU = multiGPU
        self.save_top_path = save_top_path
        self.weight_decay = weight_decay

        self.setRandomCondition()

        if self.multiGPU == 0:
            self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("Training Device: ", self.device)

        self.dataloaders_dict = self.getDataloader(train_dataset, valid_dataset, batch_size)
        self.net = self.getSetNetwork(net)
        self.criterion = criterion
        self.optimizer = self.getOptimizer(optimizer_name, lr_cnn, lr_fc)
        self.num_epochs = num_epochs
        self.str_hyperparameter = self.getStrHyperparameter(method_name, train_dataset, optimizer_name, lr_cnn, lr_fc, batch_size)

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

    def getOptimizer(self, optimizer_name, lr_cnn, lr_fc):

        if self.multiGPU == 1 and self.device == 'cuda':
            list_cnn_param_value, list_fc_param_value = self.net.module.getParamValueList()
        elif self.multiGPU == 0:
            list_cnn_param_value, list_fc_param_value = self.net.getParamValueList()

        if optimizer_name == "SGD":
            optimizer = optim.SGD([
                {"params": list_cnn_param_value, "lr": lr_cnn},
                {"params": list_fc_param_value, "lr": lr_fc},
            ], momentum=0.9, 
            weight_decay=self.weight_decay)
        elif optimizer_name == "Adam":
            optimizer = optim.Adam([
                {"params": list_cnn_param_value, "lr": lr_cnn},
                {"params": list_fc_param_value, "lr": lr_fc},
            ], weight_decay=self.weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW([
                {"params": list_cnn_param_value, "lr": lr_cnn},
                {"params": list_fc_param_value, "lr": lr_fc},
            ], weight_decay=self.weight_decay)

        print("optimizer")
        print(optimizer)
        return optimizer

    def getStrHyperparameter(self, method_name, dataset, optimizer_name, lr_cnn, lr_fc, batch_size):
        str_hyperparameter = method_name \
            + str(len(self.dataloaders_dict["train"].dataset)) + "train" \
            + str(len(self.dataloaders_dict["valid"].dataset)) + "valid" \
            + str(dataset.transform.mean) + "mean" \
            + str(dataset.transform.std) + "std" \
            + optimizer_name \
            + str(lr_cnn) + "lrcnn" \
            + str(lr_fc) + "lrfc" \
            + str(batch_size) + "batch" \
            + str(self.num_epochs) + "epoch"
        print("str_hyperparameter = ", str_hyperparameter)
        return str_hyperparameter

    def train(self):
        #Time
        start_clock = time.time()

        #Loss Record
        writer = SummaryWriter(logdir=self.save_top_path + self.str_hyperparameter)
        
        record_loss_train = []
        record_loss_valid = []

        for epoch in range(self.num_epochs):
            print("----------------------")
            print("Epoch {}/{}".format(epoch+1, self.num_epochs))
            for phase in ["train", "valid"]:
                if phase == "train":
                    self.net.train()
                else:
                    self.net.eval()
                
                ##skip
                #if (epoch == 0) and (phase== "train"):
                #    continue

                epoch_loss = 0.0
                for inputs, labels in tqdm(self.dataloaders_dict[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    ## reset gradient
                    self.optimizer.zero_grad()   #reset grad to zero (after .step())

                    ## compute gradient
                    with torch.set_grad_enabled(phase == "train"):  #compute grad only in "train"
                        ## forward
                        outputs = self.net(inputs)
                        loss = self.computeLoss(outputs, labels)
                        ## backward
                        if phase == "train":
                            loss.backward()     #accumulate gradient to each Tensor
                            self.optimizer.step()    #update param depending on current .grad
                        ## add loss
                        epoch_loss += loss.item() * inputs.size(0)
                ## average loss
                epoch_loss = epoch_loss / len(self.dataloaders_dict[phase].dataset)
                print("{} Loss: {:.4f}".format(phase, epoch_loss))
                ## record
                if phase == "train":
                    record_loss_train.append(epoch_loss)
                    writer.add_scalar("Loss/train", epoch_loss, epoch)
                    # for param_name, param_value in self.net.named_parameters():
                    #     # print(param_name, ": ", param_value.grad.abs().mean())
                    #     writer.add_scalar("Gradient/" + param_name, param_value.grad.abs().mean(), epoch)
                else:
                    record_loss_valid.append(epoch_loss)
                    writer.add_scalar("Loss/valid", epoch_loss, epoch)
            if record_loss_train and record_loss_valid:
                writer.add_scalars("Loss/train_and_val", {"train": record_loss_train[-1], "valid": record_loss_valid[-1]}, epoch)
        writer.close()
        ## save
        self.saveParam()
        self.saveGraph(record_loss_train, record_loss_valid)
        ## training time
        mins = (time.time() - start_clock) // 60
        secs = (time.time() - start_clock) % 60
        print ("training_time: ", mins, " [min] ", secs, " [sec]")

    def computeLoss(self, outputs, labels):
        loss = self.criterion(outputs, labels)
        return loss

    def saveParam(self):
        #save_path = "../../weights/" + self.str_hyperparameter + ".pth"
        save_path = self.save_top_path + self.str_hyperparameter + ".pth"
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
        graph.savefig(self.save_top_path + self.str_hyperparameter + ".png")
        plt.show()