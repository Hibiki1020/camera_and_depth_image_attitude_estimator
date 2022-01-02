import os
import cv2
import PIL.Image as PILIMAGE
import math
import numpy as np
import time
import argparse
from numpy.core.fromnumeric import argmin
from torch.utils import data
import yaml
import csv
import random
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy.stats as stats

from sklearn.mixture import GaussianMixture
from collections import OrderedDict

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as nn_functional

import sys
from common import network_mod

class InferenceMod:
    def __init__(self, CFG):
        self.CFG = CFG
        self.infer_dataset_top_directory = CFG["infer_dataset_top_directory"]
        self.csv_name = CFG["csv_name"]
        
        self.weights_top_directory = CFG["weights_top_directory"]
        self.weights_file_name = CFG["weights_file_name"]
        self.weights_path = os.path.join(self.weights_top_directory, self.weights_file_name)
        
        self.infer_log_top_directory = CFG["infer_log_top_directory"]
        self.infer_log_file_name = CFG["infer_log_file_name"]
        
        self.index_dict_name = CFG["index_dict_name"]
        self.index_dict_path = "../../index_dict/" + self.index_dict_name

        self.original_size = int(CFG["original_size"])
        self.resize = int(CFG["resize"])
        self.mean_element = float(CFG["mean_element"])
        self.std_element = float(CFG["std_element"])
        self.dim_fc_out = int(CFG["dim_fc_out"])
        self.enable_dropout = bool(CFG["enable_dropout"])
        self.dropout_rate = float(CFG["dropout_rate"])

        self.image_cv = np.empty(0)
        self.depth_cv = np.empty(0)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ",self.device)

        self.img_transform = self.getImageTransform(self.original_size, self.mean_element, self.std_element, self.resize)

        self.net = self.getNetwork(self.resize, self.weights_path, self.dim_fc_out)

        self.value_dict = []

        with open(self.index_dict_path) as fd:
            reader = csv.reader(fd)
            for row in reader:
                num = float(row[0])
                self.value_dict.append(num)

    def getImageTransform(self, original_size, mean_element, std_element, resize):

        mean = mean_element
        std = std_element
        size = (resize, resize)

        '''
        img_transform = transforms.Compose([
            transforms.CenterCrop(original_size),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])
        '''

        img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

        return img_transform

    def getNetwork(self, resize, weights_path, dim_fc_out):
        net = network_mod.Network(dim_fc_out, norm_layer=nn.BatchNorm2d,pretrained_model=weights_path)

        print(net)

        net.to(self.device)
        net.eval()

        #load
        if torch.cuda.is_available():
            state_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)
            print("GPU  ==>  GPU")
        else:
            state_dict = torch.load(weights_path, map_location={"cuda:0": "cpu"})
            print("GPU  ==>  CPU")
        

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' in k:
                k = k.replace('module.', '')
            new_state_dict[k] = v

        net.load_state_dict(new_state_dict)
        return net

    def spin(self):
        print("Load data")
        self.image_data_list, self.depth_data_list, self.ground_truth_list = self.get_data()
        
        self.result_csv = self.frame_infer(self.image_data_list, self.depth_data_list, self.ground_truth_list)
        self.save_csv(self.result_csv)

    def save_csv(self, result_csv):
        result_csv_path = os.path.join(self.infer_log_top_directory, self.infer_log_file_name)
        csv_file = open(result_csv_path, 'w')
        csv_w = csv.writer(csv_file)
        for row in result_csv:
            csv_w.writerow(row)
        csv_file.close()
        print("Save Inference Data")

    def get_data(self):
        image_data_list = []
        depth_data_list = []
        data_list = []

        csv_path = os.path.join(self.infer_dataset_top_directory, self.csv_name)

        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                #img_path = os.path.join(self.infer_dataset_top_directory, row[0])
                img_path = self.infer_dataset_top_directory + "/camera_image/" + row[0]
                #depth_path = os.path.join(self.infer_dataset_top_directory, row[1])
                depth_path = self.infer_dataset_top_directory + "/depth_image/" + row[1]
                gt_roll = float(row[5])/3.141592*180.0
                gt_pitch = float(row[6])/3.141592*180.0

                #print(img_path)

                image_data_list.append(img_path)
                depth_data_list.append(depth_path)
                tmp_row = [row[0], row[1], gt_roll, gt_pitch]
                data_list.append(tmp_row)

        return image_data_list, depth_data_list, data_list

    def prediction(self, mono_image, depth_image):
        logged_roll, logged_pitch, roll, pitch = self.net(mono_image, depth_image)

        output_roll_array = roll.to('cpu').detach().numpy().copy()
        output_pitch_array = pitch.to('cpu').detach().numpy().copy()

        return np.array(output_roll_array), np.array(output_pitch_array)

    def array_to_value_simple(self, output_array):
        max_index = int(np.argmax(output_array))
        plus_index = max_index + 1
        minus_index = max_index - 1

        value = 0.0

        for value, label in zip(output_array[0], self.value_dict):
            value += value * label

        return value

    def array_to_value_simple_hist(self, output_array):

        value = 0.0

        for i in range(len(output_array)):
            value += output_array[i]*self.value_dict[i]

        return value
    
    def transformImage(self, inference_image):
        ## color
        img_pil = self.cvToPIL(inference_image)
        img_tensor = self.img_transform(img_pil)
        inputs = img_tensor.unsqueeze_(0)
        inputs = inputs.to(self.device)
        #print(inputs)
        return inputs

    def cvToPIL(self, img_cv):
        #img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = PILIMAGE.fromarray(img_cv)
        return img_pil

    def frame_infer(self, image_data_list, depth_data_list, ground_truth_list):
        print("Start Inference")

        result_csv = []
        infer_count = 0

        for (img_path, depth_path, ground_truth) in zip(image_data_list, depth_data_list, ground_truth_list):
            print("---------Inference at " + str(infer_count + 1) + "---------")
            infer_count += 1

            mono_image = cv2.imread(img_path)
            #cv2.imshow('image', mono_image)
            #cv2.waitKey(0)
            depth_image = cv2.imread(depth_path)

            start_clock = time.time()

            result = []

            roll_result_list = []
            pitch_result_list = []

            roll_hist_array = []
            pitch_hist_array = []

            roll_value_array = []
            pitch_value_array = []

            for i in range(self.dim_fc_out):
                tmp = 0.0
                roll_hist_array.append(tmp)
                pitch_hist_array.append(tmp)

            input_mono = self.transformImage(mono_image)
            input_depth = self.transformImage(depth_image)

            roll_output_array, pitch_output_array = self.prediction(input_mono, input_depth)
            tmp_roll = self.array_to_value_simple(roll_output_array)
            tmp_pitch = self.array_to_value_simple(pitch_output_array)

            roll_hist_array += roll_output_array[0]
            pitch_hist_array += pitch_output_array[0]

            tmp_result = [tmp_roll, tmp_pitch]
                
            roll_result_list.append(tmp_roll)
            pitch_result_list.append(tmp_pitch)

            roll_value_array.append(tmp_roll)
            pitch_value_array.append(tmp_pitch)

            result.append(tmp_result)

            #roll_hist_array /= float(len(windows))
            #pitch_hist_array /= float(len(windows))

            roll_hist = self.array_to_value_simple_hist(roll_hist_array)
            pitch_hist = self.array_to_value_simple_hist(pitch_hist_array)

            np_result = np.array(result)

            #roll = np.mean(tmp_roll)
            #pitch = np.mean(tmp_pitch)

            roll = roll_hist
            pitch = pitch_hist

            np_roll_value_array = np.array(roll_value_array)
            np_pitch_value_array = np.array(pitch_value_array)

            roll_var = np.var(np_roll_value_array)
            pitch_var = np.var(np_pitch_value_array)

            print("Infered Roll:  " + str(roll) +  "[deg]")
            print("GT Roll:       " + str(ground_truth[2]) + "[deg]")
            print("Infered Pitch: " + str(pitch) + "[deg]")
            print("GT Pitch:      " + str(ground_truth[3]) + "[deg]")
            print("Roll Variance :" + str(roll_var))
            print("Pitch Variance:" + str(pitch_var))

            
            #self.show_fig(roll_hist_array, pitch_hist_array, self.value_dict, windows[1])
            


            cov = np.cov(np_result)
            

            #Image roll pitch GTroll GTpitch
            tmp_result_csv = [ground_truth[0], ground_truth[1], roll, pitch, ground_truth[2], ground_truth[3], roll_var, pitch_var]
            result_csv.append(tmp_result_csv)

            print("Period [s]: ", time.time() - start_clock)
            print("---------------------")

        return result_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./frame_infer.py")
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default='../pyyaml/infer_config.yaml',
        help='Frame Infer Config'
    )

    FLAGS, unparsed = parser.parse_known_args()

    #Load yaml file
    try:
        print("Opening frame infer config file %s", FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening frame infer config file %s", FLAGS.config)
        quit()

    estimator = InferenceMod(CFG)
    estimator.spin()

