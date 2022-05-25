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
import urllib
import pickle

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


# Grad-CAM


class GradCam:
    def __init__(self, CFG):
        self.CFG = CFG
        self.infer_dataset_top_directory = CFG["infer_dataset_top_directory"]
        self.csv_name = CFG["csv_name"]
        
        self.weights_top_directory = CFG["weights_top_directory"]
        self.weights_file_name = CFG["weights_file_name"]
        self.weights_path = os.path.join(self.weights_top_directory, self.weights_file_name)
        self.model = CFG["model"]
        
        self.index_dict_name = CFG["index_dict_name"]
        self.index_dict_path = "../../index_dict/" + self.index_dict_name

        self.original_size = int(CFG["original_size"])
        self.resize = int(CFG["resize"])
        self.mean_element = float(CFG["mean_element"])
        self.std_element = float(CFG["std_element"])
        self.dim_fc_out = int(CFG["dim_fc_out"])
        self.enable_dropout = bool(CFG["enable_dropout"])
        self.dropout_rate = float(CFG["dropout_rate"])

        self.window_original_size = int(CFG["window_original_size"])
        self.num_windows = int(CFG["num_windows"])

        self.image_cv = np.empty(0)
        self.depth_cv = np.empty(0)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ",self.device)

        self.img_transform = self.getImageTransform(self.original_size, self.mean_element, self.std_element, self.resize)

        self.net = self.getNetwork(self.model, self.resize, self.weights_path, self.dim_fc_out)


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

        img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

        return img_transform

    def getNetwork(self, model, resize, weights_path, dim_fc_out):
        net = network_mod.Network(model, dim_fc_out, norm_layer=nn.BatchNorm2d,pretrained_model=weights_path)

        print(net)
        print("Load ", model)

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
        random_num = random.randint(0, 1000)
        self.image_data_list, self.depth_data_list, self.ground_truth_list = self.get_data()

        print("Start Inference")

        result_csv = []
        infer_count = 0
        diff_total_roll = 0.0
        diff_total_pitch = 0.0

        for (img_path, depth_path, ground_truth) in zip(self.image_data_list, self.depth_data_list, self.ground_truth_list):
            print("---------Inference at " + str(infer_count + 1) + "---------")
            infer_count += 1

            mono_image = cv2.imread(img_path)
            #cv2.imshow('image', mono_image)
            #cv2.waitKey(0)
            depth_image = cv2.imread(depth_path)

            mono_windows, depth_windows = self.extract_window(mono_image, depth_image)

            start_clock = time.time()

            result = []

            roll_result_list = []
            pitch_result_list = []

            roll_hist_array = []
            pitch_hist_array = []

            roll_value_array = []
            pitch_value_array = []

            self.net.eval()

            for i in range(self.dim_fc_out):
                tmp = 0.0
                roll_hist_array.append(tmp)
                pitch_hist_array.append(tmp)

            for (mono_image, depth_image) in zip(mono_windows, depth_windows):

                input_mono = self.transformImage(mono_image)
                input_depth = self.transformImage(depth_image)

                print(input_depth.size())

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

    def array_to_value_simple(self, output_array):
        max_index = int(np.argmax(output_array))
        plus_index = max_index + 1
        minus_index = max_index - 1

        value = 0.0

        '''
        if max_index == 0:
            value = output_array[0][max_index]*self.value_dict[max_index] + output_array[0][max_index+1]*self.value_dict[max_index+1]
        elif max_index == 360: #361
            value = output_array[0][max_index]*self.value_dict[max_index] + output_array[0][max_index-1]*self.value_dict[max_index-1]
        else:
            if output_array[0][minus_index] > output_array[0][plus_index]: #一つ前のインデックスを採用
                value = output_array[0][max_index]*self.value_dict[max_index] + output_array[0][minus_index]*self.value_dict[minus_index]
            elif output_array[0][minus_index] < output_array[0][plus_index]: #一つ後のインデックスを採用
                value = output_array[0][max_index]*self.value_dict[max_index] + output_array[0][plus_index]*self.value_dict[plus_index]
        '''
        
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

    def extract_window(self, mono_image, depth_image):
        height = mono_image.shape[0]
        width = mono_image.shape[1]

        mono_windows = []
        depth_windows = []

        window_count = 0

        while window_count < self.num_windows:
            width_start = random.randint(0, int(width)-self.window_original_size)
            height_start = random.randint(0, int(height)-self.window_original_size)

            mono_window = mono_image[height_start:(height_start + self.window_original_size), width_start:(width_start + self.window_original_size)]
            depth_window = depth_image[height_start:(height_start + self.window_original_size), width_start:(width_start + self.window_original_size)]

            #cv2.imshow('window',mono_window)
            #cv2.waitKey(0)

            mono_windows.append(mono_window)
            depth_windows.append(depth_window)

            window_count += 1

        return mono_windows, depth_windows

if __name__ == '__main__':

    parser = argparse.ArgumentParser("./grad_cam.py")
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default='../pyyaml/grad_cam_config.yaml',
        help='Grad Cam Config'
    )

    FLAGS, unparsed = parser.parse_known_args()

    #Load yaml file
    try:
        print("Opening grad cam config file %s", FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening grad cam config file %s", FLAGS.config)
        quit()
    
    grad_cam = GradCam(CFG)
    grad_cam.spin()