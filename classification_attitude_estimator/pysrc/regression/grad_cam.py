import cv2
import PIL.Image as Image
import math
import numpy as np
import time
import argparse
from numpy.core.fromnumeric import argmin
import yaml
import os
import csv
import random
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy.stats as stats

from sklearn.mixture import GaussianMixture

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as nn_functional

from collections import OrderedDict

import sys
sys.path.append('../')
#from common import network_mod
from common import vgg_network_mod

#Grad CAM

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

#from gradcam.utils import visualize_cam
#from gradcam import GradCAMpp
#from torchvision.utils import make_grid, save_image


class GradCam:
    def __init__(self, CFG):
        self.CFG = CFG
        self.method_name = CFG["method_name"]
        
        self.infer_dataset_top_directory = CFG["infer_dataset_top_directory"]
        self.csv_name = CFG["csv_name"]

        self.weights_top_directory = CFG["weights_top_directory"]
        self.weights_file_name = CFG["weights_file_name"]

        self.weights_path = os.path.join(self.weights_top_directory, self.weights_file_name)

        self.infer_log_top_directory = CFG["infer_log_top_directory"]
        self.infer_log_file_name = CFG["infer_log_file_name"]

        self.index_dict_name = CFG["index_dict_name"]
        self.index_dict_path = "../../../index_dict/" + self.index_dict_name

        self.window_original_size = int(CFG["window_original_size"])
        self.original_size = int(CFG["original_size"])
        self.window_num = int(CFG["window_num"])
        self.resize = int(CFG["resize"])
        self.mean_element = float(CFG["mean_element"])
        self.std_element = float(CFG["std_element"])
        self.dim_fc_out = int(CFG["dim_fc_out"])
        self.dropout_rate = float(CFG["dropout_rate"])
        self.enable_dropout = bool(CFG["enable_dropout"])
        self.corner_threshold = int(CFG["corner_threshold"])

        self.image_cv = np.empty(0)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ",self.device)

        self.img_transform = self.getImageTransform(self.original_size, self.mean_element, self.std_element, self.resize)
        self.net, self.net_vgg, self.net_roll, self.net_pitch = self.getNetwork(self.resize, self.weights_path, self.dim_fc_out, self.dropout_rate)

        self.target_layer = self.net.cnn_feature
        self.target_layer_vgg = self.net_vgg.features
        self.target_layer_roll = self.net_roll.features
        self.target_layer_pitch = self.net_pitch.features

        self.gradcam_roll = GradCAM(model = self.net_roll, target_layers = self.target_layer_roll, use_cuda = torch.cuda.is_available())
        self.gradcam_pitch = GradCAM(model = self.net_pitch, target_layers = self.target_layer_pitch, use_cuda = torch.cuda.is_available())
        #self.gradcam = GradCAM(model = self.net_vgg, target_layers = self.target_layer_vgg, use_cuda = torch.cuda.is_available())


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

    def getNetwork(self, resize, weights_path, dim_fc_out, dropout_rate):
        net = vgg_network_mod.Network(resize, dim_fc_out, dropout_rate, use_pretrained_vgg=False)

        #print(net)

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

        net_vgg = models.vgg16(pretrained=True)
        net_vgg.to(self.device)
        net_vgg.eval()

        net_roll = models.vgg16(pretrained=False)
        net_roll.features = net.cnn_feature
        net_roll.fc = net.roll_fc
        net_roll.to(self.device)
        net_roll.eval()

        net_pitch = models.vgg16(pretrained=False)
        net_pitch.features = net.cnn_feature
        net_pitch.fc = net.pitch_fc
        net_pitch.to(self.device)
        net_pitch.eval()

        return net, net_vgg, net_roll, net_pitch

    def spin(self):
        self.image_data_list, self.ground_truth_list = self.get_data()
        self.result_csv = self.frame_infer(self.image_data_list, self.ground_truth_list)

    def get_data(self):
        image_data_list = []
        data_list = []

        csv_path = os.path.join(self.infer_dataset_top_directory, self.csv_name)

        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                #img_path = os.path.join(self.infer_dataset_top_directory, row[0])
                img_path = self.infer_dataset_top_directory + "/camera_image/" + row[0]
                
                gt_roll = float(row[5])/3.141592*180.0
                gt_pitch = float(row[6])/3.141592*180.0

                #print(img_path)

                image_data_list.append(img_path)
                tmp_row = [row[0], row[1], gt_roll, gt_pitch]
                data_list.append(tmp_row)

        return image_data_list, data_list

    def check_window(self, window): #Bottle Neck
        detector = cv2.ORB_create()
        keypoints = detector.detect(window)

        window_checker = False

        if len(keypoints) > self.corner_threshold:
            window_checker = True

        return window_checker

    def extract_window(self, image_original):
        height = image_original.shape[0]
        width = image_original.shape[1]

        windows = []
        correct_windows = []
        tmp_windows = []        

        total_window_checker = False
        window_count = 0
        error_count = 0

        while total_window_checker==False:
            width_start = random.randint(0, int(width)-self.window_original_size)
            height_start = random.randint(0, int(height)-self.window_original_size)

            window = image_original[height_start:(height_start + self.window_original_size), width_start:(width_start + self.window_original_size)]
            tmp_window_checker = True

            if tmp_window_checker == True:
                window_count += 1
                correct_windows.append(window)
                tmp_windows.append(window)

                if window_count >= self.window_num:
                    total_window_checker = True
                    windows = correct_windows
            else:
                error_count += 1
                tmp_windows.append(window)

                if error_count >=self.window_num:
                    print("Less Feature Point...")
                    total_window_checker = True
                    windows = tmp_windows

        return windows

    def cvToPIL(self, img_cv):
        #img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cv)
        return img_pil

    def transformImage(self, inference_image):
        ## color
        img_pil = self.cvToPIL(inference_image)
        img_tensor = self.img_transform(img_pil)
        inputs = img_tensor.unsqueeze_(0)
        inputs = inputs.to(self.device)
        #print(inputs)
        return inputs

    def normalize(self, v):
        l2 = np.linalg.norm(v, ord=2, axis=-1, keepdims=True)
        l2[l2==0] = 1
        return v/l2

    def array_to_value_simple_hist(self, output_array):

        value = 0.0

        for i in range(len(output_array)):
            value += output_array[i]*self.value_dict[i]


        return value
    
    def array_to_value_simple(self, output_array):
        max_index = int(np.argmax(output_array))
        plus_index = max_index + 1
        minus_index = max_index - 1

        value = 0.0

        for value, label in zip(output_array[0], self.value_dict):
            value += value * label

        return value

    def prediction(self, input_image):
        logged_output_roll_array, logged_output_pitch_array, roll_array, pitch_array = self.net(input_image)

        output_roll_array = roll_array.to('cpu').detach().numpy().copy()
        output_pitch_array = pitch_array.to('cpu').detach().numpy().copy()

        #print(output_roll_array)

        return np.array(output_roll_array), np.array(output_pitch_array)

    def frame_infer(self, image_data_list, ground_truth_list):
        print("Start Inference")

        result_csv = []

        infer_count = 0

        diff_total_roll = 0.0
        diff_total_pitch = 0.0

        images = []

        for (img_path, ground_truth) in zip(image_data_list, ground_truth_list):
            print("---------Inference at " + str(infer_count) + "---------")
            infer_count += 1
            #image_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #Load Image
            image_original = cv2.imread(img_path)
            #cv2.imshow('image',image_original)

            windows = self.extract_window(image_original)

            print("Transform input image")
            print("---------------------")

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

            for window in windows:
                inference_image = window
                input_image = self.transformImage(inference_image)
                #print(input_image.size())

                #roll_output_array, pitch_output_array = self.prediction(input_image)
                vis_image = cv2.resize(window, (224, 224)) / 255.0 #(height, width, channel), [0, 1]
                label = 0

                grayscale_cam_roll = self.gradcam_roll(input_tensor = input_image)
                grayscale_cam_roll = grayscale_cam_roll[0, :]
                visualization_roll = show_cam_on_image(vis_image, grayscale_cam_roll, use_rgb = True)

                grayscale_cam_pitch = self.gradcam_pitch(input_tensor = input_image)
                grayscale_cam_pitch = grayscale_cam_pitch[0, :]
                visualization_pitch = show_cam_on_image(vis_image, grayscale_cam_pitch, use_rgb = True)

                """
                plt.imshow(visualization_roll)
                plt.imshow(visualization_pitch)
                plt.show()
                """

                # create figure
                fig = plt.figure(figsize=(7, 5))
  
                # setting values to rows and column variables
                rows = 1
                columns = 2

                fig.add_subplot(rows, columns, 1)

                # showing image
                plt.imshow(visualization_roll)
                plt.axis('off')
                plt.title("Roll")
                
                # Adds a subplot at the 2nd position
                fig.add_subplot(rows, columns, 2)
                
                # showing image
                plt.imshow(visualization_pitch)
                plt.axis('off')
                plt.title("Pitch")

                plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser("./grad_cam.py")
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default='../../pyyaml/grad_cam_config.yaml',
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