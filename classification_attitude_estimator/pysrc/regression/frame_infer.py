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

class CNNAttitudeEstimator:
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

        self.color_img_cv = np.empty(0)

        #Using only 1 GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device ==> ", self.device)

        self.img_transform = self.getImageTransform(self.resize, self.mean_element, self.std_element)
        self.net = self.getNetwork(self.resize, self.weights_path, self.dim_fc_out, self.dropout_rate)

        if self.enable_dropout==True:
            self.do_mc_dropout()

        self.value_dict = []

        with open(self.index_dict_path) as fd:
            reader = csv.reader(fd)
            for row in reader:
                num = float(row[0])
                self.value_dict.append(num)
        
    def do_mc_dropout(self):
        #enable dropout when inference
        for module in self.net.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

    def getImageTransform(self, resize,mean_element,std_element):

        mean = mean_element
        std = std_element
        size = (resize, resize)

        '''
        img_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((mean_element,), (std_element,))
        ])

        '''
        img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

        return img_transform

    def getNetwork(self, resize, weights_path, dim_fc_out, dropout_rate):
        net = vgg_network_mod.Network(resize, dim_fc_out, dropout_rate, use_pretrained_vgg=False)

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
        self.image_data_list, self.ground_truth_list = self.get_data()
        self.result_csv = self.frame_infer(self.image_data_list, self.ground_truth_list)
        self.save_csv(self.result_csv)

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

    def prediction(self, input_image):
        logged_output_roll_array, logged_output_pitch_array, roll_array, pitch_array = self.net(input_image)

        output_roll_array = roll_array.to('cpu').detach().numpy().copy()
        output_pitch_array = pitch_array.to('cpu').detach().numpy().copy()

        #print(output_roll_array)

        return np.array(output_roll_array), np.array(output_pitch_array)

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

    def save_csv(self, result_csv):
        result_csv_path = os.path.join(self.infer_log_top_directory, self.infer_log_file_name)
        csv_file = open(result_csv_path, 'w')
        csv_w = csv.writer(csv_file)
        for row in result_csv:
            csv_w.writerow(row)
        csv_file.close()
        print("Save Inference Data")

    def numpy_2d_softmax(self, x):
        max = np.max(x,axis=1,keepdims=True) #returns max of each row and keeps same dims
        e_x = np.exp(x - max) #subtracts each row with its max value
        sum = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
        f_x = e_x / sum 
        return f_x

    def show_fig_no(self, roll_hist_array, pitch_hist_array, value_dict, image):

        np_roll_hist_array = np.array(roll_hist_array).reshape([1, self.dim_fc_out])
        np_pitch_hist_array = np.array(pitch_hist_array).reshape([1, self.dim_fc_out])

        two_hist_array = np.matmul(np_roll_hist_array.T, np_pitch_hist_array)

        fig = plt.figure(figsize=(8,6))
        plt.imshow(two_hist_array)
        plt.title("Plot 2D array")
        plt.show()


    def show_fig(self, roll_hist_array, pitch_hist_array, value_dict, image):
        plt.bar(value_dict, roll_hist_array)
        plt.show()

        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def generate_data(self, np_value_dict, hist_array):
        count_num = 500000

        data = []

        for value, hist_value in zip(np_value_dict, hist_array):
            tmp_hist_value = count_num * hist_value
            int_hist_value = int(tmp_hist_value)

            for i in range(int_hist_value):
                data.append(float(value))
        
        return data
    
    def frame_infer(self, image_data_list, ground_truth_list):
        print("Start Inference")

        result_csv = []

        infer_count = 0

        diff_total_roll = 0.0
        diff_total_pitch = 0.0

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

                roll_output_array, pitch_output_array = self.prediction(input_image)

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

            roll_hist_array /= float(len(windows))
            pitch_hist_array /= float(len(windows))

            roll_hist = self.array_to_value_simple_hist(roll_hist_array)
            pitch_hist = self.array_to_value_simple_hist(pitch_hist_array)

            #print(roll_hist)

            np_result = np.array(result)

            #roll = np.mean(tmp_roll)
            #pitch = np.mean(tmp_pitch)

            roll = roll_hist
            pitch = pitch_hist

            np_roll_value_array = np.array(roll_value_array)
            np_pitch_value_array = np.array(pitch_value_array)

            roll_var = np.var(np_roll_value_array)
            pitch_var = np.var(np_pitch_value_array)

            diff_roll = np.abs(roll - ground_truth[2])
            diff_pitch = np.abs(pitch - ground_truth[3])

            diff_total_roll += diff_roll
            diff_total_pitch += diff_pitch

            print("Infered Roll:  " + str(roll) +  "[deg]")
            print("GT Roll:       " + str(ground_truth[2]) + "[deg]")
            print("Infered Pitch: " + str(pitch) + "[deg]")
            print("GT Pitch:      " + str(ground_truth[3]) + "[deg]")
            print("Diff Roll: " + str(diff_roll) + " [deg]")
            print("Diff Pitch: " + str(diff_pitch) + " [deg]")

            
            #self.show_fig_no(roll_hist_array, pitch_hist_array, self.value_dict, mono_windows[1])
            


            cov = np.cov(np_result)
            

            #Image roll pitch GTroll GTpitch
            tmp_result_csv = [ground_truth[0], ground_truth[1], roll, pitch, ground_truth[2], ground_truth[3], diff_roll, diff_pitch]
            result_csv.append(tmp_result_csv)

            print("Period [s]: ", time.time() - start_clock)
            print("---------------------")


        print("Inference Test Has Done....")
        print("Average of Error of Roll : " + str(diff_total_roll/float(infer_count)) + " [deg]")
        print("Average of Error of Pitch: " + str(diff_total_pitch/float(infer_count)) + " [deg]")
        return result_csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./frame_infer.py")
    parser.add_argument(
        '--frame_infer_config', '-fic',
        type=str,
        required=False,
        default='../../pyyaml/frame_infer_config.yaml',
        help='Frame Infer Config'
    )

    FLAGS, unparsed = parser.parse_known_args()

    #Load yaml file
    try:
        print("Opening frame infer config file %s", FLAGS.frame_infer_config)
        CFG = yaml.safe_load(open(FLAGS.frame_infer_config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening frame infer config file %s", FLAGS.frame_infer_config)
        quit()
    
    cnn_attitude_estimator = CNNAttitudeEstimator(CFG)
    cnn_attitude_estimator.spin()