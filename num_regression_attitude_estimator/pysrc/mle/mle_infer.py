import cv2
import PIL.Image as Image
import math
import numpy as np
import time
import argparse
import yaml
import os
import csv

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as nn_functional

from collections import OrderedDict


import sys
sys.path.append('../')
from common import network_mod


class MLE_Infer:
    def __init__(self, CFG):
        self.CFG = CFG
        #contain yaml data to variance
        self.method_name = CFG["method_name"]

        self.infer_dataset_top_directory = CFG["infer_dataset_top_directory"]
        self.csv_name = CFG["csv_name"]

        self.weights_top_directory = CFG["weights_top_directory"]
        self.weights_file_name = CFG["weights_file_name"]

        self.weights_path = os.path.join(self.weights_top_directory, self.weights_file_name)

        self.infer_log_top_directory = CFG["infer_log_top_directory"]
        self.infer_log_file_name = CFG["infer_log_file_name"]

        self.resize = int(CFG["resize"])
        self.original_size = int(CFG["original_size"])
        self.mean_element = float(CFG["mean_element"])
        self.std_element = float(CFG["std_element"])
        self.dim_fc_out = int(CFG["dim_fc_out"])
        self.num_sampling = int(CFG["num_sampling"])
        self.dropout_rate = float(CFG["dropout_rate"])

        self.color_img_cv = np.empty(0)

        #Using only 1 GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device ==> ", self.device)
        self.img_transform = self.getImageTransform(self.resize, self.original_size,self.mean_element, self.std_element)
        self.net = self.getNetwork(self.resize, self.weights_path, self.dim_fc_out, self.dropout_rate)
        self.do_mc_dropout = self.enable_mc_dropout()

        self.image_data_list = []
        self.ground_truth_list = []

        self.result_csv = []

    def getImageTransform(self, resize, original_size,mean_element, std_element):
        mean = mean_element
        std = std_element
        size = (resize, resize)

        '''
        img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std )
        ])
        '''
        img_transform = transforms.Compose([
            transforms.CenterCrop(original_size),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std )
        ])

        return img_transform

    def enable_mc_dropout(self):
        for module in self.net.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()
                can_dropout = True

        return True

    def spin(self):
        self.image_data_list, self.ground_truth_list = self.get_data()
        self.result_csv = self.frame_infer(self.image_data_list, self.ground_truth_list)
        self.save_csv(self.result_csv)

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

        return image_data_list, data_list
    
    def frame_infer(self, image_data_list, ground_truth_list):
        print("Start Inference")

        result_csv = []
        infer_count = 0

        diff_total_roll = 0.0
        diff_total_pitch = 0.0

        for (img_path, ground_truth) in zip(image_data_list, ground_truth_list):
            print("---------Inference at " + str(infer_count) + "---------")
            infer_count += 1
            start_clock = time.time()

            image = cv2.imread(img_path)
            input_tensor = self.transformImage(image)
            
            roll_array, pitch_array, yaw_array= self.prediction(input_tensor)

            roll = np.average(roll_array)
            pitch = np.average(pitch_array)
            yaw = np.average(yaw_array)

            roll_var = np.var(roll_array)
            pitch_var = np.var(pitch_array)

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

            #image roll, pitch, GTroll, GTpitch
            tmp_result_csv = [ground_truth[0], ground_truth[1], roll, pitch, ground_truth[2], ground_truth[3], diff_roll, diff_pitch]
            result_csv.append(tmp_result_csv)

            print("Period [s]: ", time.time() - start_clock)
            print("---------------------")

            '''
            cv2.imshow('image',image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            '''

        print("Inference Test Has Done....")
        print("Average of Error of Roll : " + str(diff_total_roll/float(infer_count)) + " [deg]")
        print("Average of Error of Pitch: " + str(diff_total_pitch/float(infer_count)) + " [deg]")
        return result_csv

    def cvToPIL(self, img_cv):
        img_pil = Image.fromarray(img_cv)
        return img_pil
    
    def transformImage(self, image):
        img_pil = self.cvToPIL(image)
        img_tensor = self.img_transform(img_pil)
        inputs = img_tensor.unsqueeze_(0)
        inputs = inputs.to(self.device)
        #print(inputs)
        return inputs

    def prediction(self, input_tensor):
        roll_array = []
        pitch_array = []
        yaw_array = []

        for i in range(self.num_sampling):
            inferenced_x = self.net(input_tensor)
            inferenced_x = inferenced_x.to('cpu').detach().numpy().copy()

            roll_array.append(float(inferenced_x[0][0])/3.141592*180.0)
            pitch_array.append(float(inferenced_x[0][1])/3.141592*180.0)
            yaw_array.append(float(inferenced_x[0][2])/3.141592*180.0)

        return np.array(roll_array), np.array(pitch_array), np.array(yaw_array)

    def getNetwork(self, resize, weights_path, dim_fc_out, dropout_rate):
        net = network_mod.Network(resize, dim_fc_out, dropout_rate, use_pretrained_vgg=False)

        print(net)

        net.to(self.device)
        net.eval() #change inference mode

        #Load Trained Network
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
    
    def save_csv(self, result_csv):
        result_csv_path = os.path.join(self.infer_log_top_directory, self.infer_log_file_name)
        csv_file = open(result_csv_path, 'w')
        csv_w = csv.writer(csv_file)
        for row in result_csv:
            csv_w.writerow(row)
        csv_file.close()
        print("Save Inference Data")
        print(result_csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./mle_infer.py")

    parser.add_argument(
        '--mle_infer_config', '-fic',
        type=str,
        required=False,
        default='../../pyyaml/mle_infer_config.yaml',
        help='MLE infer config yaml file',
    )

    FLAGS, unparsed = parser.parse_known_args()

    try:
        print("Opening frame infer config file %s", FLAGS.mle_infer_config)
        CFG = yaml.safe_load(open(FLAGS.mle_infer_config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening frame infer config file %s", FLAGS.mle_infer_config)
        quit()
    
    mle_infer = MLE_Infer(CFG)
    mle_infer.spin()