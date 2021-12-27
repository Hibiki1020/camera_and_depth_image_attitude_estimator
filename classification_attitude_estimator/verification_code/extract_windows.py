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

def extract_window(image_original, count, window_original_size):
    height = image_original.shape[0]
    width = image_original.shape[1]

    windows = []
    correct_windows = []
    tmp_windows = []        

    total_window_checker = False
    window_count = 0
    error_count = 0

    while total_window_checker==False:
        width_start = random.randint(0, int(width)-window_original_size)
        height_start = random.randint(0, int(height)-window_original_size)

        window = image_original[height_start:(height_start + window_original_size), width_start:(width_start + window_original_size)]
        #cv2.imshow('window',window)
        #tmp_window_checker = self.check_window(window)
        tmp_window_checker = True

        if tmp_window_checker == True:
            window_count += 1
            correct_windows.append(window)
            tmp_windows.append(window)

            if window_count >= count:
                total_window_checker = True
                windows = correct_windows
            else:
                error_count += 1
                tmp_windows.append(window)

                if error_count >=count:
                    print("Less Feature Point...")
                    total_window_checker = True
                    windows = tmp_windows

    return windows

if __name__ == '__main__':
    image_path = "/media/amsl/96fde31e-3b9b-4160-8d8a-a4b913579ca21/airsim_dataset_kawai/AirSimNH/verify_data1/image30.png"
    save_root_path = "/media/amsl/96fde31e-3b9b-4160-8d8a-a4b913579ca21/misc/windows/"

    image_original = cv2.imread(image_path)

    windows = extract_window(image_original, 4, 672)

    counter = 0

    for window in windows:
        cv2.imwrite(save_root_path + "image" + str(counter) + ".png", window)
        counter += 1