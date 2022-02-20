from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import torch
from torchvision import transforms
from torchvision.transforms.transforms import Resize
import torch.nn.functional as nn_functional

class DataTransform():
    def __init__(self, resize, mean, std, original_size):
        self.mean = mean
        self.std = std
        size = (resize, resize)

        '''
        self.img_transform = transforms.Compose([
            transforms.CenterCrop(original_size),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])
        '''
        self.img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

    def __call__(self, img_pil, roll_numpy, pitch_numpy, phase="train"):
        ## img: numpy -> tensor
        img_tensor = self.img_transform(img_pil)
        
        ## roll: numpy -> tensor
        roll_numpy = roll_numpy.astype(np.float32)
        roll_tensor = torch.from_numpy(roll_numpy)

        # pitch: numpy -> tensor
        pitch_numpy = pitch_numpy.astype(np.float32)
        pitch_tensor = torch.from_numpy(pitch_numpy)

        #return img_tensor, logged_roll_tensor, logged_pitch_tensor
        return img_tensor, roll_tensor, pitch_tensor