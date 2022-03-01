from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import torch
from torchvision import transforms

class DataTransform():
    def __init__(self, original_size, resize, mean, std, hor_fov_deg=-1):
        self.resize = resize
        self.mean = mean
        self.std = std
        img_transform = transforms.Compose([
            transforms.CenterCrop(original_size),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std )
        ])
        self.hor_fov_rad = hor_fov_deg / 180.0 * math.pi

    def __call__(self, img_pil, acc_numpy, phase="train"):
        ## augemntation
        if phase == "train":
            ## mirror
            is_mirror = bool(random.getrandbits(1))
            if is_mirror:
                img_pil, acc_numpy = self.mirror(img_pil, acc_numpy)
            ## homography
            if 0 < self.hor_fov_rad < math.pi:
                img_pil, acc_numpy = self.randomHomography(img_pil, acc_numpy)
            # ## rotation
            img_pil, acc_numpy = self.randomRotation(img_pil, acc_numpy)
        ## img: numpy -> tensor
        img_tensor = self.img_transform(img_pil)
        ## acc: numpy -> tensor
        acc_numpy = acc_numpy.astype(np.float32)
        acc_numpy = acc_numpy / np.linalg.norm(acc_numpy)
        acc_tensor = torch.from_numpy(acc_numpy)
        return img_tensor, acc_tensor

    def mirror(self, img_pil, acc_numpy):
        ## image
        img_pil = ImageOps.mirror(img_pil)
        ## acc
        acc_numpy[1] = -acc_numpy[1]
        return img_pil, acc_numpy

    def randomHomography(self, img_pil, acc_numpy):
        angle_rad = random.uniform(-10.0, 10.0) / 180.0 * math.pi
        # print("hom: angle_rad/math.pi*180.0 = ", angle_rad/math.pi*180.0)
        ## image
        (w, h) = img_pil.size
        ver_fov_rad = h / w * self.hor_fov_rad
        d = h / 2 / math.tan(ver_fov_rad / 2)
        l = h / 2 / math.sin(ver_fov_rad / 2)
        l_small = d / math.cos(ver_fov_rad / 2 - abs(angle_rad))
        d_small = l_small * math.cos(ver_fov_rad / 2)
        d_large = l * math.cos(ver_fov_rad / 2 - abs(angle_rad))
        h_small = h / 2 - d * math.tan(ver_fov_rad / 2 - abs(angle_rad))
        w_small = d / d_large * w
        w_large = d / d_small * w
        if angle_rad > 0:
            points_before = [(0, h_small), (w, h_small), (0, h), (w, h)]
            points_after = [((w - w_large) / 2, 0), ((w + w_large) / 2, 0), ((w - w_small) / 2, h - h_small), ((w + w_small) / 2, h - h_small)]
        else:
            points_before = [(0, 0), (w, 0), (0, h - h_small), (w, h - h_small)]
            points_after = [((w - w_small) / 2, h_small), ((w + w_small) / 2, h_small), ((w - w_large) / 2, h), ((w + w_large) / 2, h)]
        # print("points_before = ", points_before)
        # print("points_after = ", points_after)
        coeffs = self.find_coeffs(points_after, points_before)
        img_pil = img_pil.transform(img_pil.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)
        ## acc
        acc_numpy = self.rotateVectorPitch(acc_numpy, -angle_rad)
        return img_pil, acc_numpy

    ## copy-pasted from "http://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil"
    def find_coeffs(self, pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pb).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        ret = np.array(res).reshape(8)
        return ret

    def rotateVectorPitch(self, acc_numpy, angle):
        rot = np.array([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)]
    	])
        rot_acc_numpy = np.dot(rot, acc_numpy)
        return rot_acc_numpy

    def randomRotation(self, img_pil, acc_numpy):
        angle_deg = random.uniform(-10.0, 10.0)
        angle_rad = angle_deg / 180 * math.pi
        # print("rot: angle_deg = ", angle_deg)
        ## image
        img_pil = img_pil.rotate(angle_deg)
        ## acc
        acc_numpy = self.rotateVectorRoll(acc_numpy, -angle_rad)
        return img_pil, acc_numpy

    def rotateVectorRoll(self, acc_numpy, angle):
        rot = np.array([
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)]
    	])
        rot_acc_numpy = np.dot(rot, acc_numpy)
        return rot_acc_numpy