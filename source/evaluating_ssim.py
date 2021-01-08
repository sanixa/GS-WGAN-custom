import os, sys
import numpy as np
import random
import copy
import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

from ops import exp_mov_avg
from models import *
import matplotlib.pyplot as plt

from skimage.measure import compare_ssim
import cv2
import glob

##########################################################
### main
##########################################################
def main():
    sample_conut = 10000
    g_img_list = glob.glob('./../results/generated/ResNet-mnist-iter146/*.png')
    t_img_list = glob.glob('./../results/extracted/image/*.png')

    g_rand_idx = np.random.randint(len(g_img_list), size=10000)
    t_rand_idx = np.random.randint(len(t_img_list), size=10000)

    tot = 0
    for _ in range(10):
        for i in range(sample_conut):
            g_img = cv2.imread(g_img_list[g_rand_idx[i]],0)
            t_img = cv2.imread(t_img_list[t_rand_idx[i]],0)
            s = compare_ssim(t_img, g_img)
            tot += s
    print(tot / sample_conut / 10)

if __name__ == '__main__':
    main()
