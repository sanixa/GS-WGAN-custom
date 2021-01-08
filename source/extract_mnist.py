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
import gzip

##########################################################
### main
##########################################################
def main():
    f = gzip.open('./../data/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz','r')

    image_size = 28
    num_images = 10000

    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)

    for i in range(num_images):
        plt.figure(figsize=(1,1))
        plt.imshow(data[i], cmap='gray')
        plt.axis('off')
        dpi = 150
        plt.savefig('./../results/extracted/image/samples_{}.png'.format(i), dpi=dpi, format='png')
        plt.close('all')

if __name__ == '__main__':
    main()
