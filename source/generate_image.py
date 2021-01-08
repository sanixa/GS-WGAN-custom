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
from utils import mkdir
import matplotlib.pyplot as plt
import argparse

IMG_DIM = 784
NUM_CLASSES = 10
CLIP_BOUND = 1.
SENSITIVITY = 2.
DATA_ROOT = './../data'


 

##########################################################
### main
##########################################################
def main():
    ### config
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, default=None, help='save dir')
    args = parser.parse_args()


    dataset = 'mnist'
    z_dim = 10
    batchsize = 32
    save_dir = './../results/generated/' + args.dir
    num_gpus = 1
    random_seed = 42
    print(save_dir)

    mkdir(save_dir)
    ### CUDA
    use_cuda = torch.cuda.is_available()
    devices = [torch.device("cuda:%d" % i if use_cuda else "cpu") for i in range(num_gpus)]
    device0 = devices[0]
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    ### Random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    ### Fix noise for visualization
    p = 0.5
    bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
    fix_noise = bernoulli.sample((10, z_dim)).view(10, z_dim)


    ### Set up models
    netG = torch.load(os.path.join('./../results/mnist/main/' + args.dir, 'netGS.pth'))
    netG.eval()
    netG = netG.to(device0)


 
    for iter in range(100):

        ############################
        ### Results visualization
        ############################
        p = 0.5
        bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
        fix_noise = bernoulli.sample((10, z_dim)).view(10, z_dim)
        generate_image(iter, netG, fix_noise, save_dir, device0, num_classes=10)

def generate_image(iter, netG, fix_noise, save_dir, device, num_classes=10,
                   img_w=28, img_h=28):
    batchsize = fix_noise.size()[0]
    nrows = 10
    ncols = num_classes
    figsize = (1, 1)
    noise = fix_noise.to(device)
    '''
    sample_list = []
    for class_id in range(num_classes):
        label = torch.full((nrows,), class_id, dtype=torch.int32).to(device)
        sample = netG(noise, label)
        sample = sample.view(batchsize, img_w, img_h)
        sample = sample.cpu().data.numpy()
        sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 2, 3])
    samples = np.reshape(samples, [nrows * ncols, img_w, img_h])

    plt.figure(figsize=figsize)
    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(samples[i], cmap='gray')
        plt.axis('off')
    savefig(os.path.join(save_dir, 'samples_{}.png'.format(iter)))

    del label, noise, sample
    torch.cuda.empty_cache()
    '''

    for i in range(num_classes): ##label
        label = torch.full((nrows,), i, dtype=torch.int32).to(device)
        sample = netG(noise, label)
        sample = sample.view(batchsize, img_w, img_h)
        sample = sample.cpu().data.numpy()

        plt.figure(figsize=figsize)
        for j in range(ncols):
            plt.imshow(sample[j], cmap='gray')
            plt.axis('off')
            savefig(os.path.join(save_dir, 'samples_{}.png'.format(iter*100 + i*num_classes + j)))

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi, format='png')

if __name__ == '__main__':
    main()
