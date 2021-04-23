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

##########################################################
### main
##########################################################
def main():
    ### config
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, default=None, help='save dir')
    parser.add_argument('--dataset', '-data', type=str, default=None, help='dataset')
    args = parser.parse_args()


    z_dim = 10
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


    ### Set up models
    netG = torch.load(os.path.join('./../results/' + args.dataset + '/main/' + args.dir, 'netGS.pth'))
    netG.eval()
    netG = netG.to(device0)

    if args.dataset == 'mnist':
        img = []
        label = []
        num_classes = 10
    elif args.dataset == 'cifar_10':
        img = []
        label = []
        num_classes = 10
    elif args.dataset == 'CELEBA':
        img = []
        label = []
        num_classes = 10

    for iter in range(1000):

        ############################
        ### Results visualization
        ############################
        '''
        p = 0.5
        bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
        fix_noise = bernoulli.sample((10, z_dim)).view(10, z_dim)
        '''
        ### Random seed
        random.seed(iter)
        np.random.seed(iter)
        torch.manual_seed(iter)
        ### Fix noise for visualization
        p = 0.5
        bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
        fix_noise = bernoulli.sample((1, z_dim)).view(1, z_dim)

        generate_image_unit(iter, netG, fix_noise, save_dir, device0, num_classes, args.dataset)
        img, label = generate_image_npy_unit(iter, netG, fix_noise, save_dir, img, label, device0, num_classes, args.dataset)

    img = np.array(img)
    label = np.array(label)
    print(img.shape)
    print(label.shape)
    np.save(os.path.join(save_dir, args.dir + '_img.npy'), img[1:])
    np.save(os.path.join(save_dir, args.dir + '_label.npy'), label[1:])

def generate_image_unit(iter, netG, fix_noise, save_dir, device, num_classes, dataset):
    batchsize = fix_noise.size()[0]
    nrows = 1
    figsize = (1, 1)
    noise = fix_noise.to(device)

    if dataset == 'mnist':
        img_shape = (28, 28, 1)
    elif dataset == 'cifar_10':
        img_shape = (32, 32, 1)
    elif dataset == 'CELEBA':
        img_shape = (32, 32, 1)

    for i in range(num_classes): ##label
        label = torch.full((nrows,), i, dtype=torch.int32).to(device)
        sample = netG(noise, label)
        sample = sample.view(img_shape)
        sample = sample.cpu().data.numpy()
        
        
        plt.figure(figsize=figsize)
        plt.imshow(sample, cmap='gray')
        plt.axis('off')
        savefig(os.path.join(save_dir, 'samples_{}.png'.format(iter*2 + i)))

    del label, noise, sample
    torch.cuda.empty_cache()


def generate_image_unit_npy(iter, netG, fix_noise, save_dir, img, label_l, device, num_classes, dataset):
    nrows = 1
    ncols = num_classes
    figsize = (1, 1)
    noise = fix_noise.to(device)

    if dataset == 'mnist':
        img_shape = (28, 28, 1)
    elif dataset == 'cifar_100':
        img_shape = (32, 32, 1)
    elif dataset == 'CELEBA':
        img_shape = (32, 32, 1)

    for i in range(num_classes): ##label
        label = torch.full((nrows,), i, dtype=torch.int32).to(device)
        sample = netG(noise, label)
        sample = sample.view(img_shape)
        sample = sample.cpu().data.numpy()

        img.append(sample)
        label_l.append(i)
    return img, label_l

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi, format='png')

if __name__ == '__main__':
    main()
