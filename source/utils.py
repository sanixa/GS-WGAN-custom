import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import torchvision.utils as vutils

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi, format='png')


def inf_train_gen(trainloader):
    while True:
        for images, targets in iter(trainloader):
            yield (images, targets)


def generate_image_mnist(iter, netG, fix_noise, save_dir, device, num_classes=10,
                   img_w=28, img_h=28):
    batchsize = fix_noise.size()[0]
    nrows = 10
    ncols = num_classes
    figsize = (ncols, nrows)
    noise = fix_noise.to(device)

    sample_list = []
    for class_id in range(num_classes):
        label = torch.full((nrows,), class_id, dtype=torch.long).to(device)
        sample = netG(noise, label)
        sample = sample.view(batchsize, img_w, img_h)
        sample = sample.cpu().data.numpy()
        sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 2, 3])
    samples = np.reshape(samples, [nrows * ncols, img_w, img_h])
    samples = np.clip(samples, 0, 1)

    plt.figure(figsize=figsize)
    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(samples[i], cmap='gray')
        plt.axis('off')
    savefig(os.path.join(save_dir, 'samples_{}.png'.format(iter)))

    del label, noise, sample
    torch.cuda.empty_cache()

def generate_image_cifar10(iter, netG, fix_noise, save_dir, device, num_classes=10,
                   img_w=32, img_h=32):
    batchsize = fix_noise.size()[0]
    nrows = 10
    ncols = num_classes
    figsize = (ncols, nrows)
    noise = fix_noise.to(device)

    sample_list = []
    for class_id in range(num_classes):
        label = torch.full((nrows,), class_id, dtype=torch.long).to(device)
        sample = netG(noise, label)
        sample = sample.view(batchsize, img_w, img_h)
        sample = sample.cpu().data.numpy()
        sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 2, 3])
    samples = np.reshape(samples, [nrows * ncols, img_w, img_h])
    samples = np.clip(samples, 0, 1)

    plt.figure(figsize=figsize)
    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(samples[i], cmap='gray')
        plt.axis('off')
    savefig(os.path.join(save_dir, 'samples_{}.png'.format(iter)))

    del label, noise, sample
    torch.cuda.empty_cache()

    
def generate_image_celeba(iter, netG, fixed_noise, save_dir, device, num_classes=2,
                   img_w=64, img_h=64):
    nrows = 10
    ncols = 10
    figsize = (ncols, nrows)
    noise = fixed_noise.to(device)

    sample_list = []
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    fake = np.clip(fake, 0 , 1)

    plt.figure(figsize=figsize)
    for i in range(nrows * ncols):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(np.transpose(fake[i],(1,2,0)))
        plt.axis('off')
    savefig(os.path.join(save_dir, 'samples_{}.png'.format(iter)))

    del noise, fake
    torch.cuda.empty_cache()

def get_device_id(id, num_discriminators, num_gpus):
    partitions = np.linspace(0, 1, num_gpus, endpoint=False)[1:]
    device_id = 0
    for p in partitions:
        if id <= num_discriminators * p:
            break
        device_id += 1
    return device_id


