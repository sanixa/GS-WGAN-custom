# +
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# +
def one_hot_embedding(y, num_classes=10, dtype=torch.FloatTensor):
    '''
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    '''
    scatter_dim = len(y.size())
    y_tensor = y.type(torch.LongTensor).view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)

def pixel_norm(x, eps=1e-10):
    '''
    Pixel normalization
    :param x:
    :param eps:
    :return:
    '''
    return x * torch.rsqrt(torch.mean(torch.pow(x, 2), dim=1, keepdim=True) + eps)


# -

#@torchsnooper.snoop()
class GeneratorDCGAN_cifar(nn.Module):
    def __init__(self, z_dim=10, model_dim=64, num_classes=10, outact=nn.Tanh()):
        super(GeneratorDCGAN_cifar, self).__init__()

        self.cdist = nn.CosineSimilarity(dim = 1, eps= 1e-9)
        self.grads = []
        self.grad_dict = {}

        self.model_dim = model_dim
        self.z_dim = z_dim
        self.num_classes = num_classes

        fc = nn.Linear(z_dim + num_classes, z_dim * 1 * 1)
        deconv1 = nn.ConvTranspose2d(z_dim, model_dim * 4, 4, 1, 0, bias=False)
        deconv2 = nn.ConvTranspose2d(model_dim * 4, model_dim * 2, 4, 2, 1, bias=False)
        deconv3 = nn.ConvTranspose2d(model_dim * 2, model_dim, 4, 2, 1, bias=False)
        deconv4 = nn.ConvTranspose2d(model_dim, 1, 4, 2, 1, bias=False)

        self.deconv1 = deconv1
        self.deconv2 = deconv2
        self.deconv3 = deconv3
        self.deconv4 = deconv4
        self.BN_1 = nn.BatchNorm2d(model_dim * 4)
        self.BN_2 = nn.BatchNorm2d(model_dim * 2)
        self.BN_3 = nn.BatchNorm2d(model_dim)
        self.fc = fc
        self.relu = nn.ReLU()
        self.outact = outact

        ''' reference by https://github.com/Ksuryateja/DCGAN-CIFAR10-pytorch/blob/master/gan_cifar.py
        nn.ConvTranspose2d(z_dim, model_dim * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(model_dim * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(model_dim * 8, model_dim * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(model_dim * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(model_dim * 4, model_dim * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(model_dim * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(model_dim * 2, model_dim, 4, 2, 1, bias=False),
        nn.BatchNorm2d(model_dim),
        nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        nn.ConvTranspose2d(model_dim, nc, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. (nc) x 64 x 64
        '''

    def forward(self, z, y):
        y_onehot = one_hot_embedding(y, self.num_classes)
        z_in = torch.cat([z, y_onehot], dim=1)
        output = self.fc(z_in)
        output = output.view(-1, self.z_dim, 1, 1)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv1(output)
        output = self.BN_1(output)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv2(output)
        output = self.BN_2(output)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv3(output)
        output = self.BN_3(output)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv4(output)
        output = self.outact(output)

        return output.view(-1, 32 * 32)


#@torchsnooper.snoop()
class GeneratorDCGAN_cifar_ch3(nn.Module):
    def __init__(self, z_dim=10, model_dim=64, num_classes=10, outact=nn.Tanh()):
        super(GeneratorDCGAN_cifar_ch3, self).__init__()

        self.cdist = nn.CosineSimilarity(dim = 1, eps= 1e-9)
        self.grads = []
        self.grad_dict = {}

        self.model_dim = model_dim
        self.z_dim = z_dim
        self.num_classes = num_classes

        fc = nn.Linear(z_dim + num_classes, z_dim * 1 * 1)
        deconv1 = nn.ConvTranspose2d(z_dim, model_dim * 4, 4, 1, 0, bias=False)
        deconv2 = nn.ConvTranspose2d(model_dim * 4, model_dim * 2, 4, 2, 1, bias=False)
        deconv3 = nn.ConvTranspose2d(model_dim * 2, model_dim, 4, 2, 1, bias=False)
        deconv4 = nn.ConvTranspose2d(model_dim, 3, 4, 2, 1, bias=False)

        self.deconv1 = deconv1
        self.deconv2 = deconv2
        self.deconv3 = deconv3
        self.deconv4 = deconv4
        self.BN_1 = nn.BatchNorm2d(model_dim * 4)
        self.BN_2 = nn.BatchNorm2d(model_dim * 2)
        self.BN_3 = nn.BatchNorm2d(model_dim)
        self.fc = fc
        self.relu = nn.ReLU()
        self.outact = outact

        ''' reference by https://github.com/Ksuryateja/DCGAN-CIFAR10-pytorch/blob/master/gan_cifar.py
        nn.ConvTranspose2d(z_dim, model_dim * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(model_dim * 8),
        nn.ReLU(True),
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(model_dim * 8, model_dim * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(model_dim * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(model_dim * 4, model_dim * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(model_dim * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(model_dim * 2, model_dim, 4, 2, 1, bias=False),
        nn.BatchNorm2d(model_dim),
        nn.ReLU(True),
        # state size. (ngf) x 32 x 32
        nn.ConvTranspose2d(model_dim, nc, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. (nc) x 64 x 64
        '''

    def forward(self, z, y):
        y_onehot = one_hot_embedding(y, self.num_classes)
        z_in = torch.cat([z, y_onehot], dim=1)
        output = self.fc(z_in)
        output = output.view(-1, self.z_dim, 1, 1)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv1(output)
        output = self.BN_1(output)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv2(output)
        output = self.BN_2(output)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv3(output)
        output = self.BN_3(output)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv4(output)
        output = self.outact(output)

        return output.view(-1, 3*32 * 32)


save_dir = 'results/generated/'


def save_cifar():
    
    save_dir = 'results/generated/raw_cifar/train'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    transform_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    ])

    IMG_DIM = 1024
    NUM_CLASSES = 10
    dataloader = datasets.CIFAR10
    trainset = dataloader(root=os.path.join('data', 'CIFAR10'), train=True, download=True,
                          transform=transform_train)
    
    indices = np.loadtxt('source/index_50000.txt', dtype=np.int_)
    trainset = torch.utils.data.Subset(trainset, indices)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=True)
    train_data, train_target = next(iter(train_loader))
    
    train_data = train_data.cpu().data.numpy()
    train_data = np.reshape(train_data, [50000, 32, 32])
    
    for i in range(50000):
        temp = plt.figure(figsize=(1, 1))
        plt.imshow(train_data[i], cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'samples_{}.png'.format(i)), dpi=32, format='png')
        del temp

    save_dir = 'results/generated/raw_cifar/test'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    transform_test = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    ])

    IMG_DIM = 1024
    NUM_CLASSES = 10
    dataloader = datasets.CIFAR10
    testset = dataloader(root=os.path.join('data', 'CIFAR10'), train=False, download=True,
                          transform=transform_test)
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=True)
    test_data, test_target = next(iter(test_loader))
    
    test_data = test_data.cpu().data.numpy()
    test_data = np.reshape(test_data, [10000, 32, 32])
    
    for i in range(10000):
        temp = plt.figure(figsize=(1, 1))
        plt.imshow(test_data[i], cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'samples_{}.png'.format(i)), dpi=32, format='png')
        del temp


# +
def generate_unit_image_cifar(num_classes=10,
                   img_w=32, img_h=32):
    save_dir = 'results/generated/d5000_i80000_100/10000/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    p = 0.5
    bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
    figsize = (1, 1)
    
    netG = GeneratorDCGAN_cifar(z_dim=100, model_dim=64, num_classes=10)
    netG.load_state_dict(torch.load('results/cifar_10/main/d5000_i80000_100/netGS_10000.pth'))

    sample_list = []
    for class_id in range(num_classes):
        noise = bernoulli.sample((1000, 100)).view(1000, 100)
        label = torch.full((1000,), class_id, dtype=torch.long).cuda()
        sample = netG(noise, label)
        sample = sample.view(1000, img_w, img_h)
        sample = sample.cpu().data.numpy()
        sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 2, 3])
    samples = np.reshape(samples, [10000, img_w, img_h])
    
    for i in range(10000):
        temp = plt.figure(figsize=figsize)
        plt.imshow(samples[i], cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'samples_{}.png'.format(i)), dpi=150, format='png')
        del temp
        
    del label, noise, sample
    torch.cuda.empty_cache()
    

# -

def generate_unit_image_cifar_ch3(num_classes=10,
                   img_w=32, img_h=32):
    save_dir = 'results/generated/d500_i80000_10_ch3/10000/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    p = 0.5
    bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
    figsize = (1, 1)
    
    netG = GeneratorDCGAN_cifar_ch3(z_dim=100, model_dim=64, num_classes=10)
    netG.load_state_dict(torch.load('results/cifar_10/main/d500_i80000_10_ch3/netGS_10000.pth'))

    sample_list = []
    for class_id in range(num_classes):
        noise = bernoulli.sample((1000, 100)).view(1000, 100)
        label = torch.full((1000,), class_id, dtype=torch.long).cuda()
        sample = netG(noise, label)
        sample = sample.view(1000, 3, img_w, img_h)
        sample = sample.cpu().data.numpy()
        sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 3, 4, 2])
    samples = np.reshape(samples, [10000, img_w, img_h, 3])
    
    samples = np.clip(samples, 0, 1)
    for i in range(10000):
        temp = plt.figure(figsize=figsize)
        plt.imshow(samples[i], cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'samples_{}.png'.format(i)), dpi=150, format='png')
        del temp
        
    del label, noise, sample
    torch.cuda.empty_cache()
    



def main():
    save_cifar()
    #generate_unit_image_cifar_ch3()


if __name__ == '__main__':
    main()



