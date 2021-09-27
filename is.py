from inception_score_pytorch.inception_score import inception_score

# +
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy
# +
def one_hot_embedding(y, num_classes=10, dtype=torch.cuda.FloatTensor):
    '''
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    '''
    scatter_dim = len(y.size())
    y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
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


def is_ch3():
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index]

        def __len__(self):
            return len(self.orig)
    '''
    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='data/', download=True,
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )

    IgnoreLabelDataset(cifar)
    '''
    
    p = 0.5
    bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
    netG = GeneratorDCGAN_cifar_ch3(z_dim=100, model_dim=64, num_classes=10).cuda()
    model_path = 'results/cifar_10/main/d500_i80000_10_ch3/netGS_10000.pth'
    netG.load_state_dict(torch.load(model_path))

    sample_list = []
    for class_id in range(10):
        noise = bernoulli.sample((5000, 100)).view(5000, 100).cuda()
        label = torch.full((5000,), class_id, dtype=torch.long).cuda()
        sample = netG(noise, label)
        sample = sample.view(5000, 3, 32, 32)
        sample = sample.cpu().data.numpy()
        sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 2, 3, 4])
    samples = np.reshape(samples, [50000, 3, 32, 32])

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(samples), cuda=True, batch_size=32, resize=True, splits=10))
    
    #########################################################################
    p = 0.5
    bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
    netG = GeneratorDCGAN_cifar_ch3(z_dim=100, model_dim=64, num_classes=10).cuda()
    model_path = 'results/cifar_10/main/d500_i80000_10_ch3/netGS_20000.pth'
    netG.load_state_dict(torch.load(model_path))

    sample_list = []
    for class_id in range(10):
        noise = bernoulli.sample((5000, 100)).view(5000, 100).cuda()
        label = torch.full((5000,), class_id, dtype=torch.long).cuda()
        sample = netG(noise, label)
        sample = sample.view(5000, 3, 32, 32)
        sample = sample.cpu().data.numpy()
        sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 2, 3, 4])
    samples = np.reshape(samples, [50000, 3, 32, 32])

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(samples), cuda=True, batch_size=32, resize=True, splits=10))
    
        #########################################################################
    p = 0.5
    bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
    netG = GeneratorDCGAN_cifar_ch3(z_dim=100, model_dim=64, num_classes=10).cuda()
    model_path = 'results/cifar_10/main/d500_i80000_10_ch3/netGS_30000.pth'
    netG.load_state_dict(torch.load(model_path))

    sample_list = []
    for class_id in range(10):
        noise = bernoulli.sample((5000, 100)).view(5000, 100).cuda()
        label = torch.full((5000,), class_id, dtype=torch.long).cuda()
        sample = netG(noise, label)
        sample = sample.view(5000, 3, 32, 32)
        sample = sample.cpu().data.numpy()
        sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 2, 3, 4])
    samples = np.reshape(samples, [50000, 3, 32, 32])

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(samples), cuda=True, batch_size=32, resize=True, splits=10))
    
        #########################################################################
    p = 0.5
    bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
    netG = GeneratorDCGAN_cifar_ch3(z_dim=100, model_dim=64, num_classes=10).cuda()
    model_path = 'results/cifar_10/main/d500_i80000_10_ch3/netGS_40000.pth'
    netG.load_state_dict(torch.load(model_path))

    sample_list = []
    for class_id in range(10):
        noise = bernoulli.sample((5000, 100)).view(5000, 100).cuda()
        label = torch.full((5000,), class_id, dtype=torch.long).cuda()
        sample = netG(noise, label)
        sample = sample.view(5000, 3, 32, 32)
        sample = sample.cpu().data.numpy()
        sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 2, 3, 4])
    samples = np.reshape(samples, [50000, 3, 32, 32])

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(samples), cuda=True, batch_size=32, resize=True, splits=10))
    
        #########################################################################
    p = 0.5
    bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
    netG = GeneratorDCGAN_cifar_ch3(z_dim=100, model_dim=64, num_classes=10).cuda()
    model_path = 'results/cifar_10/main/d500_i80000_10_ch3/netGS_60000.pth'
    netG.load_state_dict(torch.load(model_path))

    sample_list = []
    for class_id in range(10):
        noise = bernoulli.sample((5000, 100)).view(5000, 100).cuda()
        label = torch.full((5000,), class_id, dtype=torch.long).cuda()
        sample = netG(noise, label)
        sample = sample.view(5000, 3, 32, 32)
        sample = sample.cpu().data.numpy()
        sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 2, 3, 4])
    samples = np.reshape(samples, [50000, 3, 32, 32])

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(samples), cuda=True, batch_size=32, resize=True, splits=10))
    
        #########################################################################
    p = 0.5
    bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
    netG = GeneratorDCGAN_cifar_ch3(z_dim=100, model_dim=64, num_classes=10).cuda()
    model_path = 'results/cifar_10/main/d500_i80000_10_ch3/netGS_80000.pth'
    netG.load_state_dict(torch.load(model_path))

    sample_list = []
    for class_id in range(10):
        noise = bernoulli.sample((5000, 100)).view(5000, 100).cuda()
        label = torch.full((5000,), class_id, dtype=torch.long).cuda()
        sample = netG(noise, label)
        sample = sample.view(5000, 3, 32, 32)
        sample = sample.cpu().data.numpy()
        sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 2, 3, 4])
    samples = np.reshape(samples, [50000, 3, 32, 32])

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(samples), cuda=True, batch_size=32, resize=True, splits=10))



def is_ch1():
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index]

        def __len__(self):
            return len(self.orig)
    '''
    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    cifar = dset.CIFAR10(root='data/', download=True,
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
    )

    IgnoreLabelDataset(cifar)
    '''
    
    p = 0.5
    bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
    netG = GeneratorDCGAN_cifar(z_dim=100, model_dim=64, num_classes=10).cuda()
    netG.load_state_dict(torch.load('results/cifar_10/main/d500_i40000_10/netGS_80000.pth'))

    sample_list = []
    for class_id in range(10):
        noise = bernoulli.sample((5000, 100)).view(5000, 100).cuda()
        label = torch.full((5000,), class_id, dtype=torch.long).cuda()
        sample = netG(noise, label)
        sample = sample.view(5000, 32, 32)
        sample = sample.cpu().data.numpy()
        sample_list.append(sample)
    samples = np.transpose(np.array(sample_list), [1, 0, 2, 3])
    samples = np.reshape(samples, [50000, 1, 32, 32])
    samples = np.repeat(samples, 3, axis=1)

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(samples), cuda=True, batch_size=32, resize=True, splits=10))


if __name__ == '__main__':
    is_ch3()
