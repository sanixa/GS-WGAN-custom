# +
import os, sys
import numpy as np
import random
import torch
from torch.autograd import Variable
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST
# -

from config import *
from models import *
from utils import *

class GeneratorDCGAN(nn.Module):
    def __init__(self, z_dim=10, model_dim=64, num_classes=10, outact=nn.Sigmoid()):
        super(GeneratorDCGAN, self).__init__()

        self.cdist = nn.CosineSimilarity(dim = 1, eps= 1e-9)
        self.grads = []
        self.grad_dict = {}

        self.model_dim = model_dim
        self.z_dim = z_dim
        self.num_classes = num_classes

        fc = nn.Linear(z_dim + num_classes, 4 * 4 * 4 * model_dim)
        deconv1 = nn.ConvTranspose2d(4 * model_dim, 2 * model_dim, 5)
        deconv2 = nn.ConvTranspose2d(2 * model_dim, model_dim, 5)
        deconv3 = nn.ConvTranspose2d(model_dim, IMG_C, 8, stride=2)

        self.deconv1 = deconv1
        self.deconv2 = deconv2
        self.deconv3 = deconv3
        self.fc = fc
        self.relu = nn.ReLU()
        self.outact = outact

    def forward(self, z, y):
        y_onehot = one_hot_embedding(y, self.num_classes)
        z_in = torch.cat([z, y_onehot], dim=1)
        output = self.fc(z_in)
        output = output.view(-1, 4 * self.model_dim, 4, 4)
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv1(output)
        output = output[:, :, :7, :7]
        output = self.relu(output)
        output = pixel_norm(output)

        output = self.deconv2(output)
        output = self.relu(output).contiguous()
        output = pixel_norm(output)

        output = self.deconv3(output)
        output = self.outact(output)
        return output.view(-1, IMG_W * IMG_H)


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


def main(args):
	z_dim = args.z_dim
	model_dim = args.model_dim
	load_dir = args.load_dir
	num_gpus = args.num_gpus
	latent_type = args.latent_type
	
	use_cuda = torch.cuda.is_available()
	devices = [torch.device("cuda:%d" % i if use_cuda else "cpu") for i in range(num_gpus)]
	device0 = devices[0]
#	if use_cuda:
#		torch.set_default_tensor_type('torch.cuda.FloatTensor')

	LongTensor = torch.cuda.LongTensor

	print("loading model...")
	if args.dataset == 'mnist':
		netG = GeneratorDCGAN(z_dim=z_dim, model_dim=model_dim, num_classes=10)
	elif args.dataset == 'cifar_10':
		netG = GeneratorDCGAN_cifar(z_dim=z_dim, model_dim=model_dim, num_classes=10)
    
	#network_path = os.path.join(load_dir, 'netGS.pth')
	#netG = torch.load(load_dir)
	netG.load_state_dict(torch.load(load_dir))
	netG = netG.to(device0)
    netG.eval()

	if latent_type == 'normal':
		#fix_noise = torch.randn(10, z_dim)
		pass
	elif latent_type == 'bernoulli':
		bernoulli = torch.distributions.Bernoulli(torch.tensor([0.5]))
		
	# eps = 1000
	# dir_path = f'data/GS_eps{eps}'
	dir_path = os.path.join(args.save_dir, load_dir.split('/')[-1].split('.')[0].split('_')[1])
	if not os.path.isdir(dir_path):
		os.makedirs(dir_path, exist_ok = True)
	print(f'model path: {dir_path}')
	print("producing training image...")
	for i in tqdm(range(1, 6)):
		if latent_type == 'normal':
			noise = Variable(torch.randn(2000, z_dim).to(device0))
		elif latent_type == 'bernoulli':
			noise = Variable(bernoulli.sample((2000, z_dim)).view(2000, z_dim).to(device0))
		label = Variable(LongTensor(np.tile(np.arange(10), 200)).to(device0))
		image = Variable(netG(noise, label))

		if (i == 1):
			new_image = image.cpu().detach().numpy()
			new_label = label.cpu().detach().numpy()
			new_noise = noise.cpu().detach().numpy()
		else:
			new_image = np.concatenate((new_image, image.cpu().detach().numpy()), axis=0)
			new_label = np.concatenate((new_label, label.cpu().detach().numpy()), axis=0)
			new_noise = np.concatenate((new_noise, noise.cpu().detach().numpy()), axis=0)

	np.savez_compressed(f"{dir_path}/generated.npz", noise=new_noise, img_r01=new_image)

    
	train_set, test_set = None, None
	if args.dataset == 'mnist':
		transform = transforms.ToTensor()
		train_set = MNIST(root="./../data/MNIST", download=True, train=True, transform=transform)
		test_set = MNIST(root="./../data/MNIST", download=True, train=False, transform=transform)
	elif args.dataset == 'cifar_10':
		transform = transforms.Compose([
						transforms.Grayscale(1),
						transforms.ToTensor()])
		train_set = CIFAR10(root="./../data/CIFAR10", download=True, train=True, transform=transform)
		test_set = CIFAR10(root="./../data/CIFAR10", download=True, train=False, transform=transform)



	train_loader = torch.utils.data.DataLoader(train_set, batch_size=10000, shuffle=True)
	for data, target in train_loader:
		np.save(f"{dir_path}/train_data.npy", data.cpu().detach().numpy())
		np.save(f"{dir_path}/train_label.npy", target.cpu().detach().numpy())


	test_loader = torch.utils.data.DataLoader(test_set, batch_size=10000, shuffle=True)
	for data, target in test_loader:
		np.save(f"{dir_path}/test_data.npy", data.cpu().detach().numpy())
		np.save(f"{dir_path}/test_label.npy", target.cpu().detach().numpy())

if __name__ == '__main__':
	args = parse_arguments()
	main(args)



