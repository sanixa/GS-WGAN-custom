import os, sys
import numpy as np
import random
import torch
from torch.autograd import Variable
from tqdm import tqdm

from config import *
from models import *
from utils import *

def main(args):
	z_dim = args.z_dim
	model_dim = args.model_dim
	load_dir = args.load_dir
	num_gpus = args.num_gpus
	
	use_cuda = torch.cuda.is_available()
	devices = [torch.device("cuda:%d" % i if use_cuda else "cpu") for i in range(num_gpus)]
	device0 = devices[0]
	if use_cuda:
		torch.set_default_tensor_type('torch.cuda.FloatTensor')

	LongTensor = torch.cuda.LongTensor

	print("loading model...")
	#netG = GeneratorDCGAN_TS(z_dim=z_dim, model_dim=model_dim, num_classes=10)
	#network_path = os.path.join(load_dir, 'netGS.pth')
	netG = torch.load(load_dir)
	#netG.load_state_dict(torch.load(load_dir))
	netG = netG.to(device0)

	bernoulli = torch.distributions.Bernoulli(torch.tensor([0.5]))

	# eps = 1000
	# dir_path = f'data/GS_eps{eps}'
	dir_path = os.path.join(args.save_dir, load_dir.split('/')[-1].split('.')[0])
	if not os.path.isdir(dir_path):
		os.mkdir(dir_path)
	print(f'model path: {dir_path}')
	print("producing training image...")
	for i in tqdm(range(1, 6)):
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
	np.save(f"{dir_path}/train_data.npy", new_image)
	np.save(f"{dir_path}/train_label.npy", new_label)

	print("producing testing image...")
	for i in tqdm(range(1, 6)):
		noise = Variable(bernoulli.sample((2000, z_dim)).view(2000, z_dim).to(device0))
		label = Variable(LongTensor(np.tile(np.arange(10), 200)).to(device0))
		image = Variable(netG(noise, label))

		if (i == 1):
			new_image = image.cpu().detach().numpy()
			new_label = label.cpu().detach().numpy()
		else:
			new_image = np.concatenate((new_image, image.cpu().detach().numpy()), axis=0)
			new_label = np.concatenate((new_label, label.cpu().detach().numpy()), axis=0)

	np.save(f"{dir_path}/test_data.npy", new_image)
	np.save(f"{dir_path}/test_label.npy", new_label)


if __name__ == '__main__':
	args = parse_arguments()
	main(args)

