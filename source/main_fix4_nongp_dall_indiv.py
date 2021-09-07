import os, sys
import numpy as np
import random
import copy
import torch
import torch.autograd as autograd
from torch.autograd import Variable

from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler

from config import *
from models import *
from utils import *

from ops import exp_mov_avg
#from torchsummary import summary
from torchinfo import summary
from tqdm import tqdm

import pyvacy
#import torch.optim as optim
from pyvacy import optim, analysis, sampling

IMG_DIM = 768
NUM_CLASSES = 100
CLIP_BOUND = 1.
SENSITIVITY = 2.
DATA_ROOT = './../data'

iter_milestone = [1000, 5000, 10000, 20000]
acc_milestone = [i for i in range(10, 100, 10)]
acc_passed = [False for i in range(1, 10)]

##########################################################
### hook functions
##########################################################
def master_hook_adder(module, grad_input, grad_output):
    '''
    global hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    global dynamic_hook_function
    return dynamic_hook_function(module, grad_input, grad_output)


def dummy_hook(module, grad_input, grad_output):
    '''
    dummy hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    pass


def modify_gradnorm_conv_hook(module, grad_input, grad_output):
    '''
    gradient modification hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    ### get grad wrt. input (image)
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize  # account for the 'sum' operation in GP

    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    ### clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_image = clip_coef * grad_wrt_image
    grad_input = (grad_wrt_image.view(grad_input_shape), grad_input[1], grad_input[2])
    return tuple(grad_input)


def dp_conv_hook(module, grad_input, grad_output):
    '''
    gradient modification + noise hook

    :param module:
    :param grad_input:
    :param grad_output:
    :return:
    '''
    global noise_multiplier
    ### get grad wrt. input (image)
    grad_wrt_image = grad_input[0]
    grad_input_shape = grad_wrt_image.size()
    batchsize = grad_input_shape[0]
    clip_bound_ = CLIP_BOUND / batchsize

    grad_wrt_image = grad_wrt_image.view(batchsize, -1)
    grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)

    ### clip
    clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
    clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
    clip_coef = clip_coef.unsqueeze(-1)
    grad_wrt_image = clip_coef * grad_wrt_image

    ### add noise
    noise = clip_bound_ * noise_multiplier * SENSITIVITY * torch.randn_like(grad_wrt_image)
    grad_wrt_image = grad_wrt_image + noise
    grad_input_new = [grad_wrt_image.view(grad_input_shape)]
    for i in range(len(grad_input)-1):
        grad_input_new.append(grad_input[i+1])
    return tuple(grad_input_new)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size()[0], -1)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        #input [1, 28, 28]
        self.model = nn.Sequential(
            Flatten(),

            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, input):
        return self.model(input)

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor



def classify_training(netGS, dataset, iter):
    ### Data loaders
    if dataset == 'mnist' or dataset == 'fashionmnist':
        transform_train = transforms.Compose([
        transforms.CenterCrop((28, 28)),
        transforms.ToTensor(),
        #transforms.Grayscale(),
        ])
    elif dataset == 'cifar_100' or dataset == 'cifar_10':
        transform_train = transforms.Compose([
        transforms.CenterCrop((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        ])

    if dataset == 'mnist':
        dataloader = datasets.MNIST
        test_set = dataloader(root=os.path.join(DATA_ROOT, 'MNIST'), train=False, download=True,
                              transform=transform_train)
        IMG_DIM = 784
        NUM_CLASSES = 10
    elif dataset == 'fashionmnist':
        dataloader = datasets.FashionMNIST
        test_set = dataloader(root=os.path.join(DATA_ROOT, 'FashionMNIST'), train=False, download=True,
                              transform=transform_train)
    elif dataset == 'cifar_100':
        dataloader = datasets.CIFAR100
        test_set = dataloader(root=os.path.join(DATA_ROOT, 'CIFAR100'), train=False, download=True,
                              transform=transform_train)
        IMG_DIM = 3072
        NUM_CLASSES = 100
    elif dataset == 'cifar_10':
        IMG_DIM = 784
        NUM_CLASSES = 10
        dataloader = datasets.CIFAR10
        test_set = dataloader(root=os.path.join(DATA_ROOT, 'CIFAR10'), train=False, download=True,
                              transform=transform_train)
    else:
        raise NotImplementedError
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    netGS.eval()
    for i in tqdm(range(25)):
        gen_labels = Variable(LongTensor(np.random.randint(0, 10, 2000)))
        noise = Variable(FloatTensor(np.random.normal(0, 1, (2000, args.z_dim))))
        synthesized = netGS(noise, gen_labels)

        if (i == 0):
            new_data = synthesized.cpu().detach()
            new_label = gen_labels.cpu().detach()
        else:
            new_data = torch.cat((new_data, synthesized.cpu().detach()), 0)
            new_label = torch.cat((new_label, gen_labels.cpu().detach()), 0)

    new_data = torch.clamp(new_data, min=0., max=1.)

    C = Classifier().cuda()
    C.train()
    opt_C = torch.optim.Adam(C.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    gen_set = TensorDataset(new_data.cuda(), new_label.cuda())
    gen_loader = DataLoader(gen_set, batch_size=32, shuffle=True)

    prg_bar = tqdm(range(50))
    for epoch in prg_bar:
        train_acc = 0.0
        train_loss = 0.0
        for i, (data, label) in enumerate(gen_loader):
            pred = C(data)
            loss = criterion(pred, label)

            opt_C.zero_grad()
            loss.backward()
            opt_C.step()

            train_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == label.cpu().numpy())
            train_loss += loss.item()
        
        prg_bar.set_description(f'acc: {train_acc/gen_set.__len__():.3f}  loss: {train_loss/gen_set.__len__():.4f}')

    test_acc = 0.0
    C.eval()
    for i, (data, label) in enumerate(test_loader):
        data = Variable(data.type(FloatTensor))
        label = Variable(label.type(LongTensor))
        pred = C(data)
        test_acc += np.sum(np.argmax(pred.cpu().data.numpy(), axis=1) == label.cpu().numpy())

    test_acc /= test_set.__len__()
    test_acc *= 100
    print(f'the final result of test accuracy = {test_acc/100:.3f}')

    for i in range(9):
        if (test_acc > acc_milestone[i] and acc_passed[i] == False):
            acc_passed[i] = True
            torch.save(netGS, os.path.join(args.checkpoint, f'diff_acc/{acc_milestone[i]}.pth'))
            with open(os.path.join(args.checkpoint, f'diff_acc/result.txt'), 'a') as f:
                f.write(f"thres:{acc_milestone[i]}% iter:{iter}, acc:{test_acc:.1f}%\n")

    if (iter in iter_milestone):
        torch.save(netGS, os.path.join(args.checkpoint, f'diff_iter/{iter}.pth'))
        with open(os.path.join(args.checkpoint, f'diff_iter/result.txt'), 'a') as f:
                f.write(f"iter:{iter}, acc:{test_acc:.1f}%\n")

    del C, new_data, new_label, gen_set, gen_loader
    torch.cuda.empty_cache()

def add_noise(net):
    with torch.no_grad():
        for p_net in net.parameters():
            grad_input_shape = p_net.grad.shape
            batchsize = grad_input_shape[0]
            clip_bound_ = CLIP_BOUND / batchsize

            grad_wrt_image = p_net.grad.view(batchsize, -1)
            grad_input_norm = torch.norm(grad_wrt_image, p=2, dim=1)
        
            ### clip
            clip_coef = clip_bound_ / (grad_input_norm + 1e-10)
            clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
            clip_coef = clip_coef.unsqueeze(-1)
            grad_wrt_image = clip_coef * grad_wrt_image

            ### add noise
            global noise_multiplier
            noise = clip_bound_ * noise_multiplier * SENSITIVITY * torch.randn_like(grad_wrt_image)
            grad_wrt_image = grad_wrt_image + noise
            grad_input_new = grad_wrt_image.view(grad_input_shape)
            p_net = grad_input_new
    return net

##########################################################
### main
##########################################################
def main(args):
    ### config
    global noise_multiplier
    dataset = args.dataset
    num_discriminators = args.num_discriminators
    noise_multiplier = args.noise_multiplier
    z_dim = args.z_dim
    model_dim = args.model_dim
    batchsize = args.batchsize
    L_gp = args.L_gp
    L_epsilon = args.L_epsilon
    critic_iters = args.critic_iters
    latent_type = args.latent_type
    load_dir = args.load_dir
    save_dir = args.save_dir
    if_dp = (args.dp > 0.)
    gen_arch = args.gen_arch
    num_gpus = args.num_gpus

    ### CUDA
    use_cuda = torch.cuda.is_available()
    devices = [torch.device("cuda:%d" % i if use_cuda else "cpu") for i in range(num_gpus)]
    device0 = devices[0]
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    ### Random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    ### Fix noise for visualization
    if latent_type == 'normal':
        fix_noise = torch.randn(10, z_dim)
    elif latent_type == 'bernoulli':
        p = 0.5
        bernoulli = torch.distributions.Bernoulli(torch.tensor([p]))
        fix_noise = bernoulli.sample((10, z_dim)).view(10, z_dim)
    else:
        raise NotImplementedError

    ### Set up models
    print('gen_arch:' + gen_arch)
    netG = GeneratorDCGAN(z_dim=z_dim, model_dim=model_dim, num_classes=10)
    
    netGS = copy.deepcopy(netG)
    netD_list = []
    for i in range(num_discriminators):
        netD = DiscriminatorDCGAN()
        netD_list.append(netD)

    ### Load pre-trained discriminators
    print("load pre-training...")
    if load_dir is not None:
        for netD_id in range(num_discriminators):
            print('Load NetD ', str(netD_id))
            network_path = os.path.join(load_dir, 'netD_%d' % netD_id, 'netD.pth')
            netD = netD_list[netD_id]
            netD.load_state_dict(torch.load(network_path))

    netG = netG.to(device0)
    for netD_id, netD in enumerate(netD_list):
        device = devices[get_device_id(netD_id, num_discriminators, num_gpus)]
        netD.to(device)

    ### Set up optimizers
    optimizerD_list = []
    for i in range(num_discriminators):
        netD = netD_list[i]
        opt_D = pyvacy.optim.DPAdam(
            l2_norm_clip = 1.0,
            noise_multiplier = noise_multiplier,
            minibatch_size = args.batchsize,
            microbatch_size = 1,
            params = netD.parameters(),
            lr = 1e-4,
            betas = (0.5, 0.999),
        )
        #optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.99))
        optimizerD_list.append(opt_D)
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.99))

    ### Data loaders
    if dataset == 'mnist' or dataset == 'fashionmnist':
        transform_train = transforms.Compose([
        transforms.CenterCrop((28, 28)),
        transforms.ToTensor(),
        #transforms.Grayscale(),
        ])
    elif dataset == 'cifar_100' or dataset == 'cifar_10':
        transform_train = transforms.Compose([
        transforms.CenterCrop((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        ])

    if dataset == 'mnist':
        dataloader = datasets.MNIST
        trainset = dataloader(root=os.path.join(DATA_ROOT, 'MNIST'), train=True, download=True,
                              transform=transform_train)
        IMG_DIM = 784
        NUM_CLASSES = 10
    elif dataset == 'fashionmnist':
        dataloader = datasets.FashionMNIST
        trainset = dataloader(root=os.path.join(DATA_ROOT, 'FashionMNIST'), train=True, download=True,
                              transform=transform_train)
    elif dataset == 'cifar_100':
        dataloader = datasets.CIFAR100
        trainset = dataloader(root=os.path.join(DATA_ROOT, 'CIFAR100'), train=True, download=True,
                              transform=transform_train)
        IMG_DIM = 3072
        NUM_CLASSES = 100
    elif dataset == 'cifar_10':
        IMG_DIM = 784
        NUM_CLASSES = 10
        dataloader = datasets.CIFAR10
        trainset = dataloader(root=os.path.join(DATA_ROOT, 'CIFAR10'), train=True, download=True,
                              transform=transform_train)
    else:
        raise NotImplementedError

    print('creat indices file')
    indices_full = np.arange(len(trainset))
    np.random.shuffle(indices_full)
    #indices_full.dump(os.path.join(save_dir, 'indices.npy'))
    trainset_size = int(len(trainset) / num_discriminators)
    print('Size of the dataset: ', trainset_size)

    mini_loader, micro_loader = pyvacy.sampling.get_data_loaders(batchsize, 1, args.iterations)
    input_pipelines = []
    for i in range(num_discriminators):
        '''
        start = i * trainset_size
        end = (i + 1) * trainset_size
        indices = indices_full[start:end]
        trainloader = DataLoader(trainset, batch_size=args.batchsize, drop_last=False,
                                      num_workers=args.num_workers, sampler=SubsetRandomSampler(indices))
        #input_data = inf_train_gen(trainloader)
        input_pipelines.append(trainloader)
        '''
        start = i * trainset_size
        end = (i + 1) * trainset_size
        indices = indices_full[start:end]
        trainset_1 = torch.utils.data.Subset(trainset, indices)
        trainloader = mini_loader(trainset_1)
        input_pipelines.append(trainloader)

    ### not add noise by hook which is middle between G/D
    '''
    if if_dp:
    ### Register hook
    
        global dynamic_hook_function
        for netD in netD_list:
            netD.conv1.register_backward_hook(master_hook_adder)
    '''


    prg_bar = tqdm(range(args.iterations+1))
    for iters in prg_bar:
        #########################
        ### Update D network
        #########################
        netD_id = np.random.randint(num_discriminators, size=1)[0]
        device = devices[get_device_id(netD_id, num_discriminators, num_gpus)]
        netD = netD_list[netD_id]
        optimizerD = optimizerD_list[netD_id]
        input_data = input_pipelines[netD_id]



        for p in netD.parameters():
            p.requires_grad = True
        '''
        ### Register hook for add noise to D
        if if_dp:
            for parameter in netD.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(
                        lambda grad: grad + (1 / batchsize) * noise_multiplier * torch.randn(grad.shape) * SENSITIVITY
                        #lambda grad: grad
                    )
        '''
        optimizerD.zero_grad()
        x_mini, y_mini = next(iter(input_data))
        for real_data, real_y in micro_loader(TensorDataset(x_mini, y_mini)):
            real_data = real_data.view(-1, IMG_DIM)
            real_data = real_data.to(device)
            real_y = real_y.to(device)
            real_data_v = autograd.Variable(real_data)

            ### train with real
            dynamic_hook_function = dummy_hook
            netD.zero_grad()
            D_real_score = netD(real_data_v, real_y)
            D_real = -D_real_score.mean()

            ### train with fake
            batchsize = real_data.shape[0]
            if latent_type == 'normal':
                noise = torch.randn(batchsize, z_dim).to(device0)
            elif latent_type == 'bernoulli':
                noise = bernoulli.sample((batchsize, z_dim)).view(batchsize, z_dim).to(device0)
            else:
                raise NotImplementedError
            noisev = autograd.Variable(noise)
            fake = autograd.Variable(netG(noisev, real_y.to(device0)).data)
            inputv = fake.to(device)
            D_fake = netD(inputv, real_y.to(device))
            D_fake = D_fake.mean()

            '''
            ### train with gradient penalty
            gradient_penalty = netD.calc_gradient_penalty(real_data_v.data, fake.data, real_y, L_gp, device)
            D_cost = D_fake + D_real + gradient_penalty

            ### train with epsilon penalty
            logit_cost = L_epsilon * torch.pow(D_real_score, 2).mean()
            D_cost += logit_cost
            '''

            ### update
            optimizerD.zero_microbatch_grad()
            D_cost = D_fake + D_real
            D_cost.backward()
            #import ipdb;ipdb.set_trace()
            optimizerD.microbatch_step()

            Wasserstein_D = -D_real - D_fake
        optimizerD.step()

        del real_data, real_y, fake, noise, inputv, D_real, D_fake#, logit_cost, gradient_penalty
        torch.cuda.empty_cache()

        ############################
        # Update G network
        ###########################
        if if_dp:
            ### Sanitize the gradients passed to the Generator
            dynamic_hook_function = dp_conv_hook
        else:
            ### Only modify the gradient norm, without adding noise
            dynamic_hook_function = modify_gradnorm_conv_hook

        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        ### train with sanitized discriminator output
        if latent_type == 'normal':
            noise = torch.randn(batchsize, z_dim).to(device0)
        elif latent_type == 'bernoulli':
            noise = bernoulli.sample((batchsize, z_dim)).view(batchsize, z_dim).to(device0)
        else:
            raise NotImplementedError
        label = torch.randint(0, NUM_CLASSES, [batchsize]).to(device0)
        noisev = autograd.Variable(noise)
        fake = netG(noisev, label)
        #summary(netG, input_data=[noisev,label])
        fake = fake.to(device)
        label = label.to(device)
        G = netD(fake, label)
        G = - G.mean()

        ### update
        optimizerG.zero_grad()
        G.backward()
        G_cost = G
        optimizerG.step()

        ### update the exponential moving average
        exp_mov_avg(netGS, netG, alpha=0.999, global_step=iters)

        ############################
        ### Results visualization
        ############################
        prg_bar.set_description('iter:{}, G_cost:{:.2f}, D_cost:{:.2f}, Wasserstein:{:.2f}'.format(iters, G_cost.cpu().data,
                                                                D_cost.cpu().data,
                                                                Wasserstein_D.cpu().data
                                                                ))
        if iters % args.vis_step == 0:
            if dataset == 'mnist':
                generate_image_mnist(iters, netGS, fix_noise, save_dir, device0)
            elif dataset == 'cifar_100':
                generate_image_cifar100(iters, netGS, fix_noise, save_dir, device0)
            elif dataset == 'cifar_10':
                generate_image_mnist(iters, netGS, fix_noise, save_dir, device0)

        if iters % args.save_step == 0:
            ### save model
            torch.save(netGS.state_dict(), os.path.join(save_dir, 'netGS_%d.pth' % iters))

        del label, fake, noisev, noise, G, G_cost, D_cost
        torch.cuda.empty_cache()

        if ((iters+1) % 500 == 0):
            classify_training(netGS, dataset, iters+1)

    ### save model
    torch.save(netG, os.path.join(save_dir, 'netG.pth'))
    torch.save(netGS, os.path.join(save_dir, 'netGS.pth'))


if __name__ == '__main__':
    args = parse_arguments()
    save_config(args)
    main(args)

