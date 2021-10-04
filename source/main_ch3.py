# %matplotlib inline

# +
import os, sys
import numpy as np
import random
import copy
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
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

IMG_DIM = -1
NUM_CLASSES = -1
CLIP_BOUND = 1.
SENSITIVITY = 2.
DATA_ROOT = './../data'


# +
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


FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# -

def main(args):
    ### config
    global noise_multiplier
    dataset = args.dataset
    num_discriminators = args.num_discriminators
    noise_multiplier = args.noise_multiplier
    z_dim = args.z_dim
    if dataset == 'celeba':
        z_dim = 100
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
    if args.random_seed == 1:
        args.random_seed = np.random.randint(10000, size=1)[0]
    print('random_seed: {}'.format(args.random_seed))
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
    if dataset == 'mnist':
        netG = GeneratorDCGAN(z_dim=z_dim, model_dim=model_dim, num_classes=10)
    elif dataset == 'cifar_10':
        netG = GeneratorDCGAN_cifar_ch3(z_dim=z_dim, model_dim=model_dim, num_classes=10)
        netG.apply(weights_init)
    elif dataset == 'celeba':
        ngpu = 1
        netG = Generator(ngpu)

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.02.
        netG.apply(weights_init)

    netGS = copy.deepcopy(netG)
    netD_list = []
    for i in range(num_discriminators):
        if dataset == 'mnist':
            netD = DiscriminatorDCGAN()
        elif dataset == 'cifar_10':
            netD = DiscriminatorDCGAN_cifar_ch3()
            #netD.apply(weights_init)
        elif dataset == 'celeba':
            ngpu = 1
            netD = Discriminator_celeba(ngpu)

            # Handle multi-gpu if desired
            if (device.type == 'cuda') and (ngpu > 1):
                netD = nn.DataParallel(netD, list(range(ngpu)))

            # Apply the weights_init function to randomly initialize all weights
            #  to mean=0, stdev=0.2.
            netD.apply(weights_init)
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
        optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.99))
        optimizerD_list.append(optimizerD)
    optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.99))

    ### Data loaders
    if dataset == 'mnist':
        transform_train = transforms.Compose([
        transforms.CenterCrop((28, 28)),
        transforms.ToTensor(),
        ])
    elif dataset == 'cifar_10':
        transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif dataset == 'celeba':
        transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    if dataset == 'mnist':
        IMG_DIM = 784
        NUM_CLASSES = 10
        dataloader = datasets.MNIST
        trainset = dataloader(root=os.path.join(DATA_ROOT, 'MNIST'), train=True, download=True,
                              transform=transform_train)
    elif dataset == 'cifar_10':
        IMG_DIM = 3072
        NUM_CLASSES = 10
        dataloader = datasets.CIFAR10
        trainset = dataloader(root=os.path.join(DATA_ROOT, 'CIFAR10'), train=True, download=True,
                              transform=transform_train)
    elif dataset == 'celeba':
        IMG_DIM = 64*64*3
        NUM_CLASSES = 2
        trainset = CelebA(root=os.path.join('exp', 'datasets', 'celeba'), split='train',
            transform=transform_train, download=False)
    else:
        raise NotImplementedError
    
    ###fix sub-training set (fix to 10000 training samples)
    if args.update_train_dataset:
        if dataset == 'mnist':
            indices_full = np.arange(60000)
        elif  dataset == 'cifar_10':
            indices_full = np.arange(50000)
        elif  dataset == 'celeba':
            indices_full = len(trainset)
        np.random.shuffle(indices_full)
        indices_slice = indices_full[:20000]
        np.savetxt('index_20k.txt', indices_slice, fmt='%i')
    indices = np.loadtxt('index_20k.txt', dtype=np.int_)
    trainset = torch.utils.data.Subset(trainset, indices)


    print('creat indices file')
    indices_full = np.arange(len(trainset))
    np.random.shuffle(indices_full)
    #indices_full.dump(os.path.join(save_dir, 'indices.npy'))
    trainset_size = int(len(trainset) / num_discriminators)
    print('Size of the dataset: ', trainset_size)

    input_pipelines = []
    for i in range(num_discriminators):
        start = i * trainset_size
        end = (i + 1) * trainset_size
        indices = indices_full[start:end]
        trainloader = DataLoader(trainset, batch_size=args.batchsize, drop_last=False,
                                      num_workers=args.num_workers, sampler=SubsetRandomSampler(indices))
        #input_data = inf_train_gen(trainloader)
        input_pipelines.append(trainloader)

    if if_dp:
    ### Register hook
        global dynamic_hook_function
        for netD in netD_list:
            netD.conv1.register_backward_hook(master_hook_adder)

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

        for iter_d in range(critic_iters):
            real_data, real_y = next(iter(input_data))
            real_data = real_data.view(-1, IMG_DIM)
            real_data = real_data.to(device)
            real_y = real_y.to(device)

            ###########################
            if dataset == 'celeba':
            gender = 20
            real_y = real_y[:, gender]
            ##################################################

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

            ### train with gradient penalty
            gradient_penalty = netD.calc_gradient_penalty(real_data_v.data, fake.data, real_y, L_gp, device)
            D_cost = D_fake + D_real + gradient_penalty

            ### train with epsilon penalty
            logit_cost = L_epsilon * torch.pow(D_real_score, 2).mean()
            D_cost += logit_cost

            ### update
            D_cost.backward()
            Wasserstein_D = -D_real - D_fake
            optimizerD.step()

        del real_data, real_y, fake, noise, inputv, D_real, D_fake, logit_cost, gradient_penalty
        torch.cuda.empty_cache()
        
        for iter_d in range(1):
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
            elif dataset == 'cifar_10':
                generate_image_cifar10_ch3(str(iters+0), netGS, fix_noise, save_dir, device0)
            elif dataset == 'celeba':
                generate_image_celeba(str(iters+0), netGS, fix_noise, save_dir, device0)

        if iters % args.save_step==0:
            ### save model
            torch.save(netGS.state_dict(), os.path.join(save_dir, 'netGS_%s.pth' % str(iters+0)))
            torch.save(netD.state_dict(), os.path.join(save_dir, 'netD_%s.pth' % str(iters+0)))


        del label, fake, noisev, noise, G, G_cost, D_cost
        torch.cuda.empty_cache()

        #if ((iters+1) % 500 == 0):
        #    classify_training(netGS, dataset, iters+1)




if __name__ == '__main__':
    args = parse_arguments()
    save_config(args)
    main(args)



