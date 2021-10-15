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

from datasets.celeba import CelebA

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
    if dataset == 'cifar_100':
        z_dim = 100
    model_dim = args.model_dim
    batchsize = args.batchsize
    L_gp = args.L_gp
    L_epsilon = args.L_epsilon
    critic_iters = args.critic_iters
    latent_type = args.latent_type
    load_dir = args.load_dir
    save_dir = args.save_dir
    if_dp = (args.noise_multiplier > 0.)
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
    os.system('rm ' + os.path.join(save_dir, 'seed*'))
    os.system('touch ' + os.path.join(save_dir, 'seed=%s' % str(args.random_seed)))
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    ### Set up models
    print('gen_arch:' + gen_arch)
    if dataset == 'cifar_100':
        ngpu = 1
        netG = Generator_celeba(ngpu).to(device0)
        #netG.load_state_dict(torch.load('../results/celeba/main/d_1_2e-4_g_1_2e-4_SN_full/netG_15000.pth'))

        # Handle multi-gpu if desired
        if (device0.type == 'cuda') and (ngpu > 1):
            netG = nn.DataParallel(netG, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.02.
        netG.apply(weights_init)

    netGS = copy.deepcopy(netG).to(device0)
    if dataset == 'cifar_100':
        ngpu = 1
        netD = Discriminator_celeba(ngpu).to(device0)
        #netD.load_state_dict(torch.load('../results/celeba/main/d_1_2e-4_g_1_2e-4_SN_full/netD_15000.pth'))
        # Handle multi-gpu if desired
        if (device0.type == 'cuda') and (ngpu > 1):
            netD = nn.DataParallel(netD, list(range(ngpu)))

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        #netD.apply(weights_init)


    ### Set up optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.99))
    optimizerG = optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.99))

    ### Data loaders
    if dataset == 'cifar_100':
        transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    if dataset == 'cifar_100':
        IMG_DIM = 64*64*3
        NUM_CLASSES = 100
        trainset = datasets.CIFAR100(root=os.path.join('../data'), train=True,
            transform=transform_train, download=True)

    else:
        raise NotImplementedError
    
    ###fix sub-training set (fix to 10000 training samples)
    if args.update_train_dataset:
        indices_full = np.arange(len(trainset))
        np.random.shuffle(indices_full)
        
        '''
        #####ref
        indices = np.loadtxt('index_20k.txt', dtype=np.int_)
        remove_idx = [np.argwhere(indices_full==x) for x in indices]
        indices_ref = np.delete(indices_full, remove_idx)
        
        indices_slice = indices_ref[:20000]
        np.savetxt('index_20k_ref.txt', indices_slice, fmt='%i')   ##ref index is disjoint to original index
        '''
        '''
        ### growing dataset
        indices = np.loadtxt('index_20k.txt', dtype=np.int_)
        remove_idx = [np.argwhere(indices_full==x) for x in indices]
        indices_rest = np.delete(indices_full, remove_idx)

        indices_rest = indices_rest[:20000]
        indices_slice = np.concatenate((indices, indices_rest), axis=0)
        np.savetxt('index_40k.txt', indices_slice, fmt='%i')
        '''
        indices_slice = indices_full[:20000]
        np.savetxt('index_20k_cifar100.txt', indices_slice, fmt='%i')
    #indices = np.loadtxt('index_20k_cifar100.txt', dtype=np.int_)
    #trainset = torch.utils.data.Subset(trainset, indices)
    print(len(trainset))
    
    workers = 0
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                             shuffle=True, num_workers=workers)
        
    if if_dp:
    ### Register hook
        global dynamic_hook_function
        for netD in netD_list:
            netD.conv1.register_backward_hook(master_hook_adder)

    criterion = nn.BCELoss()
    real_label = 1.
    fake_label = 0.
    nz = 100
    fixed_noise = torch.randn(100, nz, 1, 1, device=device0)
    iters = 0
    num_epochs = 256 * 5 + 1

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (data,y) in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data.to(device0)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device0)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device0)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            iters += 1
        
            for iter_g in range(1):
            ############################
            # Update G network
            ###########################
                if if_dp:
                    ### Sanitize the gradients passed to the Generator
                    dynamic_hook_function = dp_conv_hook
                else:
                    ### Only modify the gradient norm, without adding noise
                    dynamic_hook_function = modify_gradnorm_conv_hook

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################

                noise = torch.randn(b_size, nz, 1, 1, device=device0)
                fake = netG(noise)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device0)
                
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()



            ### update the exponential moving average
            exp_mov_avg(netGS, netG, alpha=0.999, global_step=iters)

            ############################
            ### Results visualization
            ############################
            if iters % 10 ==0:
                print('iter:{}, G_cost:{:.2f}, D_cost:{:.2f}'.format(iters, errG.item(),
                                                                        errD.item(),
                                                                        ))
            if iters % args.vis_step == 0:
                if dataset == 'cifar_100':
                    generate_image_celeba(str(iters+0), netGS, fixed_noise, save_dir, device0)

            if iters % args.save_step==0:
                ### save model
                torch.save(netGS.state_dict(), os.path.join(save_dir, 'netGS_%s.pth' % str(iters+0)))
                torch.save(netD.state_dict(), os.path.join(save_dir, 'netD_%s.pth' % str(iters+0)))


        torch.cuda.empty_cache()

        #if ((iters+1) % 500 == 0):
        #    classify_training(netGS, dataset, iters+1)




if __name__ == '__main__':
    args = parse_arguments()
    save_config(args)
    main(args)


