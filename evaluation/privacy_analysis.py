import os
import sys
import numpy as np
from autodp import rdp_acct, rdp_bank

sys.path.insert(0, '../source')
from config import *


def main(config):
    delta = 1e-5
    batch_size = config['batchsize']
    prob = 1. / config['num_discriminators']  # subsampling rate
    n_steps = config['iterations']  # training iterations
    sigma = 0.4859#config['noise_multiplier']  # noise scale
    func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)

    acct = rdp_acct.anaRDPacct()
    acct.compose_subsampled_mechanism(func, prob, coeff=n_steps * batch_size)
    epsilon = acct.get_eps(delta)
    print("Privacy cost is: epsilon={}, delta={}".format(epsilon, delta))

def main():
    delta = 1e-5
    batch_size = 64
    prob = 1. / 1000  # subsampling rate
    n_steps = 20000  # training iterations
    sigma = 0.1#config['noise_multiplier']  # noise scale
    func = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma}, x)

    acct = rdp_acct.anaRDPacct()
    acct.compose_subsampled_mechanism(func, prob, coeff=n_steps * batch_size)
    epsilon = acct.get_eps(delta)
    print("Privacy cost is: epsilon={}, delta={}".format(epsilon, delta))

if __name__ == '__main__':
    #args = parse_arguments()
    #main(load_config(args))
    main()