###
#### ep_1
CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep1_1000 --load_dir checkpoint/gs_checkpoint/cifar10/eps_1/diff_iter/1000.pth --dataset cifar_10 --exp mc_attack
CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep1_5000 --load_dir checkpoint/gs_checkpoint/cifar10/eps_1/diff_iter/5000.pth --dataset cifar_10 --exp mc_attack
CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep1_10000 --load_dir checkpoint/gs_checkpoint/cifar10/eps_1/diff_iter/10000.pth --dataset cifar_10 --exp mc_attack
CUDA_VISIBLE_DEVICES=1 python mc_attack.py  --name ep1_20000 --load_dir checkpoint/gs_checkpoint/cifar10/eps_1/diff_iter/20000.pth --dataset cifar_10 --exp mc_attack