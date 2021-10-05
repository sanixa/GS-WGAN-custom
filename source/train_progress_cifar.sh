# +
#CUDA_VISIBLE_DEVICES=3,4 python main.py --dataset 'cifar_10' --load_dir ../pretrain_mnist/ResNet_default/ -ndis 200 -ngpu 2 -iter 20000 --checkpoint ./checkpoint/gs_checkpoint/cifar10/eps_1000 -noise 0.175935
#CUDA_VISIBLE_DEVICES=3,4 python main.py --dataset 'cifar_10' --load_dir ../pretrain_mnist/ResNet_default/ -ndis 200 -ngpu 2 -iter 20000 --checkpoint ./checkpoint/gs_checkpoint/cifar10/eps_100 -noise 0.24162
#CUDA_VISIBLE_DEVICES=3,4 python main.py --dataset 'cifar_10' --load_dir ../pretrain_mnist/ResNet_default/ -ndis 200 -ngpu 2 -iter 20000 --checkpoint ./checkpoint/gs_checkpoint/cifar10/eps_1 -noise 0.86
#CUDA_VISIBLE_DEVICES=3,4 python main.py --dataset 'cifar_10' --load_dir ../pretrain_mnist/ResNet_default/ -ndis 200 -ngpu 2 -iter 20000 --checkpoint ./checkpoint/gs_checkpoint/cifar10/eps_inf -noise 0
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 --exp_name main_eps_10 -noise 1.45
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 --exp_name main_eps_1 -noise 14.5
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 --exp_name main_eps_inf -noise 0
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 100 --exp_name main_eps_inf_z100 -noise 0
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 100 --exp_name main_eps_1000_z100 -noise 0.41
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 100 --exp_name main_eps_100_z100 -noise 0.531
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 100 --exp_name main_eps_10_z100 -noise 1.45
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 100 --exp_name main_eps_1_z100 -noise 14.5
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 100 --exp_name main_n_01_z100 -noise 0.1
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 100 --exp_name main_n_02_z100 -noise 0.2
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 100 --exp_name main_n_001_z100 -noise 0.01
# -

#CUDA_VISIBLE_DEVICES=0 python main_ts.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 100 --exp_name main_eps_10_ts_z100 -noise 1.45
#CUDA_VISIBLE_DEVICES=0 python main_mp.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 100 --exp_name main_eps_10_mp_z100 -noise 1.45
#CUDA_VISIBLE_DEVICES=0 python main_lap.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 100 --exp_name main_eps_10_lap_z100 -noise 1.45
#CUDA_VISIBLE_DEVICES=0 python main_ge.py --dataset 'cifar_10' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 100 --exp_name main_eps_10_ge_z100 -noise 1.45
for j in 10
		do
			CUDA_VISIBLE_DEVICES=0 python main_ch3.py -s 42 -bs 128 --dataset 'cifar_10' -latent bernoulli -ndis 1 -ngpu 1 -iter 40000 -zdim 100 --exp_name cifar_z100_bs128 -noise 0
		done
