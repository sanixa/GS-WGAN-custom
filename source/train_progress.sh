# +
#CUDA_VISIBLE_DEVICES=2 python main.py --dataset 'mnist' --load_dir ../pretrain_mnist/ResNet_default/ -ndis 200 -ngpu 1 -iter 20000 --checkpoint ./checkpoint/gs_checkpoint/mnist/eps_1000 -noise 0.41
#CUDA_VISIBLE_DEVICES=2 python main.py --dataset 'mnist' --load_dir ../pretrain_mnist/ResNet_default/ -ndis 200 -ngpu 1 -iter 20000 --checkpoint ./checkpoint/gs_checkpoint/mnist/eps_100 -noise 0.531
#CUDA_VISIBLE_DEVICES=2 python main.py --dataset 'mnist' --load_dir ../pretrain_mnist/ResNet_default/ -ndis 200 -ngpu 1 -iter 20000 --checkpoint ./checkpoint/gs_checkpoint/mnist/eps_1 -noise 14.5
#CUDA_VISIBLE_DEVICES=2 python main.py --dataset 'mnist' --load_dir ../pretrain_mnist/ResNet_default/ -ndis 200 -ngpu 1 -iter 20000 --checkpoint ./checkpoint/gs_checkpoint/mnist/eps_inf -noise 0
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'mnist' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 --exp_name main_eps_10 -noise 1.45
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'mnist' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 --exp_name main_eps_1 -noise 14.5
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'mnist' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 32 --exp_name main_eps_inf_z32 -noise 0
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'mnist' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 32 --exp_name main_eps_1000_z32 -noise 0.41
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'mnist' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 32 --exp_name main_eps_100_z32 -noise 0.531
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'mnist' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 32 --exp_name main_eps_10_z32 -noise 1.45
#CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'mnist' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 32 --exp_name main_eps_1_z32 -noise 14.5
# -

#CUDA_VISIBLE_DEVICES=0 python main_ts.py --dataset 'mnist' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 32 --exp_name main_eps_10_ts_z32 -noise 1.45
CUDA_VISIBLE_DEVICES=0 python main_lap.py --dataset 'mnist' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 32 --exp_name main_eps_10_lap_z32 -noise 1.45
#CUDA_VISIBLE_DEVICES=0 python main_mp.py --dataset 'mnist' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 32 --exp_name main_eps_10_mp_z32 -noise 1.45
#CUDA_VISIBLE_DEVICES=0 python main_ge.py --dataset 'mnist' -latent bernoulli -ndis 1000 -ngpu 1 -iter 20000 -zdim 32 --exp_name main_eps_10_ge_z32 -noise 1.45
