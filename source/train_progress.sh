#CUDA_VISIBLE_DEVICES=2 python main.py --dataset 'mnist' --load_dir ../pretrain_mnist/ResNet_default/ -ndis 200 -ngpu 1 -iter 20000 --checkpoint ./checkpoint/gs_checkpoint/mnist/eps_1000 -noise 0.175935
#CUDA_VISIBLE_DEVICES=2 python main.py --dataset 'mnist' --load_dir ../pretrain_mnist/ResNet_default/ -ndis 200 -ngpu 1 -iter 20000 --checkpoint ./checkpoint/gs_checkpoint/mnist/eps_100 -noise 0.24162
#CUDA_VISIBLE_DEVICES=2 python main.py --dataset 'mnist' --load_dir ../pretrain_mnist/ResNet_default/ -ndis 200 -ngpu 1 -iter 20000 --checkpoint ./checkpoint/gs_checkpoint/mnist/eps_1 -noise 0.86
#CUDA_VISIBLE_DEVICES=2 python main.py --dataset 'mnist' --load_dir ../pretrain_mnist/ResNet_default/ -ndis 200 -ngpu 1 -iter 20000 --checkpoint ./checkpoint/gs_checkpoint/mnist/eps_inf -noise 0
CUDA_VISIBLE_DEVICES=2 python main.py --dataset 'mnist' --load_dir ../pretrain_mnist/ResNet_default/ -ndis 200 -ngpu 1 -iter 20000 --checkpoint ./checkpoint/gs_checkpoint/mnist/eps_10 -noise 0.427
