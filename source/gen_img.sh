#!/bin/sh

if [[ "$1" == "cifar" ]]
then
#	for i in 1 10 100 1000 inf
#	do
#		python img_gen.py --load_dir checkpoint/gs_checkpoint/cifar10/eps_${i}/diff_acc/10.pth --save_dir checkpoint/gs_data/cifar/eps_${i}/diff_acc
#	done
#	python img_gen.py --load_dir checkpoint/gs_checkpoint/cifar10/eps_inf/diff_acc/20.pth --save_dir checkpoint/gs_data/cifar/eps_inf/diff_acc

	for i in 1 10 100 1000 inf
	do
		for (( j=1000; i<=20000; j=j+1000 ))
		do
			python img_gen.py --z_dim 100 --latent_type bernoulli --load_dir ../results/cifar_10/main/main_eps_${i}_z100/netGS_${j}.pth --save_dir ../results/generated/cifar_10/eps_${i}/diff_iter
		done
	done
    
	for i in 01 02 001 
	do
		for (( j=1000; i<=20000; j=j+1000 ))
		do
			python img_gen.py --z_dim 100 --latent_type bernoulli --load_dir ../results/cifar_10/main/main_n_${i}_z100/netGS_${j}.pth --save_dir ../results/generated/cifar_10/n_${i}/diff_iter
		done
	done

else
#	for i in 1 10 100 1000 inf
#	do
#		for j in 10 20 30 40 50 60 70 80 90 
#		do
##			python img_gen.py --load_dir checkpoint/gs_checkpoint/mnist/eps_${i}/diff_acc/${j}.pth --save_dir checkpoint/gs_data/mnist/eps_${i}/diff_acc
#		done
#	done

	for i in 1 10 100 1000 inf
	do
		for (( j=1000; i<=20000; j=j+1000 ))
		do
			python img_gen.py --z_dim 32 --latent_type bernoulli --load_dir ../results/mnist/main/main_eps_${i}_z32/netGS_${j}.pth --save_dir ../results/generated/mnist/eps_${i}/diff_iter
		done
	done
fi

	for i in 01 02 001 
	do
		for (( j=1000; i<=20000; j=j+1000 ))
		do
			python img_gen.py --z_dim 32 --latent_type bernoulli --load_dir ../results/mnist/main/main_n_${i}_z100/netGS_${j}.pth --save_dir ../results/generated/mnist/n_${i}/diff_iter
		done
	done
