#!/bin/sh

if [[ "$1" == "cifar" ]]
then
	for i in 1 10 100 1000 inf
	do
		python img_gen.py --load_dir checkpoint/gs_checkpoint/cifar10/eps_${i}/diff_acc/10.pth --save_dir checkpoint/gs_data/cifar/eps_${i}/diff_acc
	done
	python img_gen.py --load_dir checkpoint/gs_checkpoint/cifar10/eps_inf/diff_acc/20.pth --save_dir checkpoint/gs_data/cifar/eps_inf/diff_acc

	for i in 1 10 100 1000 inf
	do
		for j in 1000 5000 10000 20000
		do
			python img_gen.py --load_dir checkpoint/gs_checkpoint/cifar10/eps_${i}/diff_iter/${j}.pth --save_dir checkpoint/gs_data/cifar/eps_${i}/diff_iter
		done
	done

else
	for i in 1 10 100 1000 inf
	do
		for j in 10 20 30 40 50 60 70 80 90 
		do
			python img_gen.py --load_dir checkpoint/gs_checkpoint/mnist/eps_${i}/diff_acc/${j}.pth --save_dir checkpoint/gs_data/mnist/eps_${i}/diff_acc
		done
	done

	for i in 1 10 100 1000 inf
	do
		for j in 1000 5000 10000 20000
		do
			python img_gen.py --load_dir checkpoint/gs_checkpoint/mnist/eps_${i}/diff_iter/${j}.pth --save_dir checkpoint/gs_data/mnist/eps_${i}/diff_iter
		done
	done
fi
