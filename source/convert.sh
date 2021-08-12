for name in mnist cifar10
do
	for i in 1 10 100 1000 inf
	do
		for ((j=20; j <= 90; j+=10))
		do
			python convert.py --checkpoint checkpoint/gs_checkpoint/${name}/eps_${i}/diff_acc/${j}.pth
		done
		# for j in 1000 5000 10000 20000
		# do
		# 	python convert.py --checkpoint checkpoint/gs_checkpoint/${name}/eps_${i}/diff_iter/${j}.pth
		# done
	done
done