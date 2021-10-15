CUDA_VISIBLE_DEVICES=0 python main_celeba.py --dataset 'celeba' -bs 256 -ndis 1 -ngpu 1 -zdim 100 --exp_name d_1_2e-4_g_1_2e-4_SN_40k -noise 0 --save_step 5000 --update-train-dataset1
