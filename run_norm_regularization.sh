#!/bin/sh

gpu_id=0
epochs=120
robust_test_size=1024

for model in 'vgg' 'resnet' 'senet' 'regnet'
do
	CUDA_VISIBLE_DEVICES=$gpu_id python train_nd.py --model=$model --epochs=$epochs --test-size=$robust_test_size
	echo "python train_nd.py --model=$model --epochs=$epochs --test-size=$robust_test_size" >> reg_done.log
	CUDA_VISIBLE_DEVICES=$gpu_id python train_nd_inf.py --model=$model --epochs=$epochs --test-size=$robust_test_size
	echo "python train_nd_inf.py --model=$model --epochs=$epochs --test-size=$robust_test_size" >> reg_done.log
	CUDA_VISIBLE_DEVICES=$gpu_id python train_svc.py --model=$model --epochs=$epochs --test-size=$robust_test_size
	echo "python train_svc.py --model=$model --epochs=$epochs --test-size=$robust_test_size" >> reg_done.log
	CUDA_VISIBLE_DEVICES=$gpu_id python train_wd.py --model=$model --epochs=$epochs --test-size=$robust_test_size
	echo "python train_wd.py --model=$model --epochs=$epochs --test-size=$robust_test_size" >> reg_done.log
	CUDA_VISIBLE_DEVICES=$gpu_id python train_nd_proj_bn.py --model=$model --epochs=$epochs --test-size=$robust_test_size
	echo "python train_nd_proj_bn.py --model=$model --epochs=$epochs --test-size=$robust_test_size" >> reg_done.log
	CUDA_VISIBLE_DEVICES=$gpu_id python train_nd_inf_proj_bn.py --model=$model --epochs=$epochs --test-size=$robust_test_size
	echo "python train_nd_inf_proj_bn.py --model=$model --epochs=$epochs --test-size=$robust_test_size" >> reg_done.log
done

python draw_lip_again.py
python summary_acc_latex.py > ./regularization/acc_table.txt # redirect output to a file
