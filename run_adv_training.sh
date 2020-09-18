#!/bin/sh

gpu_id=0
epochs=120
robust_test_size=1024

for model in 'vgg' 'resnet' 'senet' 'regnet'
do
	CUDA_VISIBLE_DEVICES=$gpu_id python train_plain_cifar10.py --model=$model --epochs=$epochs --test-size=$robust_test_size
	echo "python train_plain_cifar10.py --model=$model --epochs=$epochs --test-size=$robust_test_size" >> adv_done.log
	CUDA_VISIBLE_DEVICES=$gpu_id python train_adv_cifar10.py --model=$model --epochs=$epochs --test-size=$robust_test_size
	echo "python train_adv_cifar10.py --model=$model --epochs=$epochs --test-size=$robust_test_size" >> adv_done.log
done

python compare_norm_individual.py
python compare_norm_group_density.py
