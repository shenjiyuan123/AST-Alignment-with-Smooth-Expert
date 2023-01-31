#!/bin/bash
#SBATCH -J mtt_cifar10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=RTXq
#SBATCH -w node31
#SBATCH --gres=gpu:2
#SBATCH --output /export/home2/jiyuan/mtt-distillation/outs/buffernewsamlllr.out




python buffer.py --dataset=CIFAR100 --model=ConvNet --train_epochs=50 --num_experts=50 --buffer_path='/export/home2/jiyuan/mtt-distillation/buffer_flat2' --data_path='/export/home2/jiyuan/mtt-distillation/data'