#!/bin/bash
#SBATCH -J mtt_cifar10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PA100q
#SBATCH -w node03
#SBATCH --gres=gpu:1
#SBATCH --output /export/home2/jiyuan/mtt-distillation/outs/buffer2.out




python buffer.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=100 --zca --buffer_path='/export/home2/jiyuan/mtt-distillation/buffer' --data_path='/export/home2/jiyuan/mtt-distillation/data'