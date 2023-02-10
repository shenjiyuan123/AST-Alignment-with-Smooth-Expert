#!/bin/bash
#SBATCH -J mtt_cifar10
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --partition=RTXA6Kq
#SBATCH -w node10
#SBATCH --gres=gpu:2
#SBATCH --output /export/home2/jiyuan/mtt-distillation/outs/test.out


python test.py --dataset CIFAR100 --img_pth logged_files/CIFAR100/auspicious-ox-263/images_5000.pt --ipc 10 --lr_net 0.04 --num_classes 100
