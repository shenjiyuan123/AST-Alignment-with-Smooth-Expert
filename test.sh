#!/bin/bash
#SBATCH -J mtt_cifar10
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --partition=RTXq
#SBATCH -w node31
#SBATCH --gres=gpu:2
#SBATCH --output /export/home2/jiyuan/mtt-distillation/outs/stilted-pond-203.out


python test.py --img_pth logged_files/CIFAR100/stilted-pond-203/images_5000.pt
