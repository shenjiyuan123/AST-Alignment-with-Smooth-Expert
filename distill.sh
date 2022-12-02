#!/bin/bash
#SBATCH -J mtt_cifar10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --partition=PA40q
#SBATCH -w node05
#SBATCH --gres=gpu:2
#SBATCH --output /export/home2/jiyuan/mtt-distillation/outs/distill.out





python distill.py --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --zca --lr_img=1000 --lr_lr=1e-04 --lr_teacher=0.01 \
                  --buffer_path='/export/home2/jiyuan/mtt-distillation/buffer' --data_path='/export/home2/jiyuan/mtt-distillation/data'