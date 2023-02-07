#!/bin/bash
#SBATCH -J mtt_cifar10
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --partition=PA40q
#SBATCH -w node06
#SBATCH --gres=gpu:1
#SBATCH --output /export/home2/jiyuan/mtt-distillation/outs/distill_flat1_test2.out


# distill 1 ipc
# python distill.py --Iteration 5000 --epoch_eval_train 1000 --num_eval 1 --lr_img=10000 --lr_teacher 0.01 --lr_lr=1e-05 --dataset=CIFAR100 --ipc=1 --syn_steps=40 --expert_epochs=3 --max_start_epoch=20 --buffer_path='buffer_flat1' --data_path='data'

# distill 10 ipc
# python distill.py --Iteration 5000 --epoch_eval_train 800 --num_eval 1 --batch_syn 800 --balance_loss  --lr_img=10000 --lr_teacher 0.01 --lr_lr=1e-05 --dataset=CIFAR100 --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=30 --buffer_path='buffer_flat1' --data_path='data'

python distill.py --Iteration 5000 --epoch_eval_train 800 --num_eval 1 --zca --balance_loss --lr_img=100 --lr_teacher 0.01 --lr_lr=1e-06 --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --buffer_path='buffer_cifa10_flat' --data_path='data'

# distill 50 ipc
# python distill.py --Iteration 5000 --epoch_eval_train 500 --num_eval 1 --lr_img=10000 --lr_teacher 0.01 --lr_lr=1e-05 --dataset=CIFAR100 --ipc=50 --batch_syn 400  --syn_steps=20 --expert_epochs=2 --max_start_epoch=35 --buffer_path='buffer_flat1' --data_path='data'