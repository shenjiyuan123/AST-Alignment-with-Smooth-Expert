#!/bin/bash
#SBATCH -J mtt_cifar10
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --partition=RTXq
#SBATCH -w node30
#SBATCH --output /export/home2/jiyuan/mtt-distillation/outs/distill_cifa100_draw_rep.out


# distill 1 ipc
# CUDA_VISIBLE_DEVICES=1 python distill.py --Iteration 5000 --epoch_eval_train 300 --num_eval 1 --pix_init avg --weight_perturb --agg_middle_loss --lr_img=100 --lr_teacher 0.01 --lr_lr=1e-05 --dataset=CIFAR100 --ipc=1 --syn_steps=40 --expert_epochs=3 --max_start_epoch=15 --buffer_path='buffer_flat2' --data_path='data'

# distill 10 ipc
CUDA_VISIBLE_DEVICES=1 python distill.py --Iteration 5000 --epoch_eval_train 300 --num_eval 1 --pix_init avg --weight_perturb --agg_middle_loss --lr_img=1000 --lr_teacher 0.01 --lr_lr=1e-05 --dataset=CIFAR100 --ipc=10 --batch_syn 128 --syn_steps=20 --expert_epochs=2 --max_start_epoch=30 --buffer_path='buffer_flat2' --data_path='data'

# python distill.py --Iteration 5000 --epoch_eval_train 1000 --num_eval 1 --weight_perturb --zca --balance_loss --agg_middle_loss --lr_img=100 --lr_teacher 0.01 --lr_lr=1e-06 --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --buffer_path='buffer_cifa10_flat' --data_path='data'

# python distill.py --Iteration 5000 --epoch_eval_train 1000 --num_eval 1 --weight_perturb --zca --balance_loss --agg_middle_loss --lr_img=1000 --lr_teacher 0.01 --lr_lr=1e-05 --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --buffer_path='buffer' --data_path='data'

# distill 50 ipc
# CUDA_VISIBLE_DEVICES=1,7 python distill.py --Iteration 5000 --epoch_eval_train 1000 --num_eval 1 --pix_init avg --lr_img=1000 --lr_teacher 0.01 --lr_lr=1e-05 --dataset=CIFAR100 --ipc=50 --batch_syn 700  --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --buffer_path='buffer_flat1' --data_path='data'


# CUDA_VISIBLE_DEVICES=1 python distill.py --Iteration 5000 --epoch_eval_train 1000 --num_eval 1 --weight_perturb --agg_middle_loss --lr_img=10000 --lr_teacher 0.01 --lr_lr=1e-04 --dataset=Tiny --ipc=1 --model ConvNetD4 --syn_steps=10 --expert_epochs=2 --max_start_epoch=10 --buffer_path='buffer' --data_path='data'
