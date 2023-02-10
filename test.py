import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, get_time, DiffAugment, ParamDiffAug, TensorDataset, epoch
import wandb
import argparse
import numpy as np
import time

def parse_args():
    parser = argparse.ArgumentParser(description='test the distilled images')

    parser.add_argument('--dataset', type=str, default='CIFAR100', help='dataset')
    parser.add_argument('--img_pth', type=str, default='logged_files/CIFAR100/comfy-universe-129/images_5000.pt')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--ipc', type=int, default=10)
    parser.add_argument('--lr_net', type=float, default=0.06)
    parser.add_argument('--num_classes', type=int, default=100)
    args = parser.parse_args()
    return args

def main(args):
    img = torch.load(args.img_pth)
    
    im_size = tuple(img.size()[-2:])
    num_classes = args.num_classes
    args.dsa_param = ParamDiffAug()
    args.dc_aug_param = None
    args.num_eval = 3
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.epoch_eval_train = 1500
    args.batch_train = 256
    args.im_size = im_size
    args.model = 'ConvNet'
    
    label = torch.tensor(np.array([np.ones(args.ipc,dtype=np.int_)*i for i in range(num_classes)]), dtype=torch.long).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    print(args.__dict__)
    
    wandb.init(sync_tensorboard=False,
            project="Distilled_Image_Test",
            job_type="CleanRepo",
            config=args
            )

    print('-------------------------\nEvaluation\nmodel_train = %s'%(args.model))
    if args.dsa:
        print('DSA augmentation strategy: \n', args.dsa_strategy)
        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
    else:
        print('DC augmentation parameters: \n', args.dc_aug_param)

    accs_test = []
    accs_train = []
    
    _, _, _, _, mean, std, _, _, testloader, _, _, _ = get_dataset(args.dataset, args.data_path, batch_size=args.batch_train, args=args)
    
    for it_eval in range(args.num_eval):
        net_eval = get_network('ConvNet', 3, num_classes, im_size).to(args.device) # get a random model

        label_syn_eval = label
        with torch.no_grad():
            image_syn_eval = img

        _, acc_train, acc_test, max_acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
        print(acc_test, max_acc_test)
        accs_test.append(acc_test)
        accs_train.append(acc_train)
        
        
    accs_test = np.array(accs_test)
    accs_train = np.array(accs_train)
    acc_test_mean = np.mean(accs_test)
    acc_test_std = np.std(accs_test)



def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, return_loss=False, texture=False):
    net = net.to(args.device)
    images_train = images_train
    labels_train = labels_train
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//3+1, Epoch*2//3+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.001)

    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=4)

    start = time.time()
    acc_train_list = []
    loss_train_list = []
    
    max_acc_test = 0.0

    for ep in tqdm(range(Epoch+1)):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=True, texture=texture)
        acc_train_list.append(acc_train)
        loss_train_list.append(loss_train)
        if 0 == ep%20:
            with torch.no_grad():
                loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
                wandb.log({'test_acc':acc_test}, step=ep)
                print(acc_test)
            if acc_test>max_acc_test:
                max_acc_test = acc_test
        if ep in lr_schedule:
            lr *= 0.5
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.8, weight_decay=0.001)


    time_train = time.time() - start

    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f, max test acc = %.4f, test loss = %.6f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test, max_acc_test, loss_test))

    if return_loss:
        return net, acc_train_list, acc_test, loss_train_list, loss_test
    else:
        return net, acc_train_list, acc_test, max_acc_test


if __name__=="__main__":
    args = parse_args()
    main(args)



