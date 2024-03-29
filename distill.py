import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
from generator import Generator
from reparam_module import ReparamModule

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", args.device)

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    wandb.init(sync_tensorboard=False,
               project="DatasetDistillation",
               job_type="CleanRepo",
               config=args,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    def get_images(c, ipc):  # get n random images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:ipc]
        return images_all[idx_shuffle]
    
    def get_repre_images(ipc):  # get n representative images each class
        ipc_visual = 10
        expert_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)
        expert_files = []
        expert_dir = os.path.join(args.buffer_path, args.dataset)
        if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
            expert_dir += "_NO_ZCA"
        expert_dir = os.path.join(expert_dir, args.model)
        path_num = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(path_num))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(path_num)))
            path_num += 1
        if path_num == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

        random.shuffle(expert_files)
        buffer = torch.load(expert_files[0])
        random.shuffle(buffer)
        expert_para = buffer[0][-1]  # get the 50th epoch's parameters
        
        expert_para_odict = {}
        for i,(k,v) in enumerate(expert_net.state_dict().items()):
            if expert_para[i].shape==v.shape:
                expert_para_odict[k] = expert_para[i]
            else:
                print(f'Error{i}!')
        expert_net.load_state_dict(expert_para_odict)
        
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        
        def distance(x,y):
            return np.sum((x-y)**2, axis=1)
        
        repre_indices = []
        fig, ax = plt.subplots(1, 5, figsize=(30,6))
        visual = True if ipc_visual<=10 else False 
        
        print("indices_class ",len(indices_class))
        
        for c in range(len(indices_class)):
            images_per_class = images_all[indices_class[c]].to(args.device)
            logist_per_class = expert_net(images_per_class)
            logist = logist_per_class.cpu().data.numpy()
            kmeans = KMeans(n_clusters=ipc_visual, random_state=0).fit(logist)
            centers = kmeans.cluster_centers_
            pseudo_labels = kmeans.labels_
            
            kmeans_ipc = KMeans(n_clusters=ipc, random_state=0).fit(logist)
            centers_ipc = kmeans_ipc.cluster_centers_
            
            for center in centers:
                # choose args.ipc/ipc_visual points as the representative initialization samples
                dis = distance(center, logist)
                # min_dis_indice = np.argmin(dis)
                min_dis_indice = np.argsort(dis)[:int(ipc/ipc_visual)].tolist()
                for tmp in min_dis_indice:
                    repre_indices.append(indices_class[c][tmp])
                    
            # visualization, only show first 5 classes
            if c<5 and visual:
                pca = PCA(n_components=2).fit(logist)
                logist_draw = pca.transform(logist)
                center_draw = pca.transform(centers)
                label_color = {0:'#0066CC', 1:'#00CCCC', 2:'#3399FF', 3:'#66FFFF', 4:'#CCFFFF', 5:'#CCE5FF', 6:'#66B2FF', 7:'#0080FF', 8:'#9999FF', 9:'#CCFFE5'}
                random_init = random.sample(list(range(logist_draw.shape[0])), ipc)
                for i, draw in enumerate(logist_draw):  
                    if i in random_init:
                        # ax[c].scatter(draw[0], draw[1], c='#202020', marker='x', s=20, zorder=2)
                        continue
                    else:
                        ax[c].scatter(draw[0], draw[1], c=label_color[pseudo_labels[i]], s=8,zorder=1)
                for draw in center_draw:
                    ax[c].scatter(draw[0], draw[1], c='r', marker='*', s=30, zorder=3)
                ax[c].set_xlabel(f'Class {c}.')

                
        if visual:
            fig.savefig('represent_new.pdf',bbox_inches='tight',pad_inches=0)
        print(centers.shape, len(repre_indices))
        return repre_indices
    

    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.ipc,dtype=np.int_)*i for i in range(num_classes)], dtype=torch.long, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    

    if args.texture:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
    else:
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        if args.texture:
            for c in range(num_classes):
                for i in range(args.canvas_size):
                    for j in range(args.canvas_size):
                        image_syn.data[c * args.ipc:(c + 1) * args.ipc, :, i * im_size[0]:(i + 1) * im_size[0],
                        j * im_size[1]:(j + 1) * im_size[1]] = torch.cat(
                            [get_images(c, 1).detach().data for s in range(args.ipc)])
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
    elif args.pix_init == 'avg':
        print('initialize synthetic data from the average of all real images')
        image_syn.data = images_all[get_repre_images(args.ipc)].detach().data
    elif args.pix_init == 'gan':
        Gen = Generator()
        dim = 100
        img_base = torch.randn(size=(num_classes, dim), dtype=torch.float, requires_grad=True)      # 10*100
        image_syn = Gen(img_base)       # 10*3*32*32
    else:
        print('initialize synthetic data from random noise')


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    
    cosine_similarity = nn.CosineSimilarity(dim=1)
    
    optimizer_img.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    kl_loss = nn.KLDivLoss(reduction="sum")
    print('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))
    
    def weightPerturb(network, alpha=1.0):
        ''' perturb the initial weight using Gaussian distribution '''
        # input: one epoch's training result
        weight = []
        for i in network:
            if len(i.shape)==4:
                d_ = torch.normal(mean=0,std=1,size=i.size())
                # Drop = nn.Dropout(p=0.0)
                # d_ = Drop(d)
                d_F = torch.norm(d_)
                d_ /= d_F
                weight_perturb = i + alpha*d_*i
                weight.append(weight_perturb)
            else:
                weight.append(i)
        return weight
    
    def weightDropout(network, p=0.1):
        ''' randomly drop part of the network weight '''
        # input: one epoch's training result
        weight = []
        layer_numbers = len(network)
        drop_layer = random.randint(0, layer_numbers-1)
        for i, layer in enumerate(network):
            out = layer
            if i == drop_layer:
                Drop = nn.Dropout(p)
                out = Drop(layer)
            weight.append(out)
        return weight

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}
    
    former = 0

    for it in range(0, args.Iteration+1):
        save_this_it = False
            
        # writer.add_scalar('Progress', it, it)
        wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            # XXX: for debug
            # continue
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs_test = []
                accs_train = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

                    eval_labs = label_syn
                    with torch.no_grad():
                        image_save = image_syn
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                    args.lr_net = syn_lr.item()
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)


        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            # XXX: for debug
            # continue
            with torch.no_grad():
                image_save = image_syn.to(args.device)

                save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))

                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

                if args.ipc <= 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)

        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        def hierarchical_sample_expert(former, interval=10):
            start_epoch = np.random.randint(0, args.max_start_epoch)
            if np.abs(start_epoch-former) > interval:
                small, large = max(former - interval, 0), min(former + interval, args.max_start_epoch)
                start_epoch = np.random.randint(small, large)
            return start_epoch
        
        # start_epoch = hierarchical_sample_expert(former)
        start_epoch = np.random.randint(0, args.max_start_epoch)
        # start_epoch = 0
        former = start_epoch
        starting_params = expert_trajectory[start_epoch]
        if args.weight_perturb:
            starting_params = weightPerturb(starting_params, alpha=1)

        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        syn_images = image_syn

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []
        
        ce_loss_total = 0.0
        
        student_to_target = {}
        match_pool = []
        middle_params = []
        middle_loss = []
        for i in range(args.expert_epochs-1):
            middle = (args.syn_steps//args.expert_epochs)*(i+1)
            student_to_target[i+1] = middle
            middle_params.append(torch.cat([p.data.to(args.device).reshape(-1) for p in expert_trajectory[start_epoch+(i+1)]], 0))
            match_pool.append(middle)
        student_to_target[args.expert_epochs] = args.syn_steps
        match_pool.append(args.syn_steps)

        for step in range(args.syn_steps):
            # XXX: for debug
            # if step>2:
            #     break
            # if use batch_syn, will random sample a batch of syn. else, sample all.
            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()


            x = syn_images[these_indices]
            this_y = label_syn[these_indices]

            if args.texture:
                x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
                
            # train student net
            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            # adaptive ce_loss to balance loss between small and large start epoch
            if args.balance_loss:
                if start_epoch >= args.max_start_epoch//2:
                    ce_loss *= math.log(start_epoch-args.max_start_epoch//2+8, 5)
                else:
                    ce_loss /= math.log(args.max_start_epoch//2-start_epoch+8, 5)

            # create computation graph so that when compute the distance loss, can have the higher order derivative products
            if args.ignore_graph:
                if step >= args.syn_steps-8:
                    grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
                else:
                    grad = torch.autograd.grad(ce_loss, student_params[-1])[0]
            else:
                grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

            # optimize the student net weights, add momentum
            if step == 0:
                b_t1 = grad
                g_t = b_t1
            else:
                b_t = args.mom * b_t1 + grad
                # no use nesterov
                g_t  = b_t
                b_t1 = b_t
            student_params.append(student_params[-1] - syn_lr * g_t)
            
            ce_loss_total += ce_loss
            
            if step+1 in match_pool and step<args.syn_steps-1 and (args.agg_middle_loss or args.adaptive_middle_loss):
                middle_target_params = middle_params.pop(0)
                param_middle_loss = torch.nn.functional.mse_loss(student_params[-1], middle_target_params, reduction="sum")
                param_middle_dist = torch.nn.functional.mse_loss(starting_params, middle_target_params, reduction="sum")
                param_middle_loss /= (param_middle_dist)
                middle_loss.append(param_middle_loss)
                
        ce_loss_total /= args.syn_steps
        wandb.log({"celoss/Inner_loss":ce_loss_total}, step=it)

        '''
        # alignment loss
        if args.align_loss:
            forward_params = student_params[-1].clone().detach()
            expert_params  = target_params
            
            sum_align_loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                # get images of each label
                batch_syn = syn_images[c*args.ipc:(c+1)*args.ipc]
                feas_syn = student_net.module.feature_forward(batch_syn, flat_param=forward_params)
                
                batch_img = get_images(c, args.align_bs).detach().to(args.device)
                batch_label = torch.ones(args.align_bs, dtype=torch.long)*c
                feas_real = student_net.module.feature_forward(batch_syn, flat_param=expert_params)
                
                # align the feature
                for layer in range(len(feas_syn)):
                    out = torch.mean(feas_syn[layer], dim=0)
                    target = torch.mean(feas_real[layer], dim=0)
                    out = F.log_softmax(out, dim=1)
                    target = F.softmax(target, dim=1)
                    loss = kl_loss(out, target)
                    # loss = torch.nn.functional.mse_loss(out, target)
                    sum_align_loss += loss
            
            # print(sum_align_loss)
            # break
            wandb.log({"align_loss":sum_align_loss}, step=it)
        '''
        
        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")


        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)

        # param_loss /= num_params
        # param_dist /= num_params
        
        wandb.log({"param_loss":param_loss, "param_dist":param_dist}, step=it)

        param_loss /= (param_dist)
        
        grand_loss = param_loss
        if args.adaptive_middle_loss:
            for num, tmp in enumerate(middle_loss):
                beta = (1/(1+len(middle_loss)))*(num+1)
                grand_loss += beta* tmp
        elif args.agg_middle_loss:
            for num, tmp in enumerate(middle_loss):
                beta = 1
                grand_loss += beta* tmp

        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()
        
        wandb.log({"Syn_Im_Gradient":torch.linalg.norm(syn_images.grad)},step=it)

        optimizer_img.step()
        optimizer_lr.step()


        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})

        # if it == args.Iteration//2:
        #     optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img*0.5, momentum=0.5)
        
        if args.align_loss:
            forward_params = student_params[-1].clone().detach()
            expert_params  = target_params
            sum_align_loss = 0.0
            
            for idx in range(args.ipc):
                batch_syn = syn_images[idx::args.ipc]
                print(len(batch_syn))
                feas_syn = student_net.module(batch_syn, flat_param=forward_params)
                feas_real = student_net.module(batch_syn, flat_param=expert_params)
                
                out = F.log_softmax(feas_syn, dim=1)
                target = F.softmax(feas_real, dim=1)
                loss = kl_loss(out, target)
                print(loss)
                sum_align_loss += loss
                
                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
            
            wandb.log({"align_loss":sum_align_loss}, step=it)

        for _ in student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--align_loss', action='store_true', help='whether use alignment loss')
    parser.add_argument('--align_alpha', type=float, default=0.1, help='the hyper-parameters for both loss')
    parser.add_argument('--align_bs', type=int, default=500, help='real images per class for the alignment loss, e.g. each class has 500 images for cifa100')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=1, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=200, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for update the inner neural network')
    parser.add_argument('--balance_loss', action='store_true', help="do balance celoss")
    parser.add_argument('--agg_middle_loss', action='store_true', help="whether match the student with the expert in the middle of syn_step")
    parser.add_argument('--adaptive_middle_loss', action='store_true', help="adaptive match the student with the expert in the middle of syn_step")
    parser.add_argument('--ignore_graph', action='store_true', help="ignore some parts of the computation graph")
    parser.add_argument('--weight_perturb', action='store_true', help="perturb the starting model parameters")

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real", "avg", "gan"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffer', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    args = parser.parse_args()

    main(args)


