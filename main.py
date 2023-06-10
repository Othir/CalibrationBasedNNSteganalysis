'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from models import *
# import ipdb
import random
import math
import torch.multiprocessing as mp

from attack import corr_loss, sign_loss,rbg_to_grayscale, corr_loss_full, sign_loss_full
from load_cifar import load_cifar

COR = 'cor'  # Correlation value encoding attack
SGN = 'sgn'  # Sign encoding attack
NO = 'no'

from LSB import LSB, LSB_random
from finetune import finetune


def main(network='vgg16', lr=0.1, attack=NO, cor_ratio=0.0, ratio=1.0, m=0, add_LSB=0, with_finetune=1, m_step=4, flr=0.1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data Cifar10
    print('==> Preparing data..')

    data_path = './data/Cifar10'
    output_path = './output'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        # root='./data', train=True, download=True, transform=transform_train)
        root=data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        # root='./data', train=False, download=True, transform=transform_test)
        root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)


    # for ra in range(16):
    for ra in range(m_step):
        lr = 0.1
        best_acc = 0  # best test accuracy
        start_epoch = 1  # start from epoch 0 or last checkpoint epoch
        # net = EfficientNetB0()
        netname='other'
        # Model
        print('==> Building model..')
        if network == 'vgg16':
            net = VGG('VGG16')
            netname = 'vgg'
        if network == 'resnet34':
            net = ResNet34()
            netname='other'
            
        # netname='vgg'

        # net = ResNet34()
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = ShuffleNetV2(1)
        # net = EfficientNetB0()
        # net = RegNetX_200MF()
        net = net.to(device) # 很慢
        print('network:{} attack:{} regular:{} ratio:{}'.format(network, attack, cor_ratio, ratio))
        # print(netname)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr,
                              momentum=0.9, weight_decay=1e-4)

        if ratio == 1.0:
            length = 0
            if netname == 'vgg':
                i = 1
                for name, params in net.named_parameters():
                    if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                        if i == 1:
                            p = params.flatten()
                            length += len(p)
                        i = 1 - i
            elif netname == 'other':
                for name, params in net.named_parameters():
                    if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                        p = params.flatten()
                        length += len(p)
        if ratio != 1.0:
            length = 0
            if netname == 'vgg':
                i = 1
                for name, params in net.named_parameters():
                    if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                        if i == 1:
                            p = params.flatten()
                            length += len(p)
                        i = 1 - i
            elif netname == 'other':
                for name, params in net.named_parameters():
                    if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                        p = params.flatten()
                        length += len(p)           

            # order
            # begin = 0
            # ad = set()
            # while(len(ad)<int(ratio * length)):
            #     ad.add(random.randint(begin, begin + int(ratio * length)))

            # random begin
            # begin = int(np.random.randint(0, int(length * (1 - ratio)), 1))
            # ad = set()
            # while(len(ad)<int(ratio * length)):
            #     ad.add(random.randint(begin, begin + int(ratio * length)))

            #random
            random_seed = random.getstate()
            ad = set()
            while(len(ad)<int(ratio * length)):
                ad.add(random.randint(0, length-1))
            ad = list(ad)
            # ad.sort()
            ad = np.array(ad)
            print(len(ad))

        if attack == COR:
            # get the gray-scaled data to be encoded
            np_seed = np.random.get_state()
            raw_data = load_cifar(10)
            raw_data = rbg_to_grayscale(raw_data).astype(np.uint8)
            raw_data = raw_data.flatten()
            print(raw_data.shape)
            raw_data = raw_data[:int(ratio * length)]
            np.random.shuffle(raw_data)
            raw_data = torch.from_numpy(raw_data)
            raw_data = raw_data.float().to(device)

        # if attack == COR:
        #     np_seed = np.random.get_state()
        #     raw_data = np.random.randint(0,2,int(ratio * length))
        #     raw_data = torch.from_numpy(raw_data)
        #     raw_data = raw_data.float().to(device)
        if attack == SGN:
            np_seed = np.random.get_state()
            secret = np.random.randint(0,2,int(ratio * length))
            secret = torch.from_numpy(secret)
            secret = secret.float().to(device)


        scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

        for epoch in range(start_epoch, start_epoch+100):
        # for epoch in range(start_epoch, start_epoch+1):
            lr = scheduler.get_lr()[0]
            print('\nlearning rate: %f' % lr)
            #Train
            print('\nEpoch: %d' % epoch)
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = net(inputs) # RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`

                loss = criterion(outputs, targets)

                if attack == COR:
                    if ratio == 1.0:
                        cor_loss = corr_loss_full(net,raw_data,device,netname)
                    else:
                        cor_loss = corr_loss(net,raw_data,ad,device,netname)
                    cor_loss *= cor_ratio
                    loss += cor_loss
                elif attack == SGN:
                    if ratio == 1.0:
                        sgn_loss = sign_loss_full(net,secret,length,device,netname)
                    else:
                        sgn_loss = sign_loss(net,secret,length,ad,device,netname)
                    sgn_loss *= cor_ratio
                    loss += sgn_loss

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

            scheduler.step()

            #Test
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # Save checkpoint.
            acc = 100.*correct/total
            if acc > best_acc:

                model_path = os.path.join(output_path, 'cifar10_{}_attack_{}_regular_{}_ratio_{}_{}.pth'.format(network, attack, cor_ratio, ratio, ra+m))
                os.makedirs(os.path.dirname(model_path), exist_ok=True)

                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    'serect_seed':{'np':np_seed, 'random':random_seed}
                }
                torch.save(state, model_path)
                # torch.save(state, './checkpoint/Cifar10/VGG16/cor_new_01/VGG16_{}_{}.pth'.format(attack, ra+50))
                best_acc = acc

        if ratio==1.0:
            if attack == COR:
                raw_data = raw_data.cpu()
                raw_data = raw_data.detach().numpy()
                length1 = 0
                length2 = 0
                r = 0
                if netname == 'vgg':
                    i = 1
                    for name, params in net.named_parameters():
                        if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                            if i == 1:
                                p = params.flatten()
                                p = p.cpu()
                                p = p.detach().numpy()
                                pp = p
                                p_min = min(p)
                                p_max = max(p)
                                length2 += len(p)

                                t = raw_data[length1:length2]
                                r_min = min(t)
                                r_max = max(t)

                                pp1 = r_min + ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                                pp2 = r_max - ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)

                                r1 = np.sum(abs(pp1 - t))
                                r2 = np.sum(abs(pp2 - t))
                                r += min(r1,r2)
                                length1 = length2
                            i = 1 - i
                    r = r/length
                    print(r)  
                elif netname == 'other':
                    for name, params in net.named_parameters():
                        if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                            p = params.flatten()
                            p = p.cpu()
                            p = p.detach().numpy()
                            pp = p
                            p_min = min(p)
                            p_max = max(p)
                            length2 += len(p)

                            t = raw_data[length1:length2]
                            r_min = min(t)
                            r_max = max(t)

                            pp1 = r_min + ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                            pp2 = r_max - ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)

                            r1 = np.sum(abs(pp1 - t))
                            r2 = np.sum(abs(pp2 - t))
                            r += min(r1,r2)
                            length1 = length2
                    r = r/length
                    print(r)  
            if attack == SGN: 
                secret = secret.cpu()
                secret = secret.detach().numpy()
                secret[secret == 0.] = -1

                length1 = 0
                inject_sum = 0

                if netname == 'vgg16':
                    i = 1
                    for name, params in net.named_parameters():
                        if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                            if i == 1:
                                p = params.flatten()
                                p = p.cpu()
                                p = p.detach().numpy()             
                                t = secret[length1:length1+len(p)]


                                t1 = t * p
                                t1[t1 >= 0] = 1
                                t1[t1 < 0] = 0

                                inject_sum += np.sum(t1)

                                length1 += len(p)
                            i = 1 - i
                    print(inject_sum, length)
                elif netname == 'other':
                    for name, params in net.named_parameters():
                        if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                            p = params.flatten()
                            p = p.cpu()
                            p = p.detach().numpy()             
                            t = secret[length1:length1+len(p)]


                            t1 = t * p
                            t1[t1 >= 0] = 1
                            t1[t1 < 0] = 0

                            inject_sum += np.sum(t1)

                            length1 += len(p)
                    print(inject_sum, length)
        else:
            if attack == COR:
                p=torch.from_numpy(np.array([])).float().to(device)
                raw_data = raw_data.cpu()
                raw_data = raw_data.detach().numpy()
                r = 0
                if netname == 'vgg':
                    i = 1
                    for name, params in net.named_parameters():
                        if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                            if i == 1:
                                # if layer not in choices:
                                p=torch.cat([p,params.flatten()])
                            i = 1 - i
                                    # sum_length += len(params.flatten()) 
                elif netname == 'other':
                    for name, params in net.named_parameters():
                        if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                            # if layer not in choices:
                            p=torch.cat([p,params.flatten()])
                p = p.cpu()
                p = p.detach().numpy()
                pp = p[ad]
                t = raw_data[:len(ad)]
                
                p_min = min(pp)
                p_max = max(pp)
                # length2 += len(p)

                # t = raw_data[length1:length2]
                r_min = min(t)
                r_max = max(t)

                # p和pp
                # pp1 = r_min + ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                # pp2 = r_max - ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                pp1 = r_min + ((pp - np.ones(len(pp)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                pp2 = r_max - ((pp - np.ones(len(pp)) * p_min) * (r_max - r_min)) / (p_max - p_min)

                r1 = np.sum(abs(pp1 - t))
                r2 = np.sum(abs(pp2 - t))
                r += min(r1,r2)
                        # length1 = length2
                # r = r/length
                r = r/len(ad)
                print(r)
            if attack == SGN:
                secret = secret.cpu()
                secret = secret.detach().numpy()
                secret[secret == 0.] = -1
            
                inject_sum = 0
                ad = list(ad)

                t=secret[:len(ad)]

                p=torch.from_numpy(np.array([])).float().to(device)
                if netname == 'vgg':
                    i = 1
                    for name, params in net.named_parameters():
                        if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                            if i == 1:
                                p = torch.cat([p,params.flatten()])
                            i = 1 - i
                elif netname == 'other':
                    for name, params in net.named_parameters():
                        if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                            p = torch.cat([p,params.flatten()]) 
                p = p.cpu()
                p = p.detach().numpy()
                pp = p[ad]

                t1 = t * pp
                t1[t1 >= 0] = 1
                t1[t1 < 0] = 0

                inject_sum = np.sum(t1)
                print(inject_sum, len(ad))
        

        if bool(add_LSB):
            save_path = os.path.join(output_path, 'add20LSB_cifar10_{}_attack_{}_regular_{}_ratio_{}_{}.pth'.format(network, attack, cor_ratio, ratio, ra+m))
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            LSB(network=network, model_path=model_path, save_path=save_path)

            print('==> Building LSB model..')
            state_dict = torch.load(save_path, map_location=device)['net']
            net.load_state_dict(state_dict, strict=True)

            if ratio==1.0:
                if attack == COR:
                    # raw_data = raw_data.cpu()
                    # raw_data = raw_data.detach().numpy()
                    length1 = 0
                    length2 = 0
                    add_LSB_r = 0
                    if netname == 'vgg':
                        i = 1
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                if i == 1:
                                    p = params.flatten()
                                    p = p.cpu()
                                    p = p.detach().numpy()
                                    pp = p
                                    p_min = min(p)
                                    p_max = max(p)
                                    length2 += len(p)

                                    t = raw_data[length1:length2]
                                    r_min = min(t)
                                    r_max = max(t)

                                    pp1 = r_min + ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                                    pp2 = r_max - ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)

                                    r1 = np.sum(abs(pp1 - t))
                                    r2 = np.sum(abs(pp2 - t))
                                    add_LSB_r += min(r1,r2)
                                    length1 = length2
                                i = 1 - i
                        add_LSB_r = add_LSB_r/length
                        print(add_LSB_r)  
                    elif netname == 'other':
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                p = params.flatten()
                                p = p.cpu()
                                p = p.detach().numpy()
                                pp = p
                                p_min = min(p)
                                p_max = max(p)
                                length2 += len(p)

                                t = raw_data[length1:length2]
                                r_min = min(t)
                                r_max = max(t)

                                pp1 = r_min + ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                                pp2 = r_max - ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)

                                r1 = np.sum(abs(pp1 - t))
                                r2 = np.sum(abs(pp2 - t))
                                add_LSB_r += min(r1,r2)
                                length1 = length2
                        add_LSB_r = add_LSB_r/length
                        print(add_LSB_r)  
                if attack == SGN: 
                    # secret = secret.cpu()
                    # secret = secret.detach().numpy()
                    # secret[secret == 0.] = -1

                    length1 = 0
                    add_LSB_inject_sum = 0

                    if netname == 'vgg16':
                        i = 1
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                if i == 1:
                                    p = params.flatten()
                                    p = p.cpu()
                                    p = p.detach().numpy()             
                                    t = secret[length1:length1+len(p)]


                                    t1 = t * p
                                    t1[t1 >= 0] = 1
                                    t1[t1 < 0] = 0

                                    add_LSB_inject_sum += np.sum(t1)

                                    length1 += len(p)
                                i = 1 - i
                        print(add_LSB_inject_sum, length)
                    elif netname == 'other':
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                p = params.flatten()
                                p = p.cpu()
                                p = p.detach().numpy()             
                                t = secret[length1:length1+len(p)]


                                t1 = t * p
                                t1[t1 >= 0] = 1
                                t1[t1 < 0] = 0

                                add_LSB_inject_sum += np.sum(t1)

                                length1 += len(p)
                        print(add_LSB_inject_sum, length)
            else:
                if attack == COR:
                    # p=torch.cat([p,params.flatten()])
                    p=torch.from_numpy(np.array([])).float().to(device)
                    # raw_data = raw_data.cpu()
                    # raw_data = raw_data.detach().numpy()
                    add_LSB_r = 0
                    if netname == 'vgg':
                        i = 1
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                if i == 1:
                                    # if layer not in choices:
                                    p=torch.cat([p,params.flatten()])
                                i = 1 - i
                                        # sum_length += len(params.flatten()) 
                    elif netname == 'other':
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                # if layer not in choices:
                                p=torch.cat([p,params.flatten()])
                    p = p.cpu()
                    p = p.detach().numpy()
                    pp = p[ad]
                    t = raw_data[:len(ad)]
                    
                    p_min = min(pp)
                    p_max = max(pp)
                    # length2 += len(p)

                    # t = raw_data[length1:length2]
                    r_min = min(t)
                    r_max = max(t)

                    # p和pp
                    # pp1 = r_min + ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                    # pp2 = r_max - ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                    pp1 = r_min + ((pp - np.ones(len(pp)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                    pp2 = r_max - ((pp - np.ones(len(pp)) * p_min) * (r_max - r_min)) / (p_max - p_min)

                    r1 = np.sum(abs(pp1 - t))
                    r2 = np.sum(abs(pp2 - t))
                    add_LSB_r += min(r1,r2)
                            # length1 = length2
                    # add_LSB_r = add_LSB_r/length
                    add_LSB_r = add_LSB_r/len(ad)
                    print(add_LSB_r)
                if attack == SGN:
                    # secret = secret.cpu()
                    # secret = secret.detach().numpy()
                    # secret[secret == 0.] = -1
                
                    add_LSB_inject_sum = 0
                    ad = list(ad)

                    t=secret[:len(ad)]

                    p=torch.from_numpy(np.array([])).float().to(device)
                    if netname == 'vgg':
                        i = 1
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                if i == 1:
                                    p = torch.cat([p,params.flatten()])
                                i = 1 - i
                    elif netname == 'other':
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                p = torch.cat([p,params.flatten()]) 
                    p = p.cpu()
                    p = p.detach().numpy()
                    pp = p[ad]

                    t1 = t * pp
                    t1[t1 >= 0] = 1
                    t1[t1 < 0] = 0

                    add_LSB_inject_sum = np.sum(t1)
                    print(add_LSB_inject_sum, len(ad))
            
            #Test
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # Save checkpoint.
            LSB_acc = 100.*correct/total


        if with_finetune:
            for iter_num in [40, 30, 20, 10]:
                save_path = os.path.join(output_path, 'finetune_lr_{}_iter_{}_cifar10_{}_attack_{}_regular_{}_ratio_{}_{}.pth'.format(flr, iter_num, network, attack, cor_ratio, ratio, ra+m))
                finetune(lr=flr, network=network, iter_num=iter_num, model_path=model_path, finetune_model_path=save_path)

            print('==> Building finetune model..')
            save_path = os.path.join(output_path, 'finetune_lr_{}_iter_10_cifar10_{}_attack_{}_regular_{}_ratio_{}_{}.pth'.format(flr, network, attack, cor_ratio, ratio, ra+m))
            state_dict = torch.load(save_path, map_location=device)['net']
            net.load_state_dict(state_dict, strict=True)

            if ratio==1.0:
                if attack == COR:
                    # raw_data = raw_data.cpu()
                    # raw_data = raw_data.detach().numpy()
                    length1 = 0
                    length2 = 0
                    finetune_r = 0
                    if netname == 'vgg':
                        i = 1
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                if i == 1:
                                    p = params.flatten()
                                    p = p.cpu()
                                    p = p.detach().numpy()
                                    pp = p
                                    p_min = min(p)
                                    p_max = max(p)
                                    length2 += len(p)

                                    t = raw_data[length1:length2]
                                    r_min = min(t)
                                    r_max = max(t)

                                    pp1 = r_min + ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                                    pp2 = r_max - ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)

                                    r1 = np.sum(abs(pp1 - t))
                                    r2 = np.sum(abs(pp2 - t))
                                    finetune_r += min(r1,r2)
                                    length1 = length2
                                i = 1 - i
                        finetune_r = finetune_r/length
                        print(finetune_r)  
                    elif netname == 'other':
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                p = params.flatten()
                                p = p.cpu()
                                p = p.detach().numpy()
                                pp = p
                                p_min = min(p)
                                p_max = max(p)
                                length2 += len(p)

                                t = raw_data[length1:length2]
                                r_min = min(t)
                                r_max = max(t)

                                pp1 = r_min + ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                                pp2 = r_max - ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)

                                r1 = np.sum(abs(pp1 - t))
                                r2 = np.sum(abs(pp2 - t))
                                finetune_r += min(r1,r2)
                                length1 = length2
                        finetune_r = finetune_r/length
                        print(finetune_r)  
                if attack == SGN: 
                    # secret = secret.cpu()
                    # secret = secret.detach().numpy()
                    # secret[secret == 0.] = -1

                    length1 = 0
                    finetune_inject_sum = 0

                    if netname == 'vgg16':
                        i = 1
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                if i == 1:
                                    p = params.flatten()
                                    p = p.cpu()
                                    p = p.detach().numpy()             
                                    t = secret[length1:length1+len(p)]


                                    t1 = t * p
                                    t1[t1 >= 0] = 1
                                    t1[t1 < 0] = 0

                                    finetune_inject_sum += np.sum(t1)

                                    length1 += len(p)
                                i = 1 - i
                        print(finetune_inject_sum, length)
                    elif netname == 'other':
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                p = params.flatten()
                                p = p.cpu()
                                p = p.detach().numpy()             
                                t = secret[length1:length1+len(p)]


                                t1 = t * p
                                t1[t1 >= 0] = 1
                                t1[t1 < 0] = 0

                                finetune_inject_sum += np.sum(t1)

                                length1 += len(p)
                        print(finetune_inject_sum, length)
            else:
                if attack == COR:
                    # p=torch.cat([p,params.flatten()])
                    p=torch.from_numpy(np.array([])).float().to(device)
                    # raw_data = raw_data.cpu()
                    # raw_data = raw_data.detach().numpy()
                    finetune_r = 0
                    if netname == 'vgg':
                        i = 1
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                if i == 1:
                                    # if layer not in choices:
                                    p=torch.cat([p,params.flatten()])
                                i = 1 - i
                                        # sum_length += len(params.flatten()) 
                    elif netname == 'other':
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                # if layer not in choices:
                                p=torch.cat([p,params.flatten()])
                    p = p.cpu()
                    p = p.detach().numpy()
                    pp = p[ad]
                    t = raw_data[:len(ad)]
                    
                    p_min = min(pp)
                    p_max = max(pp)
                    # length2 += len(p)

                    # t = raw_data[length1:length2]
                    r_min = min(t)
                    r_max = max(t)

                    # p和pp
                    # pp1 = r_min + ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                    # pp2 = r_max - ((pp - np.ones(len(p)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                    pp1 = r_min + ((pp - np.ones(len(pp)) * p_min) * (r_max - r_min)) / (p_max - p_min)
                    pp2 = r_max - ((pp - np.ones(len(pp)) * p_min) * (r_max - r_min)) / (p_max - p_min)

                    r1 = np.sum(abs(pp1 - t))
                    r2 = np.sum(abs(pp2 - t))
                    finetune_r += min(r1,r2)
                            # length1 = length2
                    # finetune_r = finetune_r/length
                    finetune_r = finetune_r/len(ad)
                    print(finetune_r)
                if attack == SGN:
                    # secret = secret.cpu()
                    # secret = secret.detach().numpy()
                    # secret[secret == 0.] = -1
                
                    finetune_inject_sum = 0
                    ad = list(ad)

                    t=secret[:len(ad)]

                    p=torch.from_numpy(np.array([])).float().to(device)
                    if netname == 'vgg':
                        i = 1
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                if i == 1:
                                    p = torch.cat([p,params.flatten()])
                                i = 1 - i
                    elif netname == 'other':
                        for name, params in net.named_parameters():
                            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                                p = torch.cat([p,params.flatten()]) 
                    p = p.cpu()
                    p = p.detach().numpy()
                    pp = p[ad]

                    t1 = t * pp
                    t1[t1 >= 0] = 1
                    t1[t1 < 0] = 0

                    finetune_inject_sum = np.sum(t1)
                    print(finetune_inject_sum, len(ad))
            
            #Test
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            # Save checkpoint.
            finetune_acc = 100.*correct/total


        if bool(add_LSB):
            if ratio == 1.0:
                print('network:{} | attack:{} | regular:{} | ratio:{} | LSB:20 \nMAE(no LSB):{} | MAE(with LSB):{} \nmodel acc(no LSB):{} | model acc(with LSB):{}'.format(network, attack, cor_ratio, ratio, r if attack==COR else 1-inject_sum/length, add_LSB_r if attack==COR else 1-add_LSB_inject_sum/length, best_acc, LSB_acc))
            else:
                print('network:{} | attack:{} | regular:{} | ratio:{} | LSB:20 \nMAE(no LSB):{} | MAE(with LSB):{} \nmodel acc(no LSB):{} | model acc(with LSB):{}'.format(network, attack, cor_ratio, ratio, r if attack==COR else 1-inject_sum/len(ad), add_LSB_r if attack==COR else 1-add_LSB_inject_sum/len(ad), best_acc, LSB_acc))
        else:
            if ratio == 1.0:
                print('network:{} | attack:{} | regular:{} | ratio:{} \nMAE(no LSB):{} \nmodel acc(no LSB):{}'.format(network, attack, cor_ratio, ratio, r if attack==COR else 1-inject_sum/length, best_acc))
            else:
                print('network:{} | attack:{} | regular:{} | ratio:{} \nMAE(no LSB):{} \nmodel acc(no LSB):{}'.format(network, attack, cor_ratio, ratio, r if attack==COR else 1-inject_sum/len(ad), best_acc))
        
        if bool(with_finetune):
            if ratio == 1.0:
                print('network:{} | attack:{} | regular:{} | ratio:{} \nMAE(without finetune):{} | MAE(with finetune):{} \nmodel acc(without finetune):{} | model acc(with finetune):{}'.format(network, attack, cor_ratio, ratio, r if attack==COR else 1-inject_sum/length, finetune_r if attack==COR else 1-finetune_inject_sum/length, best_acc, finetune_acc))
            else:
                print('network:{} | attack:{} | regular:{} | ratio:{} \nMAE(without finetune):{} | MAE(with finetune):{} \nmodel acc(without finetune):{} | model acc(with finetune):{}'.format(network, attack, cor_ratio, ratio, r if attack==COR else 1-inject_sum/len(ad), finetune_r if attack==COR else 1-finetune_inject_sum/len(ad), best_acc, finetune_acc))
        else:
            if ratio == 1.0:
                print('network:{} | attack:{} | regular:{} | ratio:{} \nMAE(without finetune):{} \nmodel acc(without finetune):{}'.format(network, attack, cor_ratio, ratio, r if attack==COR else 1-inject_sum/length, best_acc))
            else:
                print('network:{} | attack:{} | regular:{} | ratio:{} \nMAE(without finetune):{} \nmodel acc(without finetune):{}'.format(network, attack, cor_ratio, ratio, r if attack==COR else 1-inject_sum/len(ad), best_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='vgg16', type=str, help='name of the network (vgg16, resnet34, or efficientnet)')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--corr', type=float, default=0.)   # malicious term ratio
    parser.add_argument('--attack', type=str, default=NO)
    parser.add_argument('--ratio', type=float, default=1.)
    parser.add_argument('--m', type=int, default=0)
    parser.add_argument('--m_step', type=int, default=25)
    parser.add_argument('--LSB', type=int, default=0)
    parser.add_argument('--finetune', type=int, default=1)
    parser.add_argument('--flr', default=0.1, type=float, help='finetune learning rate')
    args = parser.parse_args()
    # main(network=args.net, lr=args.lr, cor_ratio=args.corr, attack=args.attack, ratio=args.ratio, m=args.m, m_step=args.m_step, add_LSB=args.LSB, with_finetune=args.finetune, flr=args.flr)
    now_ratio = 0.05
    # main(network='vgg16', lr=args.lr, cor_ratio=0.1, attack=COR, ratio=now_ratio, m=args.m, m_step=args.m_step, add_LSB=args.LSB, with_finetune=args.finetune, flr=args.flr)
    # main(network='vgg16', lr=args.lr, cor_ratio=90.0, attack=SGN, ratio=now_ratio, m=args.m, m_step=args.m_step, add_LSB=args.LSB, with_finetune=args.finetune, flr=args.flr)
    # main(network='resnet34', lr=args.lr, cor_ratio=0.1, attack=COR, ratio=now_ratio, m=args.m, m_step=args.m_step, add_LSB=args.LSB, with_finetune=args.finetune, flr=args.flr)
    # main(network='vgg16', lr=args.lr, cor_ratio=130.0, attak=SGN, ratio=now_ratio, m=args.m, m_step=args.m_step, add_LSB=args.LSB, with_finetune=args.finetune, flr=args.flr)
    main(network='efficientnetb2', lr=args.lr, cor_ratio=0.1, attack=COR, ratio=now_ratio, m=args.m, m_step=args.m_step, add_LSB=args.LSB, with_finetune=args.finetune, flr=args.flr)
    # main(network='vgg16', lr=args.lr, cor_ratio=170.0, attack=SGN, ratio=now_ratio, m=args.m, m_step=args.m_step, add_LSB=args.LSB, with_finetune=args.finetune, flr=args.flr)