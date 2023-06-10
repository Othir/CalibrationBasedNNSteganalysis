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
# from utils import progress_bar

COR = 'cor'  # Correlation value encoding attack
SGN = 'sgn'  # Sign encoding attack
NO = 'no'

def finetune(lr=0.1, network='vgg16', iter_num=20, model_path='', finetune_model_path='', resume=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_path = './data/Cifar10'

    # Data
    print('==> Preparing data..')
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
        root=data_path, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)

    # for ra in range(2):
    best_acc = 0  # best test accuracy
    start_epoch = 1  # start from epoch 0 or last checkpoint epoch
    # Model
    print('==> Building model..')
    if network == 'vgg16':
        net = VGG('VGG16')
        netname = 'vgg'
    if network == 'resnet34':
        net = ResNet34()
        netname='other'
    if network == 'efficientnetb0':
        net = EfficientNetB0()
        netname='other'
    if network == 'shufflenetv2':
        net = ShuffleNetV2(2)
        netname='other'
    if network == 'regnet':
        net = RegNetY_4GF()
        netname='other'
    if network == 'senet':
        net = SENet18()
        netname='other'
    if network == 'googlenet':
        net = GoogLeNet()
        netname='other'

    net = net.to(device)
    # if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    # if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                        momentum=0.9, weight_decay=1e-5)

#     scheduler = lr_scheduler.StepLR(optimizer, step_size=int(iter_num/2), gamma=0.1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    for epoch in range(start_epoch, start_epoch+iter_num):
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
            outputs = net(inputs)
            loss = criterion(outputs, targets)
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

        os.makedirs(os.path.dirname(finetune_model_path), exist_ok=True)
        if epoch%iter_num==0:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, finetune_model_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--m', default=0, type=int)
    args = parser.parse_args()

    m = args.m
    network = 'vgg16'
    itern = 40
    attack = 'sgn'
    corr = 50.0
    # num = 20
    lr = 0.01

    for i in range(5):
        num = i+m
        original_stego_path = './cifar10/{}/original/{}/regular_{}/ratio_0.01/{}.pth'.format(network, attack, corr, num)
        save_path = './cifar10/{}/finetune_lr_{}/attack/iter_{}/{}/regular_{}/ratio_0.01/{}.pth'.format(network, lr, itern, attack, corr, num)
        finetune(lr=lr, network=network, iter_num=itern, model_path=original_stego_path, finetune_model_path=save_path)

        original_benign_path = './cifar10/{}/original/benign/{}.pth'.format(network, num)
        save_path = './cifar10/{}/finetune_lr_{}/benign/iter_{}/{}.pth'.format(network, lr, itern, num)
        finetune(lr=lr, network=network, iter_num=itern, model_path=original_benign_path, finetune_model_path=save_path)