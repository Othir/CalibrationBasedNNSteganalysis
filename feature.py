# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_score
# from sklearn import datasets

from models import *

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

COR = 'cor'  # Correlation value encoding attack
SGN = 'sgn'  # Sign encoding attack
NO = 'no'


# calculate feature
def calc(data):
    num = len(data) # 10000个数
    mean = 0.0 # mean表示平均值,即期望.
    mean2 = 0.0 # mean2表示平方的平均值
    mean3 = 0.0 # mean3表示三次方的平均值
    for a in data:
        mean += a
        mean2 += a**2
        mean3 += a**3
    mean /= num  
    mean2 /= num
    mean3 /= num
    variance = mean2 - mean**2
    return [mean, variance, mean3]

def calc_stat(data):
    [mean, variance, mean3] = calc(data)
    std = math.sqrt(variance)
    num = len(data)
    mean4=0.0 # mean4计算峰度计算公式的分子
    for a in data:
        a -= mean
        mean4 += a**4
    mean4 /= num
 
    skew = (mean3 - 3 * mean * std**2 - mean**3) / (std**3) # 偏度计算公式
    kurt = mean4 / (std**4) # 峰度计算公式:下方为方差的平方即为标准差的四次方

    # [期望, 标准差, 偏度, 峰度], std^2为方差
    # [mean, std, skewness, kurtosis]
    return [mean, variance, skew, kurt]

def get_fea(network, model_path):

    # print('==> Building model..')
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
        net = RegNetY_400MF()
        netname='other'
    if network == 'senet':
        net = SENet18()
        netname='other'
    if network == 'googlenet':
        net = GoogLeNet()
        netname='other'
        
    # net = net.to(device)
    # if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    
    state_dict = torch.load(model_path, map_location='cpu')['net']
    net.load_state_dict(state_dict, strict=True)
    # net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path, map_location=lambda storage, loc: storage)['net'].items()})
    
    length = 0
    # p=torch.from_numpy(np.array([])).float().to(device)
    fea = []
    if netname == 'vgg':
        i = 1
        for name, params in net.named_parameters():
            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                if i == 1:
                    # p=torch.cat([p,params.flatten()])
                    p = params.flatten()
                    p = p.cpu()
                    p = p.detach().numpy()
                    fea.append(calc_stat(p)) # [mean, variance, skew, kurt]
                i = 1 - i
    elif netname == 'other':
        for name, params in net.named_parameters():
            if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
                # p=torch.cat([p,params.flatten()])
                p = params.flatten()
                p = p.cpu()
                p = p.detach().numpy()
                fea.append(calc_stat(p)) # [mean, variance, skew, kurt]
    fea = np.array(fea)

    return fea

def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
            Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
    return Filelist

def get_model_path(attack, settings, model_root):
    # deal with benign
    if attack=='benign':
        # deal with original
        original_dir = os.path.join(model_root, settings['dataset'], settings['network'], 'original', attack)
        # deal with finetune
        finetune_dir = []
        for iter in [10,20,30,40]:
            finetune_dir.append(os.path.join(model_root, settings['dataset'], settings['network'], 'finetune', 'lr_'+str(settings['lr']), 'iter_'+str(iter), attack))
    # deal with attack
    elif attack in ['cor', 'sgn']:
        # deal with original
        original_dir = os.path.join(model_root, settings['dataset'], settings['network'], 'original', attack, 'coefficient_'+str(settings['coefficient']), 'payload_'+str(settings['payload']))
        # deal with finetune
        finetune_dir = []
        for iter in [10,20,30,40]:
            finetune_dir.append(os.path.join(model_root, settings['dataset'], settings['network'], 'finetune', 'lr_'+str(settings['lr']), 'iter_'+str(iter), attack, 'coefficient_'+str(settings['coefficient']), 'payload_'+str(settings['payload'])))
    # deal with dir
    original_path = get_filelist(original_dir) # （pth数）大小的文件列表
    original_path.sort(key=lambda x:int(os.path.basename(x).split('.')[0]))
    finetune_path = [] # （4*pth数）大小的文件列表
    for iter_dir in finetune_dir:
        iter_path = get_filelist(iter_dir)
        iter_path.sort(key=lambda x:int(os.path.basename(x).split('.')[0]))
        finetune_path.append(iter_path)
    return original_path, finetune_path

# save_path: /disk/dataset/features/
# attack: ['benign', 'cor', 'sgn']
# settings: ['dataset':str, 'network':str, ''lr':float, 'coefficient':float, 'payload':float, ] 
# original特征即静态特征
# def save_fea_as_group(attack, settings, model_root='/disk/dataset/models/', save_path='/disk/dataset/test/'):
def save_fea_as_group(attack, settings, model_root='/disk/dataset/models/', save_path='/disk/dataset/features/'):
    original_path, finetune_path = get_model_path(attack, settings, model_root)

    original_fea_1234 = [] # (pth数*层数*特征维数/阶数)大小的列表
    original_fea_1 = []
    original_fea_12 = []
    original_fea_123 = []
    original_fea_13 = []
    original_fea_24 = []

    # for model_path in tqdm(original_path):
    #     # try:
    #     #     f = get_fea(network=network, model_path=original_path[j])
    #     # except:
    #     #     print('error: '+original_path[j])
    #     #     continue
    #     f = get_fea(network=settings['network'], model_path=model_path) # （层数*特征维数/阶数）大小的列表
    #     original_fea_1234.append(list(f))

    #     # fea_1 = np.concatenate([f[:,1], f[:,3]])
    #     original_fea_1.append(f[:,0])
    #     fea_12 = np.concatenate([f[:,0], f[:,1]])
    #     original_fea_12.append(list(fea_12))
    #     fea_123 = np.concatenate([f[:,0], f[:,1], f[:,2]])
    #     original_fea_123.append(list(fea_123))

    #     fea_24 = np.concatenate([f[:,1], f[:,3]])
    #     original_fea_24.append(list(fea_24))
    #     fea_13 = np.concatenate([f[:,0], f[:,2]])
    #     original_fea_13.append(list(fea_13))

        # break
    
    if attack == 'benign':
        tmp_path = os.path.join(save_path, 'moment_1234', 'original', '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        # np.save(tmp_path, original_fea_1234)
        original_fea_1234 = np.load(tmp_path)
        tmp_path = os.path.join(save_path, 'moment_1', 'original', '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        # np.save(tmp_path, original_fea_1)
        original_fea_1 = np.load(tmp_path)
        tmp_path = os.path.join(save_path, 'moment_12', 'original', '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        # np.save(tmp_path, original_fea_12)
        original_fea_12 = np.load(tmp_path)
        tmp_path = os.path.join(save_path, 'moment_123', 'original', '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        # np.save(tmp_path, original_fea_123)
        original_fea_123 = np.load(tmp_path)
        tmp_path = os.path.join(save_path, 'moment_24', 'original', '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        # np.save(tmp_path, original_fea_24)
        original_fea_24 = np.load(tmp_path)
        tmp_path = os.path.join(save_path, 'moment_13', 'original', '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        # np.save(tmp_path, original_fea_13)
        original_fea_13 = np.load(tmp_path)
    elif attack in ['cor', 'sgn']:
        tmp_path = os.path.join(save_path, 'moment_1234', 'original', '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        # np.save(tmp_path, original_fea_1234)
        original_fea_1234 = np.load(tmp_path)
        tmp_path = os.path.join(save_path, 'moment_1', 'original', '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        # np.save(tmp_path, original_fea_1)
        original_fea_1 = np.load(tmp_path)
        tmp_path = os.path.join(save_path, 'moment_12', 'original', '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        # np.save(tmp_path, original_fea_12)
        original_fea_12 = np.load(tmp_path)
        tmp_path = os.path.join(save_path, 'moment_123', 'original', '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        # np.save(tmp_path, original_fea_123)
        original_fea_123 = np.load(tmp_path)
        tmp_path = os.path.join(save_path, 'moment_24', 'original', '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        # np.save(tmp_path, original_fea_24)
        original_fea_24 = np.load(tmp_path)
        tmp_path = os.path.join(save_path, 'moment_13', 'original', '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        # np.save(tmp_path, original_fea_13)
        original_fea_13 = np.load(tmp_path)

    for i in range(4):

        iter_path = finetune_path[i]
        iter = (i+1)*10
        if settings['iter'] != iter:
            continue

        print(iter)
        finetune_fea_1234 = [] # (pth数*层数*特征维数/阶数)大小的列表
        finetune_fea_1 = []
        finetune_fea_12 = []
        finetune_fea_123 = []
        finetune_fea_13 = []
        finetune_fea_24 = []

        # combined_fea_1234 = [] # (pth数*层数*特征维数/阶数)大小的列表
        combined_fea_1234 = {'concat':[], 'differ':[], 'conDif':[], 'oriDif':[]}
        combined_fea_1 = {'concat':[], 'differ':[], 'conDif':[], 'oriDif':[]}
        combined_fea_12 = {'concat':[], 'differ':[], 'conDif':[], 'oriDif':[]}
        combined_fea_123 = {'concat':[], 'differ':[], 'conDif':[], 'oriDif':[]}
        combined_fea_13 = {'concat':[], 'differ':[], 'conDif':[], 'oriDif':[]}
        combined_fea_24 = {'concat':[], 'differ':[], 'conDif':[], 'oriDif':[]}

        for j in tqdm(range(len(iter_path))):
        # for j in tqdm(range(1)):
            model_path = iter_path[j]
            # try:
            #     f = get_fea(network=network, model_path=original_path[j])
            # except:
            #     print('error: '+original_path[j])
            #     continue
            f = get_fea(network=settings['network'], model_path=model_path) # （层数*特征维数/阶数）大小的列表
            
            # deal with finetune feature
            fea_1234 = f
            finetune_fea_1234.append(list(fea_1234))
            # finetune_fea_1.append(f[:,0])
            # fea_12 = np.concatenate([f[:,0], f[:,1]])
            # finetune_fea_12.append(list(fea_12))
            # fea_123 = np.concatenate([f[:,0], f[:,1], f[:,2]])
            # finetune_fea_123.append(list(fea_123))

            # fea_24 = np.concatenate([f[:,1], f[:,3]])
            # finetune_fea_24.append(list(fea_24))
            # fea_13 = np.concatenate([f[:,0], f[:,2]])
            # finetune_fea_13.append(list(fea_13))

            # if attack == COR: # [_, variance, _, kurt]
            #     fea = np.concatenate([fea[:,1], fea[:,3]])
            # elif attack == SGN: # [mean, _, skew, _]
            #     fea = np.concatenate([fea[:,0], fea[:,2]])

            # deal with combined feature
            # f:  original
            # ff: finetune
            f = np.array(original_fea_1234[j])
            ff = np.array(finetune_fea_1234[j])
            combined_fea_1234['concat'].append(list(f)+list(ff))
            # combined_fea_1234['differ'].append(list(ff-f))
            # combined_fea_1234['conDif'].append(list(f)+list(ff)+list(ff-f))
            # combined_fea_1234['oriDif'].append(list(f)+list(ff-f))
            f = np.array(original_fea_1[j])
            ff = np.array(finetune_fea_1[j])
            combined_fea_1['concat'].append(list(f)+list(ff))
            # combined_fea_1['differ'].append(list(ff-f))
            # combined_fea_1['conDif'].append(list(f)+list(ff)+list(ff-f))
            # combined_fea_1['oriDif'].append(list(f)+list(ff-f))
            f = np.array(original_fea_12[j])
            ff = np.array(finetune_fea_12[j])
            combined_fea_12['concat'].append(list(f)+list(ff))
            # combined_fea_12['differ'].append(list(ff-f))
            # combined_fea_12['conDif'].append(list(f)+list(ff)+list(ff-f))
            # combined_fea_12['oriDif'].append(list(f)+list(ff-f))
            f = np.array(original_fea_123[j])
            ff = np.array(finetune_fea_123[j])
            combined_fea_123['concat'].append(list(f)+list(ff))
            # combined_fea_123['differ'].append(list(ff-f))
            # combined_fea_123['conDif'].append(list(f)+list(ff)+list(ff-f))
            # combined_fea_123['oriDif'].append(list(f)+list(ff-f))
            f = np.array(original_fea_13[j])
            ff = np.array(finetune_fea_13[j])
            combined_fea_13['concat'].append(list(f)+list(ff))
            # combined_fea_13['differ'].append(list(ff-f))
            # combined_fea_13['conDif'].append(list(f)+list(ff)+list(ff-f))
            # combined_fea_13['oriDif'].append(list(f)+list(ff-f))
            f = np.array(original_fea_24[j])
            ff = np.array(finetune_fea_24[j])
            combined_fea_24['concat'].append(list(f)+list(ff))
            # combined_fea_24['differ'].append(list(ff-f))
            # combined_fea_24['conDif'].append(list(f)+list(ff)+list(ff-f))
            # combined_fea_24['oriDif'].append(list(f)+list(ff-f))

        if attack == 'benign':
            pass
        #     # save finetune feature
        #     tmp_path = os.path.join(save_path, 'moment_1234', 'finetune', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, finetune_fea_1234)
        #     tmp_path = os.path.join(save_path, 'moment_1', 'finetune', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, finetune_fea_1)
        #     tmp_path = os.path.join(save_path, 'moment_12', 'finetune', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, finetune_fea_12)
        #     tmp_path = os.path.join(save_path, 'moment_123', 'finetune', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, finetune_fea_123)
        #     tmp_path = os.path.join(save_path, 'moment_24', 'finetune', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, finetune_fea_24)
        #     tmp_path = os.path.join(save_path, 'moment_13', 'finetune', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, finetune_fea_13)
        #     # save combined feature
        #     # concat
        #     tmp_path = os.path.join(save_path, 'moment_1234', 'combined', 'concat', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_1234['concat'])
        #     tmp_path = os.path.join(save_path, 'moment_1', 'combined', 'concat', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_1['concat'])
        #     tmp_path = os.path.join(save_path, 'moment_12', 'combined', 'concat', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_12['concat'])
        #     tmp_path = os.path.join(save_path, 'moment_123', 'combined', 'concat', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_123['concat'])
        #     tmp_path = os.path.join(save_path, 'moment_24', 'combined', 'concat', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_24['concat'])
        #     tmp_path = os.path.join(save_path, 'moment_13', 'combined', 'concat', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_13['concat'])
        #     # differ
        #     tmp_path = os.path.join(save_path, 'moment_1234', 'combined', 'differ', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_1234['differ'])
        #     tmp_path = os.path.join(save_path, 'moment_1', 'combined', 'differ', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_1['differ'])
        #     tmp_path = os.path.join(save_path, 'moment_12', 'combined', 'differ', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_12['differ'])
        #     tmp_path = os.path.join(save_path, 'moment_123', 'combined', 'differ', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_123['differ'])
        #     tmp_path = os.path.join(save_path, 'moment_24', 'combined', 'differ', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_24['differ'])
        #     tmp_path = os.path.join(save_path, 'moment_13', 'combined', 'differ', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_13['differ'])
        #     # conDif
        #     tmp_path = os.path.join(save_path, 'moment_1234', 'combined', 'conDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_1234['conDif'])
        #     tmp_path = os.path.join(save_path, 'moment_1', 'combined', 'conDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_1['conDif'])
        #     tmp_path = os.path.join(save_path, 'moment_12', 'combined', 'conDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_12['conDif'])
        #     tmp_path = os.path.join(save_path, 'moment_123', 'combined', 'conDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_123['conDif'])
        #     tmp_path = os.path.join(save_path, 'moment_24', 'combined', 'conDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_24['conDif'])
        #     tmp_path = os.path.join(save_path, 'moment_13', 'combined', 'conDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_13['conDif'])
        #     # oriDif
        #     tmp_path = os.path.join(save_path, 'moment_1234', 'combined', 'oriDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_1234['oriDif'])
        #     tmp_path = os.path.join(save_path, 'moment_1', 'combined', 'oriDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_1['oriDif'])
        #     tmp_path = os.path.join(save_path, 'moment_12', 'combined', 'oriDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_12['oriDif'])
        #     tmp_path = os.path.join(save_path, 'moment_123', 'combined', 'oriDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_123['oriDif'])
        #     tmp_path = os.path.join(save_path, 'moment_24', 'combined', 'oriDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_24['oriDif'])
        #     tmp_path = os.path.join(save_path, 'moment_13', 'combined', 'oriDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}.npy'.format(settings['dataset'], settings['network'], attack))
        #     os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        #     np.save(tmp_path, combined_fea_13['oriDif'])
            
        elif attack in ['cor', 'sgn']:
            # # save finetune feature
            # tmp_path = os.path.join(save_path, 'moment_1234', 'finetune', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, finetune_fea_1234)
            # tmp_path = os.path.join(save_path, 'moment_1', 'finetune', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, finetune_fea_1)
            # tmp_path = os.path.join(save_path, 'moment_12', 'finetune', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, finetune_fea_12)
            # tmp_path = os.path.join(save_path, 'moment_123', 'finetune', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, finetune_fea_123)
            # tmp_path = os.path.join(save_path, 'moment_24', 'finetune', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, finetune_fea_24)
            # tmp_path = os.path.join(save_path, 'moment_13', 'finetune', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, finetune_fea_13)
            # save combined feature
            # concat
            tmp_path = os.path.join(save_path, 'moment_1234', 'combined', 'concat', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            np.save(tmp_path, combined_fea_1234['concat'])
            tmp_path = os.path.join(save_path, 'moment_1', 'combined', 'concat', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            np.save(tmp_path, combined_fea_1['concat'])
            tmp_path = os.path.join(save_path, 'moment_12', 'combined', 'concat', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            np.save(tmp_path, combined_fea_12['concat'])
            tmp_path = os.path.join(save_path, 'moment_123', 'combined', 'concat', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            np.save(tmp_path, combined_fea_123['concat'])
            tmp_path = os.path.join(save_path, 'moment_24', 'combined', 'concat', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            np.save(tmp_path, combined_fea_24['concat'])
            tmp_path = os.path.join(save_path, 'moment_13', 'combined', 'concat', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            np.save(tmp_path, combined_fea_13['concat'])
            # # differ
            # tmp_path = os.path.join(save_path, 'moment_1234', 'combined', 'differ', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_1234['differ'])
            # tmp_path = os.path.join(save_path, 'moment_1', 'combined', 'differ', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_1['differ'])
            # tmp_path = os.path.join(save_path, 'moment_12', 'combined', 'differ', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_12['differ'])
            # tmp_path = os.path.join(save_path, 'moment_123', 'combined', 'differ', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_123['differ'])
            # tmp_path = os.path.join(save_path, 'moment_24', 'combined', 'differ', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_24['differ'])
            # tmp_path = os.path.join(save_path, 'moment_13', 'combined', 'differ', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_13['differ'])
            # # conDif
            # tmp_path = os.path.join(save_path, 'moment_1234', 'combined', 'conDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_1234['conDif'])
            # tmp_path = os.path.join(save_path, 'moment_1', 'combined', 'conDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_1['conDif'])
            # tmp_path = os.path.join(save_path, 'moment_12', 'combined', 'conDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_12['conDif'])
            # tmp_path = os.path.join(save_path, 'moment_123', 'combined', 'conDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_123['conDif'])
            # tmp_path = os.path.join(save_path, 'moment_24', 'combined', 'conDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_24['conDif'])
            # tmp_path = os.path.join(save_path, 'moment_13', 'combined', 'conDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_13['conDif'])
            # # oriDif
            # tmp_path = os.path.join(save_path, 'moment_1234', 'combined', 'oriDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_1234['oriDif'])
            # tmp_path = os.path.join(save_path, 'moment_1', 'combined', 'oriDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_1['oriDif'])
            # tmp_path = os.path.join(save_path, 'moment_12', 'combined', 'oriDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_12['oriDif'])
            # tmp_path = os.path.join(save_path, 'moment_123', 'combined', 'oriDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_123['oriDif'])
            # tmp_path = os.path.join(save_path, 'moment_24', 'combined', 'oriDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_24['oriDif'])
            # tmp_path = os.path.join(save_path, 'moment_13', 'combined', 'oriDif', 'lr_'+str(settings['lr']), 'iter_'+str(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], attack, settings['coefficient'], settings['payload']))
            # os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            # np.save(tmp_path, combined_fea_13['oriDif'])


if __name__ == '__main__':

# def save_fea_as_group(attack, settings, model_root='/disk/dataset/models/', save_path='/disk/dataset/features/'):
# settings: ['dataset':str, 'network':str, 'lr':float, 'coefficient':float, 'payload':float, ] 

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of the dataset (cifar10, or tinyimagenet)')
    parser.add_argument('--network', type=str, default='vgg16', help='name of the network (vgg16, resnet34, or efficientnetb0)')
    parser.add_argument('--attack', type=str, default=COR)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--corr', type=float, default=1.0)   # malicious term ratio
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--model_root', type=str, default='/disk/dataset/models/')
    parser.add_argument('--save_path', type=str, default='/disk/dataset/features/')
    parser.add_argument('--iter', type=int, default=10)
    args = parser.parse_args()
    dataset = args.dataset
    network = args.network
    attack = args.attack
    lr = args.lr
    coefficient = args.corr
    payload = args.ratio
    model_root = args.model_root
    save_path = args.save_path
    iter = args.iter
    if attack == 'benign':
        settings = {'dataset':dataset, 'network':network, 'lr':lr, 'iter':iter}
    elif attack in ['cor', 'sgn']:
        settings = {'dataset':dataset, 'network':network, 'lr':lr, 'iter':iter, 'coefficient':coefficient, 'payload':payload}
    save_fea_as_group(attack, settings, model_root=model_root, save_path=save_path)



# f1      python feature.py --network vgg16 --attack benign
# f1-20   python feature.py --network vgg16 --attack benign --iter 20
# f1-30   python feature.py --network vgg16 --attack benign --iter 30
# f1-40   python feature.py --network vgg16 --attack benign --iter 40

# f2      python feature.py --network vgg16 --attack cor --corr 1.0 --ratio 0.01
# f2-20   python feature.py --network vgg16 --attack cor --corr 1.0 --ratio 0.01 --iter 20
# f2-30   python feature.py --network vgg16 --attack cor --corr 1.0 --ratio 0.01 --iter 30
# f2-40   python feature.py --network vgg16 --attack cor --corr 1.0 --ratio 0.01 --iter 40

# f3      python feature.py --network vgg16 --attack cor --corr 1.0 --ratio 0.03
# f3-20   python feature.py --network vgg16 --attack cor --corr 1.0 --ratio 0.03 --iter 20
# f3-30   python feature.py --network vgg16 --attack cor --corr 1.0 --ratio 0.03 --iter 30
# f3-40   python feature.py --network vgg16 --attack cor --corr 1.0 --ratio 0.03 --iter 40

# f4      python feature.py --network vgg16 --attack cor --corr 1.0 --ratio 0.05
# f4-20   python feature.py --network vgg16 --attack cor --corr 1.0 --ratio 0.05 --iter 20
# f4-30   python feature.py --network vgg16 --attack cor --corr 1.0 --ratio 0.05 --iter 30
# f4-40   python feature.py --network vgg16 --attack cor --corr 1.0 --ratio 0.05 --iter 40

# f5      python feature.py --network vgg16 --attack sgn --corr 50.0 --ratio 0.01
# f5-20   python feature.py --network vgg16 --attack sgn --corr 50.0 --ratio 0.01 --iter 20
# f5-30   python feature.py --network vgg16 --attack sgn --corr 50.0 --ratio 0.01 --iter 30
# f5-40   python feature.py --network vgg16 --attack sgn --corr 50.0 --ratio 0.01 --iter 40

# f6      python feature.py --network vgg16 --attack sgn --corr 50.0 --ratio 0.03
# f6-20   python feature.py --network vgg16 --attack sgn --corr 50.0 --ratio 0.03 --iter 20
# f6-30   python feature.py --network vgg16 --attack sgn --corr 50.0 --ratio 0.03 --iter 30
# f6-40   python feature.py --network vgg16 --attack sgn --corr 50.0 --ratio 0.03 --iter 40

# f7      python feature.py --network vgg16 --attack sgn --corr 50.0 --ratio 0.05
# f7-20   python feature.py --network vgg16 --attack sgn --corr 50.0 --ratio 0.05 --iter 20
# f7-30   python feature.py --network vgg16 --attack sgn --corr 50.0 --ratio 0.05 --iter 30
# f7-40   python feature.py --network vgg16 --attack sgn --corr 50.0 --ratio 0.05 --iter 40

# f8      python feature.py --network resnet34 --attack benign
# f8-20   python feature.py --network resnet34 --attack benign --iter 20
# f8-30   python feature.py --network resnet34 --attack benign --iter 30
# f8-40   python feature.py --network resnet34 --attack benign --iter 40

# f9      python feature.py --network resnet34 --attack cor --corr 1.0 --ratio 0.01
# f9-20   python feature.py --network resnet34 --attack cor --corr 1.0 --ratio 0.01 --iter 20
# f9-30   python feature.py --network resnet34 --attack cor --corr 1.0 --ratio 0.01 --iter 30
# f9-40   python feature.py --network resnet34 --attack cor --corr 1.0 --ratio 0.01 --iter 40

# conda activate pt1-8 && cd pytorch-ModelSteganalysis/train_cifar/

# f10     python feature.py --network resnet34 --attack cor --corr 1.0 --ratio 0.03
# f10-20  python feature.py --network resnet34 --attack cor --corr 1.0 --ratio 0.03 --iter 20
# f10-30  python feature.py --network resnet34 --attack cor --corr 1.0 --ratio 0.03 --iter 30
# f10-40  python feature.py --network resnet34 --attack cor --corr 1.0 --ratio 0.03 --iter 40

# f11     python feature.py --network resnet34 --attack cor --corr 1.0 --ratio 0.05
# f11-20  python feature.py --network resnet34 --attack cor --corr 1.0 --ratio 0.05 --iter 20
# f11-30  python feature.py --network resnet34 --attack cor --corr 1.0 --ratio 0.05 --iter 30
# f11-40  python feature.py --network resnet34 --attack cor --corr 1.0 --ratio 0.05 --iter 40

# f12     python feature.py --network resnet34 --attack sgn --corr 50.0 --ratio 0.01
# f12-20  python feature.py --network resnet34 --attack sgn --corr 50.0 --ratio 0.01 --iter 20
# f12-30  python feature.py --network resnet34 --attack sgn --corr 50.0 --ratio 0.01 --iter 30
# f12-40  python feature.py --network resnet34 --attack sgn --corr 50.0 --ratio 0.01 --iter 40

# f13     python feature.py --network resnet34 --attack sgn --corr 50.0 --ratio 0.03
# f13-20  python feature.py --network resnet34 --attack sgn --corr 50.0 --ratio 0.03 --iter 20
# f13-30  python feature.py --network resnet34 --attack sgn --corr 50.0 --ratio 0.03 --iter 30
# f13-40  python feature.py --network resnet34 --attack sgn --corr 50.0 --ratio 0.03 --iter 40

# f14     python feature.py --network resnet34 --attack sgn --corr 50.0 --ratio 0.05
# f14-20  python feature.py --network resnet34 --attack sgn --corr 50.0 --ratio 0.05 --iter 20
# f14-30  python feature.py --network resnet34 --attack sgn --corr 50.0 --ratio 0.05 --iter 30
# f14-40  python feature.py --network resnet34 --attack sgn --corr 50.0 --ratio 0.05 --iter 40

# conda activate pt1-8


# CUDA_VISABLE_DEVICES=0 python feature.py







    # save_fea_as_group(attack='benign', settings={'dataset':'cifar10', 'network':'vgg16', 'lr':0.1})
    # save_fea_as_group(attack='cor', settings={'dataset':'cifar10', 'network':'vgg16', 'lr':0.1, 'coefficient':1.0, 'payload':0.01})
    # save_fea_as_group(attack='cor', settings={'dataset':'cifar10', 'network':'vgg16', 'lr':0.1, 'coefficient':1.0, 'payload':0.03})
    # save_fea_as_group(attack='cor', settings={'dataset':'cifar10', 'network':'vgg16', 'lr':0.1, 'coefficient':1.0, 'payload':0.05})
    # save_fea_as_group(attack='sgn', settings={'dataset':'cifar10', 'network':'vgg16', 'lr':0.1, 'coefficient':50.0, 'payload':0.01})
    # save_fea_as_group(attack='sgn', settings={'dataset':'cifar10', 'network':'vgg16', 'lr':0.1, 'coefficient':50.0, 'payload':0.03})
    # save_fea_as_group(attack='sgn', settings={'dataset':'cifar10', 'network':'vgg16', 'lr':0.1, 'coefficient':50.0, 'payload':0.05})
    # save_fea_as_group(attack='benign', settings={'dataset':'cifar10', 'network':'resnet34', 'lr':0.1})
    # save_fea_as_group(attack='cor', settings={'dataset':'cifar10', 'network':'resnet34', 'lr':0.1, 'coefficient':1.0, 'payload':0.01})
    # save_fea_as_group(attack='cor', settings={'dataset':'cifar10', 'network':'resnet34', 'lr':0.1, 'coefficient':1.0, 'payload':0.03})
    # save_fea_as_group(attack='cor', settings={'dataset':'cifar10', 'network':'resnet34', 'lr':0.1, 'coefficient':1.0, 'payload':0.05})
    # save_fea_as_group(attack='sgn', settings={'dataset':'cifar10', 'network':'resnet34', 'lr':0.1, 'coefficient':50.0, 'payload':0.01})
    # save_fea_as_group(attack='sgn', settings={'dataset':'cifar10', 'network':'resnet34', 'lr':0.1, 'coefficient':50.0, 'payload':0.03})
    # save_fea_as_group(attack='sgn', settings={'dataset':'cifar10', 'network':'resnet34', 'lr':0.1, 'coefficient':50.0, 'payload':0.05})