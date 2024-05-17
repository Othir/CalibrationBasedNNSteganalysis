from ctypes import LibraryLoader
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# from sklearn import manifold, datasets
import pandas as pd

from sklearn import datasets, svm, metrics, model_selection, preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils import shuffle

from models import *
import os
from tqdm import tqdm
import argparse


def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
            Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)
    return Filelist

def get_data(settings, random_seed, moment='1234', feature_root='/disk/dataset/features/'):
# 输入变化，network，attack，payload，共14组

    # deal with original
    tmp_path = os.path.join(feature_root, 'moment_{}'.format(moment), 'original', '{}_{}_benign.npy'.format(settings['dataset'], settings['network']))
    tmp_fea0 = np.load(tmp_path, allow_pickle=True)
    tmp_fea0 = shuffle(tmp_fea0, random_state=random_seed)
    tmp_path = os.path.join(feature_root, 'moment_{}'.format(moment), 'original', '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], settings['attack'], settings['coefficient'], settings['payload']))
    tmp_fea1 = np.load(tmp_path, allow_pickle=True)
    tmp_fea1 = shuffle(tmp_fea1, random_state=random_seed)
    if moment=='1234':
        dim = np.array(tmp_fea0).shape
        tmp_fea0 = np.array(tmp_fea0).reshape(dim[0],dim[1]*dim[2])
        tmp_fea1 = np.array(tmp_fea1).reshape(dim[0],dim[1]*dim[2])
    train = np.array(tmp_fea0).tolist()[:485] + np.array(tmp_fea1).tolist()[:485]
    val = np.array(tmp_fea0).tolist()[485:535] + np.array(tmp_fea1).tolist()[485:535]
    test = np.array(tmp_fea0).tolist()[535:] + np.array(tmp_fea1).tolist()[535:]
    original_data = {'train':train, 'val':val, 'test':test}
    # deal with finetune
    finetune_data = {'train':[], 'val':[], 'test':[]}     
    for iter in [10,20,30,40]:
        tmp_path = os.path.join(feature_root, 'moment_{}'.format(moment), 'finetune', 'lr_{}'.format(settings['lr']), 'iter_{}'.format(iter), '{}_{}_benign.npy'.format(settings['dataset'], settings['network']))
        tmp_fea0 = np.load(tmp_path, allow_pickle=True)
        tmp_fea0 = shuffle(tmp_fea0, random_state=random_seed)
        tmp_path = os.path.join(feature_root, 'moment_{}'.format(moment), 'finetune', 'lr_{}'.format(settings['lr']), 'iter_{}'.format(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], settings['attack'], settings['coefficient'], settings['payload']))
        tmp_fea1 = np.load(tmp_path, allow_pickle=True)
        tmp_fea1 = shuffle(tmp_fea1, random_state=random_seed)
        if moment=='1234':
            dim = np.array(tmp_fea0).shape
            tmp_fea0 = np.array(tmp_fea0).reshape(dim[0],dim[1]*dim[2])
            tmp_fea1 = np.array(tmp_fea1).reshape(dim[0],dim[1]*dim[2])
        train = np.array(tmp_fea0).tolist()[:485] + np.array(tmp_fea1).tolist()[:485]
        val = np.array(tmp_fea0).tolist()[485:535] + np.array(tmp_fea1).tolist()[485:535]
        test = np.array(tmp_fea0).tolist()[535:] + np.array(tmp_fea1).tolist()[535:]
        finetune_data['train'].append(train)
        finetune_data['val'].append(val)
        finetune_data['test'].append(test)
    # deal with combine
    combined_data = {'train':{'concat':[], 'differ':[], 'conDif':[], 'oriDif':[]}, 'val':{'concat':[], 'differ':[], 'conDif':[], 'oriDif':[]}, 'test':{'concat':[], 'differ':[], 'conDif':[], 'oriDif':[]}}
    for fuse in ['differ', 'conDif', 'oriDif']:
        for iter in [10,20,30,40]:
            tmp_path = os.path.join(feature_root, 'moment_{}'.format(moment), 'combined', fuse, 'lr_{}'.format(settings['lr']), 'iter_{}'.format(iter), '{}_{}_benign.npy'.format(settings['dataset'], settings['network']))
            tmp_fea0 = np.load(tmp_path, allow_pickle=True)
            tmp_fea0 = shuffle(tmp_fea0, random_state=random_seed)
            tmp_path = os.path.join(feature_root, 'moment_{}'.format(moment), 'combined', fuse, 'lr_{}'.format(settings['lr']), 'iter_{}'.format(iter), '{}_{}_{}_coefficient_{}_payload_{}.npy'.format(settings['dataset'], settings['network'], settings['attack'], settings['coefficient'], settings['payload']))
            tmp_fea1 = np.load(tmp_path, allow_pickle=True)
            tmp_fea1 = shuffle(tmp_fea1, random_state=random_seed)
            if moment=='1234':
                if fuse == 'differ':
                    dim = np.array(tmp_fea0).shape
                    # 50,34,4
                    tmp_fea0 = np.array(tmp_fea0).reshape(dim[0],dim[1]*dim[2])
                    tmp_fea1 = np.array(tmp_fea1).reshape(dim[0],dim[1]*dim[2])
                if fuse == 'oriDif':
                    dim = np.array(tmp_fea0).shape
                    tmp_fea0 = np.array(tmp_fea0).reshape(dim[0],dim[1]*dim[2])
                    tmp_fea1 = np.array(tmp_fea1).reshape(dim[0],dim[1]*dim[2])
                    # tmp_fea0 = np.concatenate([tmp_fea0[:,:dim[1]//2,:],tmp_fea0[:,dim[1]//2:,:]],axis=2).reshape(dim[0],dim[1]*dim[2])
                    # tmp_fea1 = np.concatenate([tmp_fea1[:,:dim[1]//2,:],tmp_fea1[:,dim[1]//2:,:]],axis=2).reshape(dim[0],dim[1]*dim[2])
                if fuse == 'conDif':
                    dim = np.array(tmp_fea0).shape
                    tmp_fea0 = np.array(tmp_fea0).reshape(dim[0],dim[1]*dim[2])
                    tmp_fea1 = np.array(tmp_fea1).reshape(dim[0],dim[1]*dim[2])
                    # tmp_fea0 = np.concatenate([tmp_fea0[:,:dim[1]//3,:],tmp_fea0[:,dim[1]//3:(dim[1]//3)*2,:],tmp_fea0[:,(dim[1]//3)*2:,:]],axis=2).reshape(dim[0],dim[1]*dim[2])
                    # tmp_fea1 = np.concatenate([tmp_fea1[:,:dim[1]//3,:],tmp_fea1[:,dim[1]//3:(dim[1]//3)*2,:],tmp_fea1[:,(dim[1]//3)*2:,:]],axis=2).reshape(dim[0],dim[1]*dim[2])
                    
            train = np.array(tmp_fea0).tolist()[:485] + np.array(tmp_fea1).tolist()[:485]
            val = np.array(tmp_fea0).tolist()[485:535] + np.array(tmp_fea1).tolist()[485:535]
            test = np.array(tmp_fea0).tolist()[535:] + np.array(tmp_fea1).tolist()[535:]
            combined_data['train'][fuse].append(train)
            combined_data['val'][fuse].append(val)
            combined_data['test'][fuse].append(test)
    
    return original_data, finetune_data, combined_data

def draw_roc(fpr, tpr, settings, save_root):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % val_auc[2])
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    # plt.show()

    plt.savefig(os.path.join(save_root, '{}_{}_{}th_{}iter_tSNE_{}.svg'.format(settings['network'], settings['attack'], settings['mean_num'], settings['layer'], tSNE_iter, name)), format='svg', bbox_inches='tight')
    plt.savefig(os.path.join(save_root, '{}_{}_{}th_{}iter_tSNE_{}.jpg'.format(settings['network'], settings['attack'], settings['mean_num'], settings['layer'], tSNE_iter, name)), dpi=1000, bbox_inches='tight')

# def deal_with_concat(train, val, test):


def cls(settings, save_root, feature_root='/disk/dataset/features/', random_seed=2023):
    original_data, finetune_data, combined_data = get_data(settings, random_seed, settings['moment'], feature_root)

    train_label = [0 for i in range(485)] + [1 for i in range(485)]
    val_label = [0 for i in range(50)] + [1 for i in range(50)]
    test_label = [0 for i in range(100)] + [1 for i in range(100)]
    train_label = shuffle(train_label, random_state=random_seed+1)
    val_label = shuffle(val_label, random_state=random_seed+2)
    test_label = shuffle(test_label, random_state=random_seed+3)
    if settings['type'] == 'original':
        train_data = original_data['train']
        val_data = original_data['val']
        test_data = original_data['test']
        train_data = shuffle(train_data, random_state=random_seed+1)
        val_data = shuffle(val_data, random_state=random_seed+2)
        test_data = shuffle(test_data, random_state=random_seed+3)

        classifer = LogisticRegression(max_iter=10000, random_state=random_seed*2) # C=1.0, penalty=l2
        classifer.fit(train_data, train_label)

        if settings['test']:
            test_prob = classifer.predict_proba(test_data)
            # print(val_prob)
            fpr, tpr, thresholds = roc_curve(test_label, np.array(test_prob)[:,1], pos_label=1)
            test_auc = auc(fpr, tpr)
            # print(fpr)
            # draw_roc(fpr, tpr, settings, save_root)
            print('{} moment:{}          test AUC: {:.4f}  2A-1: {:.4f}'.format(settings['type'], settings['moment'], test_auc, 2*test_auc-1))
        else:
            val_prob = classifer.predict_proba(val_data)
            # print(val_prob)
            fpr, tpr, thresholds = roc_curve(val_label, np.array(val_prob)[:,1], pos_label=1)
            val_auc = auc(fpr, tpr)
            # print(fpr)
            # draw_roc(fpr, tpr, settings, save_root)
            print('{} moment:{}           val AUC: {:.4f}  2A-1: {:.4f}'.format(settings['type'], settings['moment'], val_auc, 2*val_auc-1))

    elif settings['type'] == 'finetune':
        train_data = finetune_data['train']
        val_data = finetune_data['val']
        test_data = finetune_data['test']
        for iter in range(4):
            # if iter != 3:
            #     continue

            train_data[iter] = shuffle(train_data[iter], random_state=random_seed+1)
            val_data[iter] = shuffle(val_data[iter], random_state=random_seed+2)
            test_data[iter] = shuffle(test_data[iter], random_state=random_seed+3)

            classifer = LogisticRegression(max_iter=10000, random_state=random_seed*2) # C=1.0, penalty=l2
            classifer.fit(train_data[iter], train_label)

            if settings['test']:
                test_prob = classifer.predict_proba(test_data[iter])
                # print(val_prob)
                fpr, tpr, thresholds = roc_curve(test_label, np.array(test_prob)[:,1], pos_label=1)
                test_auc = auc(fpr, tpr)
                # print(fpr)
                # draw_roc(fpr, tpr, settings, save_root)
                print('{} moment:{}     iter:{}     test AUC: {:.4f}  2A-1: {:.4f}'.format(settings['type'], settings['moment'], (iter+1)*10, test_auc, 2*test_auc-1))
            else:
                val_prob = classifer.predict_proba(val_data[iter])
                fpr, tpr, thresholds = roc_curve(val_label, np.array(val_prob)[:,1], pos_label=1)
                val_auc = auc(fpr, tpr)
                # draw_roc(fpr, tpr, settings, save_root)
                print('{} moment:{}     iter:{}      val AUC: {:.4f}  2A-1: {:.4f}'.format(settings['type'], settings['moment'], (iter+1)*10, val_auc, 2*val_auc-1))


    elif settings['type'] == 'combined':
        train_data = combined_data['train']
        val_data = combined_data['val']
        test_data = combined_data['test']
        # for fuse in ['concat', 'differ', 'conDif', 'oriDif']:
        # for fuse in ['differ', 'conDif', 'oriDif']:
        for fuse in ['oriDif']:
            for iter in range(4):
                # if iter != 2:
                #     continue
                # train_data[fuse][iter] = shuffle(train_data[fuse][iter], random_state=random_seed+1)
                # val_data[fuse][iter] = shuffle(val_data[fuse][iter], random_state=random_seed+2)
                # test_data[fuse][iter] = shuffle(test_data[fuse][iter], random_state=random_seed+3)
                train = shuffle(train_data[fuse][iter], random_state=random_seed+1)
                val = shuffle(val_data[fuse][iter], random_state=random_seed+2)
                test = shuffle(test_data[fuse][iter], random_state=random_seed+3)
                # print(np.array(val).shape)

                classifer = LogisticRegression(max_iter=10000, random_state=random_seed*2) # C=1.0, penalty=l2
                # print(np.array(train).shape)
                classifer.fit(train, train_label)
                

                if settings['test']:
                    test_prob = classifer.predict_proba(test)
                    # print(val_prob)
                    fpr, tpr, thresholds = roc_curve(test_label, np.array(test_prob)[:,1], pos_label=1)
                    test_auc = auc(fpr, tpr)
                    # print(fpr)
                    # draw_roc(fpr, tpr, settings, save_root)
                    print('{} moment:{}  fuse:{}      iter:{}     test AUC: {:.4f}  2A-1: {:.4f}'.format(settings['type'], settings['moment'], fuse, (iter+1)*10, test_auc, 2*test_auc-1))
                else:
                    val_prob = classifer.predict_proba(val)
                    fpr, tpr, thresholds = roc_curve(val_label, np.array(val_prob)[:,1], pos_label=1)
                    val_auc = auc(fpr, tpr)
                    # draw_roc(fpr, tpr, settings, save_root)
                    print('{} moment:{}  fuse:{}      iter:{}      val AUC: {:.4f}  2A-1: {:.4f}'.format(settings['type'], settings['moment'], fuse, (iter+1)*10, val_auc, 2*val_auc-1))

                if fuse == 'conDif':
                    dim = (np.array(train).shape[1]//3)*2
                    train = np.array(train)[:,:dim]
                    val = np.array(val)[:,:dim]
                    test = np.array(test)[:,:dim]
                    classifer = LogisticRegression(max_iter=10000, random_state=random_seed*2) # C=1.0, penalty=l2
                    classifer.fit(train, train_label)
                    if settings['test']:
                        test_prob = classifer.predict_proba(test)
                        # print(val_prob)
                        fpr, tpr, thresholds = roc_curve(test_label, np.array(test_prob)[:,1], pos_label=1)
                        test_auc = auc(fpr, tpr)
                        # print(fpr)
                        # draw_roc(fpr, tpr, settings, save_root)
                        print('{} moment:{}  fuse:concat      iter:{}     test AUC: {:.4f}  2A-1: {:.4f}'.format(settings['type'], settings['moment'], (iter+1)*10, test_auc, 2*test_auc-1))
                    else:
                        val_prob = classifer.predict_proba(val)
                        fpr, tpr, thresholds = roc_curve(val_label, np.array(val_prob)[:,1], pos_label=1)
                        val_auc = auc(fpr, tpr)
                        # draw_roc(fpr, tpr, settings, save_root)
                        print('{} moment:{}  fuse:concat      iter:{}      val AUC: {:.4f}  2A-1: {:.4f}'.format(settings['type'], settings['moment'], (iter+1)*10, val_auc, 2*val_auc-1))

    return classifer


if __name__ == '__main__':
# def main(benign_settings, attack_settings, feature_root='/disk/dataset/features/', save_root=):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='name of the dataset (cifar10, or tinyimagenet)')
    parser.add_argument('--network', type=str, default='vgg16', help='name of the network (vgg16, resnet34, or efficientnetb0)')
    parser.add_argument('--attack', type=str, default='cor')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--corr', type=float, default=1.0)   # malicious term ratio
    parser.add_argument('--ratio', type=float, default=0.05)
    parser.add_argument('--feature_root', type=str, default='/disk/dataset/features/')
    parser.add_argument('--save_root', type=str, default='/home/zhaona/pytorch-ModelSteganalysis/train_cifar/result/')
    parser.add_argument('--type', type=str, default='original')
    parser.add_argument('--moment', type=str, default='13')
    parser.add_argument('--test', type=int, default=0)
    args = parser.parse_args()
    dataset = args.dataset
    network = args.network
    attack = args.attack
    lr = args.lr
    coefficient = args.corr
    payload = args.ratio
    feature_root = args.feature_root
    save_root = args.save_root
    fea_type = args.type
    moment = args.moment
    is_test = args.test
    # settings = {'dataset':dataset, 'network':network, 'attack':attack, 'lr':lr, 'coefficient':coefficient, 'payload':payload, 'type':fea_type, 'moment':moment, 'test':is_test}

    # print('baseline')
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.01, 'type':'original', 'moment':'13', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.03, 'type':'original', 'moment':'13', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.05, 'type':'original', 'moment':'13', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.01, 'type':'original', 'moment':'24', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.03, 'type':'original', 'moment':'24', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.05, 'type':'original', 'moment':'24', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)

    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.01, 'type':'original', 'moment':'13', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.03, 'type':'original', 'moment':'13', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.05, 'type':'original', 'moment':'13', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.01, 'type':'original', 'moment':'24', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.03, 'type':'original', 'moment':'24', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.05, 'type':'original', 'moment':'24', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)

    # print('tiao can')
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.05, 'type':'combined', 'moment':'1234', 'test':0}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.05, 'type':'combined', 'moment':'1234', 'test':0}
    # cls(settings, save_root, feature_root, random_seed=1234)

    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.05, 'type':'combined', 'moment':'1', 'test':0}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.05, 'type':'combined', 'moment':'12', 'test':0}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.05, 'type':'combined', 'moment':'123', 'test':0}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.05, 'type':'combined', 'moment':'1', 'test':0}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.05, 'type':'combined', 'moment':'12', 'test':0}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.05, 'type':'combined', 'moment':'123', 'test':0}
    # cls(settings, save_root, feature_root, random_seed=1234)

    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.01, 'type':'combined', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.03, 'type':'combined', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.05, 'type':'combined', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()

    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.01, 'type':'combined', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.03, 'type':'combined', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.05, 'type':'combined', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()


    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.01, 'type':'combined', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.03, 'type':'combined', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.05, 'type':'combined', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()

    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.01, 'type':'combined', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.03, 'type':'combined', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.05, 'type':'combined', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)

    # print('original cor')
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.01, 'type':'original', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.03, 'type':'original', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.05, 'type':'original', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()

    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.01, 'type':'original', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.03, 'type':'original', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.05, 'type':'original', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()

    # print('original sgn')
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.01, 'type':'original', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.03, 'type':'original', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.05, 'type':'original', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()

    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.01, 'type':'original', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.03, 'type':'original', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.05, 'type':'original', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()



    # print('finetune cor')
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.01, 'type':'finetune', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.03, 'type':'finetune', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.05, 'type':'finetune', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()

    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.01, 'type':'finetune', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.03, 'type':'finetune', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':0.05, 'type':'finetune', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()

    # print('finetune sgn')
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.01, 'type':'finetune', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.03, 'type':'finetune', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.05, 'type':'finetune', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()

    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.01, 'type':'finetune', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.03, 'type':'finetune', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    # settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':0.05, 'type':'finetune', 'moment':'1234', 'test':1}
    # cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    
    payload = 0.01
    print(f'payload:{payload}')
    print('moment original vgg16 sgn')
    t = 'original'
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'24', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'1', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'12', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'123', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'1234', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    print()

    print('moment original resnet34 sgn')
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'24', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'1', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'12', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'123', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'1234', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    print()

    print('moment finetune vgg16 sgn')
    t = 'finetune'
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'24', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'1', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'12', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'123', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'1234', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    print()

    print('moment finetune resnet34 sgn')
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'24', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'1', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'12', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'123', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'1234', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    print()

    print('moment combined vgg16 sgn')
    t = 'combined'
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'24', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'1', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'12', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'123', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'1234', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    print()

    print('moment combined resnet34 sgn')
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'24', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'1', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'12', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'123', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'sgn', 'lr':0.1, 'coefficient':50.0, 'payload':payload, 'type':t, 'moment':'1234', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    print()

    print('moment original vgg16 cor')
    t = 'original'
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'24', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'1', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'12', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'123', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'1234', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    print()

    print('moment original resnet34 cor')
    t = 'original'
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'24', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'1', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'12', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'123', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'1234', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    print()

    print('moment finetune vgg16 cor')
    t = 'finetune'
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'24', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'1', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'12', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'123', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'1234', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    print()

    print('moment finetune resnet34 cor')
    t = 'finetune'
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'24', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'1', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'12', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'123', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'1234', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    print()

    print('moment combined vgg16 cor')
    t = 'combined'
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'24', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'1', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'12', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'123', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'vgg16', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'1234', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    print()

    print('moment combined resnet34 cor')
    t = 'combined'
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'24', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'1', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'12', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'123', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    # print()
    settings = {'dataset':dataset, 'network':'resnet34', 'attack':'cor', 'lr':0.1, 'coefficient':1.0, 'payload':payload, 'type':t, 'moment':'1234', 'test':1}
    cls(settings, save_root, feature_root, random_seed=1234)
    print()