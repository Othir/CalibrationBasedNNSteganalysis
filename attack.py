import sys
import numpy as np
# import ipdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from scipy import stats
from scipy.stats import pearsonr

def corr_loss_full(net,secret,device,netname='other'):
	cor_loss = torch.tensor(0., requires_grad=True).to(device)

	length1 = 0
	length2 = 0
	if netname == 'vgg':
		i = 1
		for name, params in net.named_parameters():
			if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
				if i == 1:
					p = params.flatten()			
					length2 += len(p)
					t = secret[length1:length2]
					t_mean = torch.mean(t)
					p_mean = torch.mean(p)
					p_m = (p - p_mean)
					t_m = (t - t_mean)
					r_num = abs(torch.sum(p_m * t_m))
					r_den = torch.sqrt(torch.sum(p_m * p_m)) * torch.sqrt(torch.sum(t_m * t_m))
					r = r_num / r_den
					cor_loss = cor_loss - r
					length1 = length2
				i = 1 - i
	elif netname == 'other':
		for name, params in net.named_parameters():
			if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
				p = params.flatten()
				length2 += len(p)
				t = secret[length1:length2]
				t_mean = torch.mean(t)
				p_mean = torch.mean(p)
				p_m = (p - p_mean)
				t_m = (t - t_mean)
				r_num = abs(torch.sum(p_m * t_m))
				r_den = torch.sqrt(torch.sum(p_m * p_m)) * torch.sqrt(torch.sum(t_m * t_m))
				r = r_num / r_den
				cor_loss = cor_loss - r
				length1 = length2
	else:
		print('error')

	return cor_loss

def sign_loss_full(net,secret,length,device,netname='other'):
	sgn_loss = torch.tensor(0., requires_grad=True).to(device)

	length1 = 0
	length2 = 0
	secret[secret == 0.] = -1.
	if netname == 'vgg':
		i = 1
		for name, params in net.named_parameters():
			if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
				if i == 1:
					p = params.flatten()
					length2 += len(p)
					zeros = torch.zeros(len(p)).to(device)
					t = secret[length1:length2]
					constraints = t * p
					penalty = torch.where(torch.gt(constraints, zeros), zeros, constraints)
					penalty = abs(penalty)
					sgn_loss = sgn_loss + torch.sum(penalty)/length
					length1 = length2
				i = 1 - i
	elif netname == 'other':
		for name, params in net.named_parameters():
			if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
				p = params.flatten()
				length2 += len(p)
				zeros = torch.zeros(len(p)).to(device)
				t = secret[length1:length2]
				constraints = t * p
				penalty = torch.where(torch.gt(constraints, zeros), zeros, constraints)
				penalty = abs(penalty)
				sgn_loss = sgn_loss + torch.sum(penalty)/length
				length1 = length2

	return sgn_loss

def rbg_to_grayscale(images):
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])

def sign_loss(net,secret,length,ad,device,netname='other'):
	sgn_loss = torch.tensor(0., requires_grad=True).to(device)

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
	pp = p[ad]
	zeros = torch.zeros(len(pp)).to(device)
	t = secret[:len(ad)]
	constraints = t * pp
	penalty = torch.where(torch.gt(constraints, zeros), zeros, constraints)
	penalty = abs(penalty)
	sgn_loss = sgn_loss + torch.sum(penalty)/len(ad)

	return sgn_loss

def corr_loss(net,secret,ad,device,netname='other'):
	cor_loss = torch.tensor(0., requires_grad=True).to(device)

	# length = 0
	p=torch.from_numpy(np.array([])).float().to(device)
	if netname == 'vgg':
		i = 1
		for name, params in net.named_parameters():
			if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
				if i == 1:
					p = torch.cat([p,params.flatten()])
				i = 1 - i						
		pp = p[ad]
		t = secret[:len(ad)]
		t_mean = torch.mean(t)
		p_mean = torch.mean(pp)
		p_m = (pp - p_mean)
		t_m = (t - t_mean)
		r_num = abs(torch.sum(p_m * t_m))
		r_den = torch.sqrt(torch.sum(p_m * p_m)) * torch.sqrt(torch.sum(t_m * t_m))
		r = r_num / r_den
		cor_loss = cor_loss - r						
	elif netname == 'other':
		for name, params in net.named_parameters():
			if 'weight' in name and 'bn' not in name and 'shortcut' not in name and 'se' not in name:
				p = torch.cat([p,params.flatten()])
		pp = p[ad]
		t = secret[:len(ad)]
		t_mean = torch.mean(t)
		p_mean = torch.mean(pp)
		p_m = (pp - p_mean)
		t_m = (t - t_mean)
		r_num = abs(torch.sum(p_m * t_m))
		r_den = torch.sqrt(torch.sum(p_m * p_m)) * torch.sqrt(torch.sum(t_m * t_m))
		r = r_num / r_den
		cor_loss = cor_loss - r
	return cor_loss