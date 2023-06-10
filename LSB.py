import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import numpy as np


from models import *
from binary_converter import float2bit, bit2float
import os

num = 1
n_lsb = 20

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = './output/ResNet34_sgn_50.0_0.pth'

def LSB(network='vgg16', model_path=MODEL_PATH, save_path='', n_lsb=n_lsb):
	print('==> Building original model..')

	if network == 'vgg16':
		net = VGG('VGG16')
	if network == 'resnet34':
		net = ResNet34()	
	if network == 'efficientnetb0':
		net = EfficientNetB0()
	if network == 'shufflenetv2':
		net = ShuffleNetV2(2)
	if network == 'regnet':
		net = RegNetY_4GF()
	if network == 'senet':
		net = SENet18()
	if network == 'googlenet':
		net = GoogLeNet()

	net = net.to(device)
	# if device == 'cuda':
	net = torch.nn.DataParallel(net)
	cudnn.benchmark = True

	state_dict = torch.load(model_path, map_location=device)['net']
	net.load_state_dict(state_dict, strict=True)
	# net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path, map_location=lambda storage, loc: storage)['net'].items()})

	for key in net.state_dict().keys():
		if 'weight' in key and 'bn' not in key and 'shortcut' not in key:
			params = net.state_dict()[key]
			# params1 = params.cpu()
			p = params.cpu()
			p = p.detach().numpy()
			pp = p.flatten()

			length = len(pp)
			s = np.random.randint(0,2,length*n_lsb)

			params2 = float2bit(pp, num_e_bits=8, num_m_bits=23, bias=127.)
			params3 = params2.numpy()

			#change
			j = 0
			for i in range(length):
				params3[i][-n_lsb:] = s[j:j+n_lsb]
				j = j + n_lsb

			#binary to float
			params4 = bit2float(params3)
			params5 = params4.reshape(p.shape)
			# params5 = params4.numpy()
			# params5 = np.resize(params5, p.shape)
			# params5 = torch.from_numpy(params5)

			net.state_dict()[key].copy_(params5)

	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	print('Saving LSB..')
	state = {
		'net': net.state_dict()
	}
	torch.save(state, save_path)


ratio = 0.1
def LSB_random(network='vgg16', model_path=MODEL_PATH, save_path='', n_lsb=n_lsb, ratio=ratio):
	print('==> Building original model..')

	if network == 'vgg16':
		net = VGG('VGG16')
	if network == 'resnet34':
		net = ResNet34()
	if network == 'efficientnet':
		net = EfficientNetB0()
	
	net = net.to(device)
	if device == 'cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True

	state_dict = torch.load(model_path, map_location=device)['net']
	net.load_state_dict(state_dict, strict=True)

	length = 0
	for key in net.state_dict().keys():
		if 'weight' in key and 'bn' not in key and 'shortcut' not in key:
			params = net.state_dict()[key]
			p = params.cpu()
			p = p.detach().numpy()
			pp = p.flatten()

			length += len(pp)

	s_length = int(ratio * length * n_lsb)
	s = np.random.randint(0,2,s_length)

	# order
	# begin = 0

	# random begin
	# begin = int(np.random.randint(0, int(length * (1 - ratio)), 1))

	ad = set()
	while(len(ad)<int(ratio * length)):
		ad.add(random.randint(0, int(length)))
	begin = min(ad)

	m = 0
	n = 0
	flag = 0
	flag_1 = 0
	sum_length = 0
	for key in net.state_dict().keys():
		if 'weight' in key and 'bn' not in key and 'shortcut' not in key:
			params = net.state_dict()[key]
			p = params.detach().numpy()
			pp = p.flatten()

			length_temp = len(pp)

			if flag == 0:
				sum_length += length_temp

			if sum_length >= begin:
				flag = 1

			if flag == 0:
				n += length_temp
			elif flag == 1:
				params1 = float2bit(pp, num_e_bits=8, num_m_bits=23, bias=127.)
				params2 = params1.numpy()

				# change
				# j = 0
				for i in range(length_temp):
					# if order or randbegin
					# if n == begin:
					#     params2[i][-n_lsb:] = s[m * n_lsb: (m + 1) * n_lsb]
					#     m += 1
					#     flag_1 = 1
					# elif flag_1 == 1:
					#     params2[i][-n_lsb:] = s[m * n_lsb: (m + 1) * n_lsb]
					#     if m + 1 == int(ratio * length):
					#         break
					#     m += 1
					# n += 1

					# random
					if n in ad:
						params2[i][-n_lsb:] = s[m * n_lsb: (m + 1) * n_lsb]
						if m + 1 == int(ratio * length):
							break
						m += 1
					n += 1

				#binary to float
				params3 = bit2float(params2)
				params4 = params3.reshape(p.shape)

				net.state_dict()[key].copy_(params4)
	print(m + 1)
	print(int(ratio * length))
	print('Saving LSB..')
	state = {
		'net': net.state_dict()
	}
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	torch.save(state, save_path)