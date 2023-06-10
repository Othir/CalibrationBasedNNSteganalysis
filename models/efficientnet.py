'''EfficientNet in PyTorch.

Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks".

Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * x.sigmoid()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels,
                             kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels,
                             kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio=1,
                 se_ratio=0.,
                 drop_rate=0.):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=channels,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2d(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(EfficientNet, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(3,
                               32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(cfg['out_channels'][-1], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size',
                                     'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    Block(in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = swish(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        dropout_rate = self.cfg['dropout_rate']
        if self.training and dropout_rate > 0:
            out = F.dropout(out, p=dropout_rate)
        out = self.linear(out)
        return out


def EfficientNetB0():
    cfg = {
        'num_blocks': [1, 2, 2, 3, 3, 4, 1],
        'expansion': [1, 6, 6, 6, 6, 6, 6],
        'out_channels': [16, 24, 40, 80, 112, 192, 320],
        'kernel_size': [3, 3, 5, 3, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 1],
        'dropout_rate': 0.2,
        'drop_connect_rate': 0.2,
    }
    return EfficientNet(cfg)


def EfficientNetB5():
    cfg = {
        'num_blocks': [1, 2, 2, 3, 3, 4, 1],
        'expansion': [1, 6, 6, 6, 6, 6, 6],
        'out_channels': [16, 24, 40, 80, 112, 192, 320],
        'kernel_size': [3, 3, 5, 3, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 1],
        'dropout_rate': 0.4,
        'drop_connect_rate': 0.2,
    }
    return EfficientNet(cfg)


def test():
    net = EfficientNetB0()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.shape)


if __name__ == '__main__':
    test()



'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from functools import partial
import torch.utils.model_zoo as model_zoo



# __all__ = ['EfficientNet', 'efficientnetB0','efficientnetB1', 'efficientnetB2', 'efficientnetB3', 'efficientnetB4', 'efficientnetB5', 'efficientnetB6', 'efficientnetB7']

class Swish(nn.Module):
    def forward(self, x):
        x = x * torch.sigmoid(x)  #nn.functional.sigmoid is deprecated, use torch.sigmoid instead
        return x

act_fn = Swish() #nn.ReLU(inplace=True)


#from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
#added groups, needed for DWConv
#"The configuration when groups == in_channels and out_channels = K * in_channels where K is a positive integer is termed in literature as depthwise convolution."


# gotta pick one of the returns for 'same' padding or a simpler ks//2 padding
def conv(ni, nf, ks=3, stride=1, groups=1, bias=False):
    #return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, groups= groups, bias=bias)
    return Conv2dSamePadding(ni, nf, kernel_size=ks, stride=stride, groups= groups, bias=bias)


#class noop(nn.Module):
  #  def __init__(self):
  #      super().__init__()
   # def forward(self,x): return x
    
def noop(x): return x

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)


# not compatible with fp16 training        
class Drop_Connect(nn.Module):
    """create a tensor mask and apply to inputs, for removing drop_ratio % of weights"""
    def __init__(self, drop_ratio=0):
        super().__init__()
        self.keep_percent = 1.0 - drop_ratio

    def forward(self, x):
        if not self.training:
            return x

        batch_size = x.size(0)
        random_tensor = self.keep_percent
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype,device=x.device)   #dtype is causing issues with fp16 training
        binary_tensor = torch.floor(random_tensor)
        output = x / self.keep_percent * binary_tensor

        return output
    
    
def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype,device=inputs.device)  # uniform [0,1)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


#added groups, needed for DWConv
#fixed batch norm momentum = 1- Tensorflow value
def conv_layer(ni, nf, ks=3, stride=1,groups=1, zero_bn=False, act=True, eps=1e-03, momentum=0.01):
    bn = nn.BatchNorm2d(nf, eps=eps, momentum=momentum)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, nf, ks, stride=stride, groups=groups), bn]
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)



class SqueezeEx(nn.Module):
    def __init__(self, ni, ns):
        super().__init__()
        

        ns = max(1, int(ns))
        
        layers = [nn.AdaptiveAvgPool2d(1),
                      conv(ni,ns,ks=1,bias=True),
                      act_fn,
                      conv(ns,ni,ks=1,bias=True),
                      nn.Sigmoid()]


        self.layers = nn.Sequential(*layers)

    def forward(self, x):
          
        return x * self.layers(x)




class MBConv(nn.Module):
    def __init__(self, ni, nf, expand_ratio, ks=3, stride=2, se = None, skip=True, drop_connect_rate=None):
        super().__init__()



        self.drop_connect_rate = drop_connect_rate
        # Expansion (only if expand ratio>1)

        ne = ni*expand_ratio
        self.conv_exp = noop if ni==ne else conv_layer(ni, ne, ks=1)

        # Depthwise Convolution (implemented using 'groups')
        # This is where ks and stride get used
        #"The configuration when groups == in_channels and out_channels = K * in_channels 
        # where K is a positive integer is termed in literature as depthwise convolution."
        # depth_multiplier=1 is default in original TF code so we keep the same number of channels

        self.dw_conv = conv_layer(ne, ne, ks=ks, stride= stride, groups=ne)


        # Squeeze and Excitation (if se ratio is specified)
        # se ratio applies to ni and not ne


        self.se = SqueezeEx(ne, ni*se) if se else noop

        # Output Conv (no relu)

        self.conv_out = conv_layer(ne, nf, ks=1, act=False)

        

        # add skip connection or not
        self.skip = skip and stride==1 and ni==nf

        # Drop connect

        #self.dc = Drop_Connect(drop_connect_rate) if drop_connect_rate else noop
        


    def forward(self, x): 
        
        self.dc = partial(drop_connect,p=self.drop_connect_rate, training=self.training) if self.drop_connect_rate else noop
        
        out = self.conv_out(self.se(self.dw_conv(self.conv_exp(x))))
        if self.skip: out = self.dc(out) + x


        return out



class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class EfficientNet(nn.Sequential):
    def __init__(self, channels, repeat, ks, stride, expand, w_mult=1.0, d_mult=1.0, se = None, drop_connect_rate = None,dropout_rate= None, c_in=3, c_out=10):

        
        repeat = [int(math.ceil(r*d_mult)) for r in repeat]
        channels = round_filters(channels, w_mult)
        
        
        stem = [conv_layer(c_in, channels[0], ks=3 ,stride=2)]

        blocks = []
        #The first block needs to take care of stride and filter size increase.

        for i in range(len(repeat)):
            blocks+= [MBConv(channels[i], channels[i+1], expand[i], ks=ks[i], stride=stride[i], se = se, drop_connect_rate=drop_connect_rate)]
            blocks+= [MBConv(channels[i+1], channels[i+1], expand[i], ks=ks[i], stride=1, se = se, drop_connect_rate=drop_connect_rate)] *(repeat[i]-1)

        dropout = nn.Dropout(p=dropout_rate) if dropout_rate else noop

        head = [conv_layer(channels[-2], channels[-1], ks=1 ,stride=1), nn.AdaptiveAvgPool2d(1), Flatten(), dropout, nn.Linear(channels[-1], c_out)]


        super().__init__(*stem,*blocks, *head)

        init_cnn(self)


        
        
def round_filters(filters, d_mult, divisor=8, min_depth=None):
    """ Calculate and round number of filters based on depth multiplier. """
    
    if not d_mult:
        return filters
    
    filters = [f*d_mult for f in filters]
    min_depth = min_depth or divisor
    new_filters = [max(min_depth, int(f + divisor / 2) // divisor * divisor) for f in filters]
    # prevent rounding by more than 10%
    new_filters = [new_filters[i] + (new_filters[i] < 0.9 * filters[i])* divisor for i in range(len(new_filters))]
    new_filters = [int(f) for f in new_filters]
    return new_filters



def EfficientNetB0():
    c = [32,16,24,40,80,112,192,320,1280]
    r = [1,2,2,3,3,4,1]
    ks = [3,3,5,3,5,5,3]
    _str = [1,2,2,2,1,2,1]
    exp = [1,6,6,6,6,6,6]
    se = 0.25
    dc=0.2
    wm, dm, do = 1.0, 1.0, 0.2
    return EfficientNet(d_mult=dm, w_mult=wm, dropout_rate=do, channels=c, repeat=r, ks=ks, stride=_str, expand=exp, se=se, drop_connect_rate=dc)


def EfficientNetB1():
    c = [32,16,24,40,80,112,192,320,1280]
    r = [1,2,2,3,3,4,1]
    ks = [3,3,5,3,5,5,3]
    _str = [1,2,2,2,1,2,1]
    exp = [1,6,6,6,6,6,6]
    se = 0.25
    dc=0.2
    wm, dm, do = 1.0, 1.1, 0.2
    return EfficientNet(d_mult=dm, w_mult=wm, dropout_rate=do, channels=c, repeat=r, ks=ks, stride=_str, expand=exp, se=se, drop_connect_rate=dc)


def EfficientNetB2():
    c = [32,16,24,40,80,112,192,320,1280]
    r = [1,2,2,3,3,4,1]
    ks = [3,3,5,3,5,5,3]
    _str = [1,2,2,2,1,2,1]
    exp = [1,6,6,6,6,6,6]
    se = 0.25
    dc=0.2
    wm, dm, do = 1.1, 1.2, 0.3
    return EfficientNet(d_mult=dm, w_mult=wm, dropout_rate=do, channels=c, repeat=r, ks=ks, stride=_str, expand=exp, se=se, drop_connect_rate=dc)


def EfficientNetB3():
    c = [32,16,24,40,80,112,192,320,1280]
    r = [1,2,2,3,3,4,1]
    ks = [3,3,5,3,5,5,3]
    _str = [1,2,2,2,1,2,1]
    exp = [1,6,6,6,6,6,6]
    se = 0.25
    dc=0.2
    wm, dm, do = 1.2, 1.4, 0.3
    return EfficientNet(d_mult=dm, w_mult=wm, dropout_rate=do, channels=c, repeat=r, ks=ks, stride=_str, expand=exp, se=se, drop_connect_rate=dc)


def EfficientNetB4():
    c = [32,16,24,40,80,112,192,320,1280]
    r = [1,2,2,3,3,4,1]
    ks = [3,3,5,3,5,5,3]
    _str = [1,2,2,2,1,2,1]
    exp = [1,6,6,6,6,6,6]
    se = 0.25
    dc=0.2
    wm, dm, do = 1.4, 1.8, 0.4
    return EfficientNet(d_mult=dm, w_mult=wm, dropout_rate=do, channels=c, repeat=r, ks=ks, stride=_str, expand=exp, se=se, drop_connect_rate=dc)


def EfficientNetB5():
    c = [32,16,24,40,80,112,192,320,1280]
    r = [1,2,2,3,3,4,1]
    ks = [3,3,5,3,5,5,3]
    _str = [1,2,2,2,1,2,1]
    exp = [1,6,6,6,6,6,6]
    se = 0.25
    dc=0.2
    wm, dm, do = 1.6, 2.2, 0.4
    return EfficientNet(d_mult=dm, w_mult=wm, dropout_rate=do, channels=c, repeat=r, ks=ks, stride=_str, expand=exp, se=se, drop_connect_rate=dc)


def EfficientNetB6():
    c = [32,16,24,40,80,112,192,320,1280]
    r = [1,2,2,3,3,4,1]
    ks = [3,3,5,3,5,5,3]
    _str = [1,2,2,2,1,2,1]
    exp = [1,6,6,6,6,6,6]
    se = 0.25
    dc=0.2
    wm, dm, do = 1.8, 2.6, 0.5
    return EfficientNet(d_mult=dm, w_mult=wm, dropout_rate=do, channels=c, repeat=r, ks=ks, stride=_str, expand=exp, se=se, drop_connect_rate=dc)


def EfficientNetB7():
    c = [32,16,24,40,80,112,192,320,1280]
    r = [1,2,2,3,3,4,1]
    ks = [3,3,5,3,5,5,3]
    _str = [1,2,2,2,1,2,1]
    exp = [1,6,6,6,6,6,6]
    se = 0.25
    dc=0.2
    wm, dm, do = 2.0, 3.1, 0.5
    return EfficientNet(d_mult=dm, w_mult=wm, dropout_rate=do, channels=c, repeat=r, ks=ks, stride=_str, expand=exp, se=se, drop_connect_rate=dc)



# if __name__ == '__main__':
# me = sys.modules[__name__]
# print(me)
# c = [32,16,24,40,80,112,192,320,1280]
# r = [1,2,2,3,3,4,1]
# ks = [3,3,5,3,5,5,3]
# _str = [1,2,2,2,1,2,1]
# exp = [1,6,6,6,6,6,6]
# se = 0.25
# do = 0.2
# dc=0.2


# # base without multipliers and dropout
# # setattr(me, 'efficientnet', partial(EfficientNet, channels=c, repeat=r, ks=ks, stride=_str, expand=exp, se=se, drop_connect_rate=dc))
# # me = partial(EfficientNet, channels=c, repeat=r, ks=ks, stride=_str, expand=exp, se=se, drop_connect_rate=dc)
# # print(me())
# # print(type(me))

# net = EfficientNet(channels=c, repeat=r, ks=ks, stride=_str, expand=exp, se=se, dropout_rate=do, drop_connect_rate=dc)
# x = torch.randn(2, 3, 32, 32)
# y = net(x)
# print(y.shape)

    # # (number, width_coefficient, depth_coefficient, dropout_rate) 
    # for n, wm, dm, do in [
    #     [ 0, 1.0, 1.0, 0.2],
    #     [ 1, 1.0, 1.1, 0.2],
    #     [ 2, 1.1, 1.2, 0.3],
    #     [ 3, 1.2, 1.4, 0.3],
    #     [ 4, 1.4, 1.8, 0.4],
    #     [ 5, 1.6, 2.2, 0.4],
    #     [ 6, 1.8, 2.6, 0.5],
    #     [ 7, 2.0, 3.1, 0.5],
    # ]:
    #     name = f'efficientnetB{n}'
    #     setattr(me, name, partial(efficientnet, d_mult=dm, w_mult=wm, dropout_rate=do))

def test():
    net = EfficientNetB5()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.shape)


# test()
'''