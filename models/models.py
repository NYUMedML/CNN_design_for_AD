# -*- coding: utf-8 -*-
"""
@author: Sheng
"""
import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init
import math
import sys
sys.path.append('Utils')

class AgeEncoding(nn.Module):
    "Implement the AE function."
    def __init__(self, d_model, dropout, out_dim,max_len=240):
        super(AgeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model)).float())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(d_model,512))
        self.fc6.add_module('lrn0_s1',nn.LayerNorm(512))
        self.fc6.add_module('fc6_s3',nn.Linear(512, out_dim))
        
        
    def forward(self, x, age_id):
        y = torch.autograd.Variable(self.pe[age_id,:], 
                         requires_grad=False)
        y = self.fc6(y)

        x += y
        return self.dropout(x)
class AgeEncoding_simple(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, out_dim,max_len=240):
        super(AgeEncoding_simple, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.fc6 = nn.Linear(out_dim+1,out_dim)
        
        
    def forward(self, x, age_id):
        age_id = (age_id -0)/(240 - 0 + 1e-6)
        y = torch.cat([x.float(),age_id.unsqueeze(-1).float()],dim=1)
        x = self.fc6(y)

        return self.dropout(x)



class NetWork(nn.Module):

    def __init__(self, in_channel=1,feat_dim=1024,expansion = 4, type_name='conv3x3x3', norm_type = 'Instance'):
        super(NetWork, self).__init__()
        

        self.conv = nn.Sequential()

        self.conv.add_module('conv0_s1',nn.Conv3d(in_channel, 4*expansion, kernel_size=1))

        if norm_type == 'Instance':
           self.conv.add_module('lrn0_s1',nn.InstanceNorm3d(4*expansion))
        else:
           self.conv.add_module('lrn0_s1',nn.BatchNorm3d(4*expansion))
        self.conv.add_module('relu0_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool0_s1',nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('conv1_s1',nn.Conv3d(4*expansion, 32*expansion, kernel_size=3,padding=0, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn1_s1',nn.InstanceNorm3d(32*expansion))
        else:
            self.conv.add_module('lrn1_s1',nn.BatchNorm3d(32*expansion))
        self.conv.add_module('relu1_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool1_s1',nn.MaxPool3d(kernel_size=3, stride=2))


        
        
        self.conv.add_module('conv2_s1',nn.Conv3d(32*expansion, 64*expansion, kernel_size=5, padding=2, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn2_s1',nn.InstanceNorm3d(64*expansion))
        else:
            self.conv.add_module('lrn2_s1',nn.BatchNorm3d(64*expansion))
        self.conv.add_module('relu2_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1',nn.MaxPool3d(kernel_size=3, stride=2))

        
        self.conv.add_module('conv3_s1',nn.Conv3d(64*expansion, 64*expansion, kernel_size=3, padding=1, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn3_s1',nn.InstanceNorm3d(64*expansion))
        else:
            self.conv.add_module('lrn2_s1',nn.BatchNorm3d(64*expansion))
        self.conv.add_module('relu3_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1',nn.MaxPool3d(kernel_size=5, stride=2))

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(64*expansion*5*5*5, feat_dim))
        self.age_encoder = AgeEncoding(512,0.1,feat_dim)


    def load(self,checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)['state_dict']
        pretrained_dict = {k[6:]: v for k, v in list(pretrained_dict.items()) if k[6:] in model_dict and 'conv3_s1' not in k and 'fc6' not in k and 'fc7' not in k and 'fc8' not in k}

        model_dict.update(pretrained_dict)
        

        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])
        return pretrained_dict.keys()

    def freeze(self, pretrained_dict_keys):
        for name, param in self.named_parameters():
            if name in pretrained_dict_keys:
                param.requires_grad = False
                

    def save(self,checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def forward(self, x, age_id):

        z = self.conv(x)
        z = self.fc6(z.view(x.shape[0],-1))
        if age_id is not None:
            z = self.age_encoder(z,age_id)
        return z


def weights_init(model):
    if type(model) in [nn.Conv3d,nn.Linear]:
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0.1)