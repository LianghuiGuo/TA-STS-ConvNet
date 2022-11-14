# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 15:05:15 2021

@author: chongxian
"""

import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
from EEG_model.TripletAttention import TripletAttention
from EEG_utils.position_embedding import positionEmbedding_CHB
   
class Dynamic_Adjacent_Net(nn.Module):
    def __init__(self, channel, inchan, reduction = 16):
        super(Dynamic_Adjacent_Net, self).__init__()
        
        self.inchan = inchan
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(inplace=False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Tanh(),
            nn.PReLU())
    
    def forward(self, x):
        x = x.reshape(1, self.inchan*self.inchan)
        y = self.fc(x).reshape(self.inchan, self.inchan)
        return y
    
class interGConv(nn.Module):
    def __init__(self, conv_length, outc, inchan):
        super(interGConv, self).__init__()
        '''
        Gconv Gconv2表示图5(a)中的W1和W2
        '''
        self.inchan = inchan
        self.outc = outc
        self.GConv = nn.Conv2d(in_channels = conv_length, out_channels = outc, kernel_size = (1, 1), 
                               stride = (1, 1), padding = (0, 0), 
                                  groups = 5, bias = False)
        self.GConv2 = nn.Conv2d(in_channels = outc, out_channels = outc, kernel_size = (1, 1), 
                               stride = (1, 1), padding = (0, 0), 
                                  groups = 5, bias = False)

        self.bn1 = nn.BatchNorm2d(outc)
        self.bn2 = nn.BatchNorm2d(outc)
        self.ELU = nn.ELU(inplace = False)
        self.initialize()
        
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ModuleList):
                for j in m:
                   if isinstance(j, nn.Linear):
                       nn.init.xavier_uniform_(j.weight, gain=1)

    def forward(self, x, L):
        x = self.bn2(self.GConv2(self.ELU(self.bn1(self.GConv(x)))))#([128, 320, 1, 19])
        y = []
        for data, edge in zip(x.split(96, 1), L.split(1, 0)):#x、L分为5组
            y.append(torch.einsum('bijk,kp->bijp', (data, edge.squeeze())))#data  ([128, 64, 1, 19] edge.squeeze()([19, 19]])
        y = torch.cat(y, 1)#([128, 320, 1, 19])
        return y


class Spatio_Dynamic_Graph_Convolutional_Net(nn.Module):
    '''
    Spatio Dynamic Graph Convolutional Net
    '''
    def __init__(self, dim, inchan, reduction ,device_number, position_embedding):
        super(Spatio_Dynamic_Graph_Convolutional_Net, self).__init__()
        self.device="cuda:"+str(device_number)
        self.reduction = reduction #reduction ratio r
        self.inchan = inchan  #input channel number
        self.bn = nn.BatchNorm2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dim=dim
        self.position_embedding = position_embedding
        self.intraGConv = interGConv(conv_length = dim*5, 
                                     outc = dim*5,  
                                     inchan = inchan)
        self.ELU = nn.ELU(inplace=False)
        if self.position_embedding:
            self.A_init = torch.from_numpy(positionEmbedding_CHB(inchan).astype('float32')).to(self.device)
        else:
            self.A_init = torch.ones((self.inchan, self.inchan), dtype = torch.float32, requires_grad = False).to(self.device)

        self.DAN = Dynamic_Adjacent_Net(self.inchan*self.inchan, self.inchan, self.reduction)

        self.self_add01 = nn.Conv2d(in_channels = dim*5, out_channels = dim*5, kernel_size = (1, 1), 
                                                    stride = 1, padding = (0, 0), groups = 5, bias = False)
        self.bn0 = nn.BatchNorm2d(dim*5)
        self.attPool = TripletAttention()
        self.importance1 = nn.Linear(inchan, 1)
        self.importance2 = nn.Linear(inchan, 1)
        self.importance3 = nn.Linear(inchan, 1)
        self.importance4 = nn.Linear(inchan, 1)
        self.importance5 = nn.Linear(inchan, 1)
        self.initialize()

        
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for j in m:
                   if isinstance(j, nn.Linear):
                       nn.init.xavier_uniform_(j.weight, gain=1)
                       
    def forward(self, feature):
        s1, s2, s3, s4 = feature.size()
        
        #adjacent matrix Ms from DAN
        Ms = self.DAN(self.A_init)
        
        #adjacent matrix A multiplied by degree matrix
        Ms_D = torch.einsum('ik,kp->ip', (Ms, torch.diag(torch.reciprocal(sum(Ms)))))
        
        #Dynamic Graph convolution
        Graph = self.intraGConv(feature, torch.stack([Ms_D, Ms_D, Ms_D, Ms_D, Ms_D])).contiguous()
        Graph = self.ELU(torch.add(Graph, feature))

        #rhythm Attentional Pooling
        A, B, C, D, E = Graph.split(self.dim, 1)
        Graph = torch.cat((A, B, C, D, E), 2)
        Graph = self.attPool(Graph)
        A, B, C, D, E = Graph[:,:,0,:], Graph[:,:,1,:], Graph[:,:,2,:], Graph[:,:,3,:], Graph[:,:,4,:]
        y = torch.cat((self.importance1(A.view(A.size(0), A.size(1), -1)).unsqueeze(-1).contiguous(),
            self.importance2(B.view(B.size(0), B.size(1), -1)).unsqueeze(-1).contiguous(),
            self.importance3(C.view(C.size(0), C.size(1), -1)).unsqueeze(-1).contiguous(),
            self.importance4(D.view(D.size(0), D.size(1), -1)).unsqueeze(-1).contiguous(),
            self.importance5(E.view(E.size(0), E.size(1), -1)).unsqueeze(-1).contiguous()), 1)

        return y, Ms, Ms_D