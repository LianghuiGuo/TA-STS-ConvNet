# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 21:38:04 2021

@author: chongxian
"""

import torch.nn as nn
import torch

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class AttentionGate(nn.Module):
    def __init__(self, kernel_size=7):
        super(AttentionGate, self).__init__()
        self.compress = ZPool()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1) // 2),
            nn.BatchNorm2d(1)
        )
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale

class TripletAttention(nn.Module):
    '''
    Triple Attention Layer
    '''
    def __init__(self):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.hw = AttentionGate()
    def forward(self, x):#(b,c,h,w)
        #height wise
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        
        #width wise
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        
        #channel wise
        x_out = self.hw(x)
        x_out = 1/3 * (x_out + x_out11 + x_out21)
        return x_out