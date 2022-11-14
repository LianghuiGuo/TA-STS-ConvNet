# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 21:59:23 2022

@author: chongxian
"""
import torch.nn as nn
from EEG_model.multi_scale_spectral_temporal_Net import Multi_Scale_Spectral_Temporal_Net
from EEG_model.spatio_dynamic_graph_convolutional_net import Spatio_Dynamic_Graph_Convolutional_Net

  
class Classification_Net(nn.Module):
    '''
    classification net
    output the seeizure predicting results
    '''
    def __init__(self, group = 1):
        super(Classification_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 96*5, out_channels = 48, kernel_size = (1, 1), 
                               stride = 1, padding = (0, 0), bias = True)
        self.bn1 = nn.BatchNorm2d(48)
        self.conv2 = nn.Conv2d(in_channels = 48, out_channels = 2, kernel_size = (1, 1), 
                               stride = 1, padding = (0, 0), bias = True)
        self.ELU = nn.ELU(inplace=False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        output1 = self.conv1(x)
        output = self.conv2(output1)
        return output

class TA_STS_ConvNet(nn.Module):
    '''
    the TA-STS-ConvNet proposed in our paper "Triple-Attention-based Spatio-Temporal-Spectral Convolutional Network for Epileptic Seizure Prediction"
    github : https://github.com/LianghuiGuo/TA-STS-ConvNet
    paper :  https://www.techrxiv.org/articles/preprint/Triple-Attention-based_Spatio-Temporal-Spectral_Convolutional_Network_for_Epileptic_Seizure_Prediction/20557074
    '''
    def __init__(self, dim, inchan, sen ,device_number, position_embedding):
        super(TA_STS_ConvNet, self).__init__()
        self.dim = dim
        self.inchan = inchan # input channel number
        self.sen = sen # 
        self.device_number=device_number
        self.position_embedding = position_embedding # whether using position embedding
        self.MS_STN = Multi_Scale_Spectral_Temporal_Net(4, self.dim, self.inchan, 1)
        self.SA_GCN=Spatio_Dynamic_Graph_Convolutional_Net(self.dim, self.inchan, self.sen, self.device_number, self.position_embedding)
        self.MLP = Classification_Net()
        self.initialize()

    def initialize(self):
        '''
        model papameter initialization
        '''
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
                       
    def forward(self, x):#(128,1,19,1280)
        #Pyramid Convolution Net + Triple Attention Fusion Net
        feature_low = self.MS_STN(x, 0).contiguous()#([128, 480, 1, 19])
        #Spatio Embedded Net
        feature_high, Ms, Ms_D=self.SA_GCN(feature_low)#([480,1,1])
        #Classification Net
        pred = self.MLP(feature_high).squeeze()#([128, 2])
        return pred, [feature_low.squeeze(), feature_high.squeeze(), Ms, Ms_D]
