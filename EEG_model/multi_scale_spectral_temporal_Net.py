# -*- coding: utf-8 -*-
"""
spectral temporal net

Created on Wed Jan  5 20:33:07 2022

@author: chongxian
"""

import torch.nn as nn
import torch
import math
from scipy import io as io
import numpy as np
from EEG_model.TripletAttention import TripletAttention

class down_sample(nn.Module):
    def __init__(self, inc, kernel_size, stride, padding):
        super(down_sample, self).__init__()
        self.conv = nn.Conv2d(in_channels = inc, out_channels = inc, kernel_size = (1, kernel_size), stride = (1, stride), padding = (0, padding), bias = False)
        self.bn = nn.BatchNorm2d(inc) 
        self.elu = nn.ELU(inplace = False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        output = self.elu(self.bn(self.conv(x)))
        return output

class conv_sample(nn.Module):
    def __init__(self, inc=9, kernel_size=11, stride=2, padding=5):
        super(conv_sample, self).__init__()
        self.conv1 = down_sample(inc, kernel_size, stride, padding)
        self.conv2 = down_sample(inc, kernel_size, stride, padding)
        self.conv3 = down_sample(inc, kernel_size, stride, padding)
        self.conv4 = down_sample(inc, kernel_size, stride, padding)
        self.conv5_1 = down_sample(inc, kernel_size, stride, padding)
        self.conv5_2 = down_sample(inc, kernel_size, stride, padding)
    def forward(self, x):
        x= self.conv1(x)
        convsample_gamma_x= self.conv2(x)
        convsample_beta_x= self.conv3(convsample_gamma_x)
        convsample_alpha_x= self.conv4(convsample_beta_x)
        convsample_theta_x= self.conv5_1(convsample_alpha_x)
        convsample_delta_x= self.conv5_2(convsample_alpha_x)
        return convsample_gamma_x, convsample_beta_x, convsample_alpha_x, convsample_theta_x, convsample_delta_x
    
class input_layer(nn.Module):
    def __init__(self, outc):
        super(input_layer, self).__init__()
        self.conv_input = nn.Conv2d(in_channels = 1, out_channels = outc, kernel_size = (1, 3), 
                                    stride = 1, padding = (0, 1), groups = 1, bias = False)
        self.bn_input = nn.BatchNorm2d(outc) 
        self.elu = nn.ELU(inplace = False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        output = self.bn_input(self.conv_input(x))
        return output

class Residual_Block(nn.Module): 
    def __init__(self, inc, outc, groups = 1):
        super(Residual_Block, self).__init__()
        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels = inc, out_channels = outc, kernel_size = 1, 
                                       stride = 1, padding = 0, groups = groups, bias = False)
        else:
          self.conv_expand = None
          
        self.conv1 = nn.Conv2d(in_channels = inc, out_channels = outc, kernel_size = (1, 3), 
                               stride = 1, padding = (0, 1), groups = groups, bias = False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(in_channels = outc, out_channels = outc, kernel_size = (1, 3), 
                               stride = 1, padding = (0, 1), groups = groups, bias = False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.elu = nn.ELU(inplace = False)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x): 
        if self.conv_expand is not None:
          identity_data = self.conv_expand(x)
        else:
          identity_data = x
        output = self.bn1(self.conv1(x))
        output = self.bn2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output 

def embedding_network(input_block, Residual_Block, num_of_layer, outc, groups = 1):
    layers = []
    layers.append(input_block(outc))
    for i in range(0, num_of_layer):
        layers.append(Residual_Block(inc = int(math.pow(2, i)*outc), outc = int(math.pow(2, i+1)*outc),
                                     groups = groups))
    return nn.Sequential(*layers) 

def self_padding(x):
    return torch.cat((x[:, :, :, -3:], x, x[:, :, :, 0:3]), 3)

class WaveletTransform(nn.Module): 
    '''
    waveConv layer
    '''
    def __init__(self, inc, params_path='./EEG_model/scaling_filter.mat', transpose = True):
        super(WaveletTransform, self).__init__()
        self.transpose = transpose        
        self.conv = nn.Conv2d(in_channels = inc, out_channels = inc*2, kernel_size = (1, 8), 
                              stride = (1, 2), padding = 0, groups = inc, bias = False)        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = io.loadmat(params_path)
                Lo_D, Hi_D = np.flip(f['Lo_D'], axis = 1).astype('float32'), np.flip(f['Hi_D'], axis = 1).astype('float32')
                m.weight.data = torch.from_numpy(np.concatenate((Lo_D, Hi_D), axis = 0)).unsqueeze(1).unsqueeze(1).repeat(inc, 1, 1, 1)            
                m.weight.requires_grad = False  
                           
    def forward(self, x): 
        out = self.conv(self_padding(x)) 
        return out[:, 0::2,:, :], out[:, 1::2, :, :]   #L, H

class Triblock(nn.Module):
    '''
    Triple Attention Fufion net
    '''
    def __init__(self, inc, outc, kernel_size, reduction = 8):
        super(Triblock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels = inc, out_channels = outc, kernel_size = (1, kernel_size), 
                               stride = (1, 1), padding = (0, kernel_size//2), 
                                 groups = 3, bias = False)
        self.bn0 = nn.BatchNorm2d(outc)        
        self.triAtt0 = TripletAttention()    
        self.conv1 = nn.Conv2d(in_channels = outc, out_channels = outc, kernel_size = (1, kernel_size), 
                               stride = (1, 1), padding = (0, kernel_size//2), 
                                  groups = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(outc)        
        self.triAtt1 = TripletAttention()    
        self.conv2 = nn.Conv2d(in_channels = outc, out_channels = outc, kernel_size = (1, kernel_size), 
                               stride = (1, 1), padding = (0, kernel_size//2), 
                                  groups = 3, bias = False)
        self.bn2 = nn.BatchNorm2d(outc)        
        self.triAtt2= TripletAttention()    
        self.elu = nn.ELU(inplace = False)
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

    def forward(self, x, flag):
        if flag == 0:
            out = self.elu(self.triAtt0(self.bn0(self.conv0(x))))
            out = self.elu(self.triAtt1(self.bn1(self.conv1(out))))
            out = self.elu(self.triAtt2(self.bn2(self.conv2(out))))
        else:
            out = self.dropout(self.elu(self.bn0(self.conv0(x))))
            out = self.dropout(self.elu(self.bn1(self.conv1(out))))
        return out
    
class Multi_Scale_Spectral_Temporal_Net(nn.Module):
    '''
    Pyramid Convolution Net + Triple Attention Fusion Net
    '''
    def __init__(self, outc, fc, inchan, num_of_layer = 1):
        super(Multi_Scale_Spectral_Temporal_Net, self).__init__() 
        self.num_of_layer = num_of_layer
        self.embedding = embedding_network(input_layer, Residual_Block, num_of_layer = num_of_layer, outc = outc) 
        
        #spectral pyramid
        self.WaveletTransform = WaveletTransform(inc = outc*int(math.pow(2, num_of_layer)) + 1)        

        #temporal pyramid
        self.downsampled_gamma = down_sample(outc*int(math.pow(2, num_of_layer))+1, 4, 4, 0)
        self.downsampled_beta = down_sample(outc*int(math.pow(2, num_of_layer))+1, 8, 8, 0)
        self.downsampled_alpha = down_sample(outc*int(math.pow(2, num_of_layer))+1, 16, 16, 0)
        self.downsampled_theta = down_sample(outc*int(math.pow(2, num_of_layer))+1, 32, 32, 0)
        self.downsampled_delta = down_sample(outc*int(math.pow(2, num_of_layer))+1, 32, 32, 0)
        
        self.conv_sample=conv_sample(outc*int(math.pow(2, num_of_layer))+1, 11, 2, 5)
        
        #triple attention fusion net
        self.TriGamma = Triblock((outc*int(math.pow(2, num_of_layer))+1)*3, fc//2, 7)
        self.TriBeta = Triblock((outc*int(math.pow(2, num_of_layer))+1)*3, fc//2, 7)
        self.TriAlpha = Triblock((outc*int(math.pow(2, num_of_layer))+1)*3, fc//2, 3)
        self.TriDelta = Triblock((outc*int(math.pow(2, num_of_layer))+1)*3, fc//2, 3)
        self.TriTheta = Triblock((outc*int(math.pow(2, num_of_layer))+1)*3, fc//2, 3)
        self.average_pooling = nn.AdaptiveAvgPool2d((inchan, 1))

        self.SEA = nn.Sequential(nn.Conv2d(in_channels = fc//2, out_channels = fc, kernel_size = (1, 1), 
                               stride = (1, 1), padding = (0, 0), 
                                 groups = 1, bias = False),
                                nn.BatchNorm2d(fc),
                                nn.ELU(inplace=False))
        self.SEB = nn.Sequential(nn.Conv2d(in_channels = fc//2, out_channels = fc, kernel_size = (1, 1), 
                               stride = (1, 1), padding = (0, 0), 
                                 groups = 1, bias = False),
                                nn.BatchNorm2d(fc),
                                nn.ELU(inplace=False))
        self.SED = nn.Sequential(nn.Conv2d(in_channels = fc//2, out_channels = fc, kernel_size = (1, 1), 
                               stride = (1, 1), padding = (0, 0), 
                                 groups = 1, bias = False),
                                nn.BatchNorm2d(fc),
                                nn.ELU(inplace=False))
        self.SET = nn.Sequential(nn.Conv2d(in_channels = fc//2, out_channels = fc, kernel_size = (1, 1), 
                               stride = (1, 1), padding = (0, 0), 
                                 groups = 1, bias = False),
                                nn.BatchNorm2d(fc),
                                nn.ELU(inplace=False))
        self.SEG = nn.Sequential(nn.Conv2d(in_channels = fc//2, out_channels = fc, kernel_size = (1, 1), 
                               stride = (1, 1), padding = (0, 0), 
                                 groups = 1, bias = False),
                                nn.BatchNorm2d(fc),
                                nn.ELU(inplace=False))

    def forward(self, x, flag):
        embedding_x = self.embedding(x)
        cat_x = torch.cat((embedding_x, x), 1)
        
        #spectral pyramid
        out, _ = self.WaveletTransform(cat_x)
        out, gamma = self.WaveletTransform(out)
        out, beta = self.WaveletTransform(out)
        out, alpha = self.WaveletTransform(out)
        delta, theta = self.WaveletTransform(out)
        
        #temporal pyramid
        downsample_gamma_x = self.downsampled_gamma(cat_x)
        downsample_beta_x = self.downsampled_beta(cat_x)
        downsample_alpha_x = self.downsampled_alpha(cat_x)
        downsample_theta_x = self.downsampled_theta(cat_x)
        downsample_delta_x = self.downsampled_delta(cat_x)
        convsample_gamma_x, convsample_beta_x, convsample_alpha_x, convsample_theta_x, convsample_delta_x = self.conv_sample(cat_x)

        #concatenate temporal and spectral features
        gamma = torch.cat((downsample_gamma_x, convsample_gamma_x, gamma), 1)
        beta = torch.cat((downsample_beta_x, convsample_beta_x, beta), 1)  
        alpha = torch.cat((downsample_alpha_x, convsample_alpha_x, alpha), 1)
        theta = torch.cat((downsample_theta_x, convsample_theta_x, theta), 1)
        delta = torch.cat((downsample_delta_x, convsample_delta_x, delta), 1)
        
        #Triple Attention Fusion Net
        x1, x2, x3, x4, x5 = self.TriGamma(gamma, flag), self.TriBeta(beta, flag), self.TriAlpha(alpha, flag), self.TriTheta(theta, flag), self.TriDelta(delta, flag)
        x1, x2, x3, x4, x5 = self.average_pooling(x1), self.average_pooling(x2), self.average_pooling(x3), self.average_pooling(x4), self.average_pooling(x5) 
        x1, x2, x3, x4, x5 = self.SEG(x1), self.SEB(x2), self.SEA(x3), self.SET(x4), self.SED(x5)
        
        return torch.cat((x1, x2, x3, x4, x5), 1).permute(0, 1, 3, 2).contiguous()
        