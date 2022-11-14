# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:57:04 2022

@author: chongxian
"""

import mne
import numpy as np
from matplotlib import pyplot as plt


# calculate distance
def Distance(vector1, vector2):
    return(np.sqrt(np.sum(np.square(vector1 - vector2))))

def positionEmbedding_CHB(inchan):
    #channel names in CHB-MIT
    CHB_ch = ['AF7', 'FT7', 'TP7', 'PO7', 'AF3', 'FC3',   #channel name in international 10-20 system
            'CP3', 'PO3', 'FCz', 'CPz', 'AF4', 'FC4',
            'CP4', 'PO4', 'AF8', 'FT8', 'TP8', 'PO8']
    
    # get 10-20 system
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    # print(ten_twenty_montage)
    
    # get distance matrix of 10-20 system
    ch_list = ten_twenty_montage.ch_names
    dis_list = ten_twenty_montage.dig
    dis_matrix = []
    for ch_name in CHB_ch:
        index = ch_list.index(ch_name)
        dis_matrix.append(dis_list[index]['r'])
    
    dis_matrix = np.array(dis_matrix)#(18, 3)
    
    # get distance set L
    L = [] #len(L)=18*17/2=153
    L_matrix = np.zeros((inchan, inchan))
    for i in range(0, dis_matrix.shape[0]):
        for j in range(i+1, dis_matrix.shape[0]):
            dis = Distance(dis_matrix[i, :], dis_matrix[j, :])
            L.append(dis)
            L_matrix[i][j]=dis
            L_matrix[j][i]=dis
    mean_dis = np.mean(L)
    
    # get adjacent matrix A
    A = np.zeros((inchan, inchan))
    for i in range(0, inchan):
        for j in range(0, inchan):
            if i == j:
                neibor_dis = []
                for k in L_matrix[i,:]:
                    if k != 0 and k <mean_dis:
                        neibor_dis.append(k)
                A[i][j] = 1/np.mean(neibor_dis)
            elif L_matrix[i][j] < mean_dis:
                A[i][j] = 1/L_matrix[i][j]
            elif L_matrix[i][j] >= mean_dis:
                A[i][j] = 0
    return A
