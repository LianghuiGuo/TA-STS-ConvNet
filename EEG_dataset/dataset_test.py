# -*- coding: utf-8 -*-
"""
Created on Thu May 27 17:37:55 2021

@author: phantom

"""

from __future__ import print_function
import torch
from torch.utils.data import Dataset
import numpy as np
from EEG_utils.eeg_utils import GetDataPath, GetDataType

class testDataset(Dataset):
    def __init__(self, dataset_name="XUANWU", i=0, using_ictal=1, patient_id=None, patient_name=None, ch_num=1, target_preictal_interval=15, step_preictal=5):
        data, label = [], []
        self.data_path=GetDataPath(dataset_name) #data path
        self.data_type=GetDataType(dataset_name) #EEG/IEEG
        self.patient_id=patient_id #patient id
        self.patient_name=patient_name #patient name
        self.ch_num=ch_num #number of channels
        self.target_preictal_interval=target_preictal_interval #how long we decide as preictal. Default set to 15 min
        self.step_preictal=step_preictal  #step of sliding window 

        #data loading
        preIctal = np.load(("%s/%s/%dmin_%dstep_%dch/preictal%d.npy" % (self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
        interIctal = np.load(("%s/%s/%dmin_%dstep_%dch/interictal%d.npy" % (self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
        Ictal = np.load(("%s/%s/%dmin_%dstep_%dch/ictal%d.npy" % (self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            
        #shape transpose
        self.preictal_length=preIctal.shape[0]
        self.interictal_length=interIctal.shape[0]
        self.ictal_length=Ictal.shape[0]
        if (len(preIctal.shape)==3):
            preIctal = preIctal.transpose(0, 2, 1)#(180, 19, 1280)
            Ictal = Ictal.transpose(0, 2, 1)#(38, 19, 1280)
            interIctal = interIctal.transpose(0, 2, 1)
        
        #whether to use ictal data for testing
        if using_ictal == 1 and Ictal.shape[0]!=0:
            preIctal = np.concatenate((preIctal, Ictal), 0)
        
        #data concat
        data.append(interIctal)
        data.append(preIctal)
        label.append(np.zeros((interIctal.shape[0], 1)))
        label.append(np.ones((preIctal.shape[0], 1)))
            
        #numpy to torch
        data, label = np.array(data), np.array(label)
        data, label = np.concatenate(data, 0), np.concatenate(label, 0) 
        if (len(preIctal.shape)==3):
            data = data[:, np.newaxis, :, :].astype('float32')
        elif (len(preIctal.shape)==4):#spectralCNN
            data = data[:, np.newaxis, :, :, :].astype('float32')
        label = label.astype('int64')
        self.x_data = torch.from_numpy(data)#([2592, 1, 19, 1280])
        self.y_data = torch.from_numpy(label)#([2592, 1])
        self.len = data.shape[0]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
