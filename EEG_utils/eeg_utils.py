# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 21:32:05 2021

@author: chongxian
"""

def GetPatientList(dataset):
    """
    Get Patient Name List
    """
    patient_list=[]
    if dataset=="XUANWU":
        patient_list={'1': 'hai dong sheng',
                      '2': 'hu jing',
                      '3': 'liu ru yue',
                      '4': 'luo yang yang',
                      '5': 'su ming'}
    elif dataset=="CHB":
        patient_list={'1' : 'chb01', 
                      '2' : 'chb02',
                      '3' : 'chb03',
                      '5' : 'chb05',
                      '6' : 'chb06',
                      '7' : 'chb07',
                      '8' : 'chb08',
                      '9' : 'chb09',
                      '10': 'chb10',
                      '11': 'chb11',
                      '13': 'chb13',
                      '14': 'chb14',
                      '16': 'chb16',
                      '17': 'chb17',
                      '18': 'chb18',
                      '20': 'chb20',
                      '21': 'chb21',
                      '22': 'chb22',
                      '23': 'chb23'}
    else:
        print("\nplease input correct dataset name\n")
        exit()
    return patient_list

def GetSeizureList(dataset):
    """
    Get Seizure List
    """
    seizure_list=[]
    if dataset=="XUANWU":
        seizure_list={'1': [0,1,2,3], 
                      '2': [0,1,2],
                      '3': [0,1,2],
                      '4': [0,1,2,3],
                      '5': [0,1,2],
                      '6': [0,1,2],
                      '7': [0,1],
                      '8': [0,1,2,3,4,5,6,7,8]}
    elif dataset=="CHB":
        seizure_list={'1' : [0,1,2,3,4,5,6], 
    		          '2' : [0,1,2],
    		          '3' : [0,1,2,3,4,5],
    		          '5' : [0,1,2,3,4],
    		          '6' : [0,1,2,3,4,5,6],
    		          '7' : [0,1,2],
    		          '8' : [0,1,2,3,4],
    		          '9' : [0,1,2,3],
                      '10': [0,1,2,3,4,5],
    		          '11': [0,1,2],
    		          '13': [0,1,2,3,4],
    				  '14': [0,1,2,3,4,5],
                      '16': [0,1,2,3,4,5,6,7],
    				  '17': [0,1,2],
    				  '18': [0,1,2,3,4,5],
    				  '20': [0,1,2,3,4,5,6,7],
    		          '21': [0,1,2,3],
    		          '22': [0,1,2],
    				  '23': [0,1,2,3,4,5,6],}
    else:
        print("\nplease input correct dataset name\n")
        exit()
    return seizure_list

def GetDataPath(dataset):
    """
    Get Data Path
    """
    data_path=""
    if dataset=="XUANWU":
        data_path=""
    elif dataset=="CHB":
        data_path="/home/al/GLH/code/seizure_predicting_seeg/code_public/data/CHB-MIT"
    else:
        print("\nplease input correct dataset name\n")
        exit()
    return data_path

def GetDataType(dataset):
    """
    Get Data Type
    """
    data_type=""
    if dataset=="XUANWU":
        data_type="SEEG"
    elif dataset=="CHB":
        data_type="EEG"
    else:
        print("\nplease input correct dataset name\n")
        exit()
    return data_type

def GetInputChannel(dataset, patient_id, ch_num):
    '''
    Get Model Input Channel Number for each patient
    '''
    if dataset=="XUANWU":#ANT
        if ch_num==0:
            ch_num_list={'1' : 111,
                         '2' : 107,
                         '3' : 107,
                         '4' : 76,
                         '5' : 91}
            return ch_num_list[str(patient_id)]
    elif dataset=="CHB":
        if ch_num!=18:
            print("\nplease input correct channel number for CHB name\n")
            return
        ch_num=18 
    else:
        print("\nplease input correct dataset name\n")
        return
    return ch_num

def GetBatchsize(dataset, patient_id):
    '''
    Get Batchsize for each patient
    '''
    batchsize=256
    if dataset=="XUANWU":#ANT
        batchsize = 256
    elif dataset=="CHB":
        batchsize = 200 if patient_id == 20 or patient_id ==21 else 256
    else:
        print("\nplease input correct dataset name\n")
        exit()
    return batchsize

import scipy.io as io
from EEG_model.TA_STS_ConvNet import TA_STS_ConvNet

def GetModel(input_channel, device_number, model_name, dataset_name, position_embedding):
    '''
    Get Model
    '''
    if model_name == "TA_STS_ConvNet":
        SE_hidden_layer = 6
        model = TA_STS_ConvNet(96, input_channel, SE_hidden_layer, device_number, position_embedding)
    else:
        print("mode name incorrect : {}".format(model_name))
        exit()
    return model

from EEG_model.seizureNetLoss import CE_Loss, FocalLoss
def GetLoss(loss):
    if loss == "CE":
        Loss = CE_Loss()
    elif loss == "FL":
        Loss = FocalLoss()
    else:
        print("Loss {} does not exist".format(loss))
        exit()
    return Loss

def mkdir(path):
    import os
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        print(path+' create successfully')
        return True
    else:
        print(path+' path already exist')
        return False
 