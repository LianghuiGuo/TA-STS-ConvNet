# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 14:37:03 2021

@author: phantom

# one example of training on CHB : 
python train.py --dataset_name=CHB --model_name=TA_STS_ConvNet --device_number=1 --patient_id=1 --step_preictal=1 --ch_num=18
"""

from __future__ import print_function
import torch.utils.data
import argparse
import numpy as np
import torch.utils.data as Data
from EEG_dataset.dataset_train import trainDataset
from EEG_dataset.dataset_test import testDataset
from EEG_trainer.trainer import Trainer
from EEG_utils.eeg_utils import GetPatientList, GetSeizureList, GetInputChannel, GetBatchsize, GetModel, GetLoss, mkdir
import torch
import random
import os


def setup_seed(seed):
    '''
    set up seed for cuda and numpy
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.benchmark = True
        
def main(train, test, LOO, patient_name, args):
    '''
    training process
    train : seizures for training. example : [1,2,3,4,5], which means seizre 1-5 are used for training
    test : seizure for testing. example : [0], which means seizre 0 is used for training
    patient_name: patient name
    args : args
    '''
    patient_id=args.patient_id
    cuda=args.cuda
    device_number=args.device_number
    seed=args.seed
    ch_num=args.ch_num
    batch_size=args.batch_size
    model_name=args.model_name
    loss=args.loss
    dataset_name=args.dataset_name
    checkpoint_dir=args.checkpoint_dir
    target_preictal_interval=args.target_preictal_interval
    transfer_learning=args.transfer_learning
    lock_model=args.lock_model
    domain_adaptation=args.domain_adaptation
    augmentation=args.augmentation
    step_preictal=args.step_preictal
    balance=args.balance
    using_ictal=args.using_ictal
    position_embedding = args.position_embedding
    
    #cuda and random seed
    cuda = cuda and torch.cuda.is_available()
    torch.cuda.set_device(device_number)
    print("set cuda device : ", device_number)
    setup_seed(seed) 
    
    #dataset loader. training set and test set
    input_channel=GetInputChannel(dataset_name, patient_id, ch_num)
    train_set = trainDataset(dataset_name, train, ite=1, augmentation = augmentation, using_ictal=using_ictal, scaler = balance, 
                        patient_id=patient_id, patient_name=patient_name, ch_num=ch_num, target_preictal_interval=target_preictal_interval, step_preictal=step_preictal)
    test_set = testDataset(dataset_name, test, using_ictal=using_ictal,
                      patient_id=patient_id, patient_name=patient_name, ch_num=ch_num, target_preictal_interval=target_preictal_interval, step_preictal=step_preictal)
    trainloader = Data.DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
    testloader = Data.DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True) 
    
    #model
    model = GetModel(input_channel, device_number, model_name, dataset_name, position_embedding)    
        
    #whether using transfer learning. load pre-trained model weights
    if transfer_learning:
        print("loading pre-trained model")
        if domain_adaptation:
            print("load Domain adaptation transfer weights")
            model.load_state_dict(torch.load('{}/{}/{}/{}/patient{}_{}_source.pth'.format(checkpoint_dir, dataset_name, model_name, patient_id, patient_id, LOO)))
        else:
            print("load transfer weights")
            model.load_state_dict(torch.load('{}/{}/{}/{}/patient{}_source.pth'.format(checkpoint_dir, dataset_name, model_name, patient_id, patient_id)))
            
    #parameter number
    print('parameters:', sum(param.numel() for param in model.parameters() if param.requires_grad))
    
    #cuda
    if cuda:
        model = model.cuda()
        
    #loss and optimizer
    print("Loss : {}".format(loss))
    Loss = GetLoss(loss)
    trainer = Trainer(model, Loss, trainloader, testloader, args)
    
    #training
    print("Training...")
    model, train_acc_list,train_loss_list = trainer.train()
    
    #mkdir and save weights
    mkdir("{}/model/{}/{}/{}".format(checkpoint_dir, dataset_name, model_name, patient_id))
    torch.save(model.state_dict(), "{}/model/{}/{}/{}/patient{}_{}.pth".format(checkpoint_dir, dataset_name, model_name, patient_id, patient_id, LOO))
    
    #testing
    print("Testing...")
    test_acc, test_loss, index_list, target_list=trainer.test()
    
    return train_acc_list,train_loss_list,test_acc, test_loss, index_list, target_list
    
if __name__ == "__main__":
    # Parse the JSON arguments
    parser = argparse.ArgumentParser(description = 'Seizure predicting on Xuanwu/CHB Dataset')
    parser.add_argument('--patient_id', type = int, default = 1, metavar = 'patient id')  
    parser.add_argument('--device_number', type = int, default = 0, metavar = 'CUDA device number')
    parser.add_argument('--ch_num', type = int, default = 15, metavar = 'number of channel')
    parser.add_argument('--model_name', type = str, default = "TA_STS_ConvNet", metavar = 'used model')
    parser.add_argument('--dataset_name', type = str, default = "CHB", metavar = 'dataset name : XUANWU / CHB') 
    parser.add_argument('--target_preictal_interval', type = int, default = 15, metavar = 'how long we decide as preictal. Default set to 15 min')
    parser.add_argument('--step_preictal', type = int, default = 1, metavar = 'step of sliding window (second)') 
    parser.add_argument('--loss', type = str, default = "CE", metavar = 'CE:cross entropy   FL:focal loss')
    parser.add_argument('--seed', type = int, default = 20221110, metavar = 'random seed')
    parser.add_argument('--transfer_learning', type = bool, default = False, metavar = "whether using transfer learning, loading pre-trained model, train on n-1, test on 1, LOOCV ") 
    parser.add_argument('--lock_model', type = bool, default = False, metavar = "whether locking the shallow layers")
    parser.add_argument('--domain_adaptation', type = bool, default = False, metavar = "whether train using domain adaption (Loss=CE+LocalMMD+GlobalMMD)") 
    parser.add_argument('--position_embedding', type = bool, default = False, metavar = "whether train use position embedding")
    parser.add_argument('--checkpoint_dir', type = str, default = '/home/al/GLH/code/seizure_predicting_seeg/model/', metavar = 'model save path')
    parser.add_argument('--log-interval', type = int, default = 4, metavar = 'N')
    parser.add_argument('--to_train', type = bool, default = True)
    parser.add_argument('--TestWhenTraining', type = int, default = 0, metavar = "whether to test when training on each epoch")
    parser.add_argument('--cuda', type = bool, default = True, metavar = "whether to use cuda")
    parser.add_argument("--augmentation", type = int, default = 1, metavar = 'whether to use data augmentation')
    parser.add_argument("--using_ictal", type = int, default = 1, metavar = 'whether to use ictal data') 
    parser.add_argument('--balance', type = int, default = 1, metavar = 'whether to balance preictal and interictal data') 
    parser.add_argument('--batch_size', type = int, default = 200, metavar = 'batchsize')
    parser.add_argument('--learning_rate', type = float, default = 0.001, metavar = 'learning rate')
    parser.add_argument('--test_every', type = int, default = 5, metavar = 'N')
    parser.add_argument('--learning_rate_decay', type = float, default = 0.8, metavar='N')
    parser.add_argument('--weight_decay', type = float, default = 5e-4, metavar = 'N')
    parser.add_argument('--num_epochs', type = int, default = 100, metavar = 'number of epochs')
    parser.add_argument('--early_stop_patience', type = int, default = 15, metavar = 'N')    
    args = parser.parse_args()
    
    #get patient list, patient id, seizure list, etc
    patient_list=GetPatientList(args.dataset_name)
    patient_id=args.patient_id
    patient_name=patient_list[str(patient_id)]
    seizure_list=GetSeizureList(args.dataset_name)
    seizure_count=len(seizure_list[str(patient_id)])
    args.batch_size = GetBatchsize(args.dataset_name, args.patient_id)
    args.checkpoint_dir = os.getcwd()# /home/al/GLH/code/seizure_predicting_seeg/no_TAL
    print("dataset : {} \npatient {} \nseizure count : {}\n".format(args.dataset_name, patient_id, seizure_count))
    
    #LOOCV for each patient.
    for LOO in seizure_list[str(patient_id)]:
        test = LOO
        train = list(set(seizure_list[str(patient_id)])-set([test]))
        train_acc_list,train_loss_list,test_acc, test_loss, index_list, target_list= main(train, test, LOO, patient_name, args)

       
