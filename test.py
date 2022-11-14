# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:49:27 2021

@author: phantom

#CHB
python test.py --dataset_name=CHB --model_name=TA_STS_ConvNet --device_number=1 --patient_id=1 --ch_num=18 --moving_average_length=9
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from EEG_dataset.dataset_test import testDataset
from EEG_eval.p_value_seizure import P_value
from EEG_utils.eeg_utils import GetPatientList, GetSeizureList, GetInputChannel, GetBatchsize, GetModel
from EEG_utils.write_to_excel import WriteToExcel, CalculateAverageToExcel
from torch.nn import functional as F
import torch.utils.data as Data
import argparse
# import scipy.io as io
import os
from sklearn.metrics import roc_auc_score,roc_curve, auc
import time

def setup_seed(seed):
    '''
    set up seed for cuda and numpy
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def smooth(a,WSZ):
    '''
    smoothing function, which is used to smooth the seizure predicting results
    a:original data. NumPy 1-D array containing the data to be smoothed
    a need to be 1-D. If not, use np.ravel() or np.squeeze() to make a transpose
    WSZ: smoothing window size needs, which must be odd number,
    as in the original MATLAB implementation
    '''
    if(WSZ%2==0):
        WSZ-=1
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


def plot_roc(labels, predict_prob):
    '''
    plot ROC curve
    labels : true labels
    predict_prob : predicted probabilities
    '''
    false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description = 'Seizure predicting on Xuanwu Dataset')
    parser.add_argument('--patient_id', type = int, default = 1, metavar = 'patient_id')
    parser.add_argument('--ch_num', type = int, default = 1, metavar = 'number of channel')
    parser.add_argument('--model_name', type = str, default = "TA_STS_ConvNet", metavar = 'N')
    parser.add_argument('--dataset_name', type = str, default = "CHB", metavar = 'XUANWU / CHB') 
    parser.add_argument('--description', type = str, default = "normal", metavar = 'normal / TL / TL-lock / DA') #normal/transfer learning/transfer learning lock/domain adaptation
    parser.add_argument('--target_preictal_interval', type = int, default = 15, metavar = 'how long we decide as preictal. Default set to 15 min') 
    parser.add_argument('--step_preictal', type = int, default = 1, metavar = 'step of sliding window (second)') 
    parser.add_argument('--device_number', type = int, default = 1, metavar = 'CUDA device number')
    parser.add_argument('--position_embedding', type = bool, default = False, metavar = "whether train use position embedding")  
    parser.add_argument('--checkpoint_dir', type = str, default = '/home/al/GLH/code/seizure_predicting_seeg/model', metavar = 'N')
    parser.add_argument('--seed', type = int, default = 20221110, metavar = 'N')
    parser.add_argument('--batch_size', type = int, default = 200, metavar = 'batchsize')
    parser.add_argument('--using_cuda', type = bool, default = True, metavar = 'whether using cuda')
    parser.add_argument('--threshold', type = float, default = 0.6, metavar = 'alarm threshold')
    parser.add_argument('--moving_average_length', type = int, default = 9, metavar = 'length of smooth window') 
    parser.add_argument('--persistence_second', type = int, default = 1, metavar = 'N')
    parser.add_argument('--tw0', type = float, default = 1/120, metavar = '1/120 hour, which is 30 seconds')
    args = parser.parse_args()
    args.model_path = os.getcwd()
    
    model_name=args.model_name
    model_path=args.checkpoint_dir
    dataset_name=args.dataset_name
    description=args.description
    patient_id=args.patient_id
    seed=args.seed
    moving_average_length=args.moving_average_length
    target_preictal_interval=args.target_preictal_interval
    step_preictal=args.step_preictal
    device_number=args.device_number
    using_cuda=args.using_cuda
    ch_num=args.ch_num
    position_embedding = args.position_embedding
    batch_size=GetBatchsize(args.dataset_name, args.patient_id)
    threshold=args.threshold
    persistence_second=args.persistence_second
    tw0=args.tw0
    tw=target_preictal_interval/60
    
    patient_list=GetPatientList(dataset_name)
    seizure_list=GetSeizureList(dataset_name)
    seizure_count=len(seizure_list[str(patient_id)])
    patient_name=patient_list[str(patient_id)]
    print("patient : {}".format(patient_id))
    print("dataset : {} | seizure : {} filter : {} | threshold : {} | persistence : {} | tw0 : {} | tw : {}".format(args.dataset_name, 
                    seizure_count, moving_average_length*step_preictal, threshold, persistence_second, tw0*3600, tw*3600))

    TP_list=[]
    FN_list=[]
    TN_list=[]
    FP_list=[]
    FPR_list=[]
    SEN_list=[]
    AUC_list=[]
    InferTime_list1 = []
    InferTime_list2 = []
    PW_count=0
    
    #LOOCV for predicting
    for i in seizure_list[str(patient_id)]:    
        #load test data
        test_set = testDataset(dataset_name, i, using_ictal=1, patient_id=patient_id, patient_name=patient_name, 
                                     ch_num=ch_num, target_preictal_interval=target_preictal_interval, step_preictal=step_preictal)
        test_loader = Data.DataLoader(dataset = test_set,batch_size = batch_size, shuffle = False)
        labels=test_set.y_data.numpy()
        preictal_length=test_set.preictal_length
        interictal_length=test_set.interictal_length
        ictal_length=test_set.ictal_length
        print("interrictal : {} | preictal : {} | ictal : {}: ".format(interictal_length, preictal_length, ictal_length))

        #get model
        input_channel=GetInputChannel(dataset_name, patient_id, ch_num)
        model = GetModel(input_channel, device_number, model_name, dataset_name, position_embedding)   
        model.load_state_dict(torch.load('{}/model/{}/{}/{}/patient{}_{}.pth'.format(model_path, dataset_name, model_name,  patient_id, patient_id, i)))

        #set cuda
        if using_cuda:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(device_number)
            model = model.cuda()	            
        
        #start predicting
        start_time1 = time.clock()
        start_time2 = time.time()
        model.eval()
        output_probablity=[]
        output_list=[]
        feature_list=[]
        with torch.no_grad():
            for k, (data, target) in enumerate(test_loader):
                if using_cuda:
                    data = data.cuda()
                    target = target.cuda()
                return_value = model(data)
                output=return_value[0]
                output_nosoftmax=output.cpu().detach().numpy()
                output=F.softmax(output, dim=1)
                output=torch.clamp(output, min=1e-9, max=1-1e-9)
                output=output.cpu().detach().numpy()
                if len(output_probablity)==0:
                    output_probablity.append(output)  
                    output_probablity=np.array(output_probablity).squeeze()
                    output_list.append(output_nosoftmax)  
                    output_list=np.array(output_list).squeeze()
                else:
                    output_probablity=np.vstack((output_probablity, output))
                    output_list=np.vstack((output_list, output_nosoftmax))
        infer_time1 = (time.clock()-start_time1)/(preictal_length+interictal_length+ictal_length)
        infer_time2 = (time.time()-start_time2)/(preictal_length+interictal_length+ictal_length)
        InferTime_list1.append(infer_time1)
        InferTime_list2.append(infer_time2)
        
        #save output probabilities
        np.save('{}/model/{}/{}/{}/probablity{}_{}.npy'.format(model_path, dataset_name, model_name, patient_id, patient_id, i), output_probablity)
        np.save('{}/model/{}/{}/{}/output{}_{}.npy'.format(model_path, dataset_name, model_name, patient_id, patient_id, i), output_list)
        np.save('{}/model/{}/{}/{}/label{}_{}.npy'.format(model_path, dataset_name, model_name, patient_id, patient_id, i), labels)
        predicting_probablity=output_probablity[:,1]
        
        #calculate AUC, draw ROC curve
        y_true=labels
        y_probablity=output_probablity
        y_score1=y_probablity[:,1]
        auc_value=roc_auc_score(y_true, y_score1)
        AUC_list.append(auc_value)
        plot_roc(y_true,y_score1)
        plt.savefig('{}/model/{}/{}/{}/ROC{}_{}.png'.format(model_path, dataset_name, model_name, patient_id, patient_id, i))
            
        #smooth the predicting results
        predicting_probablity_smooth=smooth(predicting_probablity, moving_average_length)
        predicting_result=output_probablity.argmax(axis = 1)
        np.save('{}/model/{}/{}/{}/pre_label{}_{}.npy'.format(model_path, dataset_name, model_name, patient_id, patient_id, i), predicting_result)
        np.save('{}/model/{}/{}/{}/smooth_probablity{}_{}.npy'.format(model_path, dataset_name, model_name, patient_id, patient_id, i), predicting_probablity_smooth)
        
        #calculate confusion matrix
        TP,FP,TN,FN=0,0,0,0
        for j in range(len(labels)):
            if predicting_result[j] ==1 and labels[j]==1:
                TP+=1
            elif predicting_result[j] ==0 and labels[j]==1:
                FN+=1
            elif predicting_result[j] ==0 and labels[j]==0:
                TN+=1
            else:
                FP+=1
        TP_list.append(TP)
        FN_list.append(FN)
        TN_list.append(TN)
        FP_list.append(FP)
        
        #calculate currect alarm and false alarm. calculate Sensitivity and FPR/h 
        count=0
        interval=0 #距离发作点的时间
        false_alarm_list=[]#误报时间点列表
        true_alarm_list=[]#正报时间点列表
        for index in range(len(predicting_probablity_smooth)):
			#probability is over threshold, start counting
            if predicting_probablity_smooth[index]>threshold:
                PW_count += 1
                count+=1
            else:
                count=0
            #if count is over persistence second，decide as one alarm
            if count>=persistence_second//step_preictal:
                interval=interictal_length+preictal_length-index
                #if the alarm is within 15min，True alarm
                if index >= interictal_length and index < interictal_length + preictal_length:
                    true_alarm_list.append(interval)
                #if the alarm is not within 15min，False alarm
                elif index < interictal_length:
                    false_alarm_list.append(interval)
                count=0
                
        if model_name == "spectralCNN":
            FPR=len(false_alarm_list)/((interictal_length*30+(preictal_length+ictal_length)*step_preictal)/3600)#spectralCNN
        else:
            FPR=len(false_alarm_list)/((interictal_length+preictal_length+ictal_length)*step_preictal/3600)

        FPR_list.append(FPR)
        if len(true_alarm_list) > 0:
            SEN_list.append(1)
        else:
            SEN_list.append(0)
            true_alarm_list.append(-1)
        
        print("TP {} FN {} TN {} FP {} sen {:.2%} spe {:.2%} acc {:.2%}; TA {} FA {} FPR {:.4} AUC {:.4} PT {} IT1 {:.4} IT2 {:.4}".format(TP,
              FN,TN,FP,TP/(TP+FN),TN/(TN+FP),(TP+TN)/(TP+FN+TN+FP), 
              len(true_alarm_list), len(false_alarm_list), FPR, auc_value, true_alarm_list[0], infer_time1, infer_time2))
        
        #draw predicting results
        plt.figure()
        #draw predicting probabilities
        plt.plot(predicting_probablity,linewidth = '1', color='paleturquoise')
        plt.plot(predicting_probablity_smooth,linewidth = '3', color='red')
        #draw alarm line
        plt.plot(np.ones(interictal_length+preictal_length+ictal_length)*threshold,linewidth = '2', color='black', linestyle='--')
        #draw labels
        plt.plot(np.zeros(interictal_length+preictal_length+ictal_length)-0.05,linewidth = '3', label = "ictal", color='gold')
        plt.plot(np.zeros(interictal_length+preictal_length)-0.05,linewidth = '3', label = "preictal", color='gray')
        plt.plot(np.zeros(interictal_length)-0.05,linewidth = '3', label = "interictal", color='green')
        plt.legend(labels=["probablity", "smoothed probablity", "alarm line", "ictal","preictal","interictal"], loc="upper left")
        plt.title("w {} filter {} persistence {} sen {:.4%} spe {:.4%} acc {:.4%}".format(threshold, moving_average_length, 
                  persistence_second, TP/(TP+FN),TN/(TN+FP),(TP+TN)/(TP+FN+TN+FP)))
        plt.savefig('{}/model/{}/{}/{}/patient{}_{}.png'.format(model_path, dataset_name, model_name, patient_id, patient_id, i))
    
        #draw predicting line
        plt.figure()
        plt.plot(predicting_result,linewidth = '2', color='black')#, marker='o')
        plt.savefig('{}/model/{}/{}/{}/patient{}_{}_predicting_label.png'.format(model_path, dataset_name, model_name, patient_id, patient_id, i))
        
    #calculate p-value
    N=len(SEN_list)
    n=sum(SEN_list)
    pw=PW_count/((interictal_length+preictal_length+ictal_length)*len(SEN_list))#len(SEN_list) equals to number of seizures
    p_value = P_value(tw0, tw, N, n, pw).calculate_p()
    
    print("total : TP {} FN {} TN {} FP {}".format(sum(TP_list),sum(FN_list),sum(TN_list),sum(FP_list)))
    print("true seizure {} predicted seizure {}".format(N, n))
    print("sensitivity : {:.2%}".format(sum(TP_list)/(sum(TP_list)+sum(FN_list))))
    print("specificity : {:.2%}".format(sum(TN_list)/(sum(TN_list)+sum(FP_list))))
    print("accuracy : {:.2%}".format((sum(TP_list)+sum(TN_list))/(sum(TP_list)+sum(FN_list)+sum(TN_list)+sum(FP_list))))
    print("FPR : {:.4}".format(np.mean(FPR_list)))
    print("SEN : {:.2%}".format(np.mean(SEN_list)))
    print("AUC : {:.4}".format(np.mean(AUC_list)))
    print("pw : {} {:.4}".format(PW_count, pw))
    print("p-value : {:.4}".format(p_value))
    print("Avg infer time1 (time.clock) : {:.4}".format(np.mean(InferTime_list1)))
    print("Avg infer time2 (time.time): {:.4}\n".format(np.mean(InferTime_list2)))
    
    #save results to excel
    save_data={"ID" : patient_id,
               "True Seizure": N,
               "Predict Seizure": n,
               "TP" : sum(TP_list),
               "FN" : sum(FN_list),
               "TN" : sum(TN_list),
               "FP" : sum(FP_list),
               "Sen": sum(TP_list)/(sum(TP_list)+sum(FN_list)),
               "Spe": sum(TN_list)/(sum(TN_list)+sum(FP_list)),
               "Acc": (sum(TP_list)+sum(TN_list))/(sum(TP_list)+sum(FN_list)+sum(TN_list)+sum(FP_list)),
               "AUC": np.mean(AUC_list),
               "Sn" : np.mean(SEN_list),
               "FPR": np.mean(FPR_list),
               "pw" : pw,
               "p-value": p_value}
    eval_param={"descri": description,
                "filter": moving_average_length*step_preictal,
                "threshold": threshold,
                "persistence": persistence_second}
    WriteToExcel(model_path, dataset_name, model_name, save_data, eval_param)
    if dataset_name == "CHB" and patient_id == 23 or dataset_name == "XUANWU" and patient_id == 5:
        CalculateAverageToExcel(model_path, dataset_name, model_name, save_data, eval_param)
    
