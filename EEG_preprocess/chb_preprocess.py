# -*- coding: utf-8 -*-
"""
Created on Sat May 15 22:42:15 2021

@author: phantom

#example of preprocess one patient : 
python chb_preprocess.py --patient_id=1
"""
import os
import glob
import numpy as np
import random
from chb_edf_file import CHBEdfFile
from chb_stft import getSpectral_STFT
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append('')
from EEG_utils.eeg_utils import GetInputChannel, mkdir, GetPatientList, GetDataPath

def setup_seed(seed):
    '''
    set up random seed for numpy
    '''
    np.random.seed(seed)
    random.seed(seed)
    
class CHBPatient:
    def __init__(self, patient_id, data_path, ch_num, doing_lowpass_filter, preictal_interval):
        
        self.interictal_interval=90 # 90min or longer before a seizure, decide as interictal data
        self.preictal_interval=preictal_interval #how long we decide as preictal. Default set to 15 min
        self.postictal_interval=120 # within 120min after a seizure, decide as postictal data
        self.patient_id = patient_id
        self.data_path=data_path
        self.ch_num=ch_num
        self.doing_lowpass_filter=doing_lowpass_filter
        self.patient_name=self.get_patient_name()
        
        #load edf files with seizure
        self._edf_files_seizure = list(map(
            lambda filename: CHBEdfFile(filename, self.patient_id, self.ch_num, self.doing_lowpass_filter),
            sorted(glob.glob("%s/%s/seizure/*.edf" % (self.data_path,self.patient_name)))
        ))

        #load edf files without seizures
        self._edf_files_unseizure = list(map(
            lambda filename: CHBEdfFile(filename, self.patient_id, self.ch_num, self.doing_lowpass_filter),
            sorted(glob.glob("%s/%s/unseizure/*.edf" % (self.data_path,self.patient_name)))
        ))
        
    def get_patient_name(self):
        """
        Get patient name
        """
        patient_list=GetPatientList("CHB")
        return patient_list[str(self.patient_id)]
    
    def get_seizure_time_list(self):
        """
        Get seizure time (second) in each EDF file
        for each patient, seizure times are stored in a list. [(start, end), (start, end),...]
        """
        seizure_time_list={'1' : [(2996, 3036),(1467, 1494),(1732, 1772),(1015, 1066),
                                 (1720, 1810),(327, 420),(1862, 1963)],
                           '2' : [(130, 212),(2972, 3053),(3369, 3378)],
                           # '3' : [(432, 501),(2162, 2214),(1982, 2029),(2592, 2656),
                           #        (1725, 1778)], # 3 4 34 35 36
                           '3' : [(731, 796),(432, 501),(2162, 2214),(1982, 2029),
                                  (2592, 2656),(1725, 1778)], # 2 3 4 34 35 36
                           # '3' : [(362, 414),(731, 796),(432, 501),(2162, 2214),
                           #        (1982, 2029),(2592, 2656),(1725, 1778)], # 1 2 3 4 34 35 36
                           '5' : [(417, 532),(1086, 1196),(2317, 2413),(2451, 2571),
                                 (2348, 2465)],
                           '6' : [(1724, 1738),(7461, 7476),(13525, 13540),(6211, 6231),
                                  (12500, 12516),(7799, 7811),(9387, 9403)],#1(1) 1(2) (1)3 4(2) 9 18 24
                           # '6' : [(327, 347),(6211, 6231),(12500, 12516),(10833, 10845),
                           #       (506,519),(7799, 7811),(9387, 9403)],#4(1) 4(2) 9 10 13 18 24
                           '7' : [(4920, 5006),(3285, 3381),(13688, 13831)],
                           '8' : [(2670, 2841),(2856, 3046),(2988, 3122),(2417, 2577),
                                 (2083, 2347)],
                           '9' : [(12231, 12295),(2951, 3030),(9196, 9267),(5299, 5361)],
                           # '10': [(6313, 6348),(6888, 6958),(2382, 2447),(3021, 3079),
                           #        (3801, 3877),(4618, 4707),(1383, 1437)],# 12 20 27 30 31 38 89
                           '10': [(6888, 6958),(2382, 2447),(3021, 3079),(3801, 3877),
                                  (4618, 4707),(1383, 1437)],#20 27 30 31 38 89
                           '11': [(298, 320),(2695, 2727),(1454, 2206)],
                           '13': [(2077, 2121),(934, 1004),(2474, 2491),(3339, 3401),
                                  (851, 916)],# 19 21 58 59 62(1)
# 						   '13': [(2077, 2121),(934, 1004),(142, 173),(458, 478),
# 				                  (2474, 2491)],# 19 21 40(1) 55(1) 58
                           '14': [(1986, 2000),(1372, 1392),(1911, 1925),(1838, 1879),
                                  (3239, 3259),(2833, 2849)],#3 4 6 11 17 27 
                           # '14': [(1986, 2000),(1372, 1392),(2817, 2839),(1911,1925),
                           #        (1838, 1879)],# 3 4 4 6 11
                            # '14': [(1986, 2000),(1372, 1392),(2817, 2839),(1911,1925),
                            #        (1039, 1061)],# 3 4 4 6 18
                            # '14': [(1986, 2000),(1372, 1392),(2817, 2839),(1911,1925)],# 3 4 4 6 
                            # '16': [(2290, 2299),(1120, 1129),(1854, 1868),(1214, 1220)],#10 11 14 16 
                            # '16': [(2290, 2299),(1120, 1129),(1854, 1868),(1214, 1220),
                            #        (227, 236),(1694, 1700),(3290, 3298),(627, 635),
                            #        (1909, 1916)],#10 11 14 16 17(1) 17(2) 17(4) 18(1) 18(2)
                           '16': [(2290, 2299),(1120, 1129),(1214, 1220),(227, 236),
                                  (1694, 1700),(3290, 3298),(627, 635),(1909, 1916)],#10 11 14 16 17(1) 17(2) 17(4) 18(1) 18(2)
                           '17': [(2282, 2372),(3025, 3140),(3136, 3224)],
# 						   '18': [(3477, 3527),(2087, 2155),(1908, 1963),(2196, 2264)],#用4次
                           '18': [(3477, 3527),(541, 571),(2087, 2155),(1908, 1963),
                                  (2196, 2264),(463, 509)],#用所有6次
                           # '19': [(2964, 3041),(3159, 3240)],#用2次
 						#    '19': [(299, 377),(2964, 3041),(3159, 3240)],#用所有3次
                           '20': [(94, 123),(1440, 1470),(2498, 2537),(1971, 2009),
                                  (390, 425),(1689, 1738),(2226, 2261),(1393, 1432)],
                           '21': [(1288, 1344),(2627, 2677),(2003, 2084),(2553, 2565)],
                           '22': [(3367, 3425),(3139, 3213),(1263, 1335)],
                           '23': [(3962, 4075),(325, 345),(5104, 5151),(2589, 2660),
                                  (6885, 6947),(8505, 8532),(9580, 9664)]
                               }
        
        return seizure_time_list[str(self.patient_id)]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'EDF preprocessing on CHB Dataset')
    #parser.add_argument('--data_path', type = str, default = "/data/GLH/CHB-MIT/CHB-MIT/", metavar = 'data path')
    parser.add_argument('--patient_id', type = int, default = 1, metavar = 'patient id')
    parser.add_argument('--target_preictal_interval', type = int, default = 15, metavar = 'how long we decide as preictal. Default set to 15 min') #in minute
    parser.add_argument('--seed', type = int, default = 19980702, metavar = 'random seed')
    parser.add_argument('--ch_num', type = int, default = 18, metavar = 'number of channel')
    parser.add_argument('--sfreq', type = int, default = 256, metavar = 'sample frequency')
    parser.add_argument('--window_length', type = int, default = 5, metavar = 'sliding window length')  #if stft : 30              else 5
    parser.add_argument('--preictal_step', type = int, default = 1, metavar = 'step of sliding window (second) for preictal data')  #if stft : 5               else 5
    parser.add_argument('--interictal_step', type = int, default = 1, metavar = 'step of sliding window (second) for interictal data') #if stft : 30             else 5
    parser.add_argument('--doing_STFT', type = bool, default = False, metavar = 'whether to do STFT') #if stft : True           else False
    parser.add_argument('--doing_lowpass_filter', type = bool, default = True, metavar = 'whether to do low pass filter') #if stft : False else True
    args = parser.parse_args()
    setup_seed(args.seed)
    
    patient_list=GetPatientList("CHB")
    patient_id=args.patient_id
    patient_name=patient_list[str(patient_id)]
    sfreq=args.sfreq
    window_length=args.window_length
    preictal_step=args.preictal_step
    interictal_step=args.interictal_step
    doing_STFT=args.doing_STFT
    doing_lowpass_filter=args.doing_lowpass_filter
    target_preictal_interval=args.target_preictal_interval
    preictal_interval=args.target_preictal_interval*60
    data_path=GetDataPath("CHB")
    ch_num = GetInputChannel("CHB", patient_id, args.ch_num)
    patient=CHBPatient(patient_id, data_path, ch_num, doing_lowpass_filter, target_preictal_interval)
    seizure_time_list=patient.get_seizure_time_list()
    print("\nprocessing patient : id {} {}\n seizure time {}\n".format(patient_id, patient_name, seizure_time_list))
    
    #create dir to save results
    mkdir("%s/%s/%dmin_%dstep_%dch" % (data_path, patient_name, target_preictal_interval, preictal_step, ch_num))
    
    # preprocessing ictal and preictal data
    # for each edf file with seizure, a sliding window is used to transpose data into clips
    # i is the i-th seizure of each patient
    print("clipping ictal and preictal data")
    for i, edf_file in enumerate(patient._edf_files_seizure):
        
        #preictal_interval is set to 900s(15min)
        preictal_interval=target_preictal_interval*60

        #load data from edf file
        print(edf_file.get_filepath())
        ant_data=edf_file.get_preprocessed_data()
        print("seizure {} \n shape {}".format(i+1, ant_data.shape))
        
        #如果前期不足900s， 补充数据   chb11 chb13的某一次没有补充数据
        # for a edf file with seizure, if the preictal length is less than 900s (e.g. the seizure start time is 267s), load the previous edf file (if exists) for supplement
        if seizure_time_list[i][0] < target_preictal_interval*60:
            print("seizure {} : preictal is not enough".format(i+1))
            #supplement_filepath="%s/%s/seizure-supplement/%d-supplement.edf" % (data_path, patient_name, i+1)
            supplement_filepath="%s/%s/seizure-supplement/%s-supplement.edf" % (data_path, patient_name, edf_file.get_filename())

            #如果有补充文件 则补充前期数据到900s
            #if the supplement edf file exists, load the edf file
            if os.path.exists(supplement_filepath):
                supplement_file=CHBEdfFile(supplement_filepath, patient_id, ch_num)
                print("load supplement file : {}".format(supplement_filepath))
                ant_data2=supplement_file.get_preprocessed_data()
                print("original label {}".format(seizure_time_list[i]))
                seizure_time_list[i] = (seizure_time_list[i][0] + supplement_file.get_file_duration(), seizure_time_list[i][1] + supplement_file.get_file_duration())
                print("new label {}".format(seizure_time_list[i]))
                ant_data=np.concatenate((ant_data2, ant_data))
                print("new data {}".format(ant_data.shape))

            #如果没有补充文件 则将前期时长缩短
            #if the supplement edf file does not exist, use as long as we have
            else:
                print("No supplement file")
                preictal_interval=seizure_time_list[i][0]

        #process ictal data
        ictal_list=[]
        ictal_count=0
        while seizure_time_list[i][0] + preictal_step * ictal_count + window_length <= seizure_time_list[i][1]:
            ictal_start = seizure_time_list[i][0] + preictal_step*ictal_count
            ictal_end = seizure_time_list[i][0] + preictal_step*ictal_count + window_length
            ictal_data = ant_data[ictal_start * sfreq : ictal_end * sfreq]

            #whether doing stft
            if doing_STFT:
                ictal_data=getSpectral_STFT(ictal_data) #(22, 59, 114)
            ictal_list.append(ictal_data)
            ictal_count += 1
        ictal_list=np.array(ictal_list)
        
        #save ictal data to npy file
        if doing_STFT:
            np.save("%s/%s/%dmin_%dstep_%dch/ictal%d.npy" % (data_path, patient_name, target_preictal_interval, 30, ch_num, i), ictal_list)
        else:
            save_path="%s/%s/%dmin_%dstep_%dch/ictal%d.npy" % (data_path, patient_name, target_preictal_interval, preictal_step, ch_num, i)
            print("save to {}".format(save_path))
            np.save(save_path, ictal_list)
        print("ictal shape {}".format(ictal_list.shape))
        
        #process preictal data
        preictal_list=[]
        preictal_count=0
        while seizure_time_list[i][0] + preictal_step * preictal_count + window_length - preictal_interval <= seizure_time_list[i][0]:
            preictal_start = seizure_time_list[i][0] + preictal_step*preictal_count - preictal_interval
            preictal_end = seizure_time_list[i][0] + preictal_step*preictal_count + window_length - preictal_interval
            preictal_data = ant_data[preictal_start * sfreq : preictal_end * sfreq]

            #whether doing stft
            if doing_STFT:
                preictal_data=getSpectral_STFT(preictal_data) #(22, 59, 114)
#                print("doing STFT {}".format(preictal_data.shape))
            preictal_list.append(preictal_data)
            preictal_count += 1
        preictal_list=np.array(preictal_list)
        
        #save preictal data to npy file
        if doing_STFT:
            np.save("%s/%s/%dmin_%dstep_%dch/preictal%d.npy" % (data_path, patient_name, target_preictal_interval, 30, ch_num, i), preictal_list)
        else:
            save_path="%s/%s/%dmin_%dstep_%dch/preictal%d.npy" % (data_path, patient_name, target_preictal_interval, preictal_step, ch_num, i)
            print("save to {}".format(save_path))
            np.save(save_path, preictal_list)
        print("preictal shape {}\n".format(preictal_list.shape))
    
    # preprocessing interictal data
    # for each edf file without seizure, a sliding window is used to transpose data into clips
    print("clipping interictal data")
    interictal_list_all=[]
    for i, edf_file in enumerate(patient._edf_files_unseizure):
        #load data from edf file
        print(edf_file.get_filepath())
        ant_data=edf_file.get_preprocessed_data()
        print("unseizure {} \n shape {}".format(i+1, ant_data.shape))
        
        #process interictal数据
        interictal_list=[]
        interictal_count=0
        while interictal_step*interictal_count + window_length <= edf_file.get_file_duration():
            interictal_start = interictal_step * interictal_count
            interictal_end = interictal_step*interictal_count + window_length
            interictal_data = ant_data[interictal_start * sfreq:interictal_end * sfreq]

            #whether doing stft
            if doing_STFT:
                interictal_data=getSpectral_STFT(interictal_data) #(22, 59, 114)

            interictal_list.append(interictal_data)
            interictal_count += 1
        interictal_list=np.array(interictal_list)
        print("interictal shape {}".format(interictal_list.shape))
        
        #concatenate interictal data
        if len(interictal_list_all)==0:
            interictal_list_all = interictal_list
        else:
            interictal_list_all = np.vstack((interictal_list_all, interictal_list))
        print("all interictal shape: {}".format(interictal_list_all.shape))
        
    #shuffle interictal data and divide into n gourps. n is the number of seizures of each patient
    np.random.shuffle(interictal_list_all)
    count=0
    interictal_length=len(interictal_list_all)//len(seizure_time_list)
    while (count+1) * interictal_length <= len(interictal_list_all):
        interictal_data=interictal_list_all[count * interictal_length : (count+1) * interictal_length]
        
        #save interictal data to npy file
        if doing_STFT:
            np.save("%s/%s/%dmin_%dstep_%dch/interictal%d.npy" % (data_path, patient_name, target_preictal_interval, 30, ch_num, count), interictal_data)
        else:
            save_path="%s/%s/%dmin_%dstep_%dch/interictal%d.npy" % (data_path, patient_name, target_preictal_interval, preictal_step, ch_num, count)
            print("save to {}".format(save_path))
            np.save(save_path, interictal_data)
        print("interictal count {} : {}".format(count, interictal_data.shape))
        count+=1
    
        
        
    
    
        