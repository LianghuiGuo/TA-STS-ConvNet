# from openpyxl import Workbook,load_workbook
# from openpyxl.styles import *
# import warnings
# warnings.filterwarnings('ignore')
import xlwt
import xlrd  
from xlutils.copy import copy
import argparse
import os
import numpy as np
from EEG_utils.eeg_utils import GetPatientList
    
def WriteToExcel(path, dataset, model_name, save_data, eval_param):
    #create excel
    excel_path="{}/model/{}/{}/{}_{}_filter{}_thres{:.2f}_persis{}.xlsx".format(path, dataset, model_name, model_name,
                                                                        eval_param["descri"], eval_param["filter"], eval_param["threshold"], eval_param["persistence"])
    print("excel path : {}".format(excel_path))
    if os.path.isfile(excel_path):
        print("excel already exist!")
        workbook = xlrd.open_workbook(excel_path, formatting_info=True) 
        # sheet1 = workbook.get_sheet(0) 
        # sheet1 = workbook.sheet_by_index(0)
        workbook = copy(workbook)#xlrd转为xlwt
        sheet1 = workbook.get_sheet(0) 
    else:
        print("create new excel!")
        workbook = xlwt.Workbook(encoding='utf-8') #新建工作簿
        sheet1 = workbook.add_sheet("results")  #新建sheet
        
    #write
    title_list=["ID","True Sei","Pred Sei","TP","FN","TN","FP","Sen","Spe","Acc","AUC","Sn","FPR","pw","p-value"]
    col_index=0
    for key in save_data:
        sheet1.write(0,col_index,title_list[col_index])  
        sheet1.write(save_data["ID"],col_index,save_data[key]) 
        col_index += 1
    
    #save
    workbook.save(excel_path)
    print("excel saved!")
    
def CalculateAverageToExcel(path, dataset, model_name, save_data, eval_param):
    #open excel
    excel_path="{}/model/{}/{}/{}_{}_filter{}_thres{:.2f}_persis{}.xlsx".format(path, dataset, model_name, model_name,
                                                                        eval_param["descri"], eval_param["filter"], eval_param["threshold"], eval_param["persistence"])
    print("excel path : {}".format(excel_path))
    if os.path.isfile(excel_path):
        print("excel already exist!")
        workbook = xlrd.open_workbook(excel_path, formatting_info=True) 
        sheet = workbook.sheet_by_index(0)
    else:
        print("No such excel : {}".format(excel_path))
        return
    
    #calculate average values
    title_list=["ID","True Sei","Pred Sei","TP","FN","TN","FP","Sen","Spe","Acc","AUC","Sn","FPR","pw","p-value"]
    Sn_value_list = sheet.col_values(title_list.index("Sn"))[1:]
    Sn_value_list = [i for i in Sn_value_list if i != '']
    Sn_value = np.mean(Sn_value_list)
    FPR_value_list = sheet.col_values(title_list.index("FPR"))[1:]
    FPR_value_list = [i for i in FPR_value_list if i != '']
    FPR_value = np.mean(FPR_value_list)
    print("Average value : | Sn {} | FPR {}".format(Sn_value, FPR_value))
    
    patient_count=len(GetPatientList(dataset))
    if len(Sn_value_list) != patient_count:
        print("some patients are not evaluated, total {}, but only {} evaluated".format(patient_count, len(Sn_value_list)))
        return
    else:
        print("all {} patiens are correctly evaluated!".format(patient_count))
    
    #create excel
    excel_path="{}/model/{}/{}/grid_search_Sn_persis{}.xlsx".format(path, dataset, model_name, eval_param["persistence"])
    print("excel path : {}".format(excel_path))
    if os.path.isfile(excel_path):
        print("excel already exist!")
        workbook = xlrd.open_workbook(excel_path, formatting_info=True) 
        workbook = copy(workbook)#xlrd转为xlwt
        sheet1 = workbook.get_sheet(0) 
    else:
        print("create new excel!")
        workbook = xlwt.Workbook(encoding='utf-8') #新建工作簿
        sheet1 = workbook.add_sheet("results")  #新建sheet
    
    #x label
    offset = 20
    col_list=[1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59]
    for i in range(0, len(col_list)):
        sheet1.write(0, i+1, col_list[i])  
        sheet1.write(offset, i+1, col_list[i]) 
    
    #y label
    row_list=[0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    for i in range(0, len(row_list)):
        sheet1.write(i+1, 0, row_list[i])  
        sheet1.write(i+1+offset, 0, row_list[i]) 
    
    #write Sn、FPR into excel
    row_Sn = row_list.index(eval_param["threshold"])+1
    col_Sn = col_list.index(eval_param["filter"])+1
    row_FPR = row_Sn + offset
    col_FPR = col_Sn
    sheet1.write(row_Sn, col_Sn, Sn_value)  
    sheet1.write(row_FPR, col_FPR, FPR_value) 
    
    #save
    workbook.save(excel_path)
    print("excel saved!")