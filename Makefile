#CHB-MIT

#preprocess one patient
preprocess:
	python EEG_preprocess/chb_preprocess.py --patient_id=1

#preprocess all patients
preprocess_chb:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python EEG_preprocess/chb_preprocess.py --patient_id=$$id; \
	done

#train on one patient
train:
	python train.py --dataset_name=CHB --model_name=TA_STS_ConvNet --device_number=1 --patient_id=1 --step_preictal=1 --ch_num=18

#train on all patients
train_chb:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python train.py --dataset_name=CHB --model_name=TA_STS_ConvNet --loss=CE --device_number=0 --ch_num=18 --patient_id=$$id; \
	done

#eval on one patient
eval:
	python test.py --dataset_name=CHB --model_name=TA_STS_ConvNet --device_number=1 --patient_id=1 --ch_num=18 --moving_average_length=9

#eval on all patients
eval_chb:
	for id in 1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 20 21 22 23; \
	do \
		python test.py --dataset_name=CHB --model_name=TA_STS_ConvNet --device_number=1 --patient_id=$$id --ch_num=18 --moving_average_length=9 \
	done