## **TA-STS-ConvNet**  
This is the code proposed in our paper "Triple-Attention-based Spatio-Temporal-Spectral Convolutional Network for Epileptic Seizure Prediction"
- github : https://github.com/LianghuiGuo/TA-STS-ConvNet
- paper :  https://www.techrxiv.org/articles/preprint/Triple-Attention-based_Spatio-Temporal-Spectral_Convolutional_Network_for_Epileptic_Seizure_Prediction/20557074
<!-- 分割线 -->
***

## How to use
1. **Data Preparation**
- Download CHB-MIT dataset. http://archive.physionet.org/physiobank/database/chbmit/
- For each patient of CHB-MIT, the edf files need to be separated manually into three dirs  
    - **seizure** : which stores edf files with seizure
    - **seizure-supplement** : which stores files as supplement of preictal data. e.g. for 6.edf, the preictal length is shorter than 15min, then the previous edf file is used as a supplement.
    - **unseizure** : which stores files without seizures. (at least 2 hours away from any seizure onset)
- Edf files for chb01 are provided in /data/CHB-MIT/chb01. Note that there should be more edf files in unseizure for chb01 (here chb01_07.edf ~ chb01_12.edf are provided as example).  
CHB-MIT  
└─chb01  
&emsp;    ├─seizure  
&emsp;    │  └─1.edf  
&emsp;    │  └─2.edf  
&emsp;    │  └─3.edf  
&emsp;    │  └─4.edf  
&emsp;    │  └─5.edf  
&emsp;    │  └─6.edf  
&emsp;    │  └─7.edf  
&emsp;    ├─seizure-supplement  
&emsp;    │  └─6-supplement.edf  
&emsp;    └─unseizure  
&emsp;&emsp;       └─chb01_07.edf  
&emsp;&emsp;       └─chb01_08.edf  
&emsp;&emsp;       └─chb01_09.edf  
&emsp;&emsp;       └─chb01_10.edf  
&emsp;&emsp;       └─chb01_11.edf  
&emsp;&emsp;       └─chb01_12.edf  
- For other patients, the edf files need to be separated as well.
- After preparation, you need to change some paths in code:
<pre>
    EEG_utils.eeg_utils.py : (line 91) data_path="/home/al/GLH/code/seizure_predicting_seeg/code_public/data/CHB-MIT"
    train.py : (line 135) parser.add_argument('--checkpoint_dir', type = str, default = '/home/al/GLH/code/seizure_predicting_seeg/model/', metavar = 'model save path')
    test.py : (line 82) parser.add_argument('--checkpoint_dir', type = str, default = '/home/al/GLH/code/seizure_predicting_seeg/model', metavar = 'N')
</pre>
<!-- 分割线 -->
***

2. **Data Preprocess**  
- For one patient, you can preprocess data as follow :
<pre>
    make preprocess
</pre>
the processed data should be saved in /chb01/15min_1step_18ch/
- You can also preprocess data of all the patients as follow :
<pre>
    make preprocess_chb
</pre>
<!-- 分割线 -->
***

3. **Training**
- For one patient, you run training as follow :
<pre>
    make train
</pre>
- You can also train on all the patients as follow :
<pre>
    make train_chb
</pre>

<!-- 分割线 -->
***

4. **Evaluation**
- For one patient, you run evaluation as follow :
<pre>
    make eval
</pre>
- You can also eval on all the patients as follow :
<pre>
    make eval_chb
</pre>

<!-- 分割线 -->
***
If ant possible requirement is missed, please notice me.