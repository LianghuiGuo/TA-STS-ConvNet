# -*- coding: utf-8 -*-
"""
Created on Sat May 15 11:12:52 2021

@author: phantom
"""

import numpy as np
import mne
                   
class CHBEdfFile(object):
    """
    Edf reader using pyedflib
    """
    def __init__(self, filename, patient_id=None, ch_num=1, doing_lowpass_filter=True, preload=False):
        self._filename = filename
        self._patient_id = patient_id
        self.ch_num = ch_num
        self._raw_data = mne.io.read_raw_edf(filename, preload=preload)
        self._info=self._raw_data.info
        self.doing_lowpass_filter=doing_lowpass_filter
        
    def get_filepath(self):
        """
        Name of the EDF path
        """
        return self._filename
	
    def get_filename(self):
        """
        Name of the EDF name
        """
        return self._filename.split("/")[-1].split(".")[0]

    def get_n_channels(self):
        """
        Number of channels
        """
        return self._info['nchan']

    def get_n_data_points(self):#3686400
        """
        Number of data points
        """
        return len(self._raw_data._times)

    def get_channel_names(self):
        """
        Names of channels
        """
        return self._info['ch_names']

    def get_file_duration(self):#3600
        """
        Returns the file duration in seconds
        """
        return int(round(self._raw_data._last_time))

    def get_sampling_rate(self):#1024
        """
        Get the frequency
        """
        if self._info['sfreq'] < 1:
            raise ValueError("sampling frequency is less than 1")
        return int(self._info['sfreq'])
    
    def get_preprocessed_data(self):
        """
        Get preprocessed data
        """
        #sampling frequency
        sfreq=self.get_sampling_rate()
        #data loading from edf files
        self._raw_data.load_data()
        #channel selection
        self._raw_data.pick_channels(self.get_pick_channels())
        #filter
        if self.doing_lowpass_filter:
            self._raw_data.filter(0,64)
            self._raw_data.notch_filter(np.arange(60, int((sfreq/2)//60*60+1), 60))#(channel,sample)
        #resample
        if sfreq >256:
            self._raw_data.resample(256)
        data=self._raw_data.get_data().transpose(1,0)#(sample,channel) 
        return data
    
    def get_pick_channels(self):
        """
        Get used channel names
        for CHB, use the common 18 channels
        """
        pick_channels=[]
        
        #23/28 chs -> 18chs
        if self._patient_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 20, 21, 22, 23]:
            pick_channels=['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 
                           'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4', 
                           'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2']
            
        #28 25 22 28 22 28 chs -> 18chs
        elif self._patient_id in [13, 16, 17, 18, 19]:
			
            if  self.get_n_channels() == 28:
                pick_channels=['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 
							   'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4', 
							   'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2']
				
            else: # 22/25
                pick_channels=['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 
				               'C3-P3', 'P3-O1', 'FZ-CZ', 'CZ-PZ', 'FP2-F4', 'F4-C4', 
							   'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2']
        
        return pick_channels
    
    def plot_signal(self, duration, n_channels):
        '''
        plot signals
        '''
        self._raw_data.plot(duration=duration,n_channels=n_channels)
    