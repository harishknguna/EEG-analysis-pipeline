#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:34:24 2024
 https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html
 https://mne.tools/stable/generated/mne.time_frequency.tfr_multitaper.html#mne.time_frequency.tfr_multitaper

@author: harish.gunasekaran

=============================================
08a: import single sub epochs, make group morlet wavelet TF, and plot the power 
==============================================
Imports the MNE single sub TFR object obtained in script 07 and do group
morlet wavelet TF, per condition and plot the TF power 
(baseline uncorrected and corrected) in the separte html files for each DBS condition. 

make sure to run this command in the console: %matplotlib qt
"""


import os.path as op

import mne
from mne.parallel import parallel_func
from mne.channels.montage import get_builtin_montages
from warnings import warn
from pymatreader import read_mat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from scipy import signal
from scipy import stats 
from scipy.linalg import norm
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse 
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid, inset_locator
from mne.stats import spatio_temporal_cluster_test, summarize_clusters_stc
from mne.time_frequency import tfr_multitaper, tfr_morlet

import config_for_gogait
n_subs = len(config_for_gogait.subjects_list)
fs_dir = fetch_fsaverage(verbose=True)
template_subjects_dir = op.dirname(fs_dir)

# ampliNormalization = ['AmpliNorm', 'AmpliActual']
# ampliNormalization = 'AmpliActual'

event_type = ['target']
# event_type = ['cue', 'target']
# version_list = ['GOODremove','CHANremove']
version = 'CHANremove'
ep_extension = 'TF'

waveType = 'morlet'
numType = 'complex'  # 'real' | 'complex'

tfr_freqs = np.linspace(3,40,num = 40, endpoint= True)
n_TF_freqs = len(tfr_freqs)

baseline = 'bslnNoCorr' # 'bslnCorr' | 'bslnNoCorr'

if version == 'GOODremove':
    n_chs = 128
elif version == 'CHANremove':
    n_chs = 103

for ei, evnt in enumerate(event_type):
    report = mne.Report()
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
    # condi_name = ['GOu'] 
    condi_name = ['GOc', 'GOu', 'NoGo']
    num_condi_with_eve = 6
   
   
    sampling_freq = 500 # in hz
    
 
    for ci, condi in enumerate(condi_name): 
        
        print("condition: %s" % condi)        
  
        for sub_num, subject in enumerate(config_for_gogait.subjects_list): 
            print("Processing subject: %s" % subject)
            eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
                  
            #% reading the epochs from disk
            print('Reading the epochs from disk')
            extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_'+ ep_extension +'_epo'
           
            epochs_fname = op.join(eeg_subject_dir_GOODremove,
                                      config_for_gogait.base_fname.format(**locals()))
           
            # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
            print("Input: ", epochs_fname)
            epochs = mne.read_epochs(epochs_fname, proj=True, preload=True).pick('eeg', exclude='bads')  
            epochs = epochs.set_eeg_reference(projection=True)
            info = epochs.info   # taking info,events,id from the last sub's data (same for all subs)
            
            if baseline == 'bslnCorr':
                print('Importing the TFR_power_bslnCorr from disk')
                extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_PWR_CMPLX_ABS_bslnCorr-tfr.h5'
            elif baseline == 'bslnNoCorr':
                print('Importing the TFR_power from disk')
                extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_PWR_CMPLX_ABS-tfr.h5'
                
                
            tfr_fname = op.join(eeg_subject_dir_GOODremove,
                                    config_for_gogait.base_fname_no_fif.format(**locals()))
           
            print("Output: ", tfr_fname)
            pwr_read = mne.time_frequency.read_tfrs(tfr_fname)
            
            pwr_per_sub = pwr_read[0].data.mean(axis = 0) # avg across epochs
            
            
            # store sub in dim1, chs in dim2, freq in dim 3, time in dim4 
            pwr_per_sub_exp_dim = np.expand_dims(pwr_per_sub, axis = 0) 
            
            if sub_num == 0:
                tf_pwr_array_all_sub = pwr_per_sub_exp_dim
                
            else:
                tf_pwr_array_all_sub = np.vstack((tf_pwr_array_all_sub, pwr_per_sub_exp_dim))                     
                
        # averaging TF arrays across subjects
        tf_pwr_array_avg_sub = np.mean(tf_pwr_array_all_sub, axis = 0)
        
        #%% put them in the respective MNE containers
        print('Importing the TFR_times to disk')
        extension =  'TFR_times_'+ event_type[0] +'.npy'
        tfr_fname = op.join(config_for_gogait.eeg_dir_GOODremove,
                                config_for_gogait.base_fname_generic.format(**locals()))
     
        print("Input: ", tfr_fname)
        tfr_times = np.load(tfr_fname)  
        
        power = mne.time_frequency.AverageTFR(info = info, 
                                              data = tf_pwr_array_avg_sub, 
                                              times = tfr_times, 
                                              freqs = tfr_freqs, 
                                              nave = 1, 
                                              comment = 'avgPower' ,
                                              method = 'morlet')
             
        # plot the avg TFRs 
        # don't forget %matplotlib qt
        plt.rcParams.update({'font.size': 14})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        fig.suptitle('TF power: bslnNocorr and corr')
        power.plot(baseline=None, title = 'auto',
                    exclude = 'bads', combine = 'mean', axes = ax1)
        ax1.set_title('Power bslnNoCorr')
        plt.rcParams.update({'font.size': 14})
        power.plot(baseline=(-0.5, -0.1), mode="logratio", title = 'auto',
                    exclude = 'bads', combine = 'mean', axes = ax2)
        ax2.set_title('Power bslnCorr')
        fig.tight_layout()
    
        report.add_figure(fig, title = condi + '_TF_morlet', replace = True)
                   
        plt.close('all')
        
        #%%  saving the complex epochsTFR: no baseline correction applied 
        # print('\n Writing the grand averaged complex epochsTFR to disk')
        # tfr_group_data =  power.copy() 
        # extension =  condi_name[ci] +'_' + event_type[ei] +'_' + version + '_' + waveType +'_'+ numType +'_ABS_'+ baseline + '_grand_avg-tfr.h5'
        # tfr_fname = op.join(config_for_gogait.eeg_dir_GOODremove,
        #                         config_for_gogait.base_fname_generic.format(**locals()))
       
        # print("Output: ", tfr_fname)
        # mne.time_frequency.write_tfrs(tfr_fname, tfr_group_data, overwrite=True)  
      
        
        
       
       
    # finally saving the report after the for subject loop ends.     
    print('Saving the reports to disk')  
    report.title = 'Group (n = '+ str(n_subs) + ') sub TF : ' + evnt + '_' +  version + '_' + waveType 
    extension = 'group_sub_TF'
    report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
    report.save(report_fname+'_' + evnt + '_' + version+ '_' + waveType +'_' + numType +'.html', overwrite=True)            
              
       
              

    