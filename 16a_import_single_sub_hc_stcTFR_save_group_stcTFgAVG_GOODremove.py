#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

"""
=============================================
16a. Imports single sub stcTFR files and make group stcTFR (average and SD across subjects)
==============================================
Imports single sub stcTFR files, do the norm, then do the baseline correction, 
and finally make group STC by conditions across subjects generated in the previous code 10a 

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


import config_for_gogait
n_subs = len(config_for_gogait.subjects_list)
fs_dir = fetch_fsaverage(verbose=True)
template_subjects_dir = op.dirname(fs_dir)

# version_list = ['GOODremove','CHANremove']
version = 'CHANremove'

event_type = ['target']

orientation = 'varyDipole' #  ['fixDipole', 'varyDipole' ]

# method = "dSPM"
method = 'dSPM'

# ampliNormalization = ['AmpliNormPerCondi', 'AmpliNormAccCondi', 'AmpliActual']
ampliNormalization = 'AmpliActual'

baseline = 'bslnCorr' # 'bslnCorr' | 'bslnNoCorr'
ep_extension = 'TF'
waveType = 'morlet' # 'morlet' | 'multitaper'

f1min = 3.0; f1max = 7.0 # in Hz
f2min = 13.0 ; f2max = 21.0

################# CHANGE HERE ########################################
freq_band_name ='beta_low' #'theta' |'beta_low'
######################################################################

for ei, evnt in enumerate(event_type):
    sfreq = 500
    decim = 5 # when doing tfr computation
    sfreq = sfreq/decim
    tfr_freqs = np.linspace(3,40,num = 40, endpoint= True)
    n_TF_freqs = len(tfr_freqs)
    # The files live in:
    template_subject = "fsaverage"   
    # condi_name = ['GOu']
       
    norm_kind = ['vector', 'normVec', 'normVec_zsc']  
    
    if method == 'dSPM':
        norm_kind = norm_kind[1] # selecting the normVec [1] option for dSPM  
    elif method == 'MNE':
        norm_kind = norm_kind[2] # selecting the norm_zscored [2] option for MNE
    
    condi_name = ['GOc', 'GOu', 'NoGo']  
    ncondi = len(condi_name)
    
   
    
    ## n_samples_esti = int(sampling_freq * (isi[ind_isi] + t_end - t_start + 0.002))  
    # estimate the num of time samples per condi/ISI to allocate numpy array
    
    if evnt == 'cue' and ep_extension == 'TF':
        tsec_start = 0.8 # pre-stimulus(t2/S2/S4) duration in sec
        tsec_end = 1.3 # post-stimulus (t3/S8/S16) duration in sec
    elif evnt == 'target' and ep_extension == 'TF':
        tsec_start = 1.0 # pre-stimulus(t2/S2/S4) duration in sec
        tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
    else:
        tsec_start = 0.2 # pre-stimulus(t2/S2/S4) duration in sec
        tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
       
    n_samples_esti  = int(sfreq*(tsec_start + tsec_end + 1/sfreq)) # one sample added for zeroth loc
    # n_chs = 132
    n_verticies = 20484
    n_vectors = 3
    ncondi = len(condi_name)  # nc = 3
    # evoked_array_all_sub = np.ones([n_subs, n_chs, n_samples_esti])*np.nan
    # stc_data_all_sub_all_condi = np.ones([n_subs, ncondi, n_verticies, n_samples_esti])*np.nan
    
    # goal: normalization across condition for each subject 
    # for each sub, collect all the 3 condi
    for ci, condi in enumerate(condi_name): 
        
        for sub_num, subject in enumerate(config_for_gogait.subjects_list): 
            print("Processing subject: %s" % subject)
        
            eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
                 
            print('  reading the stc numpy array from disk')
            
            extension = condi +'_' + ep_extension +'_' + 'bslnCorr_avgFreq'  +'_' + event_type[ei] + '_' +'stc_' + freq_band_name +'_' + waveType + '_' + orientation +'_' + version +'_' + method 
            
            stc_fname_array_in = op.join(eeg_subject_dir_GOODremove,
                                 config_for_gogait.base_fname_npy.format(**locals()))
            print("input: ", stc_fname_array_in)
             
            stc_data_in = np.load(stc_fname_array_in)
            stc_data_per_condi = stc_data_in.copy()
            
            # store subs in dim1, condi in dim2, vertices in dim2, time in dim3 
            stc_data_per_condi_exp_dim = np.expand_dims(stc_data_per_condi, axis = 0) 
            
            if sub_num == 0:
                stc_data_per_condi_all_sub =  stc_data_per_condi_exp_dim
            else:
                stc_data_per_condi_all_sub = np.vstack((stc_data_per_condi_all_sub, stc_data_per_condi_exp_dim))  
        
        # averaging stc across subjects
        stc_data_avg_sub_per_condi = np.mean(stc_data_per_condi_all_sub, axis = 0)
        # std stc across subjects
        stc_data_std_sub_per_condi = np.std(stc_data_per_condi_all_sub, axis = 0)
            
        # store condi in dim1, vertices in dim2, time in dim3 
        stc_data_avg_sub_per_condi_exp_dim = np.expand_dims(stc_data_avg_sub_per_condi, axis = 0) 
        stc_data_std_sub_per_condi_exp_dim = np.expand_dims(stc_data_std_sub_per_condi, axis = 0) 
        
        if ci == 0:
            stc_data_avg_sub_all_condi =   stc_data_avg_sub_per_condi_exp_dim
            stc_data_std_sub_all_condi =   stc_data_std_sub_per_condi_exp_dim
        else:
            stc_data_avg_sub_all_condi = np.vstack((stc_data_avg_sub_all_condi, stc_data_avg_sub_per_condi_exp_dim))
            stc_data_std_sub_all_condi = np.vstack((stc_data_std_sub_all_condi, stc_data_std_sub_per_condi_exp_dim))
                
    
    #%% saving the STCs (grand mean and sd) for each condi as numpy array.
    # !!! uncomment ONLY if you want to recompute the grand average!!! 
    
    # for ci, condi in enumerate(condi_name):
    #     print('\n Writing the stcGAVG to disk: %s'  % condi)
    #     extension = condi_name[ci] +'_' + event_type[ei] + '_hc_' +'stcTFgAVG_bslnCorr' + '_' + freq_band_name+ '_' + version +'_' + method +'_' + ampliNormalization 
    #     stc_fname_array = op.join(config_for_gogait.eeg_dir_GOODremove,
    #                           config_for_gogait.base_fname_avg_npy.format(**locals()))
    #     print("Output: ", stc_fname_array)
    #     stc_avg_data = stc_data_avg_sub_all_condi[ci,:,:]
    #     np.save(stc_fname_array, stc_avg_data)
        
    #     print('\n Writing the stcGSTD to disk: %s'  % condi)
    #     extension = condi_name[ci] +'_' + event_type[ei] + '_hc_' +'stcTFgSTD_bslnCorr' + '_' + freq_band_name+ '_' + version +'_' + method +'_' + ampliNormalization 
    #     stc_fname_array = op.join( config_for_gogait.eeg_dir_GOODremove,
    #                           config_for_gogait.base_fname_avg_npy.format(**locals()))
    #     print("Output: ", stc_fname_array)
    #     stc_std_data = stc_data_std_sub_all_condi[ci,:,:]
    #     np.save(stc_fname_array, stc_std_data)
        
      
   



    