#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

"""
=============================================
13b. For patients: Imports single sub STC files, do single sub contrast, and make group (average and SD across subjects)
==============================================
Imports single sub STC files, do the norm, then do the baseline correction, do the contrast  
and finally make group STC by conditions across subjects generated in the previous code 10b 

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
n_subs = len(config_for_gogait.subjects_list_patients)
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

###########################################
DBSinfo = ['OFF']  # run the code separately for ON/OFF
# dbs = DBSinfo[0]
###########################################


# event_type = ['target']
for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
    # condi_name = ['GOu']
       
    norm_kind = ['vector','normVec', 'normVec_zsc']  
    
    if method == 'dSPM':
        norm_kind = norm_kind[1] # selecting the normVec [1] option for dSPM  
    elif method == 'MNE':
        norm_kind = norm_kind[2] # selecting the norm_zscored [2] option for MNE
    
    condi_name = ['GOc', 'GOu', 'NoGo']  
    ncondi = len(condi_name)
    
    contrast_kind = ['GOu_GOc', 'NoGo_GOu']  
    ncontra = len(contrast_kind)
          
      
    ## added on 28/08/2024
    bsln = 'bslnCorr' ## 'bslnCorr' | 'NobslnCorr' 
    

    for di, dbs in enumerate(DBSinfo):
    
        ## n_samples_esti = int(sampling_freq * (isi[ind_isi] + t_end - t_start + 0.002))  
        # estimate the num of time samples per condi/ISI to allocate numpy array
        tsec_start = 0.2 # pre-stimulus(t2/S2/S4) duration in sec
        tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
        n_samples_esti  = int(500*(tsec_start + tsec_end + 1/500)) # one sample added for zeroth loc
        n_chs = 132
        n_verticies = 20484
        n_vectors = 3
        ncondi = len(condi_name)  # nc = 3
        evoked_array_all_sub = np.ones([n_subs, n_chs, n_samples_esti])*np.nan
        stc_data_all_sub_all_condi = np.ones([n_subs, ncondi, n_verticies, n_samples_esti])*np.nan
        
        # goal: normalization across condition for each subject 
        # for each sub, collect all the 3 condi
        for sub_num, subject in enumerate(config_for_gogait.subjects_list_patients):
              
            for ci, condi in enumerate(condi_name): 
           
                print("Processing subject: %s" % subject)
            
                eeg_subject_dir_GOODpatients = op.join(config_for_gogait.eeg_dir_GOODpatients, subject)
                     
                print('  reading the stc numpy array from disk')
                
                extension = condi_name[ci] +'_' + event_type[ei] + '_' + dbs +'_stc_' + orientation +'_' + version +'_' + method
                stc_fname_array_in = op.join(eeg_subject_dir_GOODpatients,
                                     config_for_gogait.base_fname_npy.format(**locals()))
                print("input: ", stc_fname_array_in)
                 
                stc_data_in = np.load(stc_fname_array_in)
                
                if norm_kind == 'vector':
                    stc_data_per_sub = stc_data_in.copy()
                    
                elif norm_kind == 'normVec':
                    if bsln == 'NobslnCorr': 
                        stc_data_in_norm = norm(stc_data_in, axis = 1)
                        stc_data_per_sub = stc_data_in_norm.copy()
             
                    elif bsln == 'bslnCorr': 
                        stc_data_in_norm = norm(stc_data_in, axis = 1)
                        # compute prestim mean and remove 
                        tmin = tsec_start # pre-stim duration
                        pre_stim_sample_size = int(sfreq * tmin )
                        stc_baseline = stc_data_in_norm[:,0:pre_stim_sample_size]
                        stc_baseline_mean = np.mean(stc_baseline, axis = 1) 
                        num_times_pts = np.shape(stc_data_in_norm)[1] 
                        stc_baseline_mean = np.expand_dims(stc_baseline_mean, axis=1)
                        mu = np.repeat(stc_baseline_mean,num_times_pts,axis = 1)
                        stc_data_in_norm_bslncorr =  stc_data_in_norm - mu
                        stc_data_per_sub = stc_data_in_norm_bslncorr.copy()
                    
                elif norm_kind == 'normVec_zsc':
                    stc_data_in_norm = norm(stc_data_in, axis = 1)
                    tmin = 0.2 # pre-stim duration
                    pre_stim_sample_size = int(sfreq * tmin )
                    stc_baseline = stc_data_in_norm[:,0:pre_stim_sample_size]
                    stc_baseline_mean = np.mean(stc_baseline, axis = 1) 
                    stc_baseline_std = np.std(stc_baseline, axis = 1) 
                    num_times_pts = np.shape(stc_data_in_norm)[1] 
                    stc_baseline_mean = np.expand_dims(stc_baseline_mean, axis=1)
                    stc_baseline_std = np.expand_dims(stc_baseline_std, axis=1)
                    mu = np.repeat(stc_baseline_mean,num_times_pts,axis = 1)
                    sig = np.repeat(stc_baseline_std,num_times_pts,axis = 1)
                    stc_data_in_norm_zscored =  (stc_data_in_norm - mu)/sig
                    stc_data_per_sub = stc_data_in_norm_zscored.copy()
                
                # store condi in dim1, vertices in dim2, time in dim3 
                stc_data_per_sub_exp_dim = np.expand_dims(stc_data_per_sub, axis = 0) 
                if ci == 0:
                    stc_data_per_sub_all_condi =  stc_data_per_sub_exp_dim
                else:
                    stc_data_per_sub_all_condi = np.vstack((stc_data_per_sub_all_condi, stc_data_per_sub_exp_dim))                     
                    
            if ampliNormalization == 'AmpliActual': # store subs in dim1, condi in dim2, vertices in dim3, time in dim4
                stc_data_all_sub_all_condi[sub_num,:,:,:] = stc_data_per_sub_all_condi.copy()                      
            elif ampliNormalization == 'AmpliNormAccCondi': ### normalization at single subject across condition, space, time
                tmax = 0.2 + 0.5 # pre-stim + post-stim duration
                sample_size = int(sfreq * tmax )
                # normalize across condition, vertices, time (upto 0.5 s)
                minSTC = np.min(np.min(np.min(stc_data_per_sub_all_condi[:,:,0:sample_size], axis = 2), axis = 1))*np.ones(np.shape(stc_data_per_sub_all_condi))
                maxSTC = np.max(np.max(np.max(stc_data_per_sub_all_condi[:,:,0:sample_size], axis = 2), axis = 1))*np.ones(np.shape(stc_data_per_sub_all_condi))
                stc_data_per_sub_all_condi_minmax = (stc_data_per_sub_all_condi - minSTC)/(maxSTC - minSTC)
                stc_data_all_sub_all_condi[sub_num,:,:,:] = stc_data_per_sub_all_condi_minmax.copy()
            elif ampliNormalization == 'AmpliNormPerCondi': ### normalization at single subject across condition, space, time
                tmax = 0.2 + 0.5 # pre-stim + post-stim duration
                sample_size = int(sfreq * tmax )
                for ci1 in range(3):
                    # normalize across vertices, time (upto 0.5 s)
                    minSTC = np.min(np.min(stc_data_per_sub_all_condi[ci1,:,0:sample_size], axis = 1))*np.ones(np.shape(stc_data_per_sub_all_condi[ci1]))
                    maxSTC = np.max(np.max(stc_data_per_sub_all_condi[ci1,:,0:sample_size], axis = 1))*np.ones(np.shape(stc_data_per_sub_all_condi[ci1]))
                    stc_data_per_sub_all_condi_minmax = (stc_data_per_sub_all_condi[ci1] - minSTC)/(maxSTC - minSTC)
                    stc_data_all_sub_all_condi[sub_num,ci1,:,:] = stc_data_per_sub_all_condi_minmax.copy()
         
        
        ## %% saving the STCs (grand mean and sd) for each contra as numpy array.
        ## !!! uncomment ONLY if you want to recompute the grand average!!! 
        # for ci, contra in enumerate(contrast_kind):
        #     if contra == 'GOu_GOc':
        #         # store subs in dim1, condi in dim2, vertices in dim3, time in dim4
        #         stc_contrast_data = stc_data_all_sub_all_condi[:,1,:,:] - stc_data_all_sub_all_condi[:,0,:,:] 
               
        #     elif contra == 'NoGo_GOu':
        #         # store subs in dim1, condi in dim2, vertices in dim3, time in dim4
        #         stc_contrast_data =  stc_data_all_sub_all_condi[:,2,:,:] - stc_data_all_sub_all_condi[:,1,:,:] 
               
        #     # averaging stc-norm across subjects
        #     stc_contrast_data_avg = np.nanmean(stc_contrast_data, axis = 0)
        #     # std dev stc-norm across subjects
        #     stc_contrast_data_std = np.nanstd(stc_contrast_data, axis = 0)
     
        #     print('\n Writing the stcGAVG to disk: %s'  % contra)
        #     extension = contrast_kind[ci] +'_' + event_type[ei] + '_pd_' +  dbs +'_' +'stcGAVG' +'_evkCondiContrast_SRC_' + version +'_' + method +'_' + ampliNormalization +'_' + bsln 
        #     stc_fname_array = op.join(config_for_gogait.eeg_dir_GOODpatients,
        #                           config_for_gogait.base_fname_avg_npy.format(**locals()))
        #     print("Output: ", stc_fname_array)
        #     np.save(stc_fname_array,  stc_contrast_data_avg)
            
        #     print('\n Writing the stcGSTD to disk: %s'  % contra)
        #     extension = contrast_kind[ci] +'_' + event_type[ei] + '_pd_' +  dbs +'_' +'stcGSTD' +'_evkCondiContrast_SRC_' + version +'_' + method +'_' + ampliNormalization +'_' + bsln 
        #     stc_fname_array = op.join(config_for_gogait.eeg_dir_GOODpatients,
        #                           config_for_gogait.base_fname_avg_npy.format(**locals()))
        #     print("Output: ", stc_fname_array)
        #     np.save(stc_fname_array, stc_contrast_data_std)
        
    
   



    