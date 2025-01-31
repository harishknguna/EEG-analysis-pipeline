#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

""" ...ON GOING... to be continued...
=============================================
21. Import group level all CONDI and exports ROI values as .csv file 
==============================================
Imports the group level condi data (GOc, GOu, NoGo), extract the values from 
the regions of interest (ROIs) used in the previous script 20c.  

Note: 
    1. run separately for HC/OFF/ON; separately for N1/N2/P3;  
    2. As an examplary, here the code is run for HC and P3
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
import seaborn as sns

import config_for_gogait


hc_list = config_for_gogait.subjects_list
n_subs_hc = 23 #len(hc_list)

pd_list = config_for_gogait.subjects_list_patients
n_subs_pd = 15 #len(pd_list)

eeg_dir_hc = config_for_gogait.eeg_dir_GOODremove
eeg_dir_pd = config_for_gogait.eeg_dir_GOODpatients


# version_list = ['GOODremove','CHANremove']
version = 'CHANremove'

event_type = ['target']

orientation = 'varyDipole' #  ['fixDipole', 'varyDipole' ]

method = "dSPM"  # "MNE" | "dSPM"

# ampliNormalization = ['AmpliNormPerCondi', 'AmpliNormAccCondi', 'AmpliActual']
ampliNormalization = 'AmpliActual'


## added on 28/08/2024
bsln = 'bslnCorr' 

################# change these two parameters ##################
stype = 'hc' # 'hc' | 'pd_off' | 'pd_on' 
time_pos = 'P3' # 'N1' | 'N2' | 'P3' 
################################################################

for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
       
    condi_name = ['GOc', 'GOu', 'NoGo']  
    ncondi = len(condi_name)
                     
               
    # for ppt on 08/03/2024
    # t1min = 0.160; t1max = 0.180  # N1
    # t2min = 0.300; t2max = 0.350  # N2 
    # t3min = 0.370; t3max = 0.450  # P3
    
    # changed on 08/08/2024
    t1min = 0.150; t1max = 0.200  # N1
    t2min = 0.250; t2max = 0.350  # N2 
    t3min = 0.370; t3max = 0.450  # P3
    
    if time_pos == 'N1': 
        tmin = t1min 
        tmax = t1max 
    elif time_pos == 'N2': 
        tmin = t2min 
        tmax = t2max 
    elif time_pos == 'P3': 
        tmin = t3min 
        tmax = t3max 
    
    ## n_samples_esti = int(sampling_freq * (isi[ind_isi] + t_end - t_start + 0.002))  
    # estimate the num of time samples per condi/ISI to allocate numpy array
    tsec_start = 0.2 # pre-stimulus(t2/S2/S4) duration in sec
    tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
    n_samples_esti  = int(500*(tsec_start + tsec_end + 1/500)) # one sample added for zeroth loc
    n_chs = 132
    n_verticies = 20484
    n_vectors = 3
    
    print('\n\n'+stype + '\n\n')
    
    if stype == 'pd_off': 
        dbs = 'OFF'
        subjects_list = pd_list
        eeg_dir = eeg_dir_pd
        n_subs = n_subs_pd
          
    elif stype == 'pd_on': 
        dbs = 'ON'
        subjects_list = pd_list
        eeg_dir = eeg_dir_pd
        n_subs = n_subs_pd
        
    elif stype == 'hc': 
        subjects_list = hc_list
        eeg_dir = eeg_dir_hc
        n_subs = n_subs_hc
    
    stc_data_all_sub_all_condi = np.ones([n_subs, ncondi, n_verticies, n_samples_esti])*np.nan
    stc_mean_acti_all_condi_all_sub = np.ones([n_subs,ncondi, 1])*np.nan
              
    ## loading the STCs (per sub) for each condi as numpy array.
    for sub_num, subject in enumerate(subjects_list): 
          
        for ci, condi in enumerate(condi_name): 
       
            print("Processing subject: %s" % subject)
            if stype == 'hc':
                eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
            else:
                eeg_subject_dir_GOODpatients = op.join(config_for_gogait.eeg_dir_GOODpatients, subject)
                 
            print('  reading the stc numpy array from disk')
            
            if stype == 'hc':
                
               extension = condi_name[ci] +'_' + event_type[ei] + '_' +'stc' + '_' + orientation +'_' + version +'_' + method
               stc_fname_array = op.join(eeg_subject_dir_GOODremove,
                                     config_for_gogait.base_fname_npy.format(**locals()))
              
            else:
               extension = condi_name[ci] +'_' + event_type[ei] + '_' + dbs +'_stc_' + orientation +'_' + version +'_' + method
               stc_fname_array = op.join(eeg_subject_dir_GOODpatients,
                                     config_for_gogait.base_fname_npy.format(**locals()))
           
            print("input: ", stc_fname_array)
            stc_data_in = np.load(stc_fname_array)
            # take the norm
            stc_data_in_norm = norm(stc_data_in, axis = 1)
            
            # do the baseline correction 
            # compute prestim mean and remove 
            #tmin = tsec_start # pre-stim duration
            pre_stim_sample_size = int(sfreq * tsec_start)
            stc_baseline = stc_data_in_norm[:,0:pre_stim_sample_size]
            stc_baseline_mean = np.mean(stc_baseline, axis = 1) 
            num_times_pts = np.shape(stc_data_in_norm)[1] 
            stc_baseline_mean = np.expand_dims(stc_baseline_mean, axis=1)
            mu = np.repeat(stc_baseline_mean,num_times_pts,axis = 1)
            stc_data_in_norm_bslncorr =  stc_data_in_norm - mu
            stc_data_per_sub = stc_data_in_norm_bslncorr.copy()
                           
            # store condi in dim1, vertices in dim2, time in dim3 
            stc_data_per_sub_exp_dim = np.expand_dims(stc_data_per_sub, axis = 0) 
            
            if ci == 0:
                stc_data_per_sub_all_condi =  stc_data_per_sub_exp_dim
            else:
                stc_data_per_sub_all_condi = np.vstack((stc_data_per_sub_all_condi, stc_data_per_sub_exp_dim))                     
                
        # store subs in dim1, condi in dim2, vertices in dim3, time in dim4
        stc_data_all_sub_all_condi[sub_num,:,:,:] = stc_data_per_sub_all_condi.copy()  
        
    
#%% extracting the values for each roi

roiType =  'roi' #'roiERP' | 'roiTFR'
# N1 ROIs
# roiNames = ['ACC_L1_HC_NoGo_GOu_N1_SRCCON',
#             'ACC_R1_HC_NoGo_GOu_N1_SRCCON',
#             'ANG_L1_HC_NoGo_GOu_theta_EL_TFRCON',
#             'BA4_L3_HC_NoGo_GOu_P3_SRCCON',
#             'BA4_L1_HC_NoGo_GOu_N1_SRCCON',
#             'BA4_R1_HC_GOu_GOc_N1_SRCCON',
#             'BA6_R1_PDOFF_GOu_GOc_N1_SENCON',  
#             'BA6_L2_PDOFF_GOu_N2', 
#             'INS_R1_PDOFF_NoGo_GOu_theta_EL_TFRCON', 
#             'PH_R1_PDOFF_NoGo_N1', 
#             'PH_R2_HC_NoGo_GOu_N1',
#             'TP_R1_HC_GOc_P3']

## P3 ROIs
roiNames = [
            'ACC_L1_HC_GOu_GOc_P3_SRCCON',
            'ACC_R1_HC_GOu_GOc_P3_SRCCON',
            'ACC_L1_HC_NoGo_GOu_P3_SRCCON',
            'ACC_R1_HC_NoGo_GOu_P3_SRCCON',
            'BA6_L1_HC_GOu_GOc_P3_SRCCON',
            'BA4_L1_HC_GOu_GOc_P3_SRCCON', 
            'BA1_L1_HC_GOu_GOc_P3_SRCCON',
            'BA1_L2_HC_NoGo_GOu_P3_SRCCON',
            'SFS_L1_HC_NoGo_GOu_P3_SRCCON',
            'SFS_R1_HC_NoGo_GOu_P3_SRCCON',
            'MFS_L1_HC_NoGo_GOu_P3_SRCCON',
            'MFS_R1_HC_NoGo_GOu_P3_SRCCON',
            'SM_L1_HC_NoGo_GOu_P3_SRCCON', 
            'TP_R2_PDON_NoGo_GOu_P3_SRCCON',
            
            ]


# Export to an Excel file with multiple sheets
# using Excel instead of CSV, as CSV files do not support multiple sheets

fpath = config_for_gogait.csv_dir
fname = 'ROI_values'+'_for_' + stype +'_at_' + time_pos+'.xlsx'
xsl_fname = op.join(fpath, fname)
# Initialize an Excel writer
with pd.ExcelWriter(xsl_fname) as writer:    
    for roiNum, roi in enumerate(roiNames): 
        total_roi_num = len(roiNames)-1
        print('\ncomputing roi = %s out of %s' %(roiNum,total_roi_num))
        print('\n' + roi + '\n')
        
       
        ## reading the label to a directory
        fpath = config_for_gogait.label_dir_aparc_a2009s
        
        
        if roi[3] == 'L' or roi[4] == 'L'or roi[5]== 'L':
            fname = roi+'-lh.label'
            hemi_small = 'lh'
        if roi[3] == 'R' or roi[4] == 'R' or roi[5]== 'R':
            fname = roi+'-rh.label'
            hemi_small = 'rh'
        
        label_fname = op.join(fpath, fname)
        func_label = mne.read_label(label_fname)  
        
      
        timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
           
                            
        ## for each ROI collect values across all sub and all condi
        for sub_num, subject in enumerate(subjects_list): # per sub
            for ci3, condi3 in enumerate(condi_name): # per condi 
                stc_data_per_sub_per_condi = stc_data_all_sub_all_condi[sub_num,ci3,:,:]  ### consider CONDI data for brain plot
                vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
                stc_per_condi = mne.SourceEstimate(stc_data_per_sub_per_condi, vertices = vertno,
                                          tstep = 1/sfreq, tmin = - 0.2,
                                          subject = 'fsaverage')  
                stc_per_condi_label = stc_per_condi.copy().crop(tmin = tmin, tmax = tmax).in_label(func_label)
                stc_label_data = stc_per_condi_label.data
                stc_label_data_exp_dim = np.expand_dims(stc_label_data, axis = 0)
                if ci3 == 0:
                    stc_label_data_condi = stc_label_data_exp_dim.copy()
                else:
                    stc_label_data_condi = np.vstack((stc_label_data_condi, stc_label_data_exp_dim.copy())) 
            
            ## AVG data: mean across time points 
            stc_label_mean_time_all_condi = stc_label_data_condi.copy().mean(axis = 2)
            ## mean activitity of label vertices
            stc_mean_acti_all_condi = stc_label_mean_time_all_condi.copy().mean(axis = 1)
            
            ## collect for all subs
            stc_mean_acti_all_condi_all_sub[sub_num,:,:] = stc_mean_acti_all_condi.copy().reshape(-1,1)
        
        ## export mean activity of this ROI at this T window to excel 
        
        # Create the pandas DataFrame with column name is provided explicitly 
        df_roi = pd.DataFrame()
        df_roi['subs'] = subjects_list
        df_roi['GOc'] = pd.Series(stc_mean_acti_all_condi_all_sub[:,0,:].flatten())
        df_roi['GOu'] = pd.Series(stc_mean_acti_all_condi_all_sub[:,1,:].flatten())
        df_roi['NoGo'] = pd.Series(stc_mean_acti_all_condi_all_sub[:,2,:].flatten())
        
        ## exporting the values as xsl file with ROIs in each page
        df_roi.to_excel(writer, sheet_name = roi, index=False)
   

#%%



           


    