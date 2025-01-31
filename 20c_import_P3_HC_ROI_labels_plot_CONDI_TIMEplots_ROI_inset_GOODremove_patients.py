#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

""" ...ON GOING... to be continued...
=============================================
20c. Import group level all CONDI and plot the timeplots across ROIs extracted
==============================================
Imports the group level condi data (GOc, GOu, NoGo), plot the time course of the 
regions of interest (ROIs) extracted in the previous script 20a.  

Note: 
    1. run separately for HC/OFF/ON; separately for N1/N2/P3; POSITIVE contrast just for P3 
    2. As an examplary, here the code is run for HC and P3, positive contrast 
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

DBSinfo = ['ON', 'OFF']  

dataKind = ['stcGAVG', 'stcGSTD']

## added on 28/08/2024
bsln = 'bslnCorr' ## 'bslnCorr' | 'NobslnCorr' 

time_pos = 'P3' ##  ['N1', 'N2', 'P3']


# event_type = ['target']
for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
    # contrast_kind = ['GOu']
       
    norm_kind = ['vector','normVec', 'normVec_zsc']  
    
    if method == 'dSPM':
        norm_kind = norm_kind[1] # selecting the normVec [1] option for dSPM  
    elif method == 'MNE':
        norm_kind = norm_kind[2] # selecting the norm_zscored [2] option for MNE
    
    # contrast_kind = ['NoGo_GOc', 'NoGo_GOu', 'GOu_GOc']  
    
    condi_name = ['GOc', 'GOu', 'NoGo']  
    ncondi = len(condi_name)
    
    contrast_kind = ['GOu_GOc', 'NoGo_GOu']  
    ncontra = len(contrast_kind)
    
    condi_name_dbs = ['GOc', 'GOu', 'NoGo']
              
               
    # for ppt on 08/03/2024
    # t1min = 0.160; t1max = 0.180  # N1
    # t2min = 0.300; t2max = 0.350  # N2 
    # t3min = 0.370; t3max = 0.450  # P3
    
    # changed on 08/08/2024
    t1min = 0.150; t1max = 0.200  # N1
    t2min = 0.250; t2max = 0.350  # N2 
    t3min = 0.370; t3max = 0.450  # P3
       
    ## n_samples_esti = int(sampling_freq * (isi[ind_isi] + t_end - t_start + 0.002))  
    # estimate the num of time samples per condi/ISI to allocate numpy array
    tsec_start = 0.2 # pre-stimulus(t2/S2/S4) duration in sec
    tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
    n_samples_esti  = int(500*(tsec_start + tsec_end + 1/500)) # one sample added for zeroth loc
    n_chs = 132
    n_verticies = 20484
    n_vectors = 3
               
    sub_type = ['hc']
            
    for stype in sub_type: 
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
                  
        ## loading evkContrasts group avg/sd across sub
        for dk, dkind in enumerate(dataKind): 
            for ci, condi in enumerate(condi_name): # three conditions
        
                print('  reading the group_'+ dkind +'_numpy array from disk')
                
                if stype == 'hc':
                    
                   extension = condi_name[ci] +'_' + event_type[ei] + '_hc_' + dkind+'_' + 'evoked_' + version +'_' + method +'_' + ampliNormalization +'_' + bsln
                   stc_fname_array = op.join(config_for_gogait.eeg_dir_GOODremove,
                                         config_for_gogait.base_fname_avg_npy.format(**locals()))
                  
                else:
                   extension = condi_name[ci] +'_' + event_type[ei] + '_pd_' + dbs +'_' + dkind+'_' + 'evoked_' + version +'_' + method +'_' + ampliNormalization +'_' + bsln
                   stc_fname_array = op.join(config_for_gogait.eeg_dir_GOODpatients,
                                         config_for_gogait.base_fname_avg_npy.format(**locals()))
               
                print("input: ", stc_fname_array)
                stc_data_in = np.load(stc_fname_array)
                               
                # store condi in dim1, vertices in dim2, time in dim3 
                stc_group_exp_dim = np.expand_dims(stc_data_in, axis = 0) 
                
                if ci == 0:
                    stc_group_sub_all_condi =  stc_group_exp_dim.copy()
                else:
                    stc_group_sub_all_condi = np.vstack((stc_group_sub_all_condi,  stc_group_exp_dim.copy()))    
            
                               
            if stype == 'pd_off' and dkind == 'stcGAVG':
                stc_condi_avg_off = stc_group_sub_all_condi.copy()
              
            elif stype == 'pd_on' and dkind == 'stcGAVG':
                stc_condi_avg_on = stc_group_sub_all_condi.copy()
              
            elif stype == 'hc'and dkind == 'stcGAVG':
                stc_condi_avg_hc = stc_group_sub_all_condi.copy()
            
            elif stype == 'pd_off' and dkind == 'stcGSTD':
                stc_condi_std_off = stc_group_sub_all_condi.copy()
              
            elif stype == 'pd_on' and dkind == 'stcGSTD':
                stc_condi_std_on = stc_group_sub_all_condi.copy()
              
            elif stype == 'hc'and dkind == 'stcGSTD':
                stc_condi_std_hc = stc_group_sub_all_condi.copy()
            
            del  stc_group_sub_all_condi
                    
    
#%% plotting contrast map
## first: do the scaling
#% plot the figures 
report = mne.Report()
scale = 'ptileScale' # 'ptileScale' | 'globalScale' | 'timeScale' | 'condiScale'
   

roiType =  'roi' #'roiERP' | 'roiTFR'
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
    
for roiNum, roi in enumerate(roiNames): 
    total_roi_num = len(roiNames)-1
    print('\ncomputing roi = %s out of %s' %(roiNum,total_roi_num))
    print('\n' + roi + '\n')
    
    ## plotting figures
    fig, axd = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, 
                            figsize=(4, 4))
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
   
    time_pos2 = 'N2'
    tmin = t2min 
    tmax = t2max 
    timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
       
    for sb, stype in enumerate(sub_type): 
       
        print('\n\n'+stype + '\n\n')
        
        if stype == 'hc': 
            subjects_list = hc_list
            eeg_dir = eeg_dir_hc
            n_subs = n_subs_hc
            
            # taking contra and condi data
            data_stcGAVG_condi = stc_condi_avg_hc.copy()
            data_stcGSTD_condi = stc_condi_std_hc.copy()
            ylim = [-1.5, 12.5]
        
        if stype == 'pd_off': 
            dbs = 'OFF'
            subjects_list = pd_list
            eeg_dir = eeg_dir_pd
            n_subs = n_subs_pd
            # taking contra and condi data
            data_stcGAVG_condi = stc_condi_avg_off.copy()
            data_stcGSTD_condi = stc_condi_std_off.copy()
            
            ylim = [-0.75, 6.25]
            
         
        elif stype == 'pd_on': 
            dbs = 'ON'
            subjects_list = pd_list
            eeg_dir = eeg_dir_pd
            n_subs = n_subs_pd
            
            # taking contra and condi data
            data_stcGAVG_condi = stc_condi_avg_on.copy()
            data_stcGSTD_condi = stc_condi_std_on.copy()
            ylim = [-0.75, 6.25]
        
        #% 2. second collect group (avg/std) activations of ROI per contrast
        for dk, dkind in enumerate(dataKind): 
            for ci4, condi4 in enumerate(condi_name):# per condi
                if dkind == 'stcGAVG':
                    stc_data_per_condi =  data_stcGAVG_condi[ci4,:,:]  ### consider CONDI data for brain plot
                elif dkind == 'stcGSTD':
                    stc_data_per_condi =  data_stcGSTD_condi[ci4,:,:]
                
                vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
                stc_per_condi = mne.SourceEstimate(stc_data_per_condi, vertices = vertno,
                                          tstep = 1/sfreq, tmin = - 0.2,
                                          subject = 'fsaverage')  
                
                stc_per_condi_label = stc_per_condi.copy().in_label(func_label)
                stc_label_data_for_condi = stc_per_condi_label.data 
                
                stc_label_data_exp_dim = np.expand_dims(stc_label_data_for_condi, axis = 0)
                if ci4 == 0:
                    stc_label_data_condi = stc_label_data_exp_dim.copy()
                else:
                    stc_label_data_condi = np.vstack((stc_label_data_condi, stc_label_data_exp_dim.copy())) 
            
            ## AVG data
            stc_mean_acti_all_condi = stc_label_data_condi.copy().mean(axis = 1)
            
            if dkind == 'stcGAVG':
                stcGAVG_mean_acti_all_condi = stc_mean_acti_all_condi.copy()
            elif dkind == 'stcGSTD':
                stcGSTD_mean_acti_all_condi = stc_mean_acti_all_condi.copy()
        
        #%
        ## export mean activity of this ROI at this T window to excel 
        # Create the pandas DataFrame with column name is provided explicitly 
        df_stcGAVG = pd.DataFrame()
        df_stcGAVG['times'] = pd.Series(stc_per_condi.times.flatten())
        df_stcGAVG['GOc'] = pd.Series(stcGAVG_mean_acti_all_condi[0,:].flatten())
        df_stcGAVG['GOu'] = pd.Series(stcGAVG_mean_acti_all_condi[1,:].flatten())
        df_stcGAVG['NoGo'] = pd.Series(stcGAVG_mean_acti_all_condi[2,:].flatten())
        # df_stcGAVG['GOu_GOc'] =  pd.Series(stcGAVG_mean_acti_all_contra[0,:].flatten())
        # df_stcGAVG['NoGo_GOu'] =  pd.Series(stcGAVG_mean_acti_all_contra[1,:].flatten())
        
        df_stcGSTD = pd.DataFrame()
        df_stcGSTD['times'] = pd.Series(stc_per_condi.times.flatten())
        df_stcGSTD['GOc'] = pd.Series(stcGSTD_mean_acti_all_condi[0,:].flatten())
        df_stcGSTD['GOu'] = pd.Series(stcGSTD_mean_acti_all_condi[1,:].flatten())
        df_stcGSTD['NoGo'] = pd.Series(stcGSTD_mean_acti_all_condi[2,:].flatten())
        # df_stcGSTD['GOu_GOc'] =  pd.Series(stcGSTD_mean_acti_all_contra[0,:].flatten())
        # df_stcGSTD['NoGo_GOu'] =  pd.Series(stcGSTD_mean_acti_all_contra[1,:].flatten())
        
        df_stcLowerCI = pd.DataFrame()
        df_stcLowerCI['GOc'] =  df_stcGAVG['GOc'] - df_stcGSTD['GOc']/np.sqrt(n_subs)
        df_stcLowerCI['GOu'] = df_stcGAVG['GOu'] - df_stcGSTD['GOu']/np.sqrt(n_subs)
        df_stcLowerCI['NoGo'] =df_stcGAVG['NoGo'] - df_stcGSTD['NoGo']/np.sqrt(n_subs)
        
        df_stcUpperCI = pd.DataFrame()
        df_stcUpperCI['GOc'] =  df_stcGAVG['GOc'] + df_stcGSTD['GOc']/np.sqrt(n_subs)
        df_stcUpperCI['GOu'] = df_stcGAVG['GOu'] + df_stcGSTD['GOu']/np.sqrt(n_subs)
        df_stcUpperCI['NoGo'] =df_stcGAVG['NoGo'] + df_stcGSTD['NoGo']/np.sqrt(n_subs)
        
        
                              
        condiContraName = condi_name +  contrast_kind
               
        font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 12,
            }
        # colors = ['xkcd:lightgreen','xkcd:salmon','xkcd:grey']
        line_colors = ['xkcd:lightgreen','xkcd:salmon','xkcd:grey','xkcd:light blue','xkcd:magenta']
        
        l1 = axd.plot(df_stcGAVG['times'], df_stcGAVG['GOc'], color = line_colors[0], linewidth=1.5, label = 'GOc')
        axd.fill_between(df_stcGAVG['times'],  df_stcLowerCI['GOc'],df_stcUpperCI['GOc'],
                         color = line_colors[0], alpha=.5)
        l2 = axd.plot(df_stcGAVG['times'], df_stcGAVG['GOu'], color = line_colors[1], linewidth=1.5, label = 'GOu')
        axd.fill_between(df_stcGAVG['times'],  df_stcLowerCI['GOu'],df_stcUpperCI['GOu'],
                         color = line_colors[1], alpha=.5)
        l3 = axd.plot(df_stcGAVG['times'], df_stcGAVG['NoGo'], color = line_colors[2], linewidth=1.5, label = 'NoGo')
        axd.fill_between(df_stcGAVG['times'],  df_stcLowerCI['NoGo'],df_stcUpperCI['NoGo'],
                         color = line_colors[2], alpha=.5)
              
       
        axd.axvspan(t1min,t1max,facecolor='blue', alpha=0.1)
        axd.axvspan(t2min,t2max,facecolor='blue', alpha=0.1)
        axd.axvspan(t3min,t3max,facecolor='magenta', alpha=0.1)
        axd.set_xlabel('time (ms)', fontdict=font)
        axd.set_ylabel('activations '+ r'$\sqrt{F}$', fontdict=font)
        
        axd.set_ylim(ylim)
        axd.grid()
        axd.set_title(stype)
        fig.suptitle(roiNames[roiNum])
        
        axd.axvline(x = 0, color = 'k')
        axd.axhline(y = 0, color = 'k')
        
        plt.legend(loc='upper right')
        plt.tight_layout()
    
        # Select stc values of the current ROI in 'times' between tmin and tmax
        df_stcGAVG_filtered =  df_stcGAVG[( df_stcGAVG['times'] >= tmin) & ( df_stcGAVG['times'] <= tmax)]
        df_stcGAVG_filtered = df_stcGAVG_filtered.drop('times', axis = 1)
        stcGAVG_max =  df_stcGAVG_filtered.max().max()
        
            
        ## Create an inset (per sub type: HC/OFF/ON) plot inside the first subplot
        ## plotting the inset of stc map: grand averaged across condi and time (N1/N2/P3)
        
        stc_data_avg_condi = data_stcGAVG_condi.mean(axis=0)
        vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
        stc_avg_condi = mne.SourceEstimate(stc_data_avg_condi, vertices = vertno,
                                  tstep = 1/sfreq, tmin = - 0.2,
                                  subject = 'fsaverage')  
        stc_grand_avg = stc_avg_condi.copy().crop(tmin = tmin, tmax = tmax).mean()
       
        ## brain views depends on ROI
            # just for N2/HC adding these two here    
        if roi[0:3] == 'SFG' or roi[0:3] == 'OCP' or roi[0:3] == 'ACC' or roi[0:3] == 'PCC' or roi[0:3] == 'BA6' or roi[0:3] == 'CUN' or  roi[0:4] == 'PCUN' or roi[0:3] == 'MCC' or roi[0:2] == 'HC' or roi[0:2] == 'PH':
            br_view = 'med'
            cbar_vertical = False
            cbar_tick_loc = 'bottom'
            cbar_wt = 0.6
            cbar_ht = 0.1
            
        elif roi[0:3] == 'SFS' or roi[0:3] == 'BA4' or roi[0:3] == 'BA1' or roi[0:2] == 'SM' or roi[0:3] == 'SPL' or roi[0:3] == 'SOS'  or roi[0:3] == 'PCG':
            br_view = 'dor'
            cbar_vertical = True
            cbar_wt = 0.1
            cbar_ht = 0.8
            cbar_tick_loc = 'left'
            
        elif roi[0:2] == 'TP' or roi[0:3] == 'OFC' or roi[0:3] == 'OCP' or roi[0:3] == 'OTG' or roi[0:2] == 'VA':
            br_view = 'ven'
            cbar_vertical = True
            cbar_wt = 0.1
            cbar_ht = 0.8
            cbar_tick_loc = 'left'
            
        elif  roi[0:3] == 'MFS' or roi[0:3] == 'MFG' or roi[0:3] == 'IFG' or roi[0:2] == 'IF'  or roi[0:3] == 'INS' or roi[0:3] == 'SMG' or roi[0:3] == 'AIP' or roi[0:3] == 'ANG' or roi[0:3] == 'STS' or roi[0:3] == 'SCG':
           # roi[0:3] == 'SFG' put this in med for N2/HC
            br_view = 'lat'
            cbar_vertical = False
            cbar_wt = 0.6
            cbar_ht = 0.1
            cbar_tick_loc = 'bottom'
        
        brain_kwargs = dict(alpha=1, background="white", cortex="low_condist")
        clrBar = True
                      
        Brain = mne.viz.get_brain_class()
        ## Download fsaverage files
        fs_dir = fetch_fsaverage(verbose=True)
        template_subjects_dir = op.dirname(fs_dir)
        brain_kwargs = dict(alpha=1, background="white",) # cortex="low_contrast")

        brain = Brain(
                    subject = "fsaverage",
                    subjects_dir = template_subjects_dir,
                    surf ="inflated", #"inflated",pial
                    views = br_view,
                    hemi = hemi_small, #"split",
                    **brain_kwargs)
        
        grown_label = mne.label.select_sources(template_subject, 
                                               func_label, location='center', 
                                               extent= 7.0, 
                                               grow_outside=True)
                                               
        brain.add_label(grown_label, color='black')
        screenshot1 = brain.screenshot()
    
    
        # stc_fig1.close() 
        inset_ax = axd.inset_axes([-0.025, 0.59, 0.4, 0.4])  # [left, bottom, width, height]
        inset_ax.imshow(screenshot1)
        inset_ax.axis('off')
        plt.show()
        brain.close() 
       
    report.add_figure(fig, title = str(roiNum) +'_' + roiNames[roiNum], replace = True)        
    plt.close('all')       
              
# finally saving the report after the for subject loop ends.     

print('Saving the reports to disk')  
report.title = 'Group_TIMEplots_ROIs_of_'+ stype +'_'+time_pos + '_'+ evnt+ ': '+ version + '_'  + method + '_' + ampliNormalization + '_' + scale +'_' + bsln
extension = 'Group_TIMEplots_ROIs_of_'+ stype +'_'+time_pos 
report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
report.save(report_fname+'_at_'+ evnt +'_' + version+ '_' + method + '_' + ampliNormalization + '_' + bsln +'.html', overwrite=True)            
              

        



           


    