#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

""" ...ON GOING... to be continued...
=============================================
20b. Import group level SRC CONTRAST and plot the contrast with ROIs 
==============================================
Imports the group level contrast data (source contrast or sensor contrast), 
plot the contrast and marks the regions of interest (ROIs) extracted in the previous script 20a.  

Note: 
    1. this code should be run several times (trial-and-error) to check if the ROI is selected
    2. run one ROI/one hemisphere/one contrast at a time 
    3. run separately for HC/OFF/ON; separately for N1/N2/P3; POSITIVE contrast just for P3 
    4. As an examplary, here the code is run for HC and P3, positive contrast 
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

hc_list = config_for_gogait.subjects_list
n_subs_hc = len(hc_list)

pd_list = config_for_gogait.subjects_list_patients
n_subs_pd = len(pd_list)

eeg_dir_hc = config_for_gogait.eeg_dir_GOODremove
eeg_dir_pd = config_for_gogait.eeg_dir_GOODpatients

# version_list = ['GOODremove','CHANremove']
version = 'CHANremove'

event_type = ['target']

orientation = 'varyDipole' #  ['fixDipole', 'varyDipole' ]

# method = "dSPM"
method = 'dSPM'

# ampliNormalization = ['AmpliNormPerCondi', 'AmpliNormAccCondi', 'AmpliActual']
ampliNormalization = 'AmpliActual'

DBSinfo = ['OFF','ON']  #, 'OFF'
contrast_kind = ['GOu_GOc', 'NoGo_GOu']
condi_name_dbs = ['GOc', 'GOu', 'NoGo']

## added on 28/08/2024
bsln = 'bslnCorr' ## 'bslnCorr' | 'NobslnCorr' 

contraType = 'SRCCON' #['SENCON', 'SRCCON'] run one at a time. 


if contraType == 'SENCON':
    evkContr = 'evkCondiContrast_SEN'
elif contraType == 'SRCCON':
    evkContr = 'evkCondiContrast_SRC'


##### all the changes are made here ##################
stype = 'hc' ## ['hc','pd_off', 'pd_on']
time_pos = 'P3' ##  ['N1', 'N2', 'P3']
cluster = 'posONLY' ## 'posONLY', 'posNEG'   
#####################################################

# event_type = ['target']
for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
    # condi_name = ['GOu']
       
    norm_kind = ['vector','normVec', 'normVec_zsc']  
    
    condi_name = ['GOc', 'GOu', 'NoGo']  
    ncondi = len(condi_name)
       
      
    # for ppt on 08/03/2024
   
    # t1min = 0.160; t1max = 0.180
    # t2min = 0.300; t2max = 0.350
    # t3min = 0.370; t3max = 0.450
    
    ## 29/07/24
    t1min = 0.150; t1max = 0.200
    t2min = 0.250; t2max = 0.350
    t3min = 0.370; t3max = 0.450
             
   
        
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
       
    #% import the STCs (grand mean and sd) for each condi as numpy array.
    ## HC: evkCondiContrast   
    for ci, contrast in enumerate(contrast_kind):
        print('\n Reading the stcGAVG to disk: %s'  % contrast)
        
        if stype == 'hc':
            extension = contrast +'_' + event_type[ei] + '_hc_' +'stcGAVG' +'_'+ evkContr +'_' +  version +'_' + method +'_' + ampliNormalization  +'_' + bsln
        else:
            extension = contrast +'_' + event_type[ei] + '_pd_' +  dbs +'_' +'stcGAVG' +'_'+ evkContr +'_' + version +'_' + method +'_' + ampliNormalization  +'_' + bsln
        
        stc_fname_array = op.join(config_for_gogait.eeg_dir_GOODremove,
                          config_for_gogait.base_fname_avg_npy.format(**locals()))
        print("input: ", stc_fname_array)
        stc_data_per_contra = np.load(stc_fname_array)    
        
        # store condi in dim1, vertices in dim2, time in dim3 
        stc_data_per_contra_exp_dim = np.expand_dims(stc_data_per_contra, axis = 0) 
        if ci == 0:
            stc_data_all_contra =  stc_data_per_contra_exp_dim
        else:
            stc_data_all_contra = np.vstack((stc_data_all_contra, stc_data_per_contra_exp_dim)) 
    
          
        
#%%  plot the figures 
report = mne.Report()
scale = 'ptileScale'  # 'ptileScale' 
print("for timewindow: %s" % time_pos) 
for ci2, condi2 in enumerate(contrast_kind):                 
    print("plotting the condition: %s" % condi2)
    
   ## figure to plot brain
    fig = plt.figure(figsize=(12,6)) #18 3
    brain_views = 3
    axes = ImageGrid(fig, (1,1,1), nrows_ncols=(brain_views, 1)) # 
    figInd = 0
    dataSTCarray = stc_data_all_contra.copy()
   
    vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
    stc = mne.SourceEstimate(dataSTCarray[ci2,:,:], vertices = vertno,
                              tstep = 1/sfreq, tmin = - 0.2,
                              subject = 'fsaverage')                
           
    # plotting the snapshots at 3 different time zones
    # # time_points_kind = ['early', 'mid', 'late']
    # time_points_kind = ['early']

    if time_pos == 'N1': 
        tmin = t1min 
        tmax = t1max 
        timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
        print('plotting T1 activations for ' + condi2)
    elif time_pos == 'N2':
        tmin = t2min 
        tmax = t2max 
        timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
        print('plotting T2 activations for ' + condi2)
    elif time_pos == 'P3':
        tmin = t3min 
        tmax = t3max 
        timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))
        print('plotting T3 activations for ' + condi2)
    

    # timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))   
    stc_cropped = stc.copy()
    stc_mean_timepts = stc_cropped.crop(tmin = tmin, tmax = tmax).mean()
 
    #% LATERAL VIEW
   
    ## percentile 
    if scale == 'ptileScale':
        vmin = 90 #96 # %tile
        vmid = 95 #97.5 # %tile
        vmax = 99 #99.95 # %tile
        if cluster == 'posNEG':
            clim=dict(kind="percent", pos_lims = [vmin, vmid, vmax]) # in percentile
        elif cluster == 'posONLY': 
            clim=dict(kind="percent", lims = [vmin, vmid, vmax]) # in percentile
        
    clrBar = True

    wt = 300 #1000
    ht = 150 #500
    stc_fig1 =  stc_mean_timepts.plot(
                        views=["lat"],
                        hemi= "split",
                        smoothing_steps=7, 
                        size=(wt, ht),
                        view_layout = 'vertical',
                        time_viewer=False,
                        show_traces=False,
                        colorbar= clrBar,
                        background='white',
                        clim = clim, # in values
                        brain_kwargs = dict(surf = 'inflated'),
                        add_data_kwargs = dict(colorbar_kwargs=
                                               dict(vertical = False,
                                                    n_labels = 3,
                                                    label_font_size = 10,
                                                    width = 0.8, 
                                                    height = 0.2,
                                                    fmt = '%.2f'
                                                    )
                                               )
                    )  
    
    
   
  
    if cluster == 'posONLY': 
        roiNames = ['SFS_L1_HC_NoGo_GOu_P3_SRCCON',
                    'SFS_R1_HC_NoGo_GOu_P3_SRCCON',
                    'MFS_L1_HC_NoGo_GOu_P3_SRCCON',
                    'MFS_R1_HC_NoGo_GOu_P3_SRCCON',]
    else:
        roiNames = ['BA4_L1_HC_GOu_GOc_P3_SRCCON', 
                    'BA1_L1_HC_GOu_GOc_P3_SRCCON',
                    'BA1_L2_HC_NoGo_GOu_P3_SRCCON',
                    'SM_L1_HC_NoGo_GOu_P3_SRCCON'] 
    
    for roiNum, roi in enumerate(roiNames): 
        if roi[3] == 'L' or roi[4] == 'L'or roi[5]== 'L':
            fname = roi+'-lh.label'
            hemi_small = 'lh'
        if roi[3] == 'R' or roi[4] == 'R' or roi[5]== 'R':
            fname = roi+'-rh.label'
            hemi_small = 'rh'
        
        ## reading the label to a directory
        fpath = config_for_gogait.label_dir_aparc_a2009s
        label_fname = op.join(fpath, fname)
        func_label = mne.read_label(label_fname) 
        grown_label = mne.label.select_sources(template_subject, 
                                               func_label, location='center', 
                                               extent= 7.0, 
                                               grow_outside=True)
                                               
        stc_fig1.add_label(grown_label, color='black')
    
    screenshot1 = stc_fig1.screenshot()
    
    
    #%  MEDIAL VIEW        
    stc_fig2 =  stc_mean_timepts.plot(
                        views=["med"],
                        hemi= "split",
                        smoothing_steps=7, 
                        size=(wt, ht),
                        view_layout = 'vertical',
                        time_viewer=False,
                        show_traces=False,
                        colorbar= False,
                        background='white',
                        clim = clim, # in values
                        brain_kwargs = dict(surf = 'inflated'),
                        
                    )  
    
    if cluster == 'posONLY':      
        roiNames = ['ACC_L1_HC_NoGo_GOu_P3_SRCCON',
                    'ACC_R1_HC_NoGo_GOu_P3_SRCCON',
                    
                    ] 
    else: 
        roiNames = ['ACC_L1_HC_GOu_GOc_P3_SRCCON',
                    'ACC_R1_HC_GOu_GOc_P3_SRCCON',
                    'BA6_L1_HC_GOu_GOc_P3_SRCCON'
                    
                    ] 
    
    for roiNum, roi in enumerate(roiNames): 
        if roi[3] == 'L' or roi[4] == 'L'or roi[5]== 'L':
            fname = roi+'-lh.label'
            hemi_small = 'lh'
        if roi[3] == 'R' or roi[4] == 'R' or roi[5]== 'R':
            fname = roi+'-rh.label'
            hemi_small = 'rh'
        
        ## reading the label to a directory
        fpath = config_for_gogait.label_dir_aparc_a2009s
        label_fname = op.join(fpath, fname)
        func_label = mne.read_label(label_fname) 
        grown_label = mne.label.select_sources(template_subject, 
                                               func_label, location='center', 
                                               extent= 7.0, 
                                               grow_outside=True)
                                               
        stc_fig2.add_label(grown_label, color='black')
        
    screenshot2 = stc_fig2.screenshot()

    #%DORSAL VENTRAL VIEW     
    ### https://docs.pyvista.org/version/stable/api/plotting/_autosummary/pyvista.Plotter.add_scalar_bar.html          
    stc_fig3 =  stc_mean_timepts.plot(
                        views=["dor", "ven"],
                        hemi= "both",
                        smoothing_steps=7, 
                        size=(wt, ht),
                        view_layout = 'horizontal',
                        time_viewer= False,
                        show_traces= False,
                        colorbar= False,
                        background='white',
                        clim = clim, # in values
                        brain_kwargs = dict(surf = 'inflated'), 
                        
                                                                             
                    )  
    if cluster == 'posONLY':    
        roiNames = ['SFS_L1_HC_NoGo_GOu_P3_SRCCON',
                    'SFS_R1_HC_NoGo_GOu_P3_SRCCON',
                    'MFS_L1_HC_NoGo_GOu_P3_SRCCON',
                    'MFS_R1_HC_NoGo_GOu_P3_SRCCON',
                     
                    ] 
    else:
        roiNames = [ 'BA4_L1_HC_GOu_GOc_P3_SRCCON',
                     'BA1_L1_HC_GOu_GOc_P3_SRCCON',
                     'BA1_L2_HC_NoGo_GOu_P3_SRCCON', 
                     'SM_L1_HC_NoGo_GOu_P3_SRCCON',
                     'TP_R2_PDON_NoGo_GOu_P3_SRCCON',
                        
                    ] 
    
    for roiNum, roi in enumerate(roiNames): 
        if roi[3] == 'L' or roi[4] == 'L'or roi[5]== 'L':
            fname = roi+'-lh.label'
            hemi_small = 'lh'
        if roi[3] == 'R' or roi[4] == 'R' or roi[5]== 'R':
            fname = roi+'-rh.label'
            hemi_small = 'rh'
        
        ## reading the label to a directory
        fpath = config_for_gogait.label_dir_aparc_a2009s
        label_fname = op.join(fpath, fname)
        func_label = mne.read_label(label_fname) 
        grown_label = mne.label.select_sources(template_subject, 
                                               func_label, location='center', 
                                               extent= 7.0, 
                                               grow_outside=True)
                                               
        stc_fig3.add_label(grown_label, color='black')
    
           
    screenshot3 = stc_fig3.screenshot()
   
    ax_ind = 0
    for ax, image in zip([axes[figInd ],axes[figInd+ 1], 
                          axes[figInd + 2 ]],
                         [screenshot1, screenshot2, screenshot3]):
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.axis('off')
        
        # ax.spines['right'].set_visible(True)
        # ax.spines['top'].set_visible(False)
        # ax.spines['left'].set_visible(True)
        # ax.spines['bottom'].set_visible(False)
        ax.imshow(image)
        ax_ind = ax_ind + 1
        
                
    
    fig.tight_layout()
    
    stc_fig1.close() 
    stc_fig2.close()
    stc_fig3.close()
        
            
    report.add_figure(fig, title= time_pos + '_' + contrast_kind[ci2], replace = True)
    plt.close('all')    


# finally saving the report after the for condi loop ends.     
print('Saving the reports to disk')  
report.title = 'Group_plot_'+ stype + '_'+contrast + '_'+time_pos + '_'+ contraType +'_' + cluster + '_at_' + evnt+ ': '+ version + '_'  + method + '_' + ampliNormalization + '_' + scale +'_' + bsln
extension = 'group_plot_'+ stype + '_'+contrast + '_'+time_pos + '_'+ contraType + '_' + cluster  
report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
report.save(report_fname+'_at_'+ evnt + '_' + version+ '_' + orientation + '_'  + method + '_' + ampliNormalization + '_' + scale +'_' + bsln+'.html', overwrite=True)                         
  
   
   



    