#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

"""
=============================================
14. Import group avg of all HC/PDOFF/PDON contrast and plot contrast of sources
==============================================
Imports the group average contrast data generated in the prev codes 13a and 13b
and plot the contrast of sources at N1/N2/P3 

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

DBSinfo = ['OFF','ON']  
contrast_kind = ['GOu_GOc', 'NoGo_GOu']
condi_name_dbs = ['GOc', 'GOu', 'NoGo']

## added on 28/08/2024
bsln = 'bslnCorr' ## 'bslnCorr' | 'NobslnCorr' 

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
                 
    #% import the STCs (grand mean and sd) for each condi as numpy array.
    ## HC: evkCondiContrast   
    for ci, contrast in enumerate(contrast_kind):
        print('\n Writing the stcGAVG to disk: %s'  % contrast)
        extension = contrast_kind[ci] +'_' + event_type[ei] + '_hc_' +'stcGAVG' +'_evkCondiContrast_SRC_' + version +'_' + method +'_' + ampliNormalization +'_' + bsln
        stc_fname_array = op.join(config_for_gogait.eeg_dir_GOODremove,
                              config_for_gogait.base_fname_avg_npy.format(**locals()))
        print("input: ", stc_fname_array)
        stc_data_per_condi_hc = np.load(stc_fname_array)    
        
        # store condi in dim1, vertices in dim2, time in dim3 
        stc_data_per_condi_hc_exp_dim = np.expand_dims(stc_data_per_condi_hc, axis = 0) 
        if ci == 0:
            stc_data_all_condi_hc =  stc_data_per_condi_hc_exp_dim
        else:
            stc_data_all_condi_hc = np.vstack((stc_data_all_condi_hc, stc_data_per_condi_hc_exp_dim)) 
    
    ## PD: evkCondiContrast   
    for di, dbs in enumerate(DBSinfo):
        for ci, condi in enumerate(contrast_kind):
            print('\n Reading the stcGAVG from disk: %s'  % condi)
            extension = contrast_kind[ci] +'_' + event_type[ei] + '_pd_' +  dbs +'_stcGAVG' +'_evkCondiContrast_SRC_' + version +'_' + method +'_' + ampliNormalization +'_' + bsln
            stc_fname_array = op.join(config_for_gogait.eeg_dir_GOODpatients,
                                  config_for_gogait.base_fname_avg_npy.format(**locals()))
            print("input: ", stc_fname_array)
            stc_data_per_condi_pd = np.load(stc_fname_array)    
            
            # store condi in dim1, vertices in dim2, time in dim3 
            stc_data_per_condi_pd_exp_dim = np.expand_dims(stc_data_per_condi_pd, axis = 0) 
            if ci == 0:
                stc_data_all_condi_pd =  stc_data_per_condi_pd_exp_dim
            else:
                stc_data_all_condi_pd = np.vstack((stc_data_all_condi_pd, stc_data_per_condi_pd_exp_dim))
            
        if dbs == 'OFF':
            stc_data_all_condi_pd_off = stc_data_all_condi_pd.copy()
        elif dbs == 'ON':
            stc_data_all_condi_pd_on = stc_data_all_condi_pd.copy()
    
    
    ## PD: evkDBSContrast        
    for ci, contrast in enumerate(condi_name_dbs):
        print('\n Reading the stcDBScontraGAVG to disk: %s '  % condi)
        extension = condi_name[ci] +'_ON_OFF_' + event_type[ei] +'_pd_'+'stcGAVG'+'_evkDBSContrast_SRC' + '_' + version +'_' + method +'_' + ampliNormalization +'_' + bsln 
        stc_fname_array = op.join(config_for_gogait.eeg_dir_GOODpatients,
                               config_for_gogait.base_fname_avg_npy.format(**locals()))
        print("input: ", stc_fname_array)
        stc_data_per_condi_hc = np.load(stc_fname_array)    
        
        # store condi in dim1, vertices in dim2, time in dim3 
        stc_data_per_condi_hc_exp_dim = np.expand_dims(stc_data_per_condi_hc, axis = 0) 
        if ci == 0:
            stc_data_all_pd_dbs =  stc_data_per_condi_hc_exp_dim
        else:
            stc_data_all_pd_dbs = np.vstack((stc_data_all_pd_dbs, stc_data_per_condi_hc_exp_dim)) 
            

#%%  plot the figures 
report = mne.Report()
cluster = 'posNEG' ## 'posONLY', 'posNEG'
scale = 'ptileScale' 
datakind = ['hc', 'pd_off', 'pd_on', 'pd_dbs']   
time_points_kind =   ['T1', 'T2', 'T3']
nT = len(time_points_kind)

## old PANEL style
## columns: N1, N2, P3

for dt, dtkd in enumerate(datakind): 
    if dtkd == 'hc': 
        contra_list = contrast_kind.copy()
        dataSTCarray = stc_data_all_condi_hc.copy()      
    elif dtkd == 'pd_off': 
        contra_list = contrast_kind.copy()
        dataSTCarray = stc_data_all_condi_pd_off.copy()     
    elif dtkd == 'pd_on': 
        contra_list = contrast_kind.copy()
        dataSTCarray = stc_data_all_condi_pd_on.copy()  
    elif dtkd == 'pd_dbs': 
        contra_list = condi_name_dbs.copy()
        dataSTCarray = stc_data_all_pd_dbs.copy()
        
    ## PLOT with common scale time point wise (rows) or condi (col) wise or both
    if scale == 'ptileScale': 
        stc_min_value_T = np.nan
        stc_max_value_T = np.nan
        
    for ci2, condi2 in enumerate(contra_list):                 
        print("plotting the condition: %s" % condi2)
        
       ## figure to plot brain
        fig = plt.figure(figsize=(9,3)) #18 3
        brain_views = 3
        axes = ImageGrid(fig, (1,1,1), nrows_ncols=(brain_views, nT)) # 
        figInd = 0
        
        for ti2, time_pos2 in enumerate(time_points_kind): 
            print("for timewindow: %s" % time_pos2)  
            # plotting the snapshots at 3 different time zones
            # depending on time, data fetched from theta (t1,t2) or beta (t3)
            
          
            if time_pos2 == 'T1': 
                tmin = t1min 
                tmax = t1max 
                timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
                print('plotting T1 activations for ' + condi2)
                
                
                  
            elif time_pos2 == 'T2':
                tmin = t2min 
                tmax = t2max 
                timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
                print('plotting T2 activations for ' + condi2)
               
                 
            elif time_pos2 == 'T3':
                tmin = t3min 
                tmax = t3max 
                timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))
                print('plotting T3 activations for ' + condi2)
                
             
            
            vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
            stc = mne.SourceEstimate(dataSTCarray[ci2,:,:], vertices = vertno,
                                      tstep = 1/sfreq, tmin = -tsec_start,
                                      subject = 'fsaverage')            
        
            # timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))   
            stc_cropped = stc.copy()
            stc_mean_timepts = stc_cropped.crop(tmin = tmin, tmax = tmax).mean()
         
            #% LATERAL VIEW
            
            ## percentile 
            if scale == 'ptileScale':
                vmin = 90 #96 # %tile
                vmid = 95 #97.5 # %tile
                vmax = 99 #99.95 # %tile
                clim = dict(kind="percent", pos_lims=[vmin, vmid, vmax])
            
            cmap = 'auto'#'bwr'
            
            clrBar = True
            
            # if ti2 == 4:
            #     clrBar = True
            # else:
            #     clrBar = False
            
            wt = 200 #1000
            ht = 100 #500
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
            
            
                   
            screenshot3 = stc_fig3.screenshot()
           
            ax_ind = 0
            for ax, image in zip([axes[figInd + ti2],axes[figInd+ nT+ ti2],axes[figInd + 2*nT + ti2]],
                                 [screenshot1, screenshot2, screenshot3]):
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.axis('off')
                ax.spines['right'].set_visible(True)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(False)
                ax.imshow(image)
                ax_ind = ax_ind + 1
                
                # if ti2 == nT-1 and ax_ind == 3:
                #     # create additional axes (for ERF and colorbar)
                #     divider = make_axes_locatable(ax)
                #     # add axes for colorbar
                #     ax_colorbar = divider.append_axes("right", size="1%", pad=0.01)
                #     image = ax.images[0]
                #     plt.colorbar(image, cax=ax_colorbar) # label =  r"$\mu$" + 'V')
                #     # ax_colorbar.set_title(r"$\mu$" + 'V', fontsize = '8')
                
            
            fig.tight_layout()
            
            stc_fig1.close() 
            stc_fig2.close()
            stc_fig3.close()
            
                
        report.add_figure(fig, title= condi2 + '_' + dtkd, replace = True)
        plt.close('all')   


# finally saving the report after the for condi loop ends.     
print('Saving the reports to disk')  
report.title = 'Group_N1N2P3_evkCondiContrast_SRC_hc_pd_all_'+ cluster +'_' + evnt + ': ' + version + '_' + orientation + '_' + method + '_' + ampliNormalization + '_' + scale +'_' + bsln
extension = 'group_N1N2P3_evkCondiContrast_SRC_hc_pd_all_' + cluster
report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
report.save(report_fname+'_at_'+ evnt + '_' + version+ '_' + orientation + '_'  + method + '_' + ampliNormalization + '_' + scale +'_' + bsln+'.html', overwrite=True)                         
  
   
   













#%%
 #%% plot the sources  
        ## step 1: find max and min values across 3 condi for each T
        
        # group_data_stat = ['average', 'stdDev']
        # for datakind in group_data_stat:
            # if datakind == 'average': 
                # dataSTCarray = stc_data_avg_sub_all_condi.copy()
            # elif datakind == 'stdDev':
                # dataSTCarray = stc_data_std_sub_all_condi.copy()

# datakind = ['hc', 'pd_off', 'pd_on', 'pd_dbs']
# time_points_kind = ['T1', 'T2', 'T3']
# events = ['N1', 'N2', 'P3']
# nT = len(time_points_kind)
# nD = len(datakind)-1
 
# ## here the for loops order are inverted 

# for dt, dtkd in enumerate(datakind):
#     if dtkd == 'hc': 
#         dataSTCarray = stc_data_all_condi_hc.copy()
#         contra_list = contrast_kind.copy()
#     elif dtkd == 'pd_off': 
#         dataSTCarray = stc_data_all_condi_pd_off.copy()
#         contra_list = contrast_kind.copy()
#     elif dtkd == 'pd_on': 
#         dataSTCarray = stc_data_all_condi_pd_on.copy()
#         contra_list = contrast_kind.copy()
#     elif dtkd == 'pd_dbs': 
#         dataSTCarray = stc_data_all_pd_dbs.copy()
#         contra_list = condi_name_dbs.copy()
        
#     stc_min_value_condi = np.ones([len(time_points_kind), len(contra_list)])*np.nan
#     stc_max_value_condi = np.ones([len(time_points_kind), len(contra_list)])*np.nan
#     print("\n\ncomputing Min/max of data: %s" % dtkd)    
#     for ti1, time_pos1 in enumerate(time_points_kind): 
#         print("for timewindow: %s" % time_pos1)
        
#         for ci1, condi1 in enumerate(contra_list): # run condi to get min/max values across 3 conditions 
#             print("computing Min/max of condition: %s" % condi1)
#             vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
#             stc = mne.SourceEstimate(dataSTCarray[ci1,:,:], vertices = vertno,
#                                       tstep = 1/sfreq, tmin = - 0.2,
#                                       subject = 'fsaverage')  
#             if time_pos1 == 'T1': 
#                 tmin = t1min 
#                 tmax = t1max 
                                   
#             elif time_pos1 == 'T2':
#                 tmin = t2min 
#                 tmax = t2max 
                                   
#             elif time_pos1 == 'T3':
#                 tmin = t3min 
#                 tmax = t3max 
        
#             # timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))   
#             stc_cropped = stc.copy()
#             stc_mean_timepts = stc_cropped.crop(tmin = tmin, tmax = tmax).mean()
#             stc_min_value_condi[ti1, ci1] = np.min(stc_mean_timepts.data)
#             stc_max_value_condi[ti1, ci1] = np.max(stc_mean_timepts.data)
     
#     if dtkd == 'hc': 
#         stc_min_hc = stc_min_value_condi.copy()
#         stc_max_hc = stc_max_value_condi.copy()
#     if dtkd == 'pd_off': 
#         stc_min_pd_off = stc_min_value_condi.copy()
#         stc_max_pd_off = stc_max_value_condi.copy()
#     elif dtkd == 'pd_on': 
#         stc_min_pd_on = stc_min_value_condi.copy()
#         stc_max_pd_on = stc_max_value_condi.copy()
#     elif dtkd == 'pd_dbs': 
#         stc_min_pd_dbs = stc_min_value_condi.copy()
#         stc_max_pd_dbs = stc_max_value_condi.copy()


# stc_min_pd = np.ones(np.shape(stc_min_value_condi)) * np.nan
# stc_max_pd = np.ones(np.shape(stc_min_value_condi)) * np.nan 

# for ti1, time_pos1 in enumerate(time_points_kind): 
#     for ci1, condi1 in enumerate(contra_list):
#         stc_min_pd[ti1, ci1] = np.min([stc_min_pd_off[ti1, ci1], stc_min_pd_on[ti1, ci1]])
#         stc_max_pd[ti1, ci1] = np.max([stc_max_pd_off[ti1, ci1], stc_max_pd_on[ti1, ci1]])

 
   