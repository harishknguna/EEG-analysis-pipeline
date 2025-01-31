#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""


"""
=============================================
17. Import group avg of all HC/PDOFF/PDON condi and plot early_theta/late_theta/beta sources
==============================================
Imports the group average data generated in the prev codes 16a and 16b
and plot the early_theta/late_theta/beta sources 

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

DBSinfo = ['ON', 'OFF']  

ep_extension = 'TF'
waveType = 'morlet' # 'morlet' | 'multitaper'

f1min = 3.0; f1max = 7.0 # in Hz
f2min = 13.0 ; f2max = 21.0


freq_band = ['theta', 'beta_low']


# for freq_band_name == 'theta': 
t1min = 0.100; 
t1max = 0.250 # 0.300 # changed on 05/07/24

t2min = 0.300 # 0.200 # changed on 05/07/24
t2max = 0.450 # 0.400
    
# for freq_band_name == 'beta_low':
t3min = 0.250
t3max = 0.450 # changed on 04/07/24


# event_type = ['target']
for fbi, freq_band_name in enumerate(freq_band): 
    for ei, evnt in enumerate(event_type):
        sfreq = 500
        decim = 5 # when doing tfr computation
        sfreq = sfreq/decim
        tfr_freqs = np.linspace(3,40,num = 40, endpoint= True)
        n_TF_freqs = len(tfr_freqs)
        
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
        n_verticies = 20484
        
        ncondi = len(condi_name)  # nc = 3
         # evoked_array_all_sub = np.ones([n_subs, n_chs, n_samples_esti])*np.nan
         # stc_data_all_sub_all_condi = np.ones([n_subs, ncondi, n_verticies, n_samples_esti])*np.nan
                      
         #% import the STCs (grand mean and sd) for each condi as numpy array.
         ## HC: 
        for ci, condi in enumerate(condi_name):
            print('\n Loading the stcTFgAVG to disk: %s'  % condi)
            extension = condi_name[ci] +'_' + event_type[ei] + '_hc_' +'stcTFgAVG_bslnCorr' + '_' + freq_band_name+ '_' + version +'_' + method +'_' + ampliNormalization 
            stc_fname_array = op.join(config_for_gogait.eeg_dir_GOODremove,
                                   config_for_gogait.base_fname_avg_npy.format(**locals()))
            stc_data_per_condi_hc = np.load(stc_fname_array)    
             
            # store condi in dim1, vertices in dim2, time in dim3 
            stc_data_per_condi_hc_exp_dim = np.expand_dims(stc_data_per_condi_hc, axis = 0) 
            if ci == 0:
                stc_data_all_condi_hc =  stc_data_per_condi_hc_exp_dim
            else:
                stc_data_all_condi_hc = np.vstack((stc_data_all_condi_hc, stc_data_per_condi_hc_exp_dim)) 
         
        if freq_band_name == 'theta': 
            stc_data_all_condi_hc_theta = stc_data_all_condi_hc.copy()
        elif freq_band_name == 'beta_low': 
            stc_data_all_condi_hc_beta_low = stc_data_all_condi_hc.copy()
         
        ## PD:
        for di, dbs in enumerate(DBSinfo):
            for ci, condi in enumerate(condi_name):
                print('\n Reading the stcTFgAVG from disk: %s'  % condi)
                extension = condi_name[ci] +'_' + event_type[ei] + '_pd_' + dbs +'_stcTFgAVG_bslnCorr' + '_' + freq_band_name+ '_' + version +'_' + method +'_' + ampliNormalization 
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
         
        if freq_band_name == 'theta': 
            stc_data_all_condi_pd_off_theta = stc_data_all_condi_pd_off.copy()
            stc_data_all_condi_pd_on_theta = stc_data_all_condi_pd_on.copy()
        elif freq_band_name == 'beta_low': 
            stc_data_all_condi_pd_off_beta_low = stc_data_all_condi_pd_off.copy()
            stc_data_all_condi_pd_on_beta_low = stc_data_all_condi_pd_on.copy()

            
#%% plot the sources  
## step 1: find max and min values across 3 condi for each T

# group_data_stat = ['average', 'stdDev']
# for datakind in group_data_stat:
    # if datakind == 'average': 
        # dataSTCarray = stc_data_avg_sub_all_condi.copy()
    # elif datakind == 'stdDev':
        # dataSTCarray = stc_data_std_sub_all_condi.copy()

## find min/max for HC and PD separately across condi:         
datakind = ['hc', 'pd_off', 'pd_on']   
time_points_kind = ['T1', 'T2', 'T3']
events = ['N1', 'N2', 'P3']
nT = len(time_points_kind)
nD = len(datakind)
 
## here the for loops order are inverted 
for dt, dtkd in enumerate(datakind): 
    stc_min_value_condi = np.ones([len(time_points_kind), len(condi_name)])*np.nan
    stc_max_value_condi = np.ones([len(time_points_kind), len(condi_name)])*np.nan
    print("\n\ncomputing Min/max of data: %s" % dtkd)
   
    for ci1, condi1 in enumerate(condi_name): # run condi to get min/max values across 3 conditions 
    
        for ti1, time_pos1 in enumerate(time_points_kind): 
            print("\nfor timewindow: %s" % time_pos1)   
            ## theta 1
            if time_pos1 == 'T1': 
                tmin = t1min 
                tmax = t1max 
                timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
                print('plotting T1 theta activations for ' + condi1)
                
                if dtkd == 'hc': 
                    dataSTCarray = stc_data_all_condi_hc_theta.copy()       
                elif dtkd == 'pd_off': 
                    dataSTCarray = stc_data_all_condi_pd_off_theta.copy()     
                elif dtkd == 'pd_on': 
                    dataSTCarray = stc_data_all_condi_pd_on_theta.copy()  
               
            ## theta 2        
            elif time_pos1 == 'T2':
                tmin = t2min 
                tmax = t2max 
                timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
                print('plotting T2 theta activations for ' + condi1)
                if dtkd == 'hc': 
                    dataSTCarray = stc_data_all_condi_hc_theta.copy()       
                elif dtkd == 'pd_off': 
                    dataSTCarray = stc_data_all_condi_pd_off_theta.copy()     
                elif dtkd == 'pd_on': 
                    dataSTCarray = stc_data_all_condi_pd_on_theta.copy()  
               
            ## beta 1        
            elif time_pos1 == 'T3':
                tmin = t3min 
                tmax = t3max 
                timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))
                print('plotting T3 beta activations for ' + condi1)
                
                if dtkd == 'hc': 
                    dataSTCarray = stc_data_all_condi_hc_beta_low.copy()       
                elif dtkd == 'pd_off': 
                    dataSTCarray = stc_data_all_condi_pd_off_beta_low.copy()     
                elif dtkd == 'pd_on': 
                    dataSTCarray = stc_data_all_condi_pd_on_beta_low.copy()  
        
            print("computing Min/max of condition: %s" % condi1)
            vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
            stc = mne.SourceEstimate(dataSTCarray[ci1,:,:], vertices = vertno,
                                      tstep = 1/sfreq, tmin = -tsec_start,
                                      subject = 'fsaverage')  
          
            # timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))   
            stc_cropped = stc.copy()
            stc_mean_timepts = stc_cropped.crop(tmin = tmin, tmax = tmax).mean()
            stc_min_value_condi[ti1, ci1] = np.min(stc_mean_timepts.data)
            stc_max_value_condi[ti1, ci1] = np.max(stc_mean_timepts.data)
            
    if dtkd == 'hc': 
        stc_min_hc = stc_min_value_condi.copy()
        stc_max_hc = stc_max_value_condi.copy()
    elif dtkd == 'pd_off': 
        stc_min_pd_off = stc_min_value_condi.copy()
        stc_max_pd_off = stc_max_value_condi.copy()
    elif dtkd == 'pd_on': 
        stc_min_pd_on = stc_min_value_condi.copy()
        stc_max_pd_on = stc_max_value_condi.copy()


stc_min_pd = np.ones(np.shape(stc_min_value_condi)) * np.nan
stc_max_pd = np.ones(np.shape(stc_min_value_condi)) * np.nan 

for ti1, time_pos1 in enumerate(time_points_kind): 
    for ci1, condi1 in enumerate(condi_name):
        stc_min_pd[ti1, ci1] = np.min([stc_min_pd_off[ti1, ci1], stc_min_pd_on[ti1, ci1]])
        stc_max_pd[ti1, ci1] = np.max([stc_max_pd_off[ti1, ci1], stc_max_pd_on[ti1, ci1]])

        
#%%  plot the figures 
report = mne.Report()
# scale = 'condiScale' # NOT WORKING
scale = 'ptileScale' 
datakind = ['hc', 'pd_off', 'pd_on']   
time_points_kind = ['T1', 'T2', 'T3']
nT = len(time_points_kind)

## old PANEL style
## columns: t1 (early-theta), t2 (late-theta), b (beta)

for dt, dtkd in enumerate(datakind): 
       
    for ci2, condi2 in enumerate(condi_name):                 
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
            
            ## theta 1
            if time_pos2 == 'T1': 
                tmin = t1min 
                tmax = t1max 
                timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
                print('plotting T1 theta activations for ' + condi2)
                
                if dtkd == 'hc': 
                    dataSTCarray = stc_data_all_condi_hc_theta.copy()  
                    stc_min = stc_min_hc
                    stc_max = stc_max_hc
                elif dtkd == 'pd_off': 
                    dataSTCarray = stc_data_all_condi_pd_off_theta.copy()  
                    stc_min = stc_min_pd
                    stc_max = stc_max_pd
                elif dtkd == 'pd_on': 
                    dataSTCarray = stc_data_all_condi_pd_on_theta.copy() 
                    stc_min = stc_min_pd
                    stc_max = stc_max_pd
               
            ## theta 2        
            elif time_pos2 == 'T2':
                tmin = t2min 
                tmax = t2max 
                timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
                print('plotting T2 theta activations for ' + condi2)
                if dtkd == 'hc': 
                    dataSTCarray = stc_data_all_condi_hc_theta.copy()  
                    stc_min = stc_min_hc
                    stc_max = stc_max_hc
                elif dtkd == 'pd_off': 
                    dataSTCarray = stc_data_all_condi_pd_off_theta.copy()  
                    stc_min = stc_min_pd
                    stc_max = stc_max_pd
                elif dtkd == 'pd_on': 
                    dataSTCarray = stc_data_all_condi_pd_on_theta.copy()  
                    stc_min = stc_min_pd
                    stc_max = stc_max_pd
               
            ## beta 1        
            elif time_pos2 == 'T3':
                tmin = t3min 
                tmax = t3max 
                timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))
                print('plotting T3 beta activations for ' + condi2)
                
                if dtkd == 'hc': 
                    dataSTCarray = stc_data_all_condi_hc_beta_low.copy()    
                    stc_min = stc_min_hc
                    stc_max = stc_max_hc
                elif dtkd == 'pd_off': 
                    dataSTCarray = stc_data_all_condi_pd_off_beta_low.copy() 
                    stc_min = stc_min_pd
                    stc_max = stc_max_pd
                elif dtkd == 'pd_on': 
                    dataSTCarray = stc_data_all_condi_pd_on_beta_low.copy() 
                    stc_min = stc_min_pd
                    stc_max = stc_max_pd
            
            
            if scale == 'condiScale':
                stc_min_value_T = np.min(stc_min, axis = 1).flatten()
                stc_max_value_T = np.max(stc_max, axis = 1).flatten()
               
            
            vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
            stc = mne.SourceEstimate(dataSTCarray[ci2,:,:], vertices = vertno,
                                      tstep = 1/sfreq, tmin = -tsec_start,
                                      subject = 'fsaverage')            
        
            # timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))   
            stc_cropped = stc.copy()
            stc_mean_timepts = stc_cropped.crop(tmin = tmin, tmax = tmax).mean()
         
            #% LATERAL VIEW
            
            ## condi wise scale scalings changed on 30/07/24
            if scale == 'condiScale':  # opposite for condi ti indices
                vmin = 0.55 * stc_max_value_T[ti2]  #   0.60
                vmid = 0.75 * stc_max_value_T[ti2]  #  0.70
                vmax = 0.95 * stc_max_value_T[ti2]  #  0.90
                # clim = dict(kind="value", lims = [vmin, vmid, vmax])
        
            elif scale == 'ptileScale':
                vmin = 90 #96 # %tile
                vmid = 95 #97.5 # %tile
                vmax = 99 #99.95 # %tile
            
            if time_pos2 == 'T1' or time_pos2 == 'T2' :
                clim = dict(kind="percent", lims=[vmin, vmid, vmax])
            elif time_pos2 == 'T3' :
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
report.title = 'Group t1t2b1_stcTFgAVG_hc_pd_all'+ '_' + evnt + '_' + version + '_' + orientation + '_' + method + '_' + ampliNormalization + '_' + scale
extension = 'group_t1t2b1_stcTFgAVG_maps_hc_pd_all'
report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
report.save(report_fname+'_at_'+ evnt + '_t1t2b1_' + version+ '_' + orientation + '_'  + method + '_' + ampliNormalization + '_' + scale + '.html', overwrite=True)                  
  
   
   



    