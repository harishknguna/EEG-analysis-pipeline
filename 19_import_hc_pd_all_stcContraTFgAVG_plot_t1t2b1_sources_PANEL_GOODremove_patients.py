#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

"""
=============================================
19. Import group avg of all HC/PDOFF/PDON contrast and plot contrast of sources
==============================================
Imports the group average contrast data generated in the prev codes 18a and 18b
and plot the contrast of sources at early-theta /late-theta/beta 

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


freq_band = ['theta', 'beta_low']


# for freq_band_name == 'theta': 
t1min = 0.100; 
t1max = 0.250 # 0.300 # changed on 05/07/24

t2min = 0.300 # 0.200 # changed on 05/07/24
t2max = 0.450 # 0.400
    
# for freq_band_name == 'beta_low':
t3min = 0.250
t3max = 0.450 # changed on 04/07/24
   

    
    
DBSinfo = ['ON', 'OFF']  
contrast_kind = ['GOu_GOc', 'NoGo_GOu']  
condi_name_dbs = ['GOc', 'GOu', 'NoGo']


# event_type = ['target']
for fbi, freq_band_name in enumerate(freq_band): 
    for ei, evnt in enumerate(event_type):
        sfreq = 500
        decim = 5 # when doing tfr computation
        sfreq = sfreq/decim
        tfr_freqs = np.linspace(3,40,num = 40, endpoint= True)
        n_TF_freqs = len(tfr_freqs)
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
        n_vectors = 3
        ncondi = len(condi_name)  # nc = 3
        # evoked_array_all_sub = np.ones([n_subs, n_chs, n_samples_esti])*np.nan
        stc_data_all_sub_all_condi = np.ones([n_subs, ncondi, n_verticies, n_samples_esti])*np.nan
                     
        #% import the STCs (grand mean and sd) for each condi as numpy array.
        ## HC: evkCondiContrast   
        for ci, contrast in enumerate(contrast_kind):
            print('\n Writing the stcGAVG to disk: %s'  % contrast)
            extension = contrast +'_' + event_type[ei] + '_hc_' +'stcContraTFgAVG_bslnCorr' + '_' + freq_band_name+ '_' + version +'_' + method +'_' + ampliNormalization 
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
        
        if freq_band_name == 'theta': 
            stc_data_all_condi_hc_theta = stc_data_all_condi_hc.copy()
        elif freq_band_name == 'beta_low': 
            stc_data_all_condi_hc_beta_low = stc_data_all_condi_hc.copy()
        
        ## PD: evkCondiContrast   
        for di, dbs in enumerate(DBSinfo):
            for ci, contra in enumerate(contrast_kind):
                print('\n Reading the stcGAVG from disk: %s'  % contra)
                extension = contra +'_' + event_type[ei] + '_pd_' + dbs +'_stcCondiContraTFgAVG_bslnCorr' + '_' + freq_band_name+ '_' + version +'_' + method +'_' + ampliNormalization 
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
        
        
        ## PD: evkDBSContrast        
        for ci, contrast in enumerate(condi_name_dbs):
            print('\n Writing the stcGAVG to disk: %s'  % contrast)
            extension = condi_name[ci] +'_ON_OFF_' + event_type[ei] +'_pd_'+'stcDBScontraTFgAVG_bslnCorr' + '_' + freq_band_name+ '_' + version +'_' + method +'_' + ampliNormalization 
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
        
        
        if freq_band_name == 'theta': 
            stc_data_all_pd_dbs_theta = stc_data_all_pd_dbs.copy()
        elif freq_band_name == 'beta_low': 
            stc_data_all_pd_dbs_beta_low = stc_data_all_pd_dbs.copy()
    
        
#%%  plot the figures 
report = mne.Report()
scale = 'ptileScale' 
datakind = ['hc', 'pd_off', 'pd_on', 'pd_dbs']   
time_points_kind =   ['T1', 'T2', 'T3']
nT = len(time_points_kind)

for dt, dtkd in enumerate(datakind): 
    if dtkd == 'hc': 
        contra_list = contrast_kind.copy()
    elif dtkd == 'pd_off': 
        contra_list = contrast_kind.copy()
    elif dtkd == 'pd_on': 
        contra_list = contrast_kind.copy()
    elif dtkd == 'pd_dbs': 
        contra_list = condi_name_dbs.copy()
        
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
            
            ## theta 1
            if time_pos2 == 'T1': 
                tmin = t1min 
                tmax = t1max 
                timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
                print('plotting T1 theta activations for ' + condi2)
                
                if dtkd == 'hc': 
                    dataSTCarray = stc_data_all_condi_hc_theta.copy()       
                elif dtkd == 'pd_off': 
                    dataSTCarray = stc_data_all_condi_pd_off_theta.copy()     
                elif dtkd == 'pd_on': 
                    dataSTCarray = stc_data_all_condi_pd_on_theta.copy()  
                elif dtkd == 'pd_dbs': 
                    dataSTCarray = stc_data_all_pd_dbs_theta.copy()
            ## theta 2        
            elif time_pos2 == 'T2':
                tmin = t2min 
                tmax = t2max 
                timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
                print('plotting T2 theta activations for ' + condi2)
                if dtkd == 'hc': 
                    dataSTCarray = stc_data_all_condi_hc_theta.copy()       
                elif dtkd == 'pd_off': 
                    dataSTCarray = stc_data_all_condi_pd_off_theta.copy()     
                elif dtkd == 'pd_on': 
                    dataSTCarray = stc_data_all_condi_pd_on_theta.copy()  
                elif dtkd == 'pd_dbs': 
                    dataSTCarray = stc_data_all_pd_dbs_theta.copy()
            ## beta 1        
            elif time_pos2 == 'T3':
                tmin = t3min 
                tmax = t3max 
                timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))
                print('plotting T3 beta activations for ' + condi2)
                
                if dtkd == 'hc': 
                    dataSTCarray = stc_data_all_condi_hc_beta_low.copy()       
                elif dtkd == 'pd_off': 
                    dataSTCarray = stc_data_all_condi_pd_off_beta_low.copy()     
                elif dtkd == 'pd_on': 
                    dataSTCarray = stc_data_all_condi_pd_on_beta_low.copy()  
                elif dtkd == 'pd_dbs': 
                    dataSTCarray = stc_data_all_pd_dbs_beta_low.copy()
            
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
                ## for contrast always use bipolar scaling
                clim = dict(kind="percent", pos_lims=[vmin, vmid, vmax])
                
            # if time_pos2 == 'T1' or time_pos2 == 'T2' :
            #     clim = dict(kind="percent", lims=[vmin, vmid, vmax])
            # elif time_pos2 == 'T3' :
            #     clim = dict(kind="percent", pos_lims=[vmin, vmid, vmax])
            
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
report.title = 'Group stcTFgAVG_TFRContrast_hc_pd_all'+ '_' + evnt + '_t1t2b1_'+ version + '_' + orientation + '_' + method + '_' + ampliNormalization + '_' + scale
extension = 'group_PANEL_stcTFgAVG_TFRContrast_maps_hc_pd_all'
report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
report.save(report_fname+'_at_'+ evnt + '_t1t2b1_' + version+ '_' + orientation + '_'  + method + '_' + ampliNormalization + '_' + scale + '_ppt.html', overwrite=True)                  
  
   
   



    