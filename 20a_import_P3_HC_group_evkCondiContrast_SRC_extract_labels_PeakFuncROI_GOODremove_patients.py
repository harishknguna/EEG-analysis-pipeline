#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

""" ...ON GOING... to be continued...
=============================================
20a. Import group level SRC CONTRAST and extract ROI labels 
==============================================
Imports the group level contrast data (source contrast or sensor contrast), 
plot the contrast and extract the regions of interest (ROIs).  

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
import seaborn as sns

import config_for_gogait

def find_closest(arr, val):
    idx = np.abs(arr - val).argmin()
    return arr[idx]

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

method = "dSPM"  # "MNE" | "dSPM"

# ampliNormalization = ['AmpliNormPerCondi', 'AmpliNormAccCondi', 'AmpliActual']
ampliNormalization = 'AmpliActual'

DBSinfo = ['ON', 'OFF']  

## added on 28/08/2024
bsln = 'bslnCorr' ## 'bslnCorr' | 'NobslnCorr' 

contraType = 'SRCCON' #['SENCON', 'SRCCON'] run one at a time. 


if contraType == 'SENCON':
    evkContr = 'evkCondiContrast_SEN'
elif contraType == 'SRCCON':
    evkContr = 'evkCondiContrast_SRC'
    
    
##### all the changes are made here ##################
stype = 'hc' ## ['hc','pd_off', 'pd_on']
contrast = 'NoGo_GOu'  ## ['GOu_GOc','NoGo_GOu']
hemi = 'RH' ## ['LH', 'RH']
time_pos = 'P3' ##  ['N1', 'N2', 'P3']
cluster = 'posONLY' ## 'posONLY', 'posNEG'   
#####################################################


# event_type = ['target']
for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
       
    norm_kind = ['vector','normVec', 'normVec_zsc']  
    
    if method == 'dSPM':
        norm_kind = norm_kind[1] # selecting the normVec [1] option for dSPM  
    elif method == 'MNE':
        norm_kind = norm_kind[2] # selecting the norm_zscored [2] option for MNE
   
    
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
        
    ## plot for subs HC and PD/ON, PD/OFF separately  
    #% reading src of fsaverage
    fs_dir = fetch_fsaverage(verbose=True)
    template_subjects_dir = op.dirname(fs_dir)
    fname_src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
    src = mne.read_source_spaces(fname_src)

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
       
            
    ## loading grand avg evkContrasts 
    stc_data_avg_sub_contra = np.ones([n_verticies, n_samples_esti])*np.nan                 
  
    print('\n Reading the stcGAVG to disk: %s'  % contrast)
    
    if stype == 'hc':
        extension = contrast +'_' + event_type[ei] + '_hc_' +'stcGAVG' +'_'+ evkContr +'_' +  version +'_' + method +'_' + ampliNormalization  +'_' + bsln
    else:
        extension = contrast +'_' + event_type[ei] + '_pd_' +  dbs +'_' +'stcGAVG' +'_'+ evkContr +'_' + version +'_' + method +'_' + ampliNormalization  +'_' + bsln
        
    ## loading the STCs (grand mean) for each contrast as numpy array.
    stc_fname_array = op.join(eeg_dir,
                           config_for_gogait.base_fname_avg_npy.format(**locals()))
    
    print("Output: ", stc_fname_array)
    stc_data_avg_sub_contra = np.load(stc_fname_array)
        
    if stype == 'pd_off':
        stc_contra_all_off = stc_data_avg_sub_contra.copy()
      
    elif stype == 'pd_on':
        stc_contra_all_on = stc_data_avg_sub_contra.copy()
      
    elif stype == 'hc':
        stc_contra_all_hc = stc_data_avg_sub_contra.copy()
    
                
        
#%%              
report = mne.Report()
scale = 'ptileScale' # 'ptileScale' | 'globalScale' | 'timeScale' | 'condiScale'

roiNames = ['arb1', 'arb2', 'arb3', 'arb4']
## roi names intialized as 'arb1'... in the first iteration but renamed properly in second iteration
    
roiNum = -1 ## intialized as -1, but iteration sum starts from 0
# roiNum = 0
   
print('\n\n'+stype + '\n\n')
if stype == 'pd_off': 
    dbs = 'OFF'
    subjects_list = pd_list
    eeg_dir = eeg_dir_pd
    n_subs = n_subs_pd
  
    dataSTCcontra = stc_contra_all_off.copy()
    
      
elif stype == 'pd_on': 
    dbs = 'ON'
    subjects_list = pd_list
    eeg_dir = eeg_dir_pd
    n_subs = n_subs_pd

    dataSTCcontra = stc_contra_all_on.copy()
   
    
elif stype == 'hc': 
    subjects_list = hc_list
    eeg_dir = eeg_dir_hc
    n_subs = n_subs_hc
    
    dataSTCcontra = stc_contra_all_hc.copy()
      

print("\nfor timewindow: %s" % time_pos)     
if time_pos == 'N1': 
    tmin = t1min 
    tmax = t1max 
    timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
    
   
elif time_pos == 'N2':
    tmin = t2min 
    tmax = t2max 
    timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000)) 
    
    
elif time_pos == 'P3':
    tmin = t3min 
    tmax = t3max 
    timeDur = str(int(tmin*1000)) + '_' + str(int(tmax*1000))
   
            
        
print("\nplotting the condition: %s" % contrast)
dataSTCarray = dataSTCcontra.copy()
vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
stc = mne.SourceEstimate(dataSTCarray, vertices = vertno,
                          tstep = 1/sfreq, tmin = - 0.2,
                          subject = 'fsaverage')                
       
  
stc_cropped = stc.copy()
stc_mean_timepts = stc_cropped.crop(tmin = tmin, tmax = tmax).mean()
 
#% LATERAL VIEW

## percentile 
if scale == 'ptileScale':
  vmin = 90 # 96 # %tile 
  vmid = 95 #97.5 # %tile
  vmax = 99 #99.95 # %tile
  if cluster == 'posNEG':
      clim=dict(kind="percent", pos_lims = [vmin, vmid, vmax]) # in percentile
  elif cluster == 'posONLY': 
      clim=dict(kind="percent", lims = [vmin, vmid, vmax]) # in percentile

clrBar = True

           
# LATERAL VIEW
#% SELECTING ROIs and plotting them
# ROIs from anat labels
# use the stc_mean to generate a functional label
# region growing is halted at 60% of the peak value within the
# anatomical label / ROI specified by aparc_label_name

aparc_label_name = 'caudalmiddlefrontal_4-rh' ## 'caudalanteriorcingulate_2-lh', 'paracentral_5-lh',
                                              ## 'postcentral_6-lh' 'postcentral_5-lh', 'supramarginal_8-lh'
## change here for hemis: rh or lh            ## 'caudalmiddlefrontal_3-lh', 'caudalmiddlefrontal_4-rh'
label = mne.read_labels_from_annot(template_subject, 
                                    parc="aparc_sub", 
                                    subjects_dir=template_subjects_dir, 
                                    regexp=aparc_label_name)[0]

stc_mean_label = stc_mean_timepts.in_label(label)

#### finding the peak ###### comment/uncomment
vth_label = np.max(stc_mean_label.rh_data)  ## change here for hemis: rh or lh  
       
### Functional labels 
stc_with_zeros =  stc_mean_label.copy()
data = stc_with_zeros.data
stc_with_zeros.data[data < vth_label] = 0.0
# ##############################

     
            
## find the labels whose value != 0.0
func_labels_all_LH,  func_labels_all_RH = mne.stc_to_label(
stc_with_zeros,
src=src,
smooth=True, ## change here
# subjects_dir=subjects_dir,
connected=True,
verbose="error",
)
   
## PLOT figures for LH and RH
   
if hemi == 'LH': 
    n_clus_lab = len(func_labels_all_LH)
elif hemi == 'RH':
    n_clus_lab = len(func_labels_all_RH)
  
#% plotting the ROI only if label meets the threshold
for li in np.arange(n_clus_lab):

    if hemi == 'LH': 
        func_label = func_labels_all_LH[li]
        hemi_small = 'lh'
    elif hemi == 'RH':
        func_label = func_labels_all_RH[li]
        hemi_small = 'rh'

    roiNum = roiNum + 1 
   
    print('\n roi number = %s' % roiNum)
   
    fig, axd = plt.subplots(figsize=(11,3))                         
    wt = 1000#1000
    ht = 250#500
    stc_fig1 =  stc_mean_timepts.plot(
        views=["lat","dor","med", "ven"],
        hemi= hemi_small ,#"split",
        smoothing_steps=7, 
        size=(wt, ht),
        view_layout = 'horizontal',
        time_viewer=False,
        show_traces=False,
        colorbar= clrBar,
        background='white',
        clim = clim, # in values
        brain_kwargs = dict(surf = 'inflated'),
        add_data_kwargs = dict(colorbar_kwargs=
                               dict(vertical = False,
                                    n_labels = 3,
                                    label_font_size = 14,
                                    width = 0.8, 
                                    height = 0.1, 
                                    fmt = '%.2f'
                                    )
                               )
    )  
      
    # show both labels
    stc_fig1.add_label(func_label, borders=True, color='black')
    screenshot1 = stc_fig1.screenshot()
    axd.imshow(screenshot1)
    axd.axis('off')
      
     
    ## exporting the label to a directory
    fpath = config_for_gogait.label_dir_aparc_a2009s
    fname = roiNames[roiNum]
    label_fname = op.join(fpath, fname)
    mne.write_label(label_fname, func_label)   
    
    
    axd.set_title(fname)
    report.add_figure(fig, title = str(roiNum) +'_' + roiNames[roiNum], replace = True)
    
    stc_fig1.close() 
    plt.close('all')       
               
# finally saving the report after the for subject loop ends.  

print('Saving the reports to disk')  
report.title = 'peak_extract_'+ stype + '_'+contrast + '_'+time_pos + '_'+hemi + '_' + contraType +'_' + cluster + '_at_' + evnt+ ': '+ version + '_'  + method + '_' + ampliNormalization + '_' + scale +'_' + bsln
extension = 'peak_extract_'+ stype + '_'+contrast + '_'+time_pos + '_'+hemi + '_' + contraType + '_' + cluster  
report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
report.save(report_fname+'_at_'+ evnt +'_' + cluster +'_' + version+ '_' + method + '_' + ampliNormalization + '_' + scale +'_' + bsln+'.html', overwrite=True)          
              

            

               
            
                  
           
                      
        
    
            
        
      
            
    
           


    