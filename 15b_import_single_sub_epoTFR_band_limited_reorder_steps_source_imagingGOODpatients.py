#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:02:43 2023

@author: harish.gunasekaran
"""
"""
=============================================
Single sub analysis: Source reconstruction of EVOKED CONTRAST using template MRI
15a.Generate STC and save the dipoles (vector format) as numpy. Report the foward and noise cov matrix

https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html#sphx-glr-auto-tutorials-forward-35-eeg-no-mri-py
https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html

==============================================

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
from scipy.linalg import norm
from mne.datasets import fetch_fsaverage
from mne.coreg import Coregistration
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_tfr_epochs 
from mne.minimum_norm import read_inverse_operator
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid, inset_locator
import config_for_gogait



ampliNormalization = 'AmpliActual'
version = 'CHANremove'
event_type = ['target']
orientation = 'varyDipole' #  ['fixDipole', 'varyDipole' ]
method = "dSPM"  # "MNE" | "dSPM"
baseline = 'bslnNoCorr' 
# corr = 'CorrApplySTC'  # 'CorrApplySTC' | 'CorrNotApplySTC'
ep_extension = 'TF'
waveType = 'morlet' # 'morlet' | 'multitaper'
numType = 'complex' # NO CHOICE, MNE allows only complex for source localization


#####################################################
DBSinfo = ['ON', 'OFF'] # 'ON' | 'OFF' 
freq_band_name = 'beta_low' # 'theta' | 'beta_low' # CHANGE HERE
condi_name = ['GOc', 'GOu', 'NoGo'] # 'GOc' | 'GOu' | 'NoGo' # CHANGE HERE

######################################################



if freq_band_name == 'theta': 
    fmin = 3.0; fmax = 7.0 # in Hz
elif freq_band_name == 'beta_low': 
    fmin = 13.0 ; fmax = 21.0


# fmin_alpha = 7.0; f2max_alpha = 12.0 
# freq_band_alpha = 'alpha'


if version == 'GOODremove':
    n_chs = 128
elif version == 'CHANremove':
    n_chs = 103


sampling_freq = 500 # in hz
tfr_freqs = np.linspace(3,40,num = 40, endpoint= True)
n_TF_freqs = len(tfr_freqs)

for ei, evnt in enumerate(event_type):

    for ci, condi in enumerate(condi_name): 
        for di, dbs in enumerate(DBSinfo):   
            print("\ncondition and dbs: %s, %s" %(condi, dbs)) 
       
            ## n_samples_esti = int(sampling_freq * (isi[ind_isi] + t_end - t_start + 0.002))  
            # estimate the num of time samples per condi/ISI to allocate numpy array
            ## added on 15/01/2024
            
            if evnt == 'cue' and ep_extension == 'TF':
                tsec_start = 0.8 # pre-stimulus(t2/S2/S4) duration in sec
                tsec_end = 1.3 # post-stimulus (t3/S8/S16) duration in sec
            elif evnt == 'target' and ep_extension == 'TF':
                tsec_start = 1.0 # pre-stimulus(t2/S2/S4) duration in sec
                tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
            else:
                tsec_start = 0.2 # pre-stimulus(t2/S2/S4) duration in sec
                tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec      
                       
            ## for subject in config_for_gogait.subjects_list: 
            ## parallel processing
            def run_tfr_source_imaging(subject):
                
                eeg_subject_dir_GOODpatients = op.join(config_for_gogait.eeg_dir_GOODpatients, subject)
                print("\nProcessing subject: %s" % subject)
                print('\nreading the epochsTFR from disk: ' + freq_band_name)
                
                                                  
                #%% reading the epochsTFR from disk ## updated 25/03/24
               
                                
                extension =  condi_name[ci] +'_' + event_type[ei] +'_' + dbs +'_' + version + '_' + waveType +'_'+ freq_band_name +'_' + 'PWR_CMPLX-tfr.h5'
                tfr_fname = op.join(eeg_subject_dir_GOODpatients,
                                        config_for_gogait.base_fname_no_fif.format(**locals()))
               
                print("\nOutput: ", tfr_fname)
                epochs_TFR_read = mne.time_frequency.read_tfrs(tfr_fname)
                epochs_TFR =  epochs_TFR_read[0]
                tfr_times = epochs_TFR.times
                               
                ## just to check
                # avgTFR =  abs(epochs_TFR.copy()).average(method='mean', dim='epochs')    
                # avgTFR_bslnCorr = avgTFR.copy().apply_baseline(baseline=(-0.5, -0.1), mode="logratio")                                     
                
                # ## baseline corrected version
                # bslncorr = avgTFR_bslnCorr.plot(
                #             exclude = 'bads', combine = 'mean')
                
                # considering epochs of only theta band
               
                
                
               
                #%% reading inverse operator per sub
                print("\n\n Reading inverse operator as FIF from disk")
               
                extension = dbs +'_' + version +'_'+ orientation +'_'+'inv' # keep _inv in the end
                    
                fname_in = op.join(eeg_subject_dir_GOODpatients,
                                     config_for_gogait.base_fname.format(**locals()))
                
                print("\ninput: ", fname_in)
                inverse_operator = read_inverse_operator(fname_in)                  
                
                #%% apply inverse method
                ## read about baseline correction : 
                ## https://github.com/mne-tools/mne-python/issues/962    
                snr = 3.0
                lambda2 = 1.0 / snr**2
                
                ## ValueError: EEG average reference (using a projector) is mandatory for modeling,
                ## use the method set_eeg_reference(projection=True)
                
                """ pick_ori = None | “normal” | “vector” """
                """ "None"- Pooling is performed by taking the norm of loose/free orientations."""
                ##  In case of a fixed source space no norm is computed leading to signed source activity. 
                ## "normal"- Only the normal to the cortical surface is kept. 
                ##  This is only implemented when working with loose orientations. 
                ## "vector" - No pooling of the orientations is done, 
                ## and the vector result will be returned in the form of a mne.VectorSourceEstimate object. 
                # epochs_TFR = mne.set_eeg_reference(epochs_TFR, projection=True) 
                
                num_eps = len(epochs_TFR)
                
                ## obtain induced power 
                ## so applying inverse func trial-by-trial and get stc for theta/beta bands
                
                for ep in range(0, num_eps):
                    
                    stc_list_freq = apply_inverse_tfr_epochs(
                                                epochs_TFR[ep],
                                                inverse_operator,
                                                lambda2,
                                                method=method,
                                                pick_ori = "vector",
                                                return_generator = False,
                                                verbose=True)
                    
                    ## step 0: convert list to array of all fbins
                    for fi, stc_freq_bin in enumerate(stc_list_freq): 
                        stc_freq_bin_array = np.array(stc_freq_bin[0].data)
                        
                        # store fbin in dim1, vertices in dim2, vectors in dim3, time in dim3 
                        stc_freq_bin_exp_dim = np.expand_dims(stc_freq_bin_array, axis = 0) 
                        
                        if fi == 0:
                            stc_freq_all_bins =  stc_freq_bin_exp_dim
                        else:
                            stc_freq_all_bins = np.vstack((stc_freq_all_bins, stc_freq_bin_exp_dim))                     
                            
                    ## step 1: taking abs to convert comp to real num rep at single trial level for each fbins
                    ## i.e., averaging power and phase components
                    stc_freq_all_bins_abs = abs(stc_freq_all_bins.copy())
                    
                    ## step 2: averaging across trials by adding extra dim (m/y consuming)
                    # ## store ep in dim1, fbin in dim2, vertices in dim3, vectors in dim4, time in dim5
                    # ## Note: use luster otherwise only one sub could be processed locally. 
                    
                    # stc_freq_all_bins_abs_exp_dim = np.expand_dims(stc_freq_all_bins_abs, axis = 0)
                                        
                    # if ep == 0:
                    #     stc_freq_all_bins_abs_all_trials =  stc_freq_all_bins_abs_exp_dim
                    # else:
                    #     stc_freq_all_bins_abs_all_trials =  np.vstack((stc_freq_all_bins_abs_all_trials, stc_freq_all_bins_abs_exp_dim)) 
                    
                    # print('\n\n epochs completed for freq: %s out of %s \n \n' %(ep+1,num_eps))
                
                # stc_freq_all_bins_avg_trials =  np.mean(stc_freq_all_bins_abs_exp_dim, axis = 0)
                    
                    ## step 2a: summing across the trials 
                                        
                    if ep == 0:
                        stc_freq_all_bins_abs_sum = stc_freq_all_bins_abs.copy()
                    else:
                        stc_freq_all_bins_abs_sum = stc_freq_all_bins_abs_sum + stc_freq_all_bins_abs.copy()
                    
                    print('\n\n epochs completed for freq: %s out of %s \n \n' %(ep+1,num_eps))  
                    
                ## step 2b: averaging across trials by dividing number of trials  
                stc_freq_all_bins_avg_trials =  stc_freq_all_bins_abs_sum/num_eps
                    
                ## step 3: taking norm for avg trials
                stc_freq_all_bins_avg_trials_norm = norm(stc_freq_all_bins_avg_trials.copy(), axis = 2)
                
                ## step 4: baseline correction for all freq bins
                stc_tfr_data = stc_freq_all_bins_avg_trials_norm.copy()
                stc_tfr_data_bslCorr = mne.baseline.rescale(stc_tfr_data, tfr_times, baseline = (-0.5, -0.1), 
                                                            mode='logratio', copy=True, picks=None, verbose=None)
                
                ## step 5: average across frequency
                stc_tfr_data_bslCorr_avg_freq = np.mean(stc_tfr_data_bslCorr.copy(), axis = 0) 
                
                #%% baseline corrected and avg freq
                print('\n\nWriting the stc to disk: bslnCorr and avg freq')
               
                extension = condi +'_' + ep_extension +'_' + 'bslnCorr_avgFreq'  +'_' + event_type[ei] +'_' + dbs +'_' +'stc_' + freq_band_name +'_' + waveType + '_' + orientation +'_' + version +'_' + method 
                stc_fname_array = op.join(eeg_subject_dir_GOODpatients,
                                      config_for_gogait.base_fname_npy.format(**locals()))
                print("Output: ", stc_fname_array)
                np.save(stc_fname_array, stc_tfr_data_bslCorr_avg_freq)
#%%                                    
            parallel, run_func, _ = parallel_func(run_tfr_source_imaging, n_jobs=config_for_gogait.N_JOBS)
            parallel(run_func(subject) for subject in config_for_gogait.subjects_list_patients)
                               
            # #%% check the stc plot
            
            # n_verticies = 20484
            # n_vectors = 3 
            # sfreq = 500
            # decim = 5 # when doing tfr computation
            # sfreq = sfreq/decim
            # vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
            # stc = mne.SourceEstimate(stc_tfr_data_bslCorr_avg_freq, vertices = vertno,
            #                           tstep = 1/sfreq, tmin = - tsec_start,
            #                           subject = 'fsaverage')
            # stc.plot()
           
           

                   
            
              
              
               
                
                
                