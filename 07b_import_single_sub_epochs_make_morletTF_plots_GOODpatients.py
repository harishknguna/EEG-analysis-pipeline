#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:34:24 2024
 https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html
 https://mne.tools/stable/generated/mne.time_frequency.tfr_multitaper.html#mne.time_frequency.tfr_multitaper

@author: harish.gunasekaran

=============================================
07b: For patients: import single sub epochs, make morlet wavelet TF, and plot the power 
==============================================
Imports the MNE epochs object obtained in script 02 and apply 
morlet wavelet TF for single subjects, per condition and plot the TF power 
(baseline uncorrected and corrected) in the separte html files for each conditions. 

make sure to run this command in the console: %matplotlib qt
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
from mne.time_frequency import tfr_multitaper, tfr_morlet

import config_for_gogait
n_subs = len(config_for_gogait.subjects_list)
fs_dir = fetch_fsaverage(verbose=True)
template_subjects_dir = op.dirname(fs_dir)

# ampliNormalization = ['AmpliNorm', 'AmpliActual']
# ampliNormalization = 'AmpliActual'

event_type = ['target']
# event_type = ['cue', 'target']
# version_list = ['GOODremove','CHANremove']
version = 'CHANremove'
ep_extension = 'TF'

waveType = 'morlet'
numType = 'complex'  # 'real' | 'complex'
 
DBSinfo = ['OFF', 'ON']  

## theta and beta band-limited parameters  
fmin_theta = 3.0; fmax_theta = 7.0 # in Hz
freq_band_theta = 'theta'

fmin_betaL = 13.0 ; fmax_betaL = 21.0
freq_band_betaL = 'beta_low'      

for ei, evnt in enumerate(event_type):
    sfreq = 500
    # The files live in:
    template_subject = "fsaverage"   
    # condi_name = ['GOu'] 
    condi_name = ['GOc', 'GOu', 'NoGo']
    num_condi_with_eve = 6
    n_chs = 132
   
    sampling_freq = 500 # in hz
    
    for di, dbs in enumerate(DBSinfo):    
    
 
        for ci, condi in enumerate(condi_name): 
            
            report = mne.Report()
            print("condition: %s" % condi)        
      
            for sub_num, subject in enumerate(config_for_gogait.subjects_list_patients): 
                print("\n\nProcessing subject: %s" % subject)
                eeg_subject_dir_GOODpatients = op.join(config_for_gogait.eeg_dir_GOODpatients, subject)
                      
                #%% reading the epochs from disk
                print('Reading the epochs from disk')
                extension =  condi_name[ci] +'_' + event_type[ei] +'_' + dbs +'_' + version + '_'+ ep_extension +'_epo'
               
                epochs_fname = op.join(eeg_subject_dir_GOODpatients,
                                          config_for_gogait.base_fname.format(**locals()))
               
                # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
                print("Input: ", epochs_fname)
                epochs = mne.read_epochs(epochs_fname, proj=True, preload=True).pick('eeg')
                epochs = epochs.set_eeg_reference(projection=True)
                info = epochs.info   # taking info,events,id from the last sub's data (same for all subs)
                
               
                #%% compute and plot the TF maps
                ## parameters of Ziri et al 2023
                # define frequencies of interest (log-spaced)
                # freqs = np.logspace(*np.log10([1, 40]), num=40)
                freqs = np.linspace(3,40,num = 40, endpoint= True)
                n_cycles = 3 # variable time window T = n_cycles/freqs (in sec)
                                # = should be equal or smaller than signal  
                
                ## (0.5 s window for 1 Hz, 0.0125 s for 40 Hz or 10 ms for 50 Hz )
                # n_cycles = 0.5* freqs / 2.0  # fixed time window or different number of cycle per frequency
                
                               
                if waveType == 'morlet' and numType == 'real': 
                    power, itc = tfr_morlet(
                        epochs,
                        freqs = freqs,
                        n_cycles = n_cycles,
                        # time_bandwidth = 2.0, # freq_bandwidth = time_bandwidth/T
                        use_fft = True,
                        return_itc = True,
                        decim = 5, ## time res = fs/decim = 500/5 = 100 or 10 ms
                        average = True, # average all the epochs
                        output = 'power', # default: returns pwr and itc
                        n_jobs = None,
                    ) 
                    
                ## use morlet for complex pwr but set average, itc = False, 
                ## """ MNE: If "complex", then average must be False."""
                
                elif waveType == 'morlet' and numType == 'complex': 
                    pwrComplex  = tfr_morlet(
                        epochs,
                        freqs = freqs,
                        n_cycles = n_cycles,
                        # time_bandwidth = 2.0, # freq_bandwidth = time_bandwidth/T
                        use_fft = True,
                        return_itc = False,
                        decim = 5, ## time res = fs/decim = 500/5 = 100 or 10 ms
                        average = False, # average all the epochs
                        output = 'complex',
                        n_jobs = None,
                    ) 
                    
    #%%         ## PLOT the single sub TF 
                # don't forget to run the magic cmd, %matplotlib qt
                if numType == 'real':
                    plt.rcParams.update({'font.size': 14})
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
                    fig.suptitle('TF power: bslnNocorr and corr')
                    power.plot(baseline=None, title = 'auto',
                                exclude = 'bads', combine = 'mean', axes = ax1)
                    ax1.set_title('Power bslnNoCorr')
                    plt.rcParams.update({'font.size': 14})
                    power.plot(baseline=(-0.5, -0.1), mode="logratio", title = 'auto',
                                exclude = 'bads', combine = 'mean', axes = ax2)
                    ax2.set_title('Power bslnCorr')
                    fig.tight_layout()
                    report.add_figure(fig, title = subject + '_TF_morlet_real', replace = True)
                
                
                elif numType == 'complex':
                    # added 29/04/2024: taking abs of complex at single trial level
                    ## taking abs(.) value to match multitaper
                    
                    pwrComplex_abs = abs(pwrComplex.copy())
                    avgTFR = pwrComplex_abs.copy().average(method='mean', dim='epochs')
                    plt.rcParams.update({'font.size': 14})
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
                    fig.suptitle('TF power: bslnNocorr and corr')
                    avgTFR.plot(baseline=None, title = 'auto',
                                exclude = 'bads', combine = 'mean', axes = ax1)
                    ax1.set_title('Power bslnNoCorr')
                    plt.rcParams.update({'font.size': 14})
                    avgTFR.plot(baseline=(-0.5, -0.1), mode="logratio", title = 'auto',
                                exclude = 'bads', combine = 'mean', axes = ax2)
                    ax2.set_title('Power bslnCorr')
                    fig.tight_layout()
                    report.add_figure(fig, title = subject + '_TF_morlet_cmplx', replace = True)
        
                plt.close('all')
                
                #%%# saving data requires more memory: use lustre; use when needed  
                # saving the tfr data in numpy array baseline NOT corrected
              
                #%   saving the complex epochsTFR: no baseline correction applied 
                print('\n Writing the complex epochsTFR to disk')
                extension =  condi_name[ci] +'_' + event_type[ei] +'_' + dbs +'_' + version + '_' + waveType +'_PWR_CMPLX-tfr.h5'
                tfr_fname = op.join(eeg_subject_dir_GOODpatients,
                                        config_for_gogait.base_fname_no_fif.format(**locals()))
               
                print("Output: ", tfr_fname)
                mne.time_frequency.write_tfrs(tfr_fname, pwrComplex, overwrite=True) 
                
                ## saving abs: added on 29/04/2024
                extension =  condi_name[ci] +'_' + event_type[ei] +'_' + dbs +'_' + version + '_' + waveType +'_PWR_CMPLX_ABS-tfr.h5'
                tfr_fname = op.join(eeg_subject_dir_GOODpatients,
                                        config_for_gogait.base_fname_no_fif.format(**locals()))
               
                print("Output: ", tfr_fname)
                mne.time_frequency.write_tfrs(tfr_fname, pwrComplex_abs, overwrite=True)  
                
                #%  saving the complex epochsTFR: baseline correction applied 
                print('\n Writing the complex epochsTFR to disk baseline corrected')
                extension =  condi_name[ci] +'_' + event_type[ei] +'_' + dbs +'_' + version + '_' + waveType +'_PWR_CMPLX_bslnCorr-tfr.h5'
                tfr_fname = op.join(eeg_subject_dir_GOODpatients,
                                        config_for_gogait.base_fname_no_fif.format(**locals()))
               
                
                print("Output: ", tfr_fname)
                pwrComplex_bslnCorr = pwrComplex.copy().apply_baseline(baseline=(-0.5, -0.1), mode="logratio")
                mne.time_frequency.write_tfrs(tfr_fname, pwrComplex_bslnCorr, overwrite=True)
                
                ## saving abs: added on 29/04/2024
    
                extension =  condi_name[ci] +'_' + event_type[ei] +'_' + dbs +'_' + version + '_' + waveType +'_PWR_CMPLX_ABS_bslnCorr-tfr.h5'
                tfr_fname = op.join(eeg_subject_dir_GOODpatients,
                                        config_for_gogait.base_fname_no_fif.format(**locals()))
               
                
                print("Output: ", tfr_fname)
                pwrComplex_abs_bslnCorr = pwrComplex_abs.copy().apply_baseline(baseline=(-0.5, -0.1), mode="logratio")
                mne.time_frequency.write_tfrs(tfr_fname, pwrComplex_abs_bslnCorr, overwrite=True)   
                
                #%% saving band-limited complex epochs used during source localizing
                
                if numType == 'complex':
                    ## considering epochs of only theta band
                   
                    epochs_TFR_theta =  pwrComplex.copy().crop(tmin=None, tmax=None, 
                                                        fmin = fmin_theta, fmax= fmax_theta) 
                   
                    ## considering epochs of only betaL band to reduce the computational burden
                    epochs_TFR_betaL =  pwrComplex.copy().crop(tmin=None, tmax=None, 
                                                        fmin = fmin_betaL , fmax= fmax_betaL) 
                   
                    
                    ##%   saving the tfr data in numpy array baseline NOT corrected
                    
                    #%   saving the complex epochsTFR: no baseline correction applied 
                    print('\n Writing the complex epochsTFR to disk')
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + dbs +'_' + version + '_' + waveType +'_theta_' + 'PWR_CMPLX-tfr.h5'
                    tfr_fname = op.join(eeg_subject_dir_GOODpatients,
                                            config_for_gogait.base_fname_no_fif.format(**locals()))
                   
                    print("Output: ", tfr_fname)
                    mne.time_frequency.write_tfrs(tfr_fname, epochs_TFR_theta, overwrite=True) 
                    
                    #%   saving the complex epochsTFR: no baseline correction applied 
                    print('\n Writing the complex epochsTFR to disk')
                    extension =  condi_name[ci] +'_' + event_type[ei] +'_' + dbs +'_' + version + '_' + waveType +'_beta_low_' + 'PWR_CMPLX-tfr.h5'
                    tfr_fname = op.join(eeg_subject_dir_GOODpatients,
                                            config_for_gogait.base_fname_no_fif.format(**locals()))
                   
                    print("Output: ", tfr_fname)
                    mne.time_frequency.write_tfrs(tfr_fname, epochs_TFR_betaL, overwrite=True) 
                    
                    
            # finally saving the report after the for subject loop ends.     
            print('\n Saving the reports to disk')  
            report.title = 'Single patients TF: ' + condi + '_' + evnt+'_' + dbs +'_' + version + '_' + waveType + '_' + numType
            #report.title = 'Group sub STC contrast at ' + evnt
            extension = 'single_patients_TF'
            report_fname = op.join(config_for_gogait.report_dir, config_for_gogait.base_fname_generic.format(**locals()))
            report.save(report_fname+'_' + condi +'_'+ evnt +'_' + dbs +'_' + version+ '_' + waveType + '_' + numType + '.html', overwrite=True)            
            
          

    