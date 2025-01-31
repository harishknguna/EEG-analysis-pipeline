#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================
02b. For Patients: Importing epoch array and saving in MNE epoch format 
==============================================
Imports the numpy array obtained in script 01 and converts into
MNE epoch format. 

"""  

import os.path as op

import mne
from mne.parallel import parallel_func
from mne.channels.montage import get_builtin_montages
from warnings import warn
# from pymatreader import read_mat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import config_for_gogait

# subject = config_for_gogait.subjects_list[0]

version_list = ['GOODremove','CHANremove']
ep_extension = 'evk'  ## change here: 'evk' | 'TF' 
## adding this extension for creating time frequency epochs


DBSinfo = ['OFF', 'ON']                  

for subject in config_for_gogait.subjects_list_patients: 
    for veri, version in enumerate(version_list):
    
        eeg_subject_dir_GOODpatients = op.join(config_for_gogait.eeg_dir_GOODpatients, subject)
        print("\n\n Processing subject: %s" % subject)
        
        for di, dbs in enumerate(DBSinfo):      
                       
            #%%            
            """ step 1: import numpy array epochs of different conditions and ISI"""    
            
            event_type = ['cue', 'target']
            condi_name = ['GOc', 'GOu', 'NoGo']
                 
            for ei, evnt in enumerate(event_type):
                for ci, condi in enumerate(condi_name):
                    print('\nImporting the sub-epochs from disk')
                    
                    ## added on 15/01/2024 (for HC) and 22/05/2024 (for PD)
                    ## importing epochs separately for evoked and TF with different prestim lengths 
                   
                    if ep_extension == 'TF':
                        extension =  condi_name[ci] +'_' + event_type[ei]  +'_' + dbs +  '_eeg_prep_'+ ep_extension+'.npy'
                    else:
                        extension =  condi_name[ci] +'_' + event_type[ei] +'_' + dbs +  '_eeg_prep.npy'
                       
                    npy_array_fname = op.join(eeg_subject_dir_GOODpatients,
                                        config_for_gogait.base_fname_no_fif.format(**locals()))   
                    
                    print("Input: ", npy_array_fname)
                    print("Output: None")       
                
                    if not op.exists(npy_array_fname):
                        warn('Run %s not found for subject %s ' %
                              (npy_array_fname, subject))
                        continue
                  
                  
                    data = np.load(npy_array_fname)   
                  
                 
                    ## importing the info from ...info.fif file
                    extension = 'info'
                    info_fname = op.join(eeg_subject_dir_GOODpatients,
                                          config_for_gogait.base_fname.format(**locals()))
                 
                    info = mne.io.read_info(info_fname)
                    
                    if version == 'CHANremove': # if needed, remove the bads from epochs
                        info['bads'] = ['Fp1', 'Fp2', 'AFp1', 'AFp2', 'AF7', 'AF8', 'F9', 'F10', 
                                               'FT9', 'FT10', 'P9', 'O9', 'PO9', 'O10', 'PO10', 'P10', 'TP9', 'TP10',
                                               'TPP9h', 'PPO9h', 'OI1h', 'OI2h', 'POO10h', 'PPO10h', 'TPP10h'] # 'Iz',
                    elif version == 'GOODremove':
                        info['bads'] = []
                        
                    
                    sampling_freq = info['sfreq']
          
                    num_events = np.shape(data)[0]
                    epoch_len_in_samples = np.shape(data)[2] 
                    events = np.column_stack(
                    (
                        np.arange(0, num_events*epoch_len_in_samples, epoch_len_in_samples),
                        np.zeros(num_events, dtype=int),
                        np.zeros(num_events, dtype=int))) # marking events at zero
                        # labeling S2 GOc event id as '1'; S4 GOu as '2'; S4 NoGo as '3'
                    if evnt == 'cue' and condi == 'GOc':
                       events[:, 2] = 1*np.ones(num_events, dtype=int)
                       event_dict = dict(S2=1)
                    elif evnt == 'cue' and condi == 'GOu':
                       events[:, 2] = 2*np.ones(num_events, dtype=int)
                       event_dict = dict(S4=2)
                    elif evnt == 'cue' and condi == 'NoGo':
                       events[:, 2] = 3*np.ones(num_events, dtype=int)
                       event_dict = dict(S4=3)
                    elif evnt == 'target' and condi == 'GOc':
                       events[:, 2] = 4*np.ones(num_events, dtype=int)
                       event_dict = dict(S8=4)
                    elif evnt == 'target' and condi == 'GOu':
                       events[:, 2] = 5*np.ones(num_events, dtype=int)
                       event_dict = dict(S8=5)
                    elif evnt == 'target' and condi == 'NoGo':
                       events[:, 2] = 6*np.ones(num_events, dtype=int)
                       event_dict = dict(S16=6)
                       
                    ## added on 15/01/2024 (for HC) and 22/05/2024 (for PD)
                    ## saving epochs separately for evoked and TF with different prestim lengths 
                    if evnt == 'cue' and ep_extension == 'TF':
                        tsec_start = 0.8 # pre-stimulus(t2/S2/S4) duration in sec
                        tsec_end = 1.3 # post-stimulus (t3/S8/S16) duration in sec
                        bslcorr = None ## don't apply bslcorr during epochs, apply in TF stage
                    elif evnt == 'target' and ep_extension == 'TF':
                        tsec_start = 1.0 # pre-stimulus(t2/S2/S4) duration in sec
                        tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
                        bslcorr = None ## don't apply bslcorr during epochs, apply in TF stage
                    else:
                        tsec_start = 0.2 # pre-stimulus(t2/S2/S4) duration in sec
                        tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec  
                        bslcorr = (-0.2,0)
             
                        
                    #  BASELINE CORRECTION APPLIED 
                    epochs = mne.EpochsArray(data, info = info, tmin = -tsec_start,  
                                            events = events, event_id = event_dict, 
                                            proj=False, baseline = bslcorr)
                    epochs.set_channel_types(config_for_gogait.set_channel_types)
                    # epochs.plot(picks = ['all'], events=events, event_id=event_dict)
                                   
                    if  ep_extension == 'evk': ## saving both epochs and evoked
                        print('\n Writing the sub-epochs numpy array to disk')
                        extension = condi_name[ci] +'_' + event_type[ei] +'_' + dbs + '_' + version +'_epo'
                        epochs_fname = op.join(eeg_subject_dir_GOODpatients,
                                                  config_for_gogait.base_fname.format(**locals()))
                       
                        # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
                        print("\n Output: ", epochs_fname)
                        epochs.save(epochs_fname, overwrite=True)     
                      
                        #%  saving the evoked data in numpy array
                        evoked = epochs.average(picks = ['all'])
                        print('\n Writing the evoked to disk')
                        extension =  condi_name[ci] +'_' + event_type[ei] +'_' + dbs + '_' +version +'_ave'
                        evoked_fname = op.join(eeg_subject_dir_GOODpatients,
                                                  config_for_gogait.base_fname.format(**locals()))
                        
                        # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
                        print("\n Output: ", evoked_fname)
                        evoked.save(evoked_fname, overwrite=True)     
                  
                    elif ep_extension == 'TF': ## saving TF epochs only 
                        print('\n Writing the sub-epochs numpy array to disk')
                        extension = condi_name[ci] +'_' + event_type[ei] +'_' + dbs + '_' + version +'_' + ep_extension + '_epo'
                        epochs_fname = op.join(eeg_subject_dir_GOODpatients,
                                                config_for_gogait.base_fname.format(**locals()))
                     
                        # epochs_fname = op.splitext(raw_fname_in)[0] + '-epo.fif'
                        print("\n Output: ", epochs_fname)
                        epochs.save(epochs_fname, overwrite=True)     
       
    
    
   
        

