#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================
01. Importing fieldtrip pre-processed eeg data
==============================================
Imports the data that has been pre-processed in fieldtrip (ft) and converts into
numpy array

"""  

import os.path as op

import mne
from mne.parallel import parallel_func
from mne.channels.montage import get_builtin_montages
from warnings import warn
from pymatreader import read_mat
import numpy as np
import pandas as pd


import config_for_gogait


ep_extension = 'evk'  ## change here: 'evk' | 'TF' 
## adding this extension for creating time frequency epochs

# def visual_inspection(subject):
for subject in config_for_gogait.subjects_list: 
    
    print("Processing subject: %s" % subject)

    # eeg_subject_dir_GOOD = op.join(config_for_gogait.eeg_dir_GOOD, subject)
    eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
   
    prepro_fname_in = op.join(eeg_subject_dir_GOODremove,
                            config_for_gogait.base_fname_in_prepro.format(**locals()))
    

    print("Input: ", prepro_fname_in)
    print("Output: None")
  
    if not op.exists(prepro_fname_in):
        raise Exception('Run %s not found for subject %s ' %
              (prepro_fname_in, subject))
        continue
    
      
    # load the mat data
    matdata = read_mat(prepro_fname_in)
    trial_struct = matdata['trial_struct']
    event_struct = matdata['event']
    df_trial = pd.DataFrame(trial_struct)   
    df_event = pd.DataFrame(event_struct) 
    time_list = matdata['reref_data']['time']
    
 #%%   
    """ step 1: Create some metadata"""
    
    n_channels = len(matdata['reref_data']['label'])
    names = matdata['reref_data']['label']
    sampling_freq = matdata['reref_data']['fsample']  # in Hertz
    ch_types = 128*["eeg"] + 3*["eog"] + ["ecg"]
    info = mne.create_info(ch_names = names, ch_types=ch_types, sfreq = sampling_freq)
    print(info)
    
    """ step 2: set eeg montage """
    
    # make montage from ch position
    # first create ch name and loc in dict format
    
    keys = matdata['reref_data']['elec']['label']
    values = matdata['reref_data']['elec']['chanpos'] 
    chpos = dict(zip(keys, values))  # channel position
    print(chpos) # {'a': 1, 'b': 2, 'c': 3}
    
    montage = mne.channels.make_dig_montage(ch_pos = chpos)
    info.set_montage(montage)
    
    # sphere size = 0.4 matches with acticap. 
    
    #%%   
    """ step 3: extract the trials data and time from matfile with equal size"""
    
    # shape (n_trials, n_channels, n_samples)
    # N.B. trials lengths are different: can't we do it in MNE
    # since MNE uses numpy array and not struct/cell like fieldTrip. 
    
    trial_data_list = matdata['reref_data']['trial'] # imports as list; unequal shapes
    time_list = matdata['reref_data']['time']
    
    # converting list to array gives error; due to unequal length trials!  
    
    # """ ValueError: setting an array element with a sequence. 
    # The requested array has an inhomogeneous shape after 2 dimensions. 
    # The detected shape was (168, 132) + inhomogeneous part."""
    
    """ # solution: normalize the trials length using nan
    # make sure all the length are equal and then convert it to numpy array! """
    
    trial_lengths = []
    time_lengths = []
       
    for trials in trial_data_list:
        trial_lengths.append(np.shape(trials))
    
    for times in time_list:
        time_lengths.append(np.shape(times))

    max_length = max(trial_lengths)[1] # find the max sample length (i.e., longest trial)
    print(max_length)
    
    # adj_trials = np.array([]).reshape(0,0,2200)
    """ for loop for normalizing data (uV) length"""
    for tr, trials in enumerate(trial_data_list):   # trials = trial_data_list[0]
       if np.shape(trials)[1] < max_length:
           len_diff = max_length - np.shape(trials)[1]
           tr_nan = np.ones([trials.shape[0],len_diff])*np.nan
           app_trials = np.hstack((trials, tr_nan))  # appended/padded trials with 'nan'
           app_trials = app_trials.reshape(1,app_trials.shape[0],app_trials.shape[1])
           if tr==0:
               adj_trials = app_trials.copy()  # length adjusted trials
               print(np.shape(adj_trials))
           else:
               adj_trials = np.concatenate((adj_trials,app_trials))
               print(np.shape(adj_trials))
        
       else:
           trials = trials.reshape(1,trials.shape[0],trials.shape[1]) # reshaping
           adj_trials = np.concatenate((adj_trials,trials))  # no padding with nan
           

    
    # converting the length of trials from samples to time
    # shape (n_trials, n_channels, n_samples)
    
    number_of_trials =  len(matdata['reref_data']['trial'])# in numbers.
    length_of_adj_trial = np.shape(adj_trials)[2] # in sample no.
    duration_of_adj_trial_sec =  length_of_adj_trial/sampling_freq # in seconds
    ## now all the trials in adj_trials matrix has the trial length of 4.4 s
    ## i.e., from -0.5 to 3.9 s 
    
    
    """ for loop for normalizing time (s) length"""
    for tr, times in enumerate(time_list):   # trials = trial_data_list[0]
       if np.shape(times)[0] < max_length:
           len_diff = max_length - np.shape(times)[0]
           tr_nan = np.ones([len_diff])*np.nan
           app_times = np.hstack((times, tr_nan))  # appended/padded trials with 'nan'
           app_times = app_times.reshape(1,app_times.shape[0])
           if tr==0:
               adj_times = app_times.copy()  # length adjusted trials
               print(np.shape(adj_times))
           else:
               adj_times = np.concatenate((adj_times,app_times))
               print(np.shape(adj_times))
        
       else:
           times = times.reshape(1,times.shape[0]) # reshaping
           adj_times = np.concatenate((adj_times,times))  # no padding with nan
 #%%          
    """ step 4:creating sample numbers matrix for each trials"""
    adj_samples_num = np.ones([number_of_trials,max_length])*np.nan
    for tr in np.arange(0,number_of_trials):
        adj_samples_num[tr,:] = trial_struct['t1'][tr] - sampling_freq * 0.5 + np.arange(0,max_length) # linearly arranging the samples 
    
    # adj_samples_num consist of n_trials x n_sample_num (eg, size = 168,2200)
    # note the two kinds of info: matrix index with size 0 to 2200 
    # and sample numbers of size ranging (x, y) of size 2200
    # np.where(adj_samples_num[tr] == trial_struct['t2'][tr])[0][0] gives matrix index corresponding to sample number
    
  #%%   
    """ step 5: creating sub-epochs and event marking each conditions with S1/S2/S4/S8/S16"""
    """ extracting data corresponding to trials with diff ISI """
    """ making a function definition to avoid code repetition """
    
      
    def data_for_subepoching (condition, event, trial_struct, adj_trials, adj_samples_num):
        
        # importing trial struct as pandas dataframe
        df = pd.DataFrame(trial_struct)   
        sampling_freq = 500 # in hz 
        # get the indices of all the trials for the condition
        array_ind  = df.loc[(df["Condition"] == condition)].index.values 
        # epoch_data = np.ones([np.shape(array_ind)[0], np.shape(adj_trials)[1], np.shape(adj_trials)[2]])*np.nan
        
        ## added on 15/01/2024 (for HC) and 22/05/2024 (for PD)
        ## exporting epochs separately for evoked and TF with different prestim lengths 
        if event == 'cue' and ep_extension == 'TF' :
            tsec_start = 0.8 # pre-stimulus(t2/S2/S4) duration in sec
            tsec_end = 1.3 # post-stimulus (t3/S8/S16) duration in sec
        elif event == 'target' and ep_extension  == 'TF' :
            tsec_start = 1.0 # pre-stimulus(t2/S2/S4) duration in sec
            tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
        else:
            tsec_start = 0.2 # pre-stimulus(t2/S2/S4) duration in sec
            tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
            
        num_samples_duration = int(500*(tsec_start + tsec_end + 1/500)) # one sample added for zeroth loc
        # for those trials (indices) extract the data at cue/target timings 
        for ti, tr in enumerate(array_ind): 
            if event == 'cue':
                t_event = np.where(adj_samples_num[tr] == trial_struct['t2'][tr])[0][0]  #gives matrix index corresponding to s2 sample number
            elif event == 'target':
                t_event = np.where(adj_samples_num[tr] == trial_struct['t3'][tr])[0][0]  #gives matrix index corresponding to s3 sample number
            
            start = t_event - int(tsec_start*sampling_freq)
            end = t_event + int(tsec_end*sampling_freq)
            # data = adj_trials[tr,:,start:end].reshape(1,np.shape(adj_trials)[1],num_samples_duration)
            
            diff = end - start # number of samples 
            diff_of_sample_dur = abs(diff - num_samples_duration)
            if diff == num_samples_duration:     
                data = adj_trials[tr,:,start:end].reshape(1,np.shape(adj_trials)[1],num_samples_duration)
            elif diff > num_samples_duration:
                data = adj_trials[tr,:,start:end-diff_of_sample_dur].reshape(1,np.shape(adj_trials)[1],num_samples_duration)
            elif diff < num_samples_duration:
                data = adj_trials[tr,:,start:end+diff_of_sample_dur].reshape(1,np.shape(adj_trials)[1],num_samples_duration)
                     
            if ti == 0:
                epoch_data = data.copy()            
            else:
                epoch_data = np.concatenate((epoch_data.copy(),data.copy()))
              
        return epoch_data  
    
    #%%  
    """ step 6: create subset of matrix trials by calling the function""" 

    event_type = ['cue', 'target']
    condi_type = ['GO only','GO', 'NO GO']
    condi_name = ['GOc', 'GOu', 'NoGo']

    for ei, evnt in enumerate(event_type):
        for ci, condi in enumerate(condi_type):

            data = data_for_subepoching(condi_type[ci], event_type[ei],
                                        matdata['trial_struct'], adj_trials, adj_samples_num)
        
            # CHANGING THE UNITS FROM uV (fieldTrip standard) to V (MNE standard)
            data = data * 1e-6
        #% saving the epoch data in numpy array
            print('  Writing the sub-epochs numpy array to disk')
            if ep_extension  == 'TF': 
                extension =  condi_name[ci] +'_' + event_type[ei] + '_eeg_prep_' + ep_extension 
            elif ep_extension  == 'evk':
                extension =  condi_name[ci] +'_' + event_type[ei] + '_eeg_prep' 
                
            npy_array_fname = op.join(eeg_subject_dir_GOODremove,
                                 config_for_gogait.base_fname_no_fif.format(**locals()))
            
            print("Output: ", npy_array_fname)
            np.save(npy_array_fname, data)  
            
        # epochs = mne.EpochsArray(data, info = info, tmin = -0.5)
        # epochs.plot()
      
        
    # ## saving the info as info.fif file in two kinds of directory: GOOD and GOODremove
    # extension = 'info'
    # info_fname = op.join(eeg_subject_dir_GOODremove,
    #                        config_for_gogait.base_fname.format(**locals()))
    # info.save(info_fname)
    
    # info_fname = op.join(eeg_subject_dir_GOOD,
    #                        config_for_gogait.base_fname.format(**locals()))
    # info.save(info_fname)    
        
  

    
    
    
    
   
        

