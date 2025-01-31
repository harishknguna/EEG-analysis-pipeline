#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:20:46 2023

@author: harish.gunasekaran
"""

"""
=============================================
10a. Single sub source reconstruction using eeg template MRI
Import single sub evoked data, apply forward model, and then inverse models
Generate STC and save the dipoles (vector format) as numpy. Report the foward and noise cov matrix

https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html#sphx-glr-auto-tutorials-forward-35-eeg-no-mri-py
https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html

==============================================
Q&A when doing source reconstruction
==============================================

1) Do you have individual MRIs?  if yes, use them, else, use a "template MRI" (average MRI: fsaverage)
2) Which method/algorithm to be used for inverse solution? MNE, dSPM, sLORETA or eLORETHA
3) EEG or MEG? To compute noise covarience, use pre-stim activities for EEG,
                                            use empty room recordings for MEG.
4) cortical sources should be placed only on surface, or volume, or both?
5) Localizing evoked or oscillatory activities. 

==============================================
Steps followed when doing source reconstruction
==============================================
1) Co-registration of head landmarks with MRI and obtain transformation matrix (trans)
2) Define source space template with certain number of distributed sources (src)
3) Define boundary element model (bem)
4) Make the forward solution using info, trans, src, bem (fwd)
5) Estimate the noise covarience (cov)
6) Compute the inverse operator using info, fwd, cov (inv)
7) Compute the source time course (stc) by applying inv operator using evk, inv, and method = dSPM 

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
from mne.datasets import fetch_fsaverage
from mne.coreg import Coregistration
from mne.minimum_norm import make_inverse_operator, apply_inverse 
from mne.minimum_norm import write_inverse_operator
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid, inset_locator

import config_for_gogait


version = 'CHANremove' 
method = "dSPM"  # "MNE" | "dSPM"

condi_name = ['GOc', 'GOu', 'NoGo']

event_type = ['target']   # 'target' | 'cue'

orientation = 'varyDipole'  # 'varyDipole' | 'fixed'

for subject in config_for_gogait.subjects_list: 
    
    eeg_subject_dir_GOODremove = op.join(config_for_gogait.eeg_dir_GOODremove, subject)
    print("Processing subject: %s" % subject)
    # just for cov estimation 
    condi_name_all = ['GOc', 'GOu', 'NoGo']
    event_cue = 'cue'
    
    """ [5.Prestim data at CUE for NOISE COV matrix] """
    """ step 5a: concatenate epochs of different conditions at CUE ONLY to compute noise cov matrix"""  
    for ci, condi in enumerate(condi_name_all):
        print('  importing epochs numpy array from disk')
        extension = condi_name_all[ci] +'_' + event_cue + '_' + version + '_epo'
        epochs_fname = op.join(eeg_subject_dir_GOODremove,
                                  config_for_gogait.base_fname.format(**locals()))
        print("Input: ", epochs_fname)
        epochs = mne.read_epochs(epochs_fname)
      
        ep_data = epochs.get_data()
        if ci == 0:
            epochs_array_for_cov = ep_data
        else:
            epochs_array_for_cov = np.vstack((epochs_array_for_cov, ep_data))
            
    
    epochs_for_cov = mne.EpochsArray(epochs_array_for_cov, info = epochs.info, 
                                     tmin = -0.2, baseline = (None,0))
                    
                
                
    #%%
    # create mne reports for saving plots 
    report = mne.Report() 
 
    for ci, condi in enumerate(condi_name):
        for ei, evnt in enumerate(event_type):
            print('  importing the epochs numpy array from disk')
            extension = condi_name[ci] +'_' + event_type[ei] + '_' + version + '_epo'
            epochs_fname = op.join(eeg_subject_dir_GOODremove,
                                      config_for_gogait.base_fname.format(**locals()))
            print("Input: ", epochs_fname)
            epochs = mne.read_epochs(epochs_fname)
            epochs = epochs.set_eeg_reference(projection=True)
            ## see below while applying inv(.) 
            ## ValueError: EEG average reference (using a projector) is mandatory for modeling,
            ## use the method set_eeg_reference(projection=True)
            
            info = epochs.info ### NOTE: simply not copying, but assigning bidirectionally 
         

            ## OPTION: standard_1005 ("works the best") OR brainproducts-RNP-BA-128
            # # Read and set the EEG electrode locations, which are already in fsaverage's
            # # space (MNI space) for standard_1020: 
            # #    standard_1005, brainproducts-RNP-BA-128
            montage_std = mne.channels.make_standard_montage("standard_1005") 
            mne.rename_channels(info, mapping = {'O9':'I1', 'O10':'I2'}, allow_duplicates=False)
            info.set_montage(montage_std, on_missing = 'warn') 
            # montage_std.plot(kind = '3d')                

            #%% Source imaging using template MRI
            ## STEP 1: compute FORWARD and INVERSE operators (only once: same across condi and event)
            ## using the standard template MRI subject, 'fsaverage'
            ## Adult template MRI (fsaverage)
            
            if ci==0 and ei==0:  
    
                # Download fsaverage files
                fs_dir = fetch_fsaverage(verbose=True)
                template_subjects_dir = op.dirname(fs_dir)
                
                # The files live in:
                template_subject = "fsaverage"
                
                ## 1a: Co-registration of landmarks of head with template MRI 
                """ [1.TRANS matrix] """
                ### define trasformation matrix 
                trans = op.join(fs_dir, "bem", "fsaverage-trans.fif")
                
                plot_kwargs = dict(
                                subject= template_subject,
                                subjects_dir= template_subjects_dir,
                                dig= "fiducials",
                                eeg=["original", "projected"],
                                show_axes=True,
                            )
                ## we are taking MNE's std trans 
                ## https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html#sphx-glr-auto-tutorials-forward-35-eeg-no-mri-py
                ## Read and set the EEG electrode locations, which are already in fsaverage's
                ## space (MNI space) for standard_1020:
    
               
    #%%            
                """ [2. SRC file] """
                # 1c: define source space 
                src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
                
                report.add_trans(trans = trans,
                                subject = template_subject,
                                subjects_dir = template_subjects_dir,
                                info = info,
                                title="template_sub_COREG")
                plt.close()
  
                #%% 1d: importing and plotting bem file of fsaverage subject [BEM]
                """ [3. BEM solution] """
                bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
                plot_bem_kwargs = dict(
                    subject = template_subject,
                    subjects_dir = template_subjects_dir,
                    brain_surfaces="white",
                    orientation="coronal",
                    slices=[50, 100, 150, 200]
                )
                
                fig_bem = mne.viz.plot_bem(**plot_bem_kwargs)
                report.add_figure(fig_bem, title = 'template_sub_BEM', replace = True)
                
                report.add_bem(subject = template_subject, 
                               subjects_dir= template_subjects_dir, 
                               title="tempMRI & tempBEM", decim=20, width=256)
                
                #%% 1e: compute foward solution 
                """ [4. FWD solution]"""
            
                
                fwd = mne.make_forward_solution(info = info, trans = trans, 
                                                src = src, bem=bem, eeg=True,
                                                mindist=5.0, n_jobs=None)
                
                report.add_forward(forward = fwd, title="template_subj_FWD")
              
            
            #%% STEP 2: Compute regularized noise covariance per subject (same for all condi)
                """ [5. COV matrix] """
                """ estimate the cov only for once (per sub) for all condi at CUE and use the same for target"""
            
           
                noise_cov = mne.compute_covariance(epochs_for_cov, tmin = -0.2, tmax=0.0,
                                                   method=["shrunk", "empirical"], 
                                                   rank=None, verbose=True)
                
            
                # fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, info = info)
                report.add_covariance(cov = noise_cov, info = epochs_for_cov.info,
                                      title= 'GOc_GOu_NoGO_cue_COV')
                
                print("Writing noise covariance matrix as FIF to disk")
                extension = version +'_cov' # keep _inv in the end
                fname_out = op.join(eeg_subject_dir_GOODremove,
                                     config_for_gogait.base_fname.format(**locals()))
                
                print("Output: ", fname_out)
                mne.write_cov(fname_out, noise_cov, overwrite = True, verbose = None)
                
            
            
                #%% STEP 3: compute the inverse per subject (same for all condi)
                
                """loose = float | ‘auto’ | dict"""
                ## Value that weights the source variances of the dipole components that are parallel (tangential) 
                ## to the cortical surface. Can be:
                ## float between 0 and 1 (inclusive). If 0, then the solution is computed with fixed orientation. 
                ## """If 1, it corresponds to free orientations.""" 
                ## 'auto' (default) Uses 0.2 for surface source spaces (unless fixed is True) and 
                ## 1.0 for other source spaces (volume or mixed). 
    
                """ fixed = bool | ‘auto’"""
                ## Use fixed source orientations normal to the cortical mantle. 
                ## If True, the loose parameter must be “auto” or 0. If ‘auto’, the loose value is used. 
                
                ## synthesis: if fixed == 'True',loose == 0, and orient = None,  signed activation (+/-)
                ##            if fixed == 'True', loose == 1, and orient = None , abs activation (+ve only)
                ##            if fixed == 'False', and loose == 0, same as case 1  if orient = None
                ##            if fixed == 'False', and loose == 1, same as case 2 with var % and amp changes 
                
               ##  if fixed == 'False', and loose == 0.2, same as case 1  if orient = "vector"
                
                # looseVal = 1 # was set 0.2 before
                
               
                if orientation == 'varyDipole':
                    looseVal = 1
                elif orientation == 'fixDipole':
                    looseVal = 0.2
                    
                
                inverse_operator = make_inverse_operator(info, fwd, noise_cov, fixed = 'False',
                                                         loose = looseVal, depth=0.8)
                report.add_inverse_operator(inverse_operator = inverse_operator, 
                                           title= 'GOc_GOu_NoGO_cue_INV')
                
                print("Writing inverse operator as FIF to disk")
                extension = version +'_'+ orientation +'_'+'inv' # keep _inv in the end
                fname_out = op.join(eeg_subject_dir_GOODremove,
                                     config_for_gogait.base_fname.format(**locals()))
                
                print("Output: ", fname_out)
                write_inverse_operator(fname_out, inverse_operator, overwrite = True, verbose = None)               
                
            
           
            #%% STEP 4: Compute inverse solution per condi and per sub
           
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
            
            # updated on 23/02/2024, this eef_ref_proj is placed above in info, to pass this info to inv operator
            evoked = epochs.average().pick("eeg")
            stc, residual = apply_inverse(
                                        evoked,
                                        inverse_operator,
                                        lambda2,
                                        method=method,
                                        pick_ori = "vector",
                                        return_residual = True,
                                        verbose=True)
            ## bug when using "vector": Stays in infinite loop when saving. overwriting existing file. 
            ## SOLUTION: saving STC files as numpy format 
            
            print('  Writing the stc to disk')
            extension = condi_name[ci] +'_' + event_type[ei] + '_' +'stc' + '_' + orientation +'_' + version +'_' + method
            stc_fname_array = op.join(eeg_subject_dir_GOODremove,
                                  config_for_gogait.base_fname_npy.format(**locals()))
            print("Output: ", stc_fname_array)
            stc_data = stc.data
            np.save(stc_fname_array, stc_data)
            
            ## NOTE: Extracting 3d sources by keeping pick_ori = "vector" and then taking norm is 
            ## equvivalent to setting the pick_ori = "normal" and directly obtaining 1d norm sources. 
            ## verified on 09/08/2024

            plt.close('all')
 
    ### finally saving the report after the for loop ends.     
    print('Saving the reports to disk')   
    report.title = 'Forward_model_and_noise_cov: ' + subject + '_at_'+ evnt +'_' + version +'_' + orientation + '_' + method 
    report_fname = op.join(config_for_gogait.report_dir_GOODremove,subject)
    report.save(report_fname+'_fwd_std1005_noisecov_cue' +'_at_'+ evnt +'_' + version +'_' + orientation + '_' + method  + '.html', overwrite=True)

    


#%% testing purpose
    # from scipy.linalg import norm
    
    # tsec_start = 0.2 # pre-stimulus(t2/S2/S4) duration in sec
    # tsec_end = 0.7 # post-stimulus (t3/S8/S16) duration in sec
    # n_samples_esti  = int(500*(tsec_start + tsec_end + 1/500)) # one sample added for zeroth loc
    # n_chs = 132
    # n_verticies = 20484
    # n_vectors = 3   
    # sfreq = 500
    # stc_data_in_norm = norm(stc.data, axis = 1)
    # stc_data_per_sub = stc_data_in_norm.copy()
    # vertno = [np.arange(0,int(n_verticies/2)), np.arange(0,int(n_verticies/2))]
    # stc = mne.SourceEstimate(stc_data_per_sub, vertices = vertno,
    #                           tstep = 1/sfreq, tmin = - 0.2,
    #                           subject = 'fsaverage')     
    # stc.plot()