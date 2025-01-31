# EEG-analysis-pipeline
EEG data analysis pipeline

This doc gives the overview of the project GOGAIT - source reconstruction - workdone by Harish between 07/2023 to 11/2024. 

working directory: /network/lustre/iss02/cenir/analyse/meeg/GOGAIT/Harish/gogait

from 19/11/2024 onwards:change in the working directory. 

study_path = '/network/iss/cenir/analyse/meeg/GOGAIT/Harish/gogait/'



Main folders
1) EEG: contains all the raw data and processed data such in numpy array format and MNE format
     processed data includes: epochs, evoked, time frequency 
     both in sensor space and source space (.stc)

2) scripts: all the python scripts and pipelines for analyzing the data are present here. 

3) reports: the outcomes of the scripts in html format are placed here and weekly reports are placed in the folder project_progress_meetings


INPUTs and OUTPUTs sample name formats (with one hc subject: BARER19, one condition: NoGo, and one event: target)
1. fieldTrip data: BARER19_All_BLOCKs.mat
2. numpy array evk: GOGAIT_BARER19_NoGo_target_eeg_prep.npy
3. numpy array TF: GOGAIT_BARER19_NoGo_target_eeg_prep_TF.npy
4. mne epoch: GOGAIT_BARER19_NoGo_target_CHANremove_epo.fif
5. mne evoked: GOGAIT_BARER19_NoGo_target_CHANremove_ave.fif
6. mne TF epochs, all freqs: GOGAIT_BARER19_NoGo_target_CHANremove_morlet_PWR_CMPLX_bslnCorr-tfr.h5
7. mne TF epochs, theta: GOGAIT_BARER19_NoGo_target_CHANremove_morlet_theta_PWR_CMPLX-tfr.h5
8. mne TF epochs, low beta: GOGAIT_BARER19_NoGo_target_CHANremove_morlet_beta_low_PWR_CMPLX-tfr.h5 
9. numpy array STC evoked: GOGAIT_BARER19_NoGo_target_stc_varyDipole_CHANremove_dSPM.npy
10. numpy array STC TF low beta: GOGAIT_BARER19_NoGo_TF_bslnCorr_avgFreq_target_stc_beta_low_morlet_varyDipole_CHANremove_dSPM.npy
11. numpy array STC evoked contrast: GOGAIT_BARER19_NoGo_GOc_target_stc_varyDipole_CHANremove_dSPM.npy
12. numpy array STC TF contrast: GOGAIT_BARER19_NoGo_GOu_TF_bslnCorr_avgFreq_target_stc_theta_morlet_varyDipole_CHANremove_dSPM.npy


Note: 

1. epochs length for evoked = -0.2 to 0.7 s (sampling freq = 500 Hz)
2. epochs length for TF = -1.0 to 0.7 s (sampling freq after decimation of 5 is 500/5 = 100 Hz)


Pre-stim baseline corrections: 

1. For evoked in sensor space: MNE applies correction at single subject level in scripts 2a and 2b, bslcorr = (-0.2, 0)
2. For evoked in source space: pre-stim mean subtraction applied at single subject level in scripts 11a and 11b , to bring the pre-stim value to zero (mne-norm of 3D dipoles made a non-zero baseline)
3. For TF in sensor space: just for sensor analysis for plotting, pre-stim log ratio, applied at single subject level in scripts 7a, 7b: baseline=(-0.5, -0.1), mode="logratio". 
4. For TF source analysis we apply just once during apply inverse stage at single subject level in scripts 15a and 15b: baseline=(-0.5, -0.1), mode="logratio"   

