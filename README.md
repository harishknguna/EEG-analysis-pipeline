# EEG-analysis-pipeline
EEG data analysis pipeline

The processing pipeline follows a clear structure, moving from raw EEG data input and preprocessing to group-level analyses, visualization, and data export. The analysis is organized into Sensor Space and Source Space analysis with specific scripts handling each step.

1. Data Input & Preprocessing
	•	Scripts (01a, 01b): Convert EEG data from FieldTrip into single-subject epochs as numpy arrays.
	•	Scripts (02a, 02b): Transform single-subject epochs into MNE format, including evoked responses.

2. ERP Analysis (Sensor Space Analysis - Blue Blocks)
	•	Single-Subject Level:
	•	Scripts (03a, 03b): Generate event-related potentials (ERPs) and topographies (N1, N2, P3).
	•	Group-Level Analysis:
	•	Scripts (04a, 04b): Perform group-level ERP analysis and save group-evoked data.

3. Time-Frequency (TF) Analysis (Sensor Space Analysis - Red Blocks)
	•	Single-Subject Level:
	•	Scripts (05a, 05b) and (07a, 07b): Perform Morlet wavelet and multitaper TF analyses, generating TF plots and data in MNE format.
	•	Group-Level Analysis:
	•	Scripts (06a, 06b) and (08a, 08b): Perform similar analyses at the group level.
	•	Scripts (09a, 09b): Generate group-level Morlet TF topoplots.

4. Source-Level Analysis (Source Space Analysis)
   Single-Subject Level:
	•	Scripts (10a, 10b): Scripts save evoked (stcEVK) data as numpy arrays.
     •	Scripts (15a, 15b): Scripts save TF (stcTFR) data as numpy arrays.
   Group-Level Analysis:
	•	Scripts (11a, 11b): Scripts save group level evoked (stcEVK) data as numpy arrays.
     •	Scripts (16a, 16b): Scripts save group level TF (stcTFR) data as numpy arrays.
   


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

