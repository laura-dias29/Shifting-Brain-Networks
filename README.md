# Shifting-Brain-Networks

This repository contains the scripts used in the analysis of the paper: 'Shifting Brain Networks: Functional MRI connectivity differences between resting-state and a multi-element passive viewing task'
The goal of this study was to test a passive viewing task created in the context of the BRAVE study, with the purpose of enhancing the shift between brain networks, when compared to a resting state approach. We hypothesize that, by viewing a complex task formed by blocks of stimuli that evoke specific networks, the correlation between the time series of regions belonging to distinct networks is stronger than the correlation of the same regions, but in a resting state acquisition.

The scripts include steps to proceed with the preprocessing, processing and statistical analysis of both resting state and passive viewing task fMRI images.

Preprocessing_fMRI_FSL_pipeline.py - Preprocessing steps performed after images were denoised using a single session approach. Steps include slice timing correction, motion correction, brain extraction and image registration, using functions from FSL’s FMRIB’s Software Library (http://www.fmrib.ox.ac.uk/fsl).

Dice_coefficient - A dice coeffiecient was used to find overlapping networks between the two approaches using the maps obtained by running a multi-session temporal concatenation ICA on both data sets. The intersection of the overlapping maps was used in the remaining steps.

Network_analysis_pt_vs_rs - Processing steps performed after the dice coefficient step to obtain spacial maps of the common networks between subjects for both approaches. In order to extract time series from common networks of the two acquisistion types, 


