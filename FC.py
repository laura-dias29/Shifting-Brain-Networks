#!/usr/bin/env python
# coding: utf-8




import numpy as np
import nibabel as nb
import os, glob, signal
import subprocess
import nipype.interfaces.fsl as fsl
from nipype.interfaces.fsl import MCFLIRT, FLIRT
from nipype.utils.filemanip import Path
from os.path import exists
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys



import subprocess
import time
import pandas as pd
from skimage import measure

from itertools import combinations
from itertools import permutations
import scipy.stats

import math
import operator
from sklearn import linear_model
from scipy import stats
import seaborn as sns
from scipy.stats import binom




m_cont_rs=np.load("masks/array_t_s_controls_rs.npy") #path to array with time series form rs-fmri
m_cont_pt=np.load("masks/array_t_s_controls_pt.npy") #path to array with time series form pt-fmri




raw_input_list_rs_hc=np.genfromtxt(r"paths_concat_input_hc_rs.txt",dtype=str)

n_rs_hc = len(raw_input_list_rs_hc)

m_rs_hc=len(raw_input_list_rs_hc)
id_num_rs_hc=np.zeros(m_rs_hc)

for k in range(m_rs_hc):
    id_num_rs_hc[k]=(raw_input_list_rs_hc[k][97:100])

print(len(id_num_rs_hc))



n_cont_rs=63
list_of_vars_len_cont_rs = [m_cont_rs[0,0,:],m_cont_rs[1,0,:],m_cont_rs[2,0,:],m_cont_rs[3,0,:],m_cont_rs[4,0,:],m_cont_rs[5,0,:],m_cont_rs[6,0,:],m_cont_rs[7,0,:],m_cont_rs[8,0,:],m_cont_rs[9,0,:],m_cont_rs[10,0,:],m_cont_rs[11,0,:],m_cont_rs[12,0,:],m_cont_rs[13,0,:],m_cont_rs[14,0,:],m_cont_rs[15,0,:],m_cont_rs[16,0,:],m_cont_rs[17,0,:],m_cont_rs[18,0,:],m_cont_rs[19,0,:],m_cont_rs[20,0,:],m_cont_rs[21,0,:],m_cont_rs[22,0,:],m_cont_rs[23,0,:],m_cont_rs[24,0,:],m_cont_rs[25,0,:],m_cont_rs[26,0,:],m_cont_rs[27,0,:],m_cont_rs[28,0,:],m_cont_rs[29,0,:],m_cont_rs[30,0,:],m_cont_rs[31,0,:],m_cont_rs[32,0,:],m_cont_rs[33,0,:],m_cont_rs[34,0,:],m_cont_rs[35,0,:],m_cont_rs[36,0,:],m_cont_rs[37,0,:],m_cont_rs[38,0,:],m_cont_rs[39,0,:],m_cont_rs[40,0,:],m_cont_rs[41,0,:],m_cont_rs[42,0,:],m_cont_rs[43,0,:]]
blobs_cont_rs=len(list_of_vars_len_cont_rs)
d_cont_rs=int((np.square(blobs_cont_rs)-blobs_cont_rs)/2)
result_cont_rs=np.zeros([n_cont_rs,d_cont_rs])
for i in range(n_cont_rs):
    list_of_vars_cont_rs = [m_cont_rs[0,i,:],m_cont_rs[1,i,:],m_cont_rs[2,i,:],m_cont_rs[3,i,:],m_cont_rs[4,i,:],m_cont_rs[5,i,:],m_cont_rs[6,i,:],m_cont_rs[7,i,:],m_cont_rs[8,i,:],m_cont_rs[9,i,:],m_cont_rs[10,i,:],m_cont_rs[11,i,:],m_cont_rs[12,i,:],m_cont_rs[13,i,:],m_cont_rs[14,i,:],m_cont_rs[15,i,:],m_cont_rs[16,i,:],m_cont_rs[17,i,:],m_cont_rs[18,i,:],m_cont_rs[19,i,:],m_cont_rs[20,i,:],m_cont_rs[21,i,:],m_cont_rs[22,i,:],m_cont_rs[23,i,:],m_cont_rs[24,i,:],m_cont_rs[25,i,:],m_cont_rs[26,i,:],m_cont_rs[27,i,:],m_cont_rs[28,i,:],m_cont_rs[29,i,:],m_cont_rs[30,i,:],m_cont_rs[31,i,:],m_cont_rs[32,i,:],m_cont_rs[33,i,:],m_cont_rs[34,i,:],m_cont_rs[35,i,:],m_cont_rs[36,i,:],m_cont_rs[37,i,:],m_cont_rs[38,i,:],m_cont_rs[39,i,:],m_cont_rs[40,i,:],m_cont_rs[41,i,:],m_cont_rs[42,i,:],m_cont_rs[43,i,:]]
    result_array_cont_rs=np.array([scipy.stats.pearsonr(*pair) for pair in combinations(list_of_vars_cont_rs, 2)])
    result_cont_rs[i]=result_array_cont_rs[:,0]


print(np.shape(result_cont_rs))





z_cont_rs=np.zeros(np.shape(result_cont_rs))
for i in range(n_cont_rs):
    for j in range(len(result_cont_rs[0,:])):
        z_cont_rs[i,j]=(0.5*(np.log(1+result_cont_rs[i,j])-np.log(1-result_cont_rs[i,j])))
    
print(np.shape(z_cont_rs))





data_frame_conect_cont_rs=pd.DataFrame(z_cont_rs,columns=['C_1_2','C_1_3','C_1_4','C_1_5','C_1_6','C_1_7','C_1_8','C_1_9','C_1_10','C_1_11','C_1_12','C_1_13','C_1_14','C_1_15','C_1_16','C_1_17','C_1_18','C_1_19','C_1_20','C_1_21','C_1_22','C_1_23','C_1_24','C_1_25','C_1_26','C_1_27','C_1_28','C_1_29','C_1_30','C_1_31','C_1_32','C_1_33','C_1_34','C_1_35','C_1_36','C_1_37','C_1_38','C_1_39','C_1_40','C_1_41','C_1_42','C_1_43','C_1_44','C_2_3','C_2_4','C_2_5','C_2_6','C_2_7','C_2_8','C_2_9','C_2_10','C_2_11','C_2_12','C_2_13','C_2_14','C_2_15','C_2_16','C_2_17','C_2_18','C_2_19','C_2_20','C_2_21','C_2_22','C_2_23','C_2_24','C_2_25','C_2_26','C_2_27','C_2_28','C_2_29','C_2_30','C_2_31','C_2_32','C_2_33','C_2_34','C_2_35','C_2_36','C_2_37','C_2_38','C_2_39','C_2_40','C_2_41','C_2_42','C_2_43','C_2_44','C_3_4','C_3_5','C_3_6','C_3_7','C_3_8','C_3_9','C_3_10','C_3_11','C_3_12','C_3_13','C_3_14','C_3_15','C_3_16','C_3_17','C_3_18','C_3_19','C_3_20','C_3_21','C_3_22','C_3_23','C_3_24','C_3_25','C_3_26','C_3_27','C_3_28','C_3_29','C_3_30','C_3_31','C_3_32','C_3_33','C_3_34','C_3_35','C_3_36','C_3_37','C_3_38','C_3_39','C_3_40','C_3_41','C_3_42','C_3_43','C_3_44','C_4_5','C_4_6','C_4_7','C_4_8','C_4_9','C_4_10','C_4_11','C_4_12','C_4_13','C_4_14','C_4_15','C_4_16','C_4_17','C_4_18','C_4_19','C_4_20','C_4_21','C_4_22','C_4_23','C_4_24','C_4_25','C_4_26','C_4_27','C_4_28','C_4_29','C_4_30','C_4_31','C_4_32','C_4_33','C_4_34','C_4_35','C_4_36','C_4_37','C_4_38','C_4_39','C_4_40','C_4_41','C_4_42','C_4_43','C_4_44','C_5_6','C_5_7','C_5_8','C_5_9','C_5_10','C_5_11','C_5_12','C_5_13','C_5_14','C_5_15','C_5_16','C_5_17','C_5_18','C_5_19','C_5_20','C_5_21','C_5_22','C_5_23','C_5_24','C_5_25','C_5_26','C_5_27','C_5_28','C_5_29','C_5_30','C_5_31','C_5_32','C_5_33','C_5_34','C_5_35','C_5_36','C_5_37','C_5_38','C_5_39','C_5_40','C_5_41','C_5_42','C_5_43','C_5_44','C_6_7','C_6_8','C_6_9','C_6_10','C_6_11','C_6_12','C_6_13','C_6_14','C_6_15','C_6_16','C_6_17','C_6_18','C_6_19','C_6_20','C_6_21','C_6_22','C_6_23','C_6_24','C_6_25','C_6_26','C_6_27','C_6_28','C_6_29','C_6_30','C_6_31','C_6_32','C_6_33','C_6_34','C_6_35','C_6_36','C_6_37','C_6_38','C_6_39','C_6_40','C_6_41','C_6_42','C_6_43','C_6_44','C_7_8','C_7_9','C_7_10','C_7_11','C_7_12','C_7_13','C_7_14','C_7_15','C_7_16','C_7_17','C_7_18','C_7_19','C_7_20','C_7_21','C_7_22','C_7_23','C_7_24','C_7_25','C_7_26','C_7_27','C_7_28','C_7_29','C_7_30','C_7_31','C_7_32','C_7_33','C_7_34','C_7_35','C_7_36','C_7_37','C_7_38','C_7_39','C_7_40','C_7_41','C_7_42','C_7_43','C_7_44','C_8_9','C_8_10','C_8_11','C_8_12','C_8_13','C_8_14','C_8_15','C_8_16','C_8_17','C_8_18','C_8_19','C_8_20','C_8_21','C_8_22','C_8_23','C_8_24','C_8_25','C_8_26','C_8_27','C_8_28','C_8_29','C_8_30','C_8_31','C_8_32','C_8_33','C_8_34','C_8_35','C_8_36','C_8_37','C_8_38','C_8_39','C_8_40','C_8_41','C_8_42','C_8_43','C_8_44','C_9_10','C_9_11','C_9_12','C_9_13','C_9_14','C_9_15','C_9_16','C_9_17','C_9_18','C_9_19','C_9_20','C_9_21','C_9_22','C_9_23','C_9_24','C_9_25','C_9_26','C_9_27','C_9_28','C_9_29','C_9_30','C_9_31','C_9_32','C_9_33','C_9_34','C_9_35','C_9_36','C_9_37','C_9_38','C_9_39','C_9_40','C_9_41','C_9_42','C_9_43','C_9_44','C_10_11','C_10_12','C_10_13','C_10_14','C_10_15','C_10_16','C_10_17','C_10_18','C_10_19','C_10_20','C_10_21','C_10_22','C_10_23','C_10_24','C_10_25','C_10_26','C_10_27','C_10_28','C_10_29','C_10_30','C_10_31','C_10_32','C_10_33','C_10_34','C_10_35','C_10_36','C_10_37','C_10_38','C_10_39','C_10_40','C_10_41','C_10_42','C_10_43','C_10_44','C_11_12','C_11_13','C_11_14','C_11_15','C_11_16','C_11_17','C_11_18','C_11_19','C_11_20','C_11_21','C_11_22','C_11_23','C_11_24','C_11_25','C_11_26','C_11_27','C_11_28','C_11_29','C_11_30','C_11_31','C_11_32','C_11_33','C_11_34','C_11_35','C_11_36','C_11_37','C_11_38','C_11_39','C_11_40','C_11_41','C_11_42','C_11_43','C_11_44','C_12_13','C_12_14','C_12_15','C_12_16','C_12_17','C_12_18','C_12_19','C_12_20','C_12_21','C_12_22','C_12_23','C_12_24','C_12_25','C_12_26','C_12_27','C_12_28','C_12_29','C_12_30','C_12_31','C_12_32','C_12_33','C_12_34','C_12_35','C_12_36','C_12_37','C_12_38','C_12_39','C_12_40','C_12_41','C_12_42','C_12_43','C_12_44','C_13_14','C_13_15','C_13_16','C_13_17','C_13_18','C_13_19','C_13_20','C_13_21','C_13_22','C_13_23','C_13_24','C_13_25','C_13_26','C_13_27','C_13_28','C_13_29','C_13_30','C_13_31','C_13_32','C_13_33','C_13_34','C_13_35','C_13_36','C_13_37','C_13_38','C_13_39','C_13_40','C_13_41','C_13_42','C_13_43','C_13_44','C_14_15','C_14_16','C_14_17','C_14_18','C_14_19','C_14_20','C_14_21','C_14_22','C_14_23','C_14_24','C_14_25','C_14_26','C_14_27','C_14_28','C_14_29','C_14_30','C_14_31','C_14_32','C_14_33','C_14_34','C_14_35','C_14_36','C_14_37','C_14_38','C_14_39','C_14_40','C_14_41','C_14_42','C_14_43','C_14_44','C_15_16','C_15_17','C_15_18','C_15_19','C_15_20','C_15_21','C_15_22','C_15_23','C_15_24','C_15_25','C_15_26','C_15_27','C_15_28','C_15_29','C_15_30','C_15_31','C_15_32','C_15_33','C_15_34','C_15_35','C_15_36','C_15_37','C_15_38','C_15_39','C_15_40','C_15_41','C_15_42','C_15_43','C_15_44','C_16_17','C_16_18','C_16_19','C_16_20','C_16_21','C_16_22','C_16_23','C_16_24','C_16_25','C_16_26','C_16_27','C_16_28','C_16_29','C_16_30','C_16_31','C_16_32','C_16_33','C_16_34','C_16_35','C_16_36','C_16_37','C_16_38','C_16_39','C_16_40','C_16_41','C_16_42','C_16_43','C_16_44','C_17_18','C_17_19','C_17_20','C_17_21','C_17_22','C_17_23','C_17_24','C_17_25','C_17_26','C_17_27','C_17_28','C_17_29','C_17_30','C_17_31','C_17_32','C_17_33','C_17_34','C_17_35','C_17_36','C_17_37','C_17_38','C_17_39','C_17_40','C_17_41','C_17_42','C_17_43','C_17_44','C_18_19','C_18_20','C_18_21','C_18_22','C_18_23','C_18_24','C_18_25','C_18_26','C_18_27','C_18_28','C_18_29','C_18_30','C_18_31','C_18_32','C_18_33','C_18_34','C_18_35','C_18_36','C_18_37','C_18_38','C_18_39','C_18_40','C_18_41','C_18_42','C_18_43','C_18_44','C_19_20','C_19_21','C_19_22','C_19_23','C_19_24','C_19_25','C_19_26','C_19_27','C_19_28','C_19_29','C_19_30','C_19_31','C_19_32','C_19_33','C_19_34','C_19_35','C_19_36','C_19_37','C_19_38','C_19_39','C_19_40','C_19_41','C_19_42','C_19_43','C_19_44','C_20_21','C_20_22','C_20_23','C_20_24','C_20_25','C_20_26','C_20_27','C_20_28','C_20_29','C_20_30','C_20_31','C_20_32','C_20_33','C_20_34','C_20_35','C_20_36','C_20_37','C_20_38','C_20_39','C_20_40','C_20_41','C_20_42','C_20_43','C_20_44','C_21_22','C_21_23','C_21_24','C_21_25','C_21_26','C_21_27','C_21_28','C_21_29','C_21_30','C_21_31','C_21_32','C_21_33','C_21_34','C_21_35','C_21_36','C_21_37','C_21_38','C_21_39','C_21_40','C_21_41','C_21_42','C_21_43','C_21_44','C_22_23','C_22_24','C_22_25','C_22_26','C_22_27','C_22_28','C_22_29','C_22_30','C_22_31','C_22_32','C_22_33','C_22_34','C_22_35','C_22_36','C_22_37','C_22_38','C_22_39','C_22_40','C_22_41','C_22_42','C_22_43','C_22_44','C_23_24','C_23_25','C_23_26','C_23_27','C_23_28','C_23_29','C_23_30','C_23_31','C_23_32','C_23_33','C_23_34','C_23_35','C_23_36','C_23_37','C_23_38','C_23_39','C_23_40','C_23_41','C_23_42','C_23_43','C_23_44','C_24_25','C_24_26','C_24_27','C_24_28','C_24_29','C_24_30','C_24_31','C_24_32','C_24_33','C_24_34','C_24_35','C_24_36','C_24_37','C_24_38','C_24_39','C_24_40','C_24_41','C_24_42','C_24_43','C_24_44','C_25_26','C_25_27','C_25_28','C_25_29','C_25_30','C_25_31','C_25_32','C_25_33','C_25_34','C_25_35','C_25_36','C_25_37','C_25_38','C_25_39','C_25_40','C_25_41','C_25_42','C_25_43','C_25_44','C_26_27','C_26_28','C_26_29','C_26_30','C_26_31','C_26_32','C_26_33','C_26_34','C_26_35','C_26_36','C_26_37','C_26_38','C_26_39','C_26_40','C_26_41','C_26_42','C_26_43','C_26_44','C_27_28','C_27_29','C_27_30','C_27_31','C_27_32','C_27_33','C_27_34','C_27_35','C_27_36','C_27_37','C_27_38','C_27_39','C_27_40','C_27_41','C_27_42','C_27_43','C_27_44','C_28_29','C_28_30','C_28_31','C_28_32','C_28_33','C_28_34','C_28_35','C_28_36','C_28_37','C_28_38','C_28_39','C_28_40','C_28_41','C_28_42','C_28_43','C_28_44','C_29_30','C_29_31','C_29_32','C_29_33','C_29_34','C_29_35','C_29_36','C_29_37','C_29_38','C_29_39','C_29_40','C_29_41','C_29_42','C_29_43','C_29_44','C_30_31','C_30_32','C_30_33','C_30_34','C_30_35','C_30_36','C_30_37','C_30_38','C_30_39','C_30_40','C_30_41','C_30_42','C_30_43','C_30_44','C_31_32','C_31_33','C_31_34','C_31_35','C_31_36','C_31_37','C_31_38','C_31_39','C_31_40','C_31_41','C_31_42','C_31_43','C_31_44','C_32_33','C_32_34','C_32_35','C_32_36','C_32_37','C_32_38','C_32_39','C_32_40','C_32_41','C_32_42','C_32_43','C_32_44','C_33_34','C_33_35','C_33_36','C_33_37','C_33_38','C_33_39','C_33_40','C_33_41','C_33_42','C_33_43','C_33_44','C_34_35','C_34_36','C_34_37','C_34_38','C_34_39','C_34_40','C_34_41','C_34_42','C_34_43','C_34_44','C_35_36','C_35_37','C_35_38','C_35_39','C_35_40','C_35_41','C_35_42','C_35_43','C_35_44','C_36_37','C_36_38','C_36_39','C_36_40','C_36_41','C_36_42','C_36_43','C_36_44','C_37_38','C_37_39','C_37_40','C_37_41','C_37_42','C_37_43','C_37_44','C_38_39','C_38_40','C_38_41','C_38_42','C_38_43','C_38_44','C_39_40','C_39_41','C_39_42','C_39_43','C_39_44','C_40_41','C_40_42','C_40_43','C_40_44','C_41_42','C_41_43','C_41_44','C_42_43','C_42_44','C_43_44'])


#data_frame_conect_cont_rs['IDS'] = id_num_m_c
#cols = data_frame_conect_cont_rs.columns.tolist()
#cols = cols[-1:] + cols[:-1]
#data_frame_conect_cont_rs=data_frame_conect_cont_rs[cols]




data_frame_conect_cont_rs.to_csv('data_frame_con_hc_rs.csv')  





n_cont_pt=63
list_of_vars_len_cont_pt = [m_cont_pt[0,0,:],m_cont_pt[1,0,:],m_cont_pt[2,0,:],m_cont_pt[3,0,:],m_cont_pt[4,0,:],m_cont_pt[5,0,:],m_cont_pt[6,0,:],m_cont_pt[7,0,:],m_cont_pt[8,0,:],m_cont_pt[9,0,:],m_cont_pt[10,0,:],m_cont_pt[11,0,:],m_cont_pt[12,0,:],m_cont_pt[13,0,:],m_cont_pt[14,0,:],m_cont_pt[15,0,:],m_cont_pt[16,0,:],m_cont_pt[17,0,:],m_cont_pt[18,0,:],m_cont_pt[19,0,:],m_cont_pt[20,0,:],m_cont_pt[21,0,:],m_cont_pt[22,0,:],m_cont_pt[23,0,:],m_cont_pt[24,0,:],m_cont_pt[25,0,:],m_cont_pt[26,0,:],m_cont_pt[27,0,:],m_cont_pt[28,0,:],m_cont_pt[29,0,:],m_cont_pt[30,0,:],m_cont_pt[31,0,:],m_cont_pt[32,0,:],m_cont_pt[33,0,:],m_cont_pt[34,0,:],m_cont_pt[35,0,:],m_cont_pt[36,0,:],m_cont_pt[37,0,:],m_cont_pt[38,0,:],m_cont_pt[39,0,:],m_cont_pt[40,0,:],m_cont_pt[41,0,:],m_cont_pt[42,0,:],m_cont_pt[43,0,:]]
blobs_cont_pt=len(list_of_vars_len_cont_pt)
d_cont_pt=int((np.square(blobs_cont_pt)-blobs_cont_pt)/2)
result_cont_pt=np.zeros([n_cont_pt,d_cont_pt])
for i in range(n_cont_pt):
    list_of_vars_cont_pt = [m_cont_pt[0,i,:],m_cont_pt[1,i,:],m_cont_pt[2,i,:],m_cont_pt[3,i,:],m_cont_pt[4,i,:],m_cont_pt[5,i,:],m_cont_pt[6,i,:],m_cont_pt[7,i,:],m_cont_pt[8,i,:],m_cont_pt[9,i,:],m_cont_pt[10,i,:],m_cont_pt[11,i,:],m_cont_pt[12,i,:],m_cont_pt[13,i,:],m_cont_pt[14,i,:],m_cont_pt[15,i,:],m_cont_pt[16,i,:],m_cont_pt[17,i,:],m_cont_pt[18,i,:],m_cont_pt[19,i,:],m_cont_pt[20,i,:],m_cont_pt[21,i,:],m_cont_pt[22,i,:],m_cont_pt[23,i,:],m_cont_pt[24,i,:],m_cont_pt[25,i,:],m_cont_pt[26,i,:],m_cont_pt[27,i,:],m_cont_pt[28,i,:],m_cont_pt[29,i,:],m_cont_pt[30,i,:],m_cont_pt[31,i,:],m_cont_pt[32,i,:],m_cont_pt[33,i,:],m_cont_pt[34,i,:],m_cont_pt[35,i,:],m_cont_pt[36,i,:],m_cont_pt[37,i,:],m_cont_pt[38,i,:],m_cont_pt[39,i,:],m_cont_pt[40,i,:],m_cont_pt[41,i,:],m_cont_pt[42,i,:],m_cont_pt[43,i,:]]
    result_array_cont_pt=np.array([scipy.stats.pearsonr(*pair) for pair in combinations(list_of_vars_cont_pt, 2)])
    result_cont_pt[i]=result_array_cont_pt[:,0]


print(np.shape(result_cont_pt))





z_cont_pt=np.zeros(np.shape(result_cont_pt))
for i in range(n_cont_pt):
    for j in range(len(result_cont_pt[0,:])):
        z_cont_pt[i,j]=(0.5*(np.log(1+result_cont_pt[i,j])-np.log(1-result_cont_pt[i,j])))
    
print(np.shape(z_cont_pt))





data_frame_conect_cont_pt=pd.DataFrame(z_cont_pt,columns=['C_1_2','C_1_3','C_1_4','C_1_5','C_1_6','C_1_7','C_1_8','C_1_9','C_1_10','C_1_11','C_1_12','C_1_13','C_1_14','C_1_15','C_1_16','C_1_17','C_1_18','C_1_19','C_1_20','C_1_21','C_1_22','C_1_23','C_1_24','C_1_25','C_1_26','C_1_27','C_1_28','C_1_29','C_1_30','C_1_31','C_1_32','C_1_33','C_1_34','C_1_35','C_1_36','C_1_37','C_1_38','C_1_39','C_1_40','C_1_41','C_1_42','C_1_43','C_1_44','C_2_3','C_2_4','C_2_5','C_2_6','C_2_7','C_2_8','C_2_9','C_2_10','C_2_11','C_2_12','C_2_13','C_2_14','C_2_15','C_2_16','C_2_17','C_2_18','C_2_19','C_2_20','C_2_21','C_2_22','C_2_23','C_2_24','C_2_25','C_2_26','C_2_27','C_2_28','C_2_29','C_2_30','C_2_31','C_2_32','C_2_33','C_2_34','C_2_35','C_2_36','C_2_37','C_2_38','C_2_39','C_2_40','C_2_41','C_2_42','C_2_43','C_2_44','C_3_4','C_3_5','C_3_6','C_3_7','C_3_8','C_3_9','C_3_10','C_3_11','C_3_12','C_3_13','C_3_14','C_3_15','C_3_16','C_3_17','C_3_18','C_3_19','C_3_20','C_3_21','C_3_22','C_3_23','C_3_24','C_3_25','C_3_26','C_3_27','C_3_28','C_3_29','C_3_30','C_3_31','C_3_32','C_3_33','C_3_34','C_3_35','C_3_36','C_3_37','C_3_38','C_3_39','C_3_40','C_3_41','C_3_42','C_3_43','C_3_44','C_4_5','C_4_6','C_4_7','C_4_8','C_4_9','C_4_10','C_4_11','C_4_12','C_4_13','C_4_14','C_4_15','C_4_16','C_4_17','C_4_18','C_4_19','C_4_20','C_4_21','C_4_22','C_4_23','C_4_24','C_4_25','C_4_26','C_4_27','C_4_28','C_4_29','C_4_30','C_4_31','C_4_32','C_4_33','C_4_34','C_4_35','C_4_36','C_4_37','C_4_38','C_4_39','C_4_40','C_4_41','C_4_42','C_4_43','C_4_44','C_5_6','C_5_7','C_5_8','C_5_9','C_5_10','C_5_11','C_5_12','C_5_13','C_5_14','C_5_15','C_5_16','C_5_17','C_5_18','C_5_19','C_5_20','C_5_21','C_5_22','C_5_23','C_5_24','C_5_25','C_5_26','C_5_27','C_5_28','C_5_29','C_5_30','C_5_31','C_5_32','C_5_33','C_5_34','C_5_35','C_5_36','C_5_37','C_5_38','C_5_39','C_5_40','C_5_41','C_5_42','C_5_43','C_5_44','C_6_7','C_6_8','C_6_9','C_6_10','C_6_11','C_6_12','C_6_13','C_6_14','C_6_15','C_6_16','C_6_17','C_6_18','C_6_19','C_6_20','C_6_21','C_6_22','C_6_23','C_6_24','C_6_25','C_6_26','C_6_27','C_6_28','C_6_29','C_6_30','C_6_31','C_6_32','C_6_33','C_6_34','C_6_35','C_6_36','C_6_37','C_6_38','C_6_39','C_6_40','C_6_41','C_6_42','C_6_43','C_6_44','C_7_8','C_7_9','C_7_10','C_7_11','C_7_12','C_7_13','C_7_14','C_7_15','C_7_16','C_7_17','C_7_18','C_7_19','C_7_20','C_7_21','C_7_22','C_7_23','C_7_24','C_7_25','C_7_26','C_7_27','C_7_28','C_7_29','C_7_30','C_7_31','C_7_32','C_7_33','C_7_34','C_7_35','C_7_36','C_7_37','C_7_38','C_7_39','C_7_40','C_7_41','C_7_42','C_7_43','C_7_44','C_8_9','C_8_10','C_8_11','C_8_12','C_8_13','C_8_14','C_8_15','C_8_16','C_8_17','C_8_18','C_8_19','C_8_20','C_8_21','C_8_22','C_8_23','C_8_24','C_8_25','C_8_26','C_8_27','C_8_28','C_8_29','C_8_30','C_8_31','C_8_32','C_8_33','C_8_34','C_8_35','C_8_36','C_8_37','C_8_38','C_8_39','C_8_40','C_8_41','C_8_42','C_8_43','C_8_44','C_9_10','C_9_11','C_9_12','C_9_13','C_9_14','C_9_15','C_9_16','C_9_17','C_9_18','C_9_19','C_9_20','C_9_21','C_9_22','C_9_23','C_9_24','C_9_25','C_9_26','C_9_27','C_9_28','C_9_29','C_9_30','C_9_31','C_9_32','C_9_33','C_9_34','C_9_35','C_9_36','C_9_37','C_9_38','C_9_39','C_9_40','C_9_41','C_9_42','C_9_43','C_9_44','C_10_11','C_10_12','C_10_13','C_10_14','C_10_15','C_10_16','C_10_17','C_10_18','C_10_19','C_10_20','C_10_21','C_10_22','C_10_23','C_10_24','C_10_25','C_10_26','C_10_27','C_10_28','C_10_29','C_10_30','C_10_31','C_10_32','C_10_33','C_10_34','C_10_35','C_10_36','C_10_37','C_10_38','C_10_39','C_10_40','C_10_41','C_10_42','C_10_43','C_10_44','C_11_12','C_11_13','C_11_14','C_11_15','C_11_16','C_11_17','C_11_18','C_11_19','C_11_20','C_11_21','C_11_22','C_11_23','C_11_24','C_11_25','C_11_26','C_11_27','C_11_28','C_11_29','C_11_30','C_11_31','C_11_32','C_11_33','C_11_34','C_11_35','C_11_36','C_11_37','C_11_38','C_11_39','C_11_40','C_11_41','C_11_42','C_11_43','C_11_44','C_12_13','C_12_14','C_12_15','C_12_16','C_12_17','C_12_18','C_12_19','C_12_20','C_12_21','C_12_22','C_12_23','C_12_24','C_12_25','C_12_26','C_12_27','C_12_28','C_12_29','C_12_30','C_12_31','C_12_32','C_12_33','C_12_34','C_12_35','C_12_36','C_12_37','C_12_38','C_12_39','C_12_40','C_12_41','C_12_42','C_12_43','C_12_44','C_13_14','C_13_15','C_13_16','C_13_17','C_13_18','C_13_19','C_13_20','C_13_21','C_13_22','C_13_23','C_13_24','C_13_25','C_13_26','C_13_27','C_13_28','C_13_29','C_13_30','C_13_31','C_13_32','C_13_33','C_13_34','C_13_35','C_13_36','C_13_37','C_13_38','C_13_39','C_13_40','C_13_41','C_13_42','C_13_43','C_13_44','C_14_15','C_14_16','C_14_17','C_14_18','C_14_19','C_14_20','C_14_21','C_14_22','C_14_23','C_14_24','C_14_25','C_14_26','C_14_27','C_14_28','C_14_29','C_14_30','C_14_31','C_14_32','C_14_33','C_14_34','C_14_35','C_14_36','C_14_37','C_14_38','C_14_39','C_14_40','C_14_41','C_14_42','C_14_43','C_14_44','C_15_16','C_15_17','C_15_18','C_15_19','C_15_20','C_15_21','C_15_22','C_15_23','C_15_24','C_15_25','C_15_26','C_15_27','C_15_28','C_15_29','C_15_30','C_15_31','C_15_32','C_15_33','C_15_34','C_15_35','C_15_36','C_15_37','C_15_38','C_15_39','C_15_40','C_15_41','C_15_42','C_15_43','C_15_44','C_16_17','C_16_18','C_16_19','C_16_20','C_16_21','C_16_22','C_16_23','C_16_24','C_16_25','C_16_26','C_16_27','C_16_28','C_16_29','C_16_30','C_16_31','C_16_32','C_16_33','C_16_34','C_16_35','C_16_36','C_16_37','C_16_38','C_16_39','C_16_40','C_16_41','C_16_42','C_16_43','C_16_44','C_17_18','C_17_19','C_17_20','C_17_21','C_17_22','C_17_23','C_17_24','C_17_25','C_17_26','C_17_27','C_17_28','C_17_29','C_17_30','C_17_31','C_17_32','C_17_33','C_17_34','C_17_35','C_17_36','C_17_37','C_17_38','C_17_39','C_17_40','C_17_41','C_17_42','C_17_43','C_17_44','C_18_19','C_18_20','C_18_21','C_18_22','C_18_23','C_18_24','C_18_25','C_18_26','C_18_27','C_18_28','C_18_29','C_18_30','C_18_31','C_18_32','C_18_33','C_18_34','C_18_35','C_18_36','C_18_37','C_18_38','C_18_39','C_18_40','C_18_41','C_18_42','C_18_43','C_18_44','C_19_20','C_19_21','C_19_22','C_19_23','C_19_24','C_19_25','C_19_26','C_19_27','C_19_28','C_19_29','C_19_30','C_19_31','C_19_32','C_19_33','C_19_34','C_19_35','C_19_36','C_19_37','C_19_38','C_19_39','C_19_40','C_19_41','C_19_42','C_19_43','C_19_44','C_20_21','C_20_22','C_20_23','C_20_24','C_20_25','C_20_26','C_20_27','C_20_28','C_20_29','C_20_30','C_20_31','C_20_32','C_20_33','C_20_34','C_20_35','C_20_36','C_20_37','C_20_38','C_20_39','C_20_40','C_20_41','C_20_42','C_20_43','C_20_44','C_21_22','C_21_23','C_21_24','C_21_25','C_21_26','C_21_27','C_21_28','C_21_29','C_21_30','C_21_31','C_21_32','C_21_33','C_21_34','C_21_35','C_21_36','C_21_37','C_21_38','C_21_39','C_21_40','C_21_41','C_21_42','C_21_43','C_21_44','C_22_23','C_22_24','C_22_25','C_22_26','C_22_27','C_22_28','C_22_29','C_22_30','C_22_31','C_22_32','C_22_33','C_22_34','C_22_35','C_22_36','C_22_37','C_22_38','C_22_39','C_22_40','C_22_41','C_22_42','C_22_43','C_22_44','C_23_24','C_23_25','C_23_26','C_23_27','C_23_28','C_23_29','C_23_30','C_23_31','C_23_32','C_23_33','C_23_34','C_23_35','C_23_36','C_23_37','C_23_38','C_23_39','C_23_40','C_23_41','C_23_42','C_23_43','C_23_44','C_24_25','C_24_26','C_24_27','C_24_28','C_24_29','C_24_30','C_24_31','C_24_32','C_24_33','C_24_34','C_24_35','C_24_36','C_24_37','C_24_38','C_24_39','C_24_40','C_24_41','C_24_42','C_24_43','C_24_44','C_25_26','C_25_27','C_25_28','C_25_29','C_25_30','C_25_31','C_25_32','C_25_33','C_25_34','C_25_35','C_25_36','C_25_37','C_25_38','C_25_39','C_25_40','C_25_41','C_25_42','C_25_43','C_25_44','C_26_27','C_26_28','C_26_29','C_26_30','C_26_31','C_26_32','C_26_33','C_26_34','C_26_35','C_26_36','C_26_37','C_26_38','C_26_39','C_26_40','C_26_41','C_26_42','C_26_43','C_26_44','C_27_28','C_27_29','C_27_30','C_27_31','C_27_32','C_27_33','C_27_34','C_27_35','C_27_36','C_27_37','C_27_38','C_27_39','C_27_40','C_27_41','C_27_42','C_27_43','C_27_44','C_28_29','C_28_30','C_28_31','C_28_32','C_28_33','C_28_34','C_28_35','C_28_36','C_28_37','C_28_38','C_28_39','C_28_40','C_28_41','C_28_42','C_28_43','C_28_44','C_29_30','C_29_31','C_29_32','C_29_33','C_29_34','C_29_35','C_29_36','C_29_37','C_29_38','C_29_39','C_29_40','C_29_41','C_29_42','C_29_43','C_29_44','C_30_31','C_30_32','C_30_33','C_30_34','C_30_35','C_30_36','C_30_37','C_30_38','C_30_39','C_30_40','C_30_41','C_30_42','C_30_43','C_30_44','C_31_32','C_31_33','C_31_34','C_31_35','C_31_36','C_31_37','C_31_38','C_31_39','C_31_40','C_31_41','C_31_42','C_31_43','C_31_44','C_32_33','C_32_34','C_32_35','C_32_36','C_32_37','C_32_38','C_32_39','C_32_40','C_32_41','C_32_42','C_32_43','C_32_44','C_33_34','C_33_35','C_33_36','C_33_37','C_33_38','C_33_39','C_33_40','C_33_41','C_33_42','C_33_43','C_33_44','C_34_35','C_34_36','C_34_37','C_34_38','C_34_39','C_34_40','C_34_41','C_34_42','C_34_43','C_34_44','C_35_36','C_35_37','C_35_38','C_35_39','C_35_40','C_35_41','C_35_42','C_35_43','C_35_44','C_36_37','C_36_38','C_36_39','C_36_40','C_36_41','C_36_42','C_36_43','C_36_44','C_37_38','C_37_39','C_37_40','C_37_41','C_37_42','C_37_43','C_37_44','C_38_39','C_38_40','C_38_41','C_38_42','C_38_43','C_38_44','C_39_40','C_39_41','C_39_42','C_39_43','C_39_44','C_40_41','C_40_42','C_40_43','C_40_44','C_41_42','C_41_43','C_41_44','C_42_43','C_42_44','C_43_44'])


#data_frame_conect_cont_pt['IDS'] = id_num_m_c
#cols = data_frame_conect_cont_pt.columns.tolist()
#cols = cols[-1:] + cols[:-1]
#data_frame_conect_cont_pt=data_frame_conect_cont_pt[cols]





data_frame_conect_cont_pt.to_csv('data_frame_con_hc_pt.csv')  




# # T-Test pt vs rs




s_rs=np.zeros(946)
s_pt=np.zeros(946)
i=0
for column_rs in data_frame_conect_cont_rs.columns[0:946]:
    
    r=stats.shapiro(data_frame_conect_cont_rs[column_rs])
    p=stats.shapiro(data_frame_conect_cont_pt[column_rs])
    s_rs[i]=r[1]
    s_pt[i]=p[1]
    i+=1   
    





p_rs_out=[]
for i in range(len(s_rs)):
    if s_rs[i]<0.05:
        p_rs_out.append(s_rs)





p_pt_out=[]
for i in range(len(s_pt)):
    if s_pt[i]<0.05:
        p_pt_out.append(s_pt)




from numpy.lib import recfunctions as rfn
t=np.zeros(946)
p_value=np.zeros(946)

i=0
for column_rs in data_frame_conect_cont_rs.columns[0:946]:
    
    t_test=stats.ttest_rel(data_frame_conect_cont_rs[column_rs], data_frame_conect_cont_pt[column_rs])
    
        
    t[i]=t_test[0]
    p_value[i]=t_test[1]
    i+=1
print(p_value)





for i in range(len(p_value)):
    if p_value[i]<0.05/946:
        print(i)





N=946
q = 0.05
i = np.arange(1, N+1)

#p_value_s=np.sort(p_value)    
below = p_value < (q * i / N) 
max_below = np.max(np.where(below)) 

print('p_i:', p_value[max_below])

print('i:', max_below + 1) 





import statsmodels.stats.multitest as smt
smt.multipletests(p_value, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)





import statsmodels.stats.multitest as smt
smt.multipletests(p_value, alpha=0.05, method='holm', is_sorted=False, returnsorted=False)





ind=[]
p_value_thresh=np.zeros((len(p_value)))
for i in range(len(p_value)):
    if p_value[i]<=0.007306009042864063:
        if t[i]>0:
            p_value_thresh[i]=1
        else:
            p_value_thresh[i]=-1
        ind.append(i)





n = 44


t_test_matrix = np.zeros((n,n)) # Initialize nxn matrix
    
triu = np.triu_indices(n,k=1) # Find upper right indices of a triangular nxn matrix
tril = np.tril_indices(n, k=-1)
t_test_matrix[triu] = t # Assign list values to upper right matrix
t_test_matrix[tril] = t_test_matrix.T[tril] # Make the matrix symmetric

print(np.shape(t_test_matrix))



#plt.imshow(t_test_matrix, cmap='hot', interpolation='nearest')

#plt.colorbar(label="t_value")
#plt.show()




ax = sns.heatmap(t_test_matrix,cmap="coolwarm",center=0)




n = 44


t_test_p_t = np.zeros((n,n)) # Initialize nxn matrix
    
triu = np.triu_indices(n,k=1) # Find upper right indices of a triangular nxn matrix
tril = np.tril_indices(n, k=-1)
t_test_p_t[triu] = p_value_thresh # Assign list values to upper right matrix
t_test_p_t[tril] = t_test_p_t.T[tril] # Make the matrix symmetric

print(np.shape(t_test_p_t))




ax_p = sns.heatmap(t_test_p_t,cmap="coolwarm",center=0)





fig, (ax1) = plt.subplots(1, 1, figsize=(12,7))
sns.heatmap(t_test_matrix,cmap="coolwarm",center=0,ax=ax1,cbar_kws={'label': 't_values'})
ax1.set_title('Connections between all regions')
plt.savefig('color_map.png')
fig, (ax2) = plt.subplots(1, 1, figsize=(9.65,7))
sns.heatmap(t_test_p_t,cmap="coolwarm",cbar=False)
ax2.set_title('Connections between all regions')
plt.savefig('color_map_2.png')



