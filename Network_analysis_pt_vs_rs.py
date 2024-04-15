#!/usr/bin/env python
# coding: utf-8

# # Passive task vs Resting state

# ### 1-Inferior frontal
# ### 2-Cerebellar
# ### 3-Motor
# ### 4-LFP
# ### 5-DMN(Frontal)
# ### 6-Sensorimotor
# ### 7-Anterior Vision
# ### 8-RFP
# ### 9-Precuneus
# ### 10-DMN(Precuneus)
# ### 11-Executive control
# ### 12-Sensory
# ### 13-Cerebellar occipital

# In[81]:


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

import pandas as pd
from skimage import measure

from itertools import combinations
from itertools import permutations
import scipy.stats


# In[82]:


z_s_hc_rs_list=np.genfromtxt(r"/home/r085233/Paths_dc/z_scores_hc_rs.txt",dtype=str)
os.chdir("/home/r085233/masks")
comp=25
for k in range (comp):
    z_s_hc_rs_struct=nb.load(Path(z_s_hc_rs_list[k]))
    z_s_hc_rs=z_s_hc_rs_struct.get_fdata()


# In[83]:


#Upload the components of interest
mask_int=np.load('array_intersec_dice_coef.npy')
print(mask_int)


# In[84]:


networks=np.zeros((13,91,109,91))
networks[0]=mask_int[24]
networks[1]=mask_int[14]
networks[2]=mask_int[20]
networks[3]=mask_int[7]
networks[4]=mask_int[1]
networks[5]=mask_int[15]
networks[6]=mask_int[18]
networks[7]=mask_int[5]
networks[8]=mask_int[6]
networks[9]=mask_int[11]
networks[10]=mask_int[8]
networks[11]=mask_int[9]
networks[12]=mask_int[4]
print(np.shape(networks))


# In[85]:


def labels_net(thresh_c,component_c_struct,net):
    poslabel=measure.label(thresh_c)
    thresh_c_image_label=nb.Nifti1Image(poslabel,affine=component_c_struct.affine)
    nb.save(thresh_c_image_label,'mask_label'+net+'.nii.gz')
    size_label=np.shape(poslabel)
    l=np.max(poslabel)
    list_blobs=list(range(1,l+1))
    
    return(poslabel,size_label,l,list_blobs)


# In[86]:


net_inf_front, size_label_inf_front, l_inf_front, list_inf_front=labels_net(networks[0],z_s_hc_rs_struct,'_int_inf_front')
print(l_inf_front)


# In[87]:


net_LFP, size_label_LFP, l_LFP, list_LFP=labels_net(networks[3],z_s_hc_rs_struct,'int_LFP')
print(l_LFP)


# In[88]:


net_cer, size_label_cer, l_cer, list_cer=labels_net(networks[1],z_s_hc_rs_struct,'int_cer')


# In[89]:


net_DMN_1, size_label_DMN_1, l_DMN_1, list_DMN_1=labels_net(networks[4],z_s_hc_rs_struct,'int_DMN_1')


# In[90]:


net_RFP, size_label_RFP, l_RFP, list_RFP=labels_net(networks[7],z_s_hc_rs_struct,'int_RFP')


# In[91]:


net_motor, size_label_motor, l_motor, list_motor=labels_net(networks[2],z_s_hc_rs_struct,'int_motor')


# In[92]:


net_sen_motor, size_label_sen_motor, l_sen_motor, list_sen_motor=labels_net(networks[5],z_s_hc_rs_struct,'int_sen_motor')


# In[93]:


net_ant_vis, size_label_ant_vis, l_ant_vis, list_ant_vis=labels_net(networks[6],z_s_hc_rs_struct,'int_ant_vis')


# In[94]:


net_DMN_2, size_label_DMN_2, l_DMN_2, list_DMN_2=labels_net(networks[9],z_s_hc_rs_struct,'int_DMN_2')


# In[95]:


net_prec, size_label_prec, l_prec, list_prec=labels_net(networks[8],z_s_hc_rs_struct,'int_prec')


# In[96]:


net_ex_cont, size_label_ex_cont, l_ex_cont, list_ex_cont=labels_net(networks[10],z_s_hc_rs_struct,'int_ex_cont')


# In[97]:


net_cer_occ, size_label_cer_occ, l_cer_occ, list_cer_occ=labels_net(networks[12],z_s_hc_rs_struct,'int_cer_occ')


# In[98]:


net_sens, size_label_sens, l_sens, list_sens=labels_net(networks[11],z_s_hc_rs_struct,'int_sens')


# In[99]:


def mask_net (l,size_label,poslabel,component_c_struct,net):
    labels=list(range(1,l+1))
    mask_labels=np.zeros((len(labels),size_label[0],size_label[1],size_label[2]))
    for i in range(len(labels)):
        mask_labels[i,:,:,:]=poslabel==labels[i]
        new_image = nb.Nifti1Image(mask_labels[i],affine=component_c_struct.affine)
        nb.save(new_image, 'mask_trial'+net+'_'+str(labels[i])+'.nii.gz')
    return(mask_labels)


# In[100]:


mask_inf_front=mask_net(l_inf_front, size_label_inf_front, net_inf_front,z_s_hc_rs_struct,'_int_inf_front')
print(l_inf_front)


# In[101]:


mask_LFP=mask_net(l_LFP, size_label_LFP, net_LFP,z_s_hc_rs_struct,'_int_LFP')
print(l_LFP)


# In[102]:


mask_cer=mask_net(l_cer, size_label_cer, net_cer,z_s_hc_rs_struct,'_int_cer')
print(l_cer)


# In[103]:


mask_DMN_1=mask_net(l_DMN_1, size_label_DMN_1, net_DMN_1,z_s_hc_rs_struct,'_int_DMN_1')
print(l_DMN_1,)


# In[104]:


mask_RFP=mask_net(l_RFP, size_label_RFP, net_RFP,z_s_hc_rs_struct,'_int_RFP')
print(l_RFP)


# In[105]:


mask_motor=mask_net(l_motor, size_label_motor, net_motor,z_s_hc_rs_struct,'_int_motor')
print(l_motor)


# In[106]:


mask_ant_vis=mask_net(l_ant_vis, size_label_ant_vis, net_ant_vis,z_s_hc_rs_struct,'_int_ant_vis')
print(l_ant_vis)


# In[107]:


mask_DMN_2=mask_net(l_DMN_2, size_label_DMN_2, net_DMN_2,z_s_hc_rs_struct,'_int_DMN_2')
print(l_DMN_2)


# In[108]:


mask_prec=mask_net(l_prec, size_label_prec, net_prec,z_s_hc_rs_struct,'_int_prec')
print(l_prec)


# In[109]:


mask_ex_cont=mask_net(l_ex_cont, size_label_ex_cont, net_ex_cont,z_s_hc_rs_struct,'_int_ex_cont')
print(l_ex_cont)


# In[110]:


mask_cer_occ=mask_net(l_cer_occ, size_label_cer_occ, net_cer_occ,z_s_hc_rs_struct,'_int_cer_occ')
print(l_cer_occ)


# In[111]:


mask_sens=mask_net(l_sens, size_label_sens, net_sens,z_s_hc_rs_struct,'_int_sens')
print(l_sens)


# In[112]:


mask_sen_motor=mask_net(l_sen_motor, size_label_sen_motor, net_sen_motor,z_s_hc_rs_struct,'_int_sen_motor')
print(l_sen_motor)


# In[113]:


list_cont_rs=np.genfromtxt("/home/r085233/concat_paths/paths_concat_input_hc_rs.txt",dtype=str)
list_cont_pt=np.genfromtxt("/home/r085233/concat_paths/paths_concat_input_hc_pt.txt",dtype=str)

n=len(list_cont_rs)
print(n)


# In[ ]:


def time_series_extract(n_par,n_vol,mask_labels,blob,list_out):
    t_s_c=np.zeros([n_par,n_vol])
    for a in range (n_par):
        struct_f=nb.load(Path(list_out[a]))
        f=struct_f.get_fdata()
        t_series=np.zeros(len(f[0,0,0,:]))
        for t in range(len(f[0,0,0,:])):
            mult=np.multiply(mask_labels[blob-1,:,:,:],f[:,:,:,t])
            t_series[t]=np.sum(mult)/np.sum(mask_labels[blob-1,:,:,:])
        t_s_c[a]=t_series
    return(t_s_c)


# In[114]:


t_c=l_inf_front+l_LFP+l_cer+l_DMN_1+l_RFP+l_motor+l_ant_vis+l_DMN_2+l_prec+l_ex_cont+l_cer_occ+l_sens+l_sen_motor
print(t_c)


# In[115]:


matrix_t_s_controls_rs=np.zeros([t_c-8,n,196])
print(np.shape(matrix_t_s_controls_rs))


# In[ ]:


ts_DMN_1_1_rs=time_series_extract(n,196,mask_DMN_1,list_DMN_1[0],list_cont_rs)
matrix_t_s_controls_rs[0,:,:]=ts_DMN_1_1_rs


# In[ ]:


ts_DMN_1_2_rs=time_series_extract(n,196,mask_DMN_1,list_DMN_1[1],list_cont_rs)
matrix_t_s_controls_rs[1,:,:]=ts_DMN_1_2_rs


# In[ ]:


ts_DMN_1_3_rs=time_series_extract(n,196,mask_DMN_1,list_DMN_1[2],list_cont_rs)
matrix_t_s_controls_rs[2,:,:]=ts_DMN_1_3_rs


# In[ ]:


ts_DMN_1_4_rs=time_series_extract(n,196,mask_DMN_1,list_DMN_1[3],list_cont_rs)
matrix_t_s_controls_rs[3,:,:]=ts_DMN_1_4_rs


# In[ ]:


ts_DMN_2_1_rs=time_series_extract(n,196,mask_DMN_2,list_DMN_2[0],list_cont_rs)
matrix_t_s_controls_rs[4,:,:]=ts_DMN_2_1_rs


# In[ ]:


ts_DMN_2_2_rs=time_series_extract(n,196,mask_DMN_2,list_DMN_2[1],list_cont_rs)
matrix_t_s_controls_rs[5,:,:]=ts_DMN_2_2_rs


# In[ ]:


ts_DMN_2_3_rs=time_series_extract(n,196,mask_DMN_2,list_DMN_2[2],list_cont_rs)
matrix_t_s_controls_rs[6,:,:]=ts_DMN_2_3_rs


# In[ ]:


ts_LFP_2_rs=time_series_extract(n,196,mask_LFP,list_LFP[1],list_cont_rs)
matrix_t_s_controls_rs[7,:,:]=ts_LFP_2_rs


# In[ ]:


ts_LFP_3_rs=time_series_extract(n,196,mask_LFP,list_LFP[2],list_cont_rs)
matrix_t_s_controls_rs[8,:,:]=ts_LFP_3_rs


# In[ ]:


ts_LFP_4_rs=time_series_extract(n,196,mask_LFP,list_LFP[3],list_cont_rs)
matrix_t_s_controls_rs[9,:,:]=ts_LFP_4_rs


# In[ ]:


ts_LFP_5_rs=time_series_extract(n,196,mask_LFP,list_LFP[4],list_cont_rs)
matrix_t_s_controls_rs[10,:,:]=ts_LFP_5_rs


# In[ ]:


ts_LFP_6_rs=time_series_extract(n,196,mask_LFP,list_LFP[5],list_cont_rs)
matrix_t_s_controls_rs[11,:,:]=ts_LFP_6_rs


# In[ ]:


ts_LFP_7_rs=time_series_extract(n,196,mask_LFP,list_LFP[6],list_cont_rs)
matrix_t_s_controls_rs[12,:,:]=ts_LFP_7_rs


# In[ ]:


ts_RFP_1_rs=time_series_extract(n,196,mask_RFP,list_LFP[0],list_cont_rs)
matrix_t_s_controls_rs[13,:,:]=ts_RFP_1_rs


# In[ ]:


ts_RFP_2_rs=time_series_extract(n,196,mask_RFP,list_LFP[1],list_cont_rs)
matrix_t_s_controls_rs[14,:,:]=ts_RFP_2_rs


# In[ ]:


ts_RFP_3_rs=time_series_extract(n,196,mask_RFP,list_LFP[2],list_cont_rs)
matrix_t_s_controls_rs[15,:,:]=ts_RFP_3_rs


# In[ ]:


ts_RFP_6_rs=time_series_extract(n,196,mask_RFP,list_LFP[5],list_cont_rs)
matrix_t_s_controls_rs[16,:,:]=ts_RFP_6_rs


# In[ ]:


ts_ant_vis_1_rs=time_series_extract(n,196,mask_ant_vis,list_ant_vis[0],list_cont_rs)
matrix_t_s_controls_rs[17,:,:]=ts_ant_vis_1_rs


# In[ ]:


ts_cer_1_rs=time_series_extract(n,196,mask_cer,list_cer[0],list_cont_rs)
matrix_t_s_controls_rs[18,:,:]=ts_cer_1_rs


# In[ ]:


ts_ex_cont_1_rs=time_series_extract(n,196,mask_ex_cont,list_ex_cont[0],list_cont_rs)
matrix_t_s_controls_rs[19,:,:]=ts_ex_cont_1_rs


# In[ ]:


ts_ex_cont_2_rs=time_series_extract(n,196,mask_ex_cont,list_ex_cont[1],list_cont_rs)
matrix_t_s_controls_rs[20,:,:]=ts_ex_cont_2_rs


# In[ ]:


ts_ex_cont_3_rs=time_series_extract(n,196,mask_ex_cont,list_ex_cont[2],list_cont_rs)
matrix_t_s_controls_rs[21,:,:]=ts_ex_cont_3_rs


# In[ ]:


ts_ex_cont_5_rs=time_series_extract(n,196,mask_ex_cont,list_ex_cont[4],list_cont_rs)
matrix_t_s_controls_rs[22,:,:]=ts_ex_cont_5_rs


# In[ ]:


ts_ex_cont_6_rs=time_series_extract(n,196,mask_ex_cont,list_ex_cont[5],list_cont_rs)
matrix_t_s_controls_rs[23,:,:]=ts_ex_cont_6_rs


# In[ ]:


ts_inf_front_1_rs=time_series_extract(n,196,mask_inf_front,list_inf_front[0],list_cont_rs)
matrix_t_s_controls_rs[24,:,:]=ts_inf_front_1_rs


# In[ ]:


ts_motor_1_rs=time_series_extract(n,196,mask_motor,list_motor[0],list_cont_rs)
matrix_t_s_controls_rs[25,:,:]=ts_motor_1_rs


# In[ ]:


ts_motor_2_rs=time_series_extract(n,196,mask_motor,list_motor[1],list_cont_rs)
matrix_t_s_controls_rs[26,:,:]=ts_motor_2_rs


# In[ ]:


ts_motor_3_rs=time_series_extract(n,196,mask_motor,list_motor[2],list_cont_rs)
matrix_t_s_controls_rs[27,:,:]=ts_motor_3_rs


# In[ ]:


ts_motor_4_rs=time_series_extract(n,196,mask_motor,list_motor[3],list_cont_rs)
matrix_t_s_controls_rs[28,:,:]=ts_motor_4_rs


# In[ ]:


ts_prec_1_rs=time_series_extract(n,196,mask_prec,list_prec[0],list_cont_rs)
matrix_t_s_controls_rs[29,:,:]=ts_prec_1_rs


# In[ ]:


ts_prec_3_rs=time_series_extract(n,196,mask_prec,list_prec[2],list_cont_rs)
matrix_t_s_controls_rs[30,:,:]=ts_prec_3_rs


# In[ ]:


ts_prec_4_rs=time_series_extract(n,196,mask_prec,list_prec[3],list_cont_rs)
matrix_t_s_controls_rs[31,:,:]=ts_prec_4_rs


# In[ ]:


ts_prec_5_rs=time_series_extract(n,196,mask_prec,list_prec[4],list_cont_rs)
matrix_t_s_controls_rs[32,:,:]=ts_prec_5_rs


# In[ ]:


ts_prec_6_rs=time_series_extract(n,196,mask_prec,list_prec[5],list_cont_rs)
matrix_t_s_controls_rs[33,:,:]=ts_prec_6_rs


# In[ ]:


ts_prec_8_rs=time_series_extract(n,196,mask_prec,list_prec[7],list_cont_rs)
matrix_t_s_controls_rs[34,:,:]=ts_prec_8_rs


# In[ ]:


ts_prec_9_rs=time_series_extract(n,196,mask_prec,list_prec[8],list_cont_rs)
matrix_t_s_controls_rs[35,:,:]=ts_prec_9_rs


# In[ ]:


ts_sen_motor_1_rs=time_series_extract(n,196,mask_sen_motor,list_sen_motor[0],list_cont_rs)
matrix_t_s_controls_rs[36,:,:]=ts_sen_motor_1_rs


# In[ ]:


ts_sen_motor_2_rs=time_series_extract(n,196,mask_sen_motor,list_sen_motor[1],list_cont_rs)
matrix_t_s_controls_rs[37,:,:]=ts_sen_motor_2_rs


# In[ ]:


ts_sen_motor_3_rs=time_series_extract(n,196,mask_sen_motor,list_sen_motor[2],list_cont_rs)
matrix_t_s_controls_rs[38,:,:]=ts_sen_motor_3_rs


# In[ ]:


ts_sens_1_rs=time_series_extract(n,196,mask_sens,list_sens[0],list_cont_rs)
matrix_t_s_controls_rs[39,:,:]=ts_sens_1_rs


# In[ ]:


ts_sens_2_rs=time_series_extract(n,196,mask_sens,list_sens[1],list_cont_rs)
matrix_t_s_controls_rs[40,:,:]=ts_sens_2_rs


# In[ ]:


ts_sens_3_rs=time_series_extract(n,196,mask_sens,list_sens[2],list_cont_rs)
matrix_t_s_controls_rs[41,:,:]=ts_sens_3_rs


# In[ ]:


ts_cer_occ_1_rs=time_series_extract(n,196,mask_cer_occ,list_cer_occ[0],list_cont_rs)
matrix_t_s_controls_rs[42,:,:]=ts_cer_occ_1_rs


# In[ ]:


ts_cer_occ_2_rs=time_series_extract(n,196,mask_cer_occ,list_cer_occ[1],list_cont_rs)
matrix_t_s_controls_rs[43,:,:]=ts_cer_occ_2_rs


# In[ ]:


np.save("array_t_s_controls_rs.npy",matrix_t_s_controls_rs)


# In[116]:


n_pt=len(list_cont_pt)
matrix_t_s_controls_pt=np.zeros([t_c-8,n_pt,210])
print(np.shape(matrix_t_s_controls_pt))


# In[ ]:


ts_DMN_1_1_pt=time_series_extract(n,210,mask_DMN_1,list_DMN_1[0],list_cont_pt)
matrix_t_s_controls_pt[0,:,:]=ts_DMN_1_1_pt

ts_DMN_1_2_pt=time_series_extract(n,210,mask_DMN_1,list_DMN_1[1],list_cont_pt)
matrix_t_s_controls_pt[1,:,:]=ts_DMN_1_2_pt

ts_DMN_1_3_pt=time_series_extract(n,210,mask_DMN_1,list_DMN_1[2],list_cont_pt)
matrix_t_s_controls_pt[2,:,:]=ts_DMN_1_3_pt

ts_DMN_1_4_pt=time_series_extract(n,210,mask_DMN_1,list_DMN_1[3],list_cont_pt)
matrix_t_s_controls_pt[3,:,:]=ts_DMN_1_4_pt

ts_DMN_2_1_pt=time_series_extract(n,210,mask_DMN_2,list_DMN_2[0],list_cont_pt)
matrix_t_s_controls_pt[4,:,:]=ts_DMN_2_1_pt

ts_DMN_2_2_pt=time_series_extract(n,210,mask_DMN_2,list_DMN_2[1],list_cont_pt)
matrix_t_s_controls_pt[5,:,:]=ts_DMN_2_2_pt

ts_DMN_2_3_pt=time_series_extract(n,210,mask_DMN_2,list_DMN_2[2],list_cont_pt)
matrix_t_s_controls_pt[6,:,:]=ts_DMN_2_3_pt

ts_LFP_2_pt=time_series_extract(n,210,mask_LFP,list_LFP[1],list_cont_pt)
matrix_t_s_controls_pt[7,:,:]=ts_LFP_2_pt

ts_LFP_3_pt=time_series_extract(n,210,mask_LFP,list_LFP[2],list_cont_pt)
matrix_t_s_controls_pt[8,:,:]=ts_LFP_3_pt

ts_LFP_4_pt=time_series_extract(n,210,mask_LFP,list_LFP[3],list_cont_pt)
matrix_t_s_controls_pt[9,:,:]=ts_LFP_4_pt

ts_LFP_5_pt=time_series_extract(n,210,mask_LFP,list_LFP[4],list_cont_pt)
matrix_t_s_controls_pt[10,:,:]=ts_LFP_5_pt

ts_LFP_6_pt=time_series_extract(n,210,mask_LFP,list_LFP[5],list_cont_pt)
matrix_t_s_controls_pt[11,:,:]=ts_LFP_6_pt

ts_LFP_7_pt=time_series_extract(n,210,mask_LFP,list_LFP[6],list_cont_pt)
matrix_t_s_controls_pt[12,:,:]=ts_LFP_7_pt

ts_RFP_1_pt=time_series_extract(n,210,mask_RFP,list_LFP[0],list_cont_pt)
matrix_t_s_controls_pt[13,:,:]=ts_RFP_1_pt

ts_RFP_2_pt=time_series_extract(n,210,mask_RFP,list_LFP[1],list_cont_pt)
matrix_t_s_controls_pt[14,:,:]=ts_RFP_2_pt

ts_RFP_3_pt=time_series_extract(n,210,mask_RFP,list_LFP[2],list_cont_pt)
matrix_t_s_controls_pt[15,:,:]=ts_RFP_3_pt

ts_RFP_6_pt=time_series_extract(n,210,mask_RFP,list_LFP[5],list_cont_pt)
matrix_t_s_controls_pt[16,:,:]=ts_RFP_6_pt

ts_ant_vis_1_pt=time_series_extract(n,210,mask_ant_vis,list_ant_vis[0],list_cont_pt)
matrix_t_s_controls_pt[17,:,:]=ts_ant_vis_1_pt

ts_cer_1_pt=time_series_extract(n,210,mask_cer,list_cer[0],list_cont_pt)
matrix_t_s_controls_pt[18,:,:]=ts_cer_1_pt

ts_ex_cont_1_pt=time_series_extract(n,210,mask_ex_cont,list_ex_cont[0],list_cont_pt)
matrix_t_s_controls_pt[19,:,:]=ts_ex_cont_1_pt


ts_ex_cont_2_pt=time_series_extract(n,210,mask_ex_cont,list_ex_cont[1],list_cont_pt)
matrix_t_s_controls_pt[20,:,:]=ts_ex_cont_2_pt

ts_ex_cont_3_pt=time_series_extract(n,210,mask_ex_cont,list_ex_cont[2],list_cont_pt)
matrix_t_s_controls_pt[21,:,:]=ts_ex_cont_3_pt

ts_ex_cont_5_pt=time_series_extract(n,210,mask_ex_cont,list_ex_cont[4],list_cont_pt)
matrix_t_s_controls_pt[22,:,:]=ts_ex_cont_5_pt

ts_ex_cont_6_pt=time_series_extract(n,210,mask_ex_cont,list_ex_cont[5],list_cont_pt)
matrix_t_s_controls_pt[23,:,:]=ts_ex_cont_6_pt

ts_inf_front_1_pt=time_series_extract(n,210,mask_inf_front,list_inf_front[0],list_cont_pt)
matrix_t_s_controls_pt[24,:,:]=ts_inf_front_1_pt

ts_motor_1_pt=time_series_extract(n,210,mask_motor,list_motor[0],list_cont_pt)
matrix_t_s_controls_pt[25,:,:]=ts_motor_1_pt

ts_motor_2_pt=time_series_extract(n,210,mask_motor,list_motor[1],list_cont_pt)
matrix_t_s_controls_pt[26,:,:]=ts_motor_2_pt

ts_motor_3_pt=time_series_extract(n,210,mask_motor,list_motor[2],list_cont_pt)
matrix_t_s_controls_pt[27,:,:]=ts_motor_3_pt

ts_motor_4_pt=time_series_extract(n,210,mask_motor,list_motor[3],list_cont_pt)
matrix_t_s_controls_pt[28,:,:]=ts_motor_4_pt

ts_prec_1_pt=time_series_extract(n,210,mask_prec,list_prec[0],list_cont_pt)
matrix_t_s_controls_pt[29,:,:]=ts_prec_1_pt

ts_prec_3_pt=time_series_extract(n,210,mask_prec,list_prec[2],list_cont_pt)
matrix_t_s_controls_pt[30,:,:]=ts_prec_3_pt

ts_prec_4_pt=time_series_extract(n,210,mask_prec,list_prec[3],list_cont_pt)
matrix_t_s_controls_pt[31,:,:]=ts_prec_4_pt

ts_prec_5_pt=time_series_extract(n,210,mask_prec,list_prec[4],list_cont_pt)
matrix_t_s_controls_pt[32,:,:]=ts_prec_5_pt

ts_prec_6_pt=time_series_extract(n,210,mask_prec,list_prec[5],list_cont_pt)
matrix_t_s_controls_pt[33,:,:]=ts_prec_6_pt

ts_prec_8_pt=time_series_extract(n,210,mask_prec,list_prec[7],list_cont_pt)
matrix_t_s_controls_pt[34,:,:]=ts_prec_8_pt

ts_prec_9_pt=time_series_extract(n,210,mask_prec,list_prec[8],list_cont_pt)
matrix_t_s_controls_pt[35,:,:]=ts_prec_9_pt

ts_sen_motor_1_pt=time_series_extract(n,210,mask_sen_motor,list_sen_motor[0],list_cont_pt)
matrix_t_s_controls_pt[36,:,:]=ts_sen_motor_1_pt

ts_sen_motor_2_pt=time_series_extract(n,210,mask_sen_motor,list_sen_motor[1],list_cont_pt)
matrix_t_s_controls_pt[37,:,:]=ts_sen_motor_2_pt

ts_sen_motor_3_pt=time_series_extract(n,210,mask_sen_motor,list_sen_motor[2],list_cont_pt)
matrix_t_s_controls_pt[38,:,:]=ts_sen_motor_3_pt

ts_sens_1_pt=time_series_extract(n,210,mask_sens,list_sens[0],list_cont_pt)
matrix_t_s_controls_pt[39,:,:]=ts_sens_1_pt

ts_sens_2_pt=time_series_extract(n,210,mask_sens,list_sens[1],list_cont_pt)
matrix_t_s_controls_pt[40,:,:]=ts_sens_2_pt

ts_sens_3_pt=time_series_extract(n,210,mask_sens,list_sens[2],list_cont_pt)
matrix_t_s_controls_pt[41,:,:]=ts_sens_3_pt

ts_cer_occ_1_pt=time_series_extract(n,210,mask_cer_occ,list_cer_occ[0],list_cont_pt)
matrix_t_s_controls_pt[42,:,:]=ts_cer_occ_1_pt

ts_cer_occ_2_pt=time_series_extract(n,210,mask_cer_occ,list_cer_occ[1],list_cont_pt)
matrix_t_s_controls_pt[43,:,:]=ts_cer_occ_2_pt


# In[ ]:


np.save("array_t_s_controls_pt.npy",matrix_t_s_controls_pt)

