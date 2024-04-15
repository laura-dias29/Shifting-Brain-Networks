#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# # Resting State

# # Labels
# 
# 
# ### 1-DMN
# ### 8-DMN
# ### 3-right frontoparietal
# ### 6-left frontoparietal
# ### 23-insula

# ## Load masks according to D.C.

# In[2]:


#Upload the components of interest

DMN_1_struct=nb.load('/home/r085233/out_melodic_rs.gica/groupmelodic.ica/stats/thresh_zstat1.nii.gz')
DMN_1=DMN_1_struct.get_fdata()

DMN_2_struct=nb.load('/home/r085233/out_melodic_rs.gica/groupmelodic.ica/stats/thresh_zstat8.nii.gz')
DMN_2=DMN_2_struct.get_fdata()

RFP_struct=nb.load('/home/r085233/out_melodic_rs.gica/groupmelodic.ica/stats/thresh_zstat3.nii.gz')
RFP=RFP_struct.get_fdata()

LFP_struct=nb.load('/home/r085233/out_melodic_rs.gica/groupmelodic.ica/stats/thresh_zstat6.nii.gz')
LFP=LFP_struct.get_fdata()
                      
insular_struct=nb.load('/home/r085233/out_melodic_rs.gica/groupmelodic.ica/stats/thresh_zstat23.nii.gz')
insular=insular_struct.get_fdata()


# In[3]:



def thresh_c (c,component_c_struct,t):
    thresh_c=np.zeros(np.shape(c))

    thresh_c[:,:,:]=c>=t

    thresh_c_image=nb.Nifti1Image(thresh_c,affine=component_c_struct.affine)
    
    
    return(thresh_c)


# In[4]:


thresh_DMN_1=thresh_c(DMN_1,DMN_1_struct,5)


# In[5]:


thresh_DMN_2=thresh_c(DMN_2,DMN_2_struct,5)


# In[6]:


thresh_RFP=thresh_c(RFP,RFP_struct,5)


# In[7]:


thresh_LFP=thresh_c(LFP,LFP_struct,5)


# In[8]:


thresh_insular=thresh_c(insular,insular_struct,5)


# In[9]:


def labels(thresh_c,component_c_struct,net):
    poslabel=measure.label(thresh_c)
    thresh_c_image_label=nb.Nifti1Image(poslabel,affine=component_c_struct.affine)
    nb.save(thresh_c_image_label,'mask_label'+net+'.nii.gz')
    size_label=np.shape(poslabel)
    l=np.max(poslabel)
    list_blobs=list(range(1,l+1))
    
    return(poslabel,size_label,l,list_blobs)
    


# In[10]:


os.chdir('/home/r085233/masks')


# In[11]:


poslabel_DMN_1, size_label_DMN_1, l_DMN_1, list_DMN_1=labels(thresh_DMN_1,DMN_1_struct,'_DMN_1')
print(l_DMN_1)


# In[12]:


poslabel_DMN_2, size_label_DMN_2, l_DMN_2, list_DMN_2=labels(thresh_DMN_2,DMN_2_struct,'_DMN_2')
print(l_DMN_2)


# In[13]:


poslabel_RFP, size_label_RFP, l_RFP, list_RFP=labels(thresh_RFP,RFP_struct,'_RFP')
print(l_RFP)


# In[14]:


poslabel_LFP, size_label_LFP, l_LFP, list_LFP=labels(thresh_LFP,LFP_struct,'_LFP')
print(l_LFP)


# In[15]:


poslabel_insular, size_label_insular, l_insular, list_insular=labels(thresh_insular,insular_struct,'_insular')
print(l_insular)


# In[16]:


def mask (l,size_label,poslabel,component_c_struct,NET):
    labels=list(range(1,l+1))
    mask_labels=np.zeros((len(labels),size_label[0],size_label[1],size_label[2]))
    for i in range(len(labels)):
        mask_labels[i,:,:,:]=poslabel==labels[i]
        new_image = nb.Nifti1Image(mask_labels[i],affine=component_c_struct.affine)
        nb.save(new_image, 'mask_trial'+NET+'_'+str(labels[i])+'.nii.gz')
    return(mask_labels)


# In[17]:


mask_labels_DMN_1=mask(l_DMN_1,size_label_DMN_1,poslabel_DMN_1,DMN_1_struct,'_DMN_1')


# In[18]:


mask_labels_DMN_2=mask(l_DMN_2,size_label_DMN_2,poslabel_DMN_2,DMN_2_struct,'_DMN_2')


# In[19]:


mask_labels_RFP=mask(l_RFP,size_label_RFP,poslabel_RFP,RFP_struct,'_RFP')


# In[20]:


mask_labels_LFP=mask(l_LFP,size_label_LFP,poslabel_LFP,LFP_struct,'_LFP')


# In[21]:


mask_labels_insular=mask(l_insular,size_label_insular,poslabel_insular,insular_struct,'_insular')


# In[ ]:





# In[22]:


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


# In[23]:


list_rs=np.genfromtxt(r"/home/r085233/paths_rs.txt",dtype=str)


n_rs=len(list_rs)
n_regions=l_DMN_1+l_DMN_2+l_RFP+l_LFP+l_insular
print(n_rs)


# In[24]:


matrix_t_s=np.zeros([n_regions-4,n_rs,196])
print(np.shape(matrix_t_s))


# In[ ]:


ts_DMN_1_1=time_series_extract(n_rs,196,mask_labels_DMN_1,list_DMN_1[0],list_rs)
matrix_t_s[0,:,:]=ts_DMN_1_1


# In[ ]:


ts_DMN_1_2=time_series_extract(n_rs,196,mask_labels_DMN_1,list_DMN_1[1],list_rs)
matrix_t_s[1,:,:]=ts_DMN_1_2


# In[ ]:


ts_DMN_1_3=time_series_extract(n_rs,196,mask_labels_DMN_1,list_DMN_1[2],list_rs)
matrix_t_s[2,:,:]=ts_DMN_1_3


# In[ ]:


ts_DMN_1_4=time_series_extract(n_rs,196,mask_labels_DMN_1,list_DMN_1[3],list_rs)
matrix_t_s[3,:,:]=ts_DMN_1_4


# In[ ]:


ts_DMN_1_5=time_series_extract(n_rs,196,mask_labels_DMN_1,list_DMN_1[4],list_rs)
matrix_t_s[4,:,:]=ts_DMN_1_5


# In[ ]:


ts_DMN_1_6=time_series_extract(n_rs,196,mask_labels_DMN_1,list_DMN_1[5],list_rs)
matrix_t_s[5,:,:]=ts_DMN_1_6


# In[ ]:


ts_DMN_1_7=time_series_extract(n_rs,196,mask_labels_DMN_1,list_DMN_1[6],list_rs)
matrix_t_s[6,:,:]=ts_DMN_1_7


# In[ ]:


ts_DMN_2_1=time_series_extract(n_rs,196,mask_labels_DMN_2,list_DMN_2[0],list_rs)
matrix_t_s[7,:,:]=ts_DMN_2_1


# In[ ]:


ts_DMN_2_2=time_series_extract(n_rs,196,mask_labels_DMN_2,list_DMN_2[1],list_rs)
matrix_t_s[8,:,:]=ts_DMN_2_2


# In[ ]:


ts_DMN_2_3=time_series_extract(n_rs,196,mask_labels_DMN_2,list_DMN_2[2],list_rs)
matrix_t_s[9,:,:]=ts_DMN_2_3


# In[ ]:


ts_LFP_1=time_series_extract(n_rs,196,mask_labels_LFP,list_LFP[0],list_rs)
matrix_t_s[10,:,:]=ts_LFP_1


# In[ ]:


ts_LFP_2=time_series_extract(n_rs,196,mask_labels_LFP,list_LFP[1],list_rs)
matrix_t_s[11,:,:]=ts_LFP_2


# In[ ]:


ts_LFP_3=time_series_extract(n_rs,196,mask_labels_LFP,list_LFP[2],list_rs)
matrix_t_s[12,:,:]=ts_LFP_3


# In[ ]:


ts_LFP_4=time_series_extract(n_rs,196,mask_labels_LFP,list_LFP[3],list_rs)
matrix_t_s[13,:,:]=ts_LFP_4


# In[ ]:


ts_LFP_5=time_series_extract(n_rs,196,mask_labels_LFP,list_LFP[4],list_rs)
matrix_t_s[14,:,:]=ts_LFP_5


# In[ ]:


ts_LFP_6=time_series_extract(n_rs,196,mask_labels_LFP,list_LFP[5],list_rs)
matrix_t_s[15,:,:]=ts_LFP_6


# In[ ]:


ts_RFP_1=time_series_extract(n_rs,196,mask_labels_RFP,list_RFP[0],list_rs)
matrix_t_s[16,:,:]=ts_RFP_1


# In[ ]:


ts_RFP_2=time_series_extract(n_rs,196,mask_labels_RFP,list_RFP[1],list_rs)
matrix_t_s[17,:,:]=ts_RFP_2


# In[ ]:


ts_RFP_3=time_series_extract(n_rs,196,mask_labels_RFP,list_RFP[2],list_rs)
matrix_t_s[18,:,:]=ts_RFP_3


# In[ ]:


ts_RFP_5=time_series_extract(n_rs,196,mask_labels_RFP,list_RFP[4],list_rs)
matrix_t_s[19,:,:]=ts_RFP_5


# In[ ]:


ts_RFP_7=time_series_extract(n_rs,196,mask_labels_RFP,list_RFP[6],list_rs)
matrix_t_s[20,:,:]=ts_RFP_7


# In[ ]:


ts_insular_1=time_series_extract(n_rs,196,mask_labels_insular,list_insular[0],list_rs)
matrix_t_s[21,:,:]=ts_insular_1


# In[ ]:


ts_insular_3=time_series_extract(n_rs,196,mask_labels_insular,list_insular[2],list_rs)
matrix_t_s[22,:,:]=ts_insular_3


# In[ ]:


ts_insular_4=time_series_extract(n_rs,196,mask_labels_insular,list_insular[3],list_rs)
matrix_t_s[23,:,:]=ts_insular_4


# In[ ]:


ts_insular_5=time_series_extract(n_rs,196,mask_labels_insular,list_insular[4],list_rs)
matrix_t_s[24,:,:]=ts_insular_5


# In[ ]:


ts_insular_6=time_series_extract(n_rs,196,mask_labels_insular,list_insular[5],list_rs)
matrix_t_s[25,:,:]=ts_insular_6


# In[ ]:





# In[ ]:


np.save("array_t_s_rs.npy",matrix_t_s)

