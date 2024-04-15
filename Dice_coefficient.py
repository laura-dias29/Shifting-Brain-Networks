#!/usr/bin/env python
# coding: utf-8

# # Dice coefficient - Name Networks

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
import math
import operator


# In[2]:


def dice_coef2(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union==0: return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union


# In[3]:


GenR25_MetaICA_2mm=nb.load("/home/r085233/metaICA500/metaICA500/GenR130/GenR25_MetaICA_2mm.nii.gz")
genr=GenR25_MetaICA_2mm.get_fdata()


# In[4]:


z_s_rs_list=np.genfromtxt(r"/home/r085233/Paths_dc/z_scores_rs.txt",dtype=str)
z_s_hc_rs_list=np.genfromtxt(r"/home/r085233/Paths_dc/z_scores_hc_rs.txt",dtype=str)
z_s_hc_pt_list=np.genfromtxt(r"/home/r085233/Paths_dc/z_scores_hc_pt.txt",dtype=str)


# In[5]:


def thresh_z (c,component_c_struct,num_c,t,state):
    thresh_c=np.zeros(np.shape(c))

    thresh_c[:,:,:]=c>=t

    thresh_c_image=nb.Nifti1Image(thresh_c,affine=component_c_struct.affine)
    nb.save(thresh_c_image,'mask_'+str(num_c)+state+'.nii.gz')
    
    return(thresh_c)


# In[6]:


os.chdir("/home/r085233/masks")
comp=25

thresh_z_s_hc_rs=np.zeros([comp,91,109,91])
thresh_z_s_hc_pt=np.zeros([comp,91,109,91])
thresh_z_s_rs=np.zeros([comp,91,109,91])

for k in range (comp):
    z_s_hc_rs_struct=nb.load(Path(z_s_hc_rs_list[k]))
    z_s_hc_rs=z_s_hc_rs_struct.get_fdata()
    
    z_s_hc_pt_struct=nb.load(Path(z_s_hc_pt_list[k]))
    z_s_hc_pt=z_s_hc_pt_struct.get_fdata()
    
    z_s_rs_struct=nb.load(Path(z_s_rs_list[k]))
    z_s_rs=z_s_rs_struct.get_fdata()
    
    
    thresh_z_s_hc_rs[k]=thresh_z(z_s_hc_rs,z_s_hc_rs_struct,k+1,5,'_hc_rs')
    thresh_z_s_hc_pt[k]=thresh_z(z_s_hc_pt,z_s_hc_pt_struct,k+1,5,'_hc_pt')
    thresh_z_s_rs[k]=thresh_z(z_s_rs,z_s_rs_struct,k+1,5,'rs')


# In[7]:


np.shape(genr[:,:,:,1])


# In[8]:



thresh_comp_genr=np.zeros([comp,91,109,91])
for k in range (comp):
    thresh_comp_genr[k]=thresh_z(genr[:,:,:,k],GenR25_MetaICA_2mm,k+1,3.09,'genr')


# In[9]:


dice_coef=np.zeros(comp)
index=np.zeros((comp,2))


for k in range (comp):
    dice_coef[k]=(dice_coef2(thresh_comp_genr[k],thresh_z_s_rs[0]))
    index[k,:]=[k,0]
    for n in range(comp):
        if dice_coef[k]<dice_coef2(thresh_comp_genr[k],thresh_z_s_rs[n]):
            dice_coef[k]=dice_coef2(thresh_comp_genr[k],thresh_z_s_rs[n])
            index[k,:]=[k,n]
            #print(k,n,dice_coef[k])
            
    
                
print(index)    


# In[10]:


dice_coef_hc_rs=np.zeros(comp)
index_hc_rs=np.zeros((comp,2))


for k in range (comp):
    dice_coef_hc_rs[k]=(dice_coef2(thresh_z_s_hc_rs[k],thresh_comp_genr[0]))
    index_hc_rs[k,:]=[k,0]
    for n in range(comp):
        if dice_coef_hc_rs[k]<dice_coef2(thresh_z_s_hc_rs[k],thresh_comp_genr[n]):
            dice_coef_hc_rs[k]=dice_coef2(thresh_z_s_hc_rs[k],thresh_comp_genr[n])
            index_hc_rs[k,:]=[k,n]
            #print(k,n,dice_coef[k])
            
    
                
print(index_hc_rs)    


# In[11]:


enumerate_object = enumerate(dice_coef_hc_rs)
sorted_pairs = sorted(enumerate_object, key=operator.itemgetter(1))
sorted_indices = [index for index, element in sorted_pairs]
print(sorted_indices)


# In[12]:


np.sort(dice_coef_hc_rs)


# In[13]:


dice_coef_hc_pt=np.zeros(comp)
index_hc_pt=np.zeros((comp,2))


for k in range (comp):
    dice_coef_hc_pt[k]=(dice_coef2(thresh_comp_genr[k],thresh_z_s_hc_pt[0]))
    index_hc_pt[k,:]=[k,0]
    for n in range(comp):
        if dice_coef_hc_pt[k]<dice_coef2(thresh_comp_genr[k],thresh_z_s_hc_pt[n]):
            dice_coef_hc_pt[k]=dice_coef2(thresh_comp_genr[k],thresh_z_s_hc_pt[n])
            index_hc_pt[k,:]=[k,n]
            #print(k,n,dice_coef[k])
            
    
                
print(index_hc_pt)    


# In[14]:


dice_coef_hc_rs_pt=np.zeros(comp)
index_hc_rs_pt=np.zeros((comp,2))


for k in range (comp):
    dice_coef_hc_rs_pt[k]=(dice_coef2(thresh_z_s_hc_rs[k],thresh_z_s_hc_pt[0]))
    index_hc_rs_pt[k,:]=[k,0]
    for n in range(comp):
        if dice_coef_hc_rs_pt[k]<dice_coef2(thresh_z_s_hc_rs[k],thresh_z_s_hc_pt[n]):
            dice_coef_hc_rs_pt[k]=dice_coef2(thresh_z_s_hc_rs[k],thresh_z_s_hc_pt[n])
            index_hc_rs_pt[k,:]=[k,n]
            #print(k,n,dice_coef[k])
            
    
                
print(index_hc_rs_pt)


# In[15]:


enumerate_object = enumerate(dice_coef_hc_rs_pt)
sorted_pairs = sorted(enumerate_object, key=operator.itemgetter(1))
sorted_indices = [index for index, element in sorted_pairs]
print(sorted_indices)


# In[16]:


ord=np.sort(dice_coef_hc_rs_pt)
print(ord)


# In[17]:


#### multiply matched images
mask_int=np.zeros(np.shape(thresh_z_s_hc_rs))
s=sorted_indices[::-1]
for i in range(len(s)):
    pt=int(index_hc_rs_pt[i,1])
    mult=np.multiply(thresh_z_s_hc_rs[i,:,:,:],thresh_z_s_hc_pt[pt,:,:,:])
    mask_int[i,:,:,:]=mult

np.save('array_intersec_dice_coef.npy',mask_int)
    

print(np.shape(mask_int))


# In[ ]:





# In[ ]:





# In[ ]:




