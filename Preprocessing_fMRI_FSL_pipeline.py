#!/usr/bin/env python
# coding: utf-8

# # Preprocessing Workflow
# 
# ### 1.Denoising(single session ICA)
# ### 2. Time shifting 
# ### 3. Motion correction
# ### 4. Brain extraction
# ### 5. Image registration
# ### 6. See what images to remove due to motion




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





'Cretate paths for the text files where the paths for all the resting state, anatomy, and passive task text files'
'Cretate paths for the text files where the paths for all the output text files'
'Create a function for each of the preprocessing steps'
'Loop over all the paths and perform the steps for each participant'



'Create paths'


den_input_list=np.genfromtxt(r'/denoised_inputs_rs.txt',dtype=str)
out_list_ts=np.genfromtxt(r"/out_ts_rs.txt",dtype=str)
out_list_tsmc=np.genfromtxt(r"/out_ts_mcf_rs.txt",dtype=str)
out_list_tsmc_bet=np.genfromtxt(r"/out_ts_mcf_bet_rs.txt",dtype=str)
out_list_tsmc_bet_flirt=np.genfromtxt(r"/out_ts_mcf_bet_flirt_rs.txt",dtype=str)
anat_bet_list=np.genfromtxt(r"/anat_bet.txt",dtype=str)
A_B_list=np.genfromtxt(r"/mat_A_B.txt",dtype=str)
B_C_list=np.genfromtxt(r"/mat_B_C.txt",dtype=str)
A_C_list=np.genfromtxt(r"/mat_A_C.txt",dtype=str)



den_input_list_pt=np.genfromtxt(r'/denoised_inputs_pt.txt',dtype=str)
out_list_ts_pt=np.genfromtxt(r"/out_ts_pt.txt",dtype=str)
out_list_tsmc_pt=np.genfromtxt(r"/out_ts_mcf_pt.txt",dtype=str)
out_list_tsmc_bet_pt=np.genfromtxt(r"/out_ts_mcf_bet_pt.txt",dtype=str)
out_list_tsmc_bet_flirt_pt=np.genfromtxt(r"/out_ts_mcf_bet_flirt_pt.txt",dtype=str)
A_B_list_pt=np.genfromtxt(r"/mat_A_B_pt.txt",dtype=str)
B_C_list_pt=np.genfromtxt(r"/mat_B_C_pt.txt",dtype=str)
A_C_list_pt=np.genfromtxt(r"/mat_A_C_pt.txt",dtype=str)


n = len(den_input_list)
n_pt = len(den_input_list_pt)




#Try to do Slicetimer
def ts (den_input_list,out_list_ts,n):
    for j in range(n):
        inp=den_input_list[j]
        pinp=Path(inp)
        out_ts=out_list_ts[j]
        pout_ts=Path(out_ts)
        st = fsl.SliceTimer()
        st.inputs.in_file = pinp
        st.inputs.out_file = pout_ts
        st.inputs.interleaved = True
        result = st.run() 
    return(result)




#Try motion correction with mcflirt
def mc (out_list_ts,out_list_tsmc,n):
    for k in range(n):
        out_ts=out_list_ts[k]
        pout_ts=Path(out_ts)
        out_tsmc=out_list_tsmc[k]
        pout_tsmc=Path(out_tsmc)
        mcflt = fsl.MCFLIRT(in_file=pout_ts,cost='normcorr',out_file=pout_tsmc,save_plots=True)#can I define the output file?
        res = mcflt.run() 
    return (res)



#Try to do BET
def bet (out_list_tsmc,out_list_tsmc_bet,n):
    for i in range(n):
        out_tsmc=out_list_tsmc[i]
        pout_tsmc=Path(out_tsmc)
        out_tsmc_bet=out_list_tsmc_bet[i]
        pout=Path(out_tsmc_bet)
        res = os.system('/mnt/appl/tools/fsl/6.0.3/bin/bet '+str(out_tsmc)+' '+str(out_tsmc_bet)+' -F -f 0.5 -g 0 ')
    return (res)





#Try to use FLIRT
def flirt (out_list_tsmc_bet,out_list_tsmc_bet_flirt,anat_bet_list,A_B_list,B_C_list,A_C_list,n):
    for a in range (n):
        out_tsmc_bet=out_list_tsmc_bet[a]
        anat=anat_bet_list[a]
        out_tsmc_bet_flirt=out_list_tsmc_bet_flirt[a]
        l_A_B=A_B_list[a]
        l_B_C=B_C_list[a]
        l_A_C=A_C_list[a]
        res=os.system('flirt -in '+str(out_tsmc_bet)+' -ref '+str(anat)+' -omat '+str(l_A_B)+' ; flirt -in '+str(anat)+' -ref /mnt/data/software/tools/fsl/6.0.0/data/standard/MNI152_T1_2mm_brain.nii.gz -omat '+str(l_B_C)+' ; convert_xfm -omat '+str(l_A_C)+' -concat '+str(l_B_C)+" "+ str(l_A_B)+' ; flirt -in '+str(out_tsmc_bet)+' -ref /mnt/data/software/tools/fsl/6.0.0/data/standard/MNI152_T1_2mm_brain.nii.gz -out '+str(out_tsmc_bet_flirt)+' -applyxfm -init '+str(l_A_C))
        print(anat)
        print(out_tsmc_bet_flirt)
    return (res)





#ts(den_input_list,out_list_ts,n)





#ts(den_input_list_pt,out_list_ts_pt,n_pt)





#mc(out_list_ts,out_list_tsmc,n)




#mc(out_list_ts_pt,out_list_tsmc_pt,n_pt)




#bet(out_list_tsmc,out_list_tsmc_bet,n)





bet(out_list_tsmc_pt,out_list_tsmc_bet_pt,n_pt)




flirt(out_list_tsmc_bet,out_list_tsmc_bet_flirt,anat_bet_list,A_B_list,B_C_list,A_C_list,n)





flirt(out_list_tsmc_bet_pt,out_list_tsmc_bet_flirt_pt,anat_bet_list,A_B_list_pt,B_C_list_pt,A_C_list_pt,n_pt)


# # Exclude Motion outliers after preprocessing




paths_motion_list=np.genfromtxt(r"/denoised_rs.txt",dtype=str)
m_l=len(paths_motion_list)
id_num=np.zeros(m_l)

paths_motion_list_pt=np.genfromtxt(r"/denoised_pt.txt",dtype=str)
m_l_pt=len(paths_motion_list_pt)
id_num_pt=np.zeros(m_l_pt)




path_motion_list_hc_rs=np.genfromtxt(r"/denoised_hc_rs.txt",dtype=str)
path_motion_list_hc_pt=np.genfromtxt(r"/denoised_hc_pt.txt",dtype=str)
m_l_hc=len(path_motion_list_hc_rs)


for k in range(m_l):
    id_num[k]=(paths_motion_list[k][93:96])



for k in range(m_l_pt):
    id_num_pt[k]=(paths_motion_list_pt[k][96:99])



id_num_m_c=np.zeros(m_l_hc)

for k in range(m_l_hc):
    id_num_m_c[k]=(path_motion_list_hc_rs[k][93:96])

    

print(id_num_m)






def motion_out(num_vol,len_list,path,id_,state):
    
    t_2_t=np.zeros(num_vol-1)
    final_frame=np.zeros([len_list,4])
    header_list = ['rotx','roty','rotz','transx','transy','transz']
    for i in range(len_list):
        for j in range(num_vol):
            path_par=Path(path[i])
            file = pd.read_csv(path_par, header=None, delimiter = "  ")
            df = pd.DataFrame(file,index=range(0,num_vol-1))
            df.columns=['rotx','roty','rotz','transx','transy','transz']
            max_x=df['transx'].max()
            max_y=df['transy'].max()
            max_z=df['transz'].max()
            min_x=df['transx'].min()
            min_y=df['transy'].min()
            min_z=df['transz'].min()
            max_mov=np.sqrt((max_x-min_x)**2+(max_y-min_y)**2+(max_z-min_z)**2)
            if j < (num_vol-1):
                t_2_t[j]=np.sqrt((df['transx'].values[i+1]-df['transx'].values[i])**2 + (df['transy'].values[i+1]-df['transy'].values[i])**2 + (df['transz'].values[i+1]-df['transz'].values[i])**2)
                mean=np.mean(t_2_t)
                std=np.std(t_2_t)

            final_frame[i,0]=id_[i]
            final_frame[i,1]=max_mov
            final_frame[i,2]=mean
            final_frame[i,3]=std
           
        

    final=pd.DataFrame(final_frame)
    final.columns=['Participants','Maximum_movement','Mean_movement','std']
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

    final.reset_index().plot(x ='index', y='Maximum_movement', kind = 'scatter', ax=axes[0], ylim=[0,4])
    final.reset_index().plot(x ='index', y='Mean_movement', kind = 'scatter',ax=axes[1],ylim=[-0.01,0.5])       
    plt.savefig('graph_scatter'+str(state)+'.png')
    
    return(final)





motion_out(196,m_l,paths_motion_list,id_num,'motion_rs_denoised')




#motion_out(210,m_l_pt,paths_motion_list_pt,id_num_pt,'motion_pt_denoised')





#motion_out(196,m_l_hc,path_motion_list_hc_rs,id_num_m_c,'motion_rs_denoised_hc')





#motion_out(210,m_l_hc,path_motion_list_hc_pt,id_num_m_c,'motion_pt_denoised_hc')







