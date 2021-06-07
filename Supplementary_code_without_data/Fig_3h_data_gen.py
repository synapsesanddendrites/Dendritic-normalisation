#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 22:51:59 2021

@author: Alex
"""
n_epoch=100
n_rep=1000000

joint_list=[]
net_sp=network_sp.Network([784,100,10],0.2,0.15)
W=net_sp.weights

n_range=[100,200]
w_range=[0.6,1]
mids=np.linspace(0.7,0.9,50)
# L0,L1,L2, L1L2, L0L1, L0L2
grids=[]

for ward in range(6):
    grids.append(np.zeros([50,50]))
counts=np.zeros([50,50])

for ward in range(n_rep):
    if ward%1000==0:
        print(ward)
    net_sp=network_sp.Network([784,100,10],0.2,0.15)
    W=net_sp.weights
    n_aff=np.count_nonzero(W[0],1)
    n_sum=np.sum(abs(W[0]),1)
    mean_weights=n_sum/n_aff
    
    ok1=n_aff>=125
    ok2=n_aff<175
    ok3=mean_weights>0.7
    ok4=mean_weights<0.9
    inds=ok1*ok2*ok3*ok4
    
    n_aff=n_aff[inds]
    mean_weights=mean_weights[inds]
    weight_sums=n_sum[inds]
    weight_sq_sums=np.sum(W[0]**2,1)[inds]
    
    naff_locs=n_aff-125
    wght_locs=np.zeros(len(naff_locs))
    for bear in range(len(naff_locs)):
        wght_locs[bear]=int(np.argmin(abs(mean_weights[bear]-mids)))
    wght_locs=wght_locs.astype(int)
        
    L0_vals=np.zeros(len(naff_locs))
    L1_vals=np.zeros(len(naff_locs))
    L2_vals=np.zeros(len(naff_locs))
    L12_vals=np.zeros(len(naff_locs))
    L01_vals=np.zeros(len(naff_locs))
    L02_vals=np.zeros(len(naff_locs))
    
    for bear in range(len(naff_locs)):
        L0_vals[bear]=n_aff[bear]
        L1_vals[bear]=weight_sums[bear]
        L2_vals[bear]=np.sqrt(weight_sq_sums[bear])
        
        L12_vals[bear]=L1_vals[bear]*L2_vals[bear]
        L01_vals[bear]=L0_vals[bear]*L1_vals[bear]
        L02_vals[bear]=L0_vals[bear]*L2_vals[bear]
    
    for bear in range(len(naff_locs)):
        counts[naff_locs[bear],wght_locs[bear]]+=1
    
        L0_grid=grids[0]
        L0_grid[naff_locs[bear],wght_locs[bear]]=L0_grid[naff_locs[bear],wght_locs[bear]]+(L0_vals[bear]-L0_grid[naff_locs[bear],wght_locs[bear]])/counts[naff_locs[bear],wght_locs[bear]]
        grids[0]=L0_grid
    
        L1_grid=grids[1]
        L1_grid[naff_locs[bear],wght_locs[bear]]=L1_grid[naff_locs[bear],wght_locs[bear]]+(L1_vals[bear]-L1_grid[naff_locs[bear],wght_locs[bear]])/counts[naff_locs[bear],wght_locs[bear]]
        grids[1]=L1_grid
        
        L2_grid=grids[2]
        L2_grid[naff_locs[bear],wght_locs[bear]]=L2_grid[naff_locs[bear],wght_locs[bear]]+(L2_vals[bear]-L2_grid[naff_locs[bear],wght_locs[bear]])/counts[naff_locs[bear],wght_locs[bear]]
        grids[2]=L2_grid
        
        L12_grid=grids[3]
        L12_grid[naff_locs[bear],wght_locs[bear]]=L12_grid[naff_locs[bear],wght_locs[bear]]+(L12_vals[bear]-L12_grid[naff_locs[bear],wght_locs[bear]])/counts[naff_locs[bear],wght_locs[bear]]
        grids[3]=L12_grid
        
        L01_grid=grids[4]
        L01_grid[naff_locs[bear],wght_locs[bear]]=L01_grid[naff_locs[bear],wght_locs[bear]]+(L01_vals[bear]-L01_grid[naff_locs[bear],wght_locs[bear]])/counts[naff_locs[bear],wght_locs[bear]]
        grids[4]=L01_grid
        
        L02_grid=grids[5]
        L02_grid[naff_locs[bear],wght_locs[bear]]=L02_grid[naff_locs[bear],wght_locs[bear]]+(L02_vals[bear]-L02_grid[naff_locs[bear],wght_locs[bear]])/counts[naff_locs[bear],wght_locs[bear]]
        grids[5]=L02_grid
#%% 
import pickle
with open('sim_data/Fig_3_h_grids.pkl','wb') as f:
    pickle.dump(grids,f)
    