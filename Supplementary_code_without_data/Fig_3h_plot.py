#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:29:04 2021

@author: Alex
"""
grids = pickle.load( open('sim_data/Fig_3_h_grids.pkl', "rb" ) )
res=10

grads=[]
for ward in grids:
    grads.append(1/ward.T)

max_grad=0
for grad in grads:
    i_max=np.max(grad)
    
    if i_max>max_grad:
        max_grad=i_max

normed_grads=[]
for grad in grads:
    curr_max=np.max(grad)
    normed_grads.append(grad/curr_max)
    

min_grad=1
for grad in normed_grads:
    i_min=np.min(grad)
    if i_min<min_grad:
        min_grad=i_min
        

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 3, figsize=(3.54331,2.3622,), sharex=True)
sns.despine


sns.heatmap(normed_grads[0], vmin=min_grad,vmax=1,cbar=False,xticklabels=False,yticklabels=False,cmap='jet',ax=axes[0,0])
sns.heatmap(normed_grads[1], vmin=min_grad,vmax=1,cbar=False,xticklabels=False,yticklabels=False,cmap='jet',ax=axes[0,1])
sns.heatmap(normed_grads[2], vmin=min_grad,vmax=1,cbar=False,xticklabels=False,yticklabels=False,cmap='jet',ax=axes[0,2])
sns.heatmap(normed_grads[3], vmin=min_grad,vmax=1,cbar=False,xticklabels=False,yticklabels=False,cmap='jet',ax=axes[1,0])
sns.heatmap(normed_grads[4], vmin=min_grad,vmax=1,cbar=False,xticklabels=False,yticklabels=False,cmap='jet',ax=axes[1,1])
sns.heatmap(normed_grads[5], vmin=min_grad,vmax=1,cbar=False,xticklabels=False,yticklabels=False,cmap='jet',ax=axes[1,2])

plt.savefig('fig/Fig_3_h.eps', format='eps')


sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 2, figsize=(2.3622,1.1811), sharex=True)
sns.despine
sns.heatmap(normed_grads[0], vmin=min_grad,vmax=1,cbar=True,xticklabels=False,yticklabels=False,cmap='jet',ax=axes[0])
plt.savefig('fig/Fig_3_h_Colourbar.eps', format='eps')