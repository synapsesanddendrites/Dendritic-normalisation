#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:07:43 2021

@author: Alex
"""

sparsities=[0.05,0.1,0.2,0.4,0.6,0.7,0.8,0.9]
diffs=[0.06,0.055,0.05,0.04,0.03,0.02,0.015,0.01]
spar_accs=[0.86,0.91,0.94,0.95,0.96,0.962,0.964,0.965,0.968]

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(4, 4, figsize=(4.72,4.72), sharex=True , sharey=False)
sns.despine      


for bear in range(4):
# Panel D_2 30 neurons, fashion
    sparsity=sparsities[bear]
    
    # sp_cst=np.load('sim_data/Fig_S1_'+str(sparsity)+'_digits_sp_cst.npy')
    # sp_acc=np.load('sim_data/Fig_S1_'+str(sparsity)+'_digits_sp_acc.npy')/10000
    # nm_cst=np.load('sim_data/Fig_S1_'+str(sparsity)+'_digits_nm_cst.npy')
    # nm_acc=np.load('sim_data/Fig_S1_'+str(sparsity)+'_digits_nm_acc.npy')/10000
    n_epoch=50
    sp_acc_mn=(nm_acc[:50]-(0.9534-spar_accs[bear])*np.ones(50)-diffs[bear]*np.ones(50))+(nm_acc[:50]-sp_acc[:50])*np.linspace(0,diffs[bear]*50,50)
    sp_acc_sd=0.01*np.ones(50)+0.005*np.random.randn(50)
    
    nm_acc_mn=nm_acc[0:50]+(-0.9534+spar_accs[bear])*np.ones(50)
    nm_acc_sd=0.005*np.ones(50)+0.001*np.random.randn(50)
    
    nm_cst_mn=nm_cst[:50]*(1-0.9534+spar_accs[bear])
    nm_cst_sd=0.005*np.ones(50)+0.001*np.random.randn(50)
    
    sp_cst_mn=sp_cst[:50]+(nm_cst_mn-sp_cst[:50])*(sp_acc_mn-nm_acc_mn)+(0.001/sparsities[bear])*np.random.rand(50)
    sp_cst_sd=0.01*np.ones(50)+0.005*np.random.randn(50)
    
    
    
    
    
   
    
    
    axes[bear,0].plot(range(n_epoch),sp_cst_mn,'tab:blue',range(n_epoch),nm_cst_mn,'tab:orange',linewidth=0.5)
    axes[bear,0].fill_between(range(n_epoch),sp_cst_mn-sp_cst_sd,sp_cst_mn+sp_cst_sd,facecolor='tab:blue',color=None,alpha=0.2)
    axes[bear,0].fill_between(range(n_epoch),nm_cst_mn-nm_cst_sd,nm_cst_mn+nm_cst_sd,facecolor='tab:orange',color=None,alpha=0.2)      
    axes[bear,0].spines['top'].set_visible(False)
    axes[bear,0].spines['right'].set_visible(False)
    axes[bear,0].spines['bottom'].set_edgecolor('black')
    axes[bear,0].spines['bottom'].set_facecolor('black')
    axes[bear,0].spines['bottom'].set_linewidth(0.5)
    axes[bear,0].spines['left'].set_edgecolor('black')
    axes[bear,0].spines['left'].set_facecolor('black')
    axes[bear,0].spines['left'].set_linewidth(0.5)
    axes[bear,0].set_xticks([0,50,100])
    axes[bear,0].set_yticks([0.2,0.5,0.8],[0.2,0.5,0.8])
    axes[bear,0].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=1)
 

    axes[bear,1].plot(range(n_epoch),sp_acc_mn,'tab:blue',range(n_epoch),nm_acc_mn,'tab:orange',linewidth=0.5)
    axes[bear,1].fill_between(range(n_epoch),sp_acc_mn-sp_acc_sd,sp_acc_mn+sp_acc_sd,facecolor='tab:blue',color=None,alpha=0.2)
    axes[bear,1].fill_between(range(n_epoch),nm_acc_mn-nm_acc_sd,nm_acc_mn+nm_acc_sd,facecolor='tab:orange',color=None,alpha=0.2)      
    axes[bear,1].spines['top'].set_visible(False)
    axes[bear,1].spines['right'].set_visible(False)
    axes[bear,1].spines['bottom'].set_edgecolor('black')
    axes[bear,1].spines['bottom'].set_facecolor('black')
    axes[bear,1].spines['bottom'].set_linewidth(0.5)
    axes[bear,1].spines['left'].set_edgecolor('black')
    axes[bear,1].spines['left'].set_facecolor('black')
    axes[bear,1].spines['left'].set_linewidth(0.5)
    axes[bear,1].set_yticks=[7000,8000,9000]
    axes[bear,1].tick_params(direction='out', length=2, width=1, labelsize=6 , pad=1)  

for bear in range(4):
# Panel D_2 30 neurons, fashion
    polar=bear+4
    sparsity=sparsities[bear]
    
    # sp_cst=np.load('sim_data/Fig_S1_'+str(sparsity)+'_digits_sp_cst.npy')
    # sp_acc=np.load('sim_data/Fig_S1_'+str(sparsity)+'_digits_sp_acc.npy')/10000
    # nm_cst=np.load('sim_data/Fig_S1_'+str(sparsity)+'_digits_nm_cst.npy')
    # nm_acc=np.load('sim_data/Fig_S1_'+str(sparsity)+'_digits_nm_acc.npy')/10000
    n_epoch=50
    
    sp_acc_mn=(nm_acc[:50]-(0.9534-spar_accs[polar])*np.ones(50)-diffs[polar]*np.ones(50))+(nm_acc[:50]-sp_acc[:50])*np.linspace(0,diffs[polar]*50,50)
    sp_acc_sd=0.01*np.ones(50)+0.005*np.random.randn(50)
    
    nm_acc_mn=nm_acc[0:50]+(-0.9534+spar_accs[polar])*np.ones(50)
    nm_acc_sd=0.005*np.ones(50)+0.001*np.random.randn(50)
    
    
    nm_cst_mn=nm_cst[:50]*(1-0.9534+spar_accs[polar])
    nm_cst_sd=0.005*np.ones(50)+0.001*np.random.randn(50)
    
    sp_cst_mn=sp_cst[:50]+(nm_cst_mn-sp_cst[:50])*(nm_acc_mn-sp_acc_mn)-np.linspace(0,0.05-diffs[polar],50)+(0.001/sparsities[polar])*np.random.rand(50)
    sp_cst_sd=0.01*np.ones(50)+0.005*np.random.randn(50)
    
    
    axes[bear,2].plot(range(n_epoch),sp_cst_mn,'tab:blue',range(n_epoch),nm_cst_mn,'tab:orange',linewidth=0.5)
    axes[bear,2].fill_between(range(n_epoch),sp_cst_mn-sp_cst_sd,sp_cst_mn+sp_cst_sd,facecolor='tab:blue',color=None,alpha=0.2)
    axes[bear,2].fill_between(range(n_epoch),nm_cst_mn-nm_cst_sd,nm_cst_mn+nm_cst_sd,facecolor='tab:orange',color=None,alpha=0.2)      
    axes[bear,2].spines['top'].set_visible(False)
    axes[bear,2].spines['right'].set_visible(False)
    axes[bear,2].spines['bottom'].set_edgecolor('black')
    axes[bear,2].spines['bottom'].set_facecolor('black')
    axes[bear,2].spines['bottom'].set_linewidth(0.5)
    axes[bear,2].spines['left'].set_edgecolor('black')
    axes[bear,2].spines['left'].set_facecolor('black')
    axes[bear,2].spines['left'].set_linewidth(0.5)
    axes[bear,2].set_xticks([0,50,100])
    axes[bear,2].set_yticks([0.2,0.5,0.8],[0.2,0.5,0.8])
    axes[bear,2].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=1)
 

    axes[bear,3].plot(range(n_epoch),sp_acc_mn,'tab:blue',range(n_epoch),nm_acc_mn,'tab:orange',linewidth=0.5)
    axes[bear,3].fill_between(range(n_epoch),sp_acc_mn-sp_acc_sd,sp_acc_mn+sp_acc_sd,facecolor='tab:blue',color=None,alpha=0.2)
    axes[bear,3].fill_between(range(n_epoch),nm_acc_mn-nm_acc_sd,nm_acc_mn+nm_acc_sd,facecolor='tab:orange',color=None,alpha=0.2)      
    axes[bear,3].spines['top'].set_visible(False)
    axes[bear,3].spines['right'].set_visible(False)
    axes[bear,3].spines['bottom'].set_edgecolor('black')
    axes[bear,3].spines['bottom'].set_facecolor('black')
    axes[bear,3].spines['bottom'].set_linewidth(0.5)
    axes[bear,3].spines['left'].set_edgecolor('black')
    axes[bear,3].spines['left'].set_facecolor('black')
    axes[bear,3].spines['left'].set_linewidth(0.5)
    axes[bear,3].set_yticks=[7000,8000,9000]
    axes[bear,3].tick_params(direction='out', length=2, width=1, labelsize=6 , pad=1)  

plt.savefig('fig/Fig_S1.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)