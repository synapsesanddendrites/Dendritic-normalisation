#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:21:03 2019

@author: Alex
"""
###############################################################################
###############################################################################
###############################################################################
# Digits with 30 hidden neurons

n_epoch=50
nrep=10
sparsities=[0.1,0.2,0.4,0.6,0.8]

for sparsity in sparsities:
    pool = mp.Pool(mp.cpu_count()) # Multiprocessing         
    multi_locs=[pool.apply(sp_func, args=([30],n_epoch,dig_training,dig_test,sparsity)) for dummy_ind in range(nrep)] # Check all_funcs for defaults
    pool.close()
    sp_all_accs=np.zeros([nrep,n_epoch])
    sp_all_costs=np.zeros([nrep,n_epoch])
    for rep_ind in range(nrep):
        sp_all_accs[rep_ind]=multi_locs[rep_ind][0]
        sp_all_costs[rep_ind]=multi_locs[rep_ind][1]
    
    
    pool = mp.Pool(mp.cpu_count()) # Multiprocessing         
    multi_locs=[pool.apply(nm_func, args=([30],n_epoch,dig_training,dig_test,sparsity)) for dummy_ind in range(nrep)] # Check all_funcs for defaults
    pool.close()
    nm_all_accs=np.zeros([nrep,n_epoch])
    nm_all_costs=np.zeros([nrep,n_epoch])
    for rep_ind in range(nrep):
        nm_all_accs[rep_ind]=multi_locs[rep_ind][0]
        nm_all_costs[rep_ind]=multi_locs[rep_ind][1]
        
        
    np.save('sim_data/Fig_S1_'+str(sparsity)+'_digits_sp_cst',sp_all_costs)
    np.save('sim_data/Fig_S1_'+str(sparsity)+'_digits_sp_acc',sp_all_accs)
    np.save('sim_data/Fig_S1_'+str(sparsity)+'_digits_nm_cst',nm_all_costs)
    np.save('sim_data/Fig_S1_'+str(sparsity)+'_digits_nm_acc',nm_all_accs)

#%%
###############################################################################
###############################################################################
###############################################################################
# Fashion with differen sparsitiies

epochs=50
n_rep=10


L0_accs=np.zeros([5,50,n_rep])
L2_accs=np.zeros([5,50,n_rep])
L0_costs=np.zeros([5,50,n_rep])
L2_costs=np.zeros([5,50,n_rep])

for rep_ind in range(n_rep):
    # Constant excitability
    for sp_ind in range(5):
        print('L0-norm, rep '+str(rep_ind)+': '+str(sp_ind))
        sp_var=sparsities[sp_ind]
        runfile('/Users/Alex/Dendritic_normalisation/network_nm_L0_sparschng.py', wdir='/Users/Alex/Dendritic_normalisation',current_namespace=True)
        i_accs=accuracies
        for ind in range(50):
            L0_accs[sp_ind,ind,rep_ind]=i_accs[ind]
        i_costs=costs
        for ind in range(50):
            L0_costs[sp_ind,ind,rep_ind]=i_costs[ind]
        


        print('L'+str(2)+'-norm, rep '+str(rep_ind)+': '+str(sp_ind))
        runfile('/Users/Alex/Dendritic_normalisation/network_nm_L2_sparschng.py', wdir='/Users/Alex/Dendritic_normalisation',current_namespace=True)
        i_accs=accuracies
        for ind in range(50):
            L2_accs[sp_ind,ind,rep_ind]=i_accs[ind]
        i_costs=costs
        for ind in range(50):
            L2_costs[sp_ind,ind,rep_ind]=i_costs[ind]
    # Variable excitability
   
    
np.save('sim_data/Fig_S1_sparsitit_L0_acc',L0_accs)
np.save('sim_data/Fig_S1_sparsitit_L2_acc',L0_accs)
np.save('sim_data/Fig_S1_sparsitit_L0_cost',L0_costs)
np.save('sim_data/Fig_S1_sparsitit_L2_cost',L0_costs)
