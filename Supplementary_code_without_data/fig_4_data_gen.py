#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:51:58 2020

@author: Alex
"""
goodrun_des=100
key_weights=[1,5,10,25,100]

curr_goodrun=0
sp_allerrs=list()
sp_allaccs=list()
sp_allweights=list()
while curr_goodrun<goodrun_des:
    try:
        runfile('/Users/Alex/Dendritic_normalisation/rec_sp.py', wdir='/Users/Alex/Dendritic_normalisation',current_namespace=True)
        if accs_sp[3]>=0.95:
            sp_allerrs.append(errs_sp[1:])
            sp_allaccs.append(accs_sp)
            
            these_keys=list()
            for ward in range(len(key_weights)):
                these_keys.append(weights_sp[key_weights[ward]-1])
            sp_allweights.append(these_keys)
            curr_goodrun+=1
    except Exception as error:
        print('Oops')
        
curr_goodrun=0
nm_allerrs=list()
nm_allaccs=list()
nm_allweights=list()
while curr_goodrun<goodrun_des:
    try:
        runfile('/Users/Alex/Dendritic_normalisation/rec_nm.py', wdir='/Users/Alex/Dendritic_normalisation',current_namespace=True)
        if accs_nm[3]>=0.95:
            nm_allerrs.append(errs_nm[1:])
            nm_allaccs.append(accs_nm)
        
            these_keys=list()
            for ward in range(len(key_weights)):
                these_keys.append(weights_nm[key_weights[ward]-1])
            nm_allweights.append(these_keys)
            curr_goodrun+=1
    except Exception as error:
        print('Oops')

gd_ind=0
sp_accs=np.zeros([100,goodrun_des])
sp_errs=np.zeros([100,goodrun_des])
for ward in range(len(sp_allaccs)):
    if len(sp_allaccs[ward])==100:
        sp_accs[:,gd_ind]=sp_allaccs[ward]
        sp_errs[:,gd_ind]=sp_allerrs[ward]
        gd_ind+=1
np.save('sim_data/Fig_4_rec_sp_accs',sp_accs)
np.save('sim_data/Fig_4_rec_sp_errs',sp_errs)

gd_ind=0
nm_accs=np.zeros([100,goodrun_des])
nm_errs=np.zeros([100,goodrun_des])
for ward in range(len(nm_allaccs)):
    if len(nm_allaccs[ward])==100:
        nm_accs[:,gd_ind]=nm_allaccs[ward]
        nm_errs[:,gd_ind]=nm_allerrs[ward]
        gd_ind+=1
np.save('sim_data/Fig_4_rec_nm_accs',nm_accs)
np.save('sim_data/Fig_4_rec_nm_errs',nm_errs)

weights_dict={'Control':sp_allweights,'Normed':nm_allweights}
with open('sim_data/Fig_4_rec_weight_dict.pkl', 'wb') as f:
    pickle.dump(weights_dict, f)
    
#%%
goodrun_des=25
key_weights=[1,5,10,25,100]

curr_goodrun=0
sp_allerrs=list()
sp_allaccs=list()
sp_allweights=list()
while curr_goodrun<goodrun_des:
    print('Control',curr_goodrun)
    try:
        runfile('/Users/Alex/Dendritic_normalisation/recff_sp_4.py', wdir='/Users/Alex/Dendritic_normalisation',current_namespace=True)
        if accs_sp[30]>=0.25:
            sp_allerrs.append(errs_sp[1:])
            sp_allaccs.append(accs_sp)
            
            these_keys=list()
            for ward in range(len(key_weights)):
                these_keys.append(weights_sp[key_weights[ward]-1])
            sp_allweights.append(these_keys)
            curr_goodrun+=1
    except Exception as error:
        print('Oops')
        
curr_goodrun=0
nm_allerrs=list()
nm_allaccs=list()
nm_allweights=list()
while curr_goodrun<goodrun_des:
    print('Normed',curr_goodrun)
    try:
        runfile('/Users/Alex/Dendritic_normalisation/recff_nm_4.py', wdir='/Users/Alex/Dendritic_normalisation',current_namespace=True)
        if accs_nm[30]>=0.25:
            nm_allerrs.append(errs_nm[1:])
            nm_allaccs.append(accs_nm)
        
            these_keys=list()
            for ward in range(len(key_weights)):
                these_keys.append(weights_nm[key_weights[ward]-1])
            nm_allweights.append(these_keys)
            curr_goodrun+=1
    except Exception as error:
        print('Oops')

sp_accs=np.zeros([100,goodrun_des])
sp_errs=np.zeros([100,goodrun_des])
for ward in range(len(sp_allaccs)):
    sp_accs[:,ward]=sp_allaccs[ward][0:-1]
    sp_errs[:,ward]=sp_allerrs[ward]
np.save('sim_data/Fig_4_recff_sp_accs',sp_accs)
np.save('sim_data/Fig_4_recff_sp_errs',sp_errs)

nm_accs=np.zeros([100,goodrun_des])
nm_errs=np.zeros([100,goodrun_des])
for ward in range(len(nm_allaccs)):
    nm_accs[:,ward]=nm_allaccs[ward][0:-1]    
    nm_errs[:,ward]=nm_allerrs[ward]
    
np.save('sim_data/Fig_4_recff_nm_accs',nm_accs)
np.save('sim_data/Fig_4_recff_nm_errs',nm_errs)

weights_dict={'Control':sp_allweights,'Normed':nm_allweights}
with open('sim_data/Fig_4_recff_weight_dict.pkl', 'wb') as f:
    pickle.dump(weights_dict, f)

