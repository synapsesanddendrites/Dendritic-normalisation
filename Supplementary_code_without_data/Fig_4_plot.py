#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:50:07 2021

@author: Alex
"""

sp_cst=np.load('sim_data/Fig_4_rec_sp_errs.npy')
nm_cst=np.load('sim_data/Fig_4_rec_nm_errs.npy')
sp_acc=np.load('sim_data/Fig_4_rec_sp_accs.npy')
nm_acc=np.load('sim_data/Fig_4_rec_nm_accs.npy')

sp_cst_mn=np.mean(sp_cst,1)
nm_cst_mn=np.mean(nm_cst,1)
sp_cst_sd=np.sqrt(np.var(sp_cst,1))
nm_cst_sd=np.sqrt(np.var(nm_cst,1))


sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 1, figsize=(1.1811,2.3622,), sharex=True)
sns.despine      
    
axes[0].semilogy(range(100),sp_cst_mn,'tab:blue',range(100),nm_cst_mn,'tab:orange',linewidth=0.5)
axes[0].fill_between(range(100),sp_cst_mn-sp_cst_sd,sp_cst_mn+sp_cst_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[0].fill_between(range(100),nm_cst_mn-nm_cst_sd,nm_cst_mn+nm_cst_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_edgecolor('black')
axes[0].spines['bottom'].set_facecolor('black')
axes[0].spines['bottom'].set_linewidth(0.5)
axes[0].spines['left'].set_edgecolor('black')
axes[0].spines['left'].set_facecolor('black')
axes[0].spines['left'].set_linewidth(0.5)
axes[0].set_xticks([0,50,100])
axes[0].set_yticks([10,100,1000],[10,100,1000])
axes[0].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=0.5)
 

sp_acc_mn=np.mean(sp_acc,1)
nm_acc_mn=np.mean(nm_acc,1)
sp_acc_sd=np.sqrt(np.var(sp_acc,1))
nm_acc_sd=np.sqrt(np.var(nm_acc,1))

axes[1].plot(range(100),sp_acc_mn,'tab:blue',range(100),nm_acc_mn,'tab:orange',linewidth=0.5)
axes[1].fill_between(range(100),sp_acc_mn-sp_acc_sd,sp_acc_mn+sp_acc_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[1].fill_between(range(100),nm_acc_mn-nm_acc_sd,nm_acc_mn+nm_acc_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_edgecolor('black')
axes[1].spines['bottom'].set_facecolor('black')
axes[1].spines['bottom'].set_linewidth(0.5)
axes[1].spines['left'].set_edgecolor('black')
axes[1].spines['left'].set_facecolor('black')
axes[1].spines['left'].set_linewidth(0.5)
axes[1].set_xticks([0,50,100])
axes[1].set_yticks([0.6,1],[0.6,1])
axes[1].set_ylim([0.6,1.1])
axes[1].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=0.5)
 

plt.savefig('fig/Fig_4_b.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)


#%%
weights_dict = pickle.load( open('sim_data/Fig_4_rec_weight_dict.pkl', "rb" ) )
res=10

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 1, figsize=(1.1811,2.3622,), sharex=True , sharey=True)
sns.despine      

key_inds=[0,4,9,24,99]

all_sp_naff=np.zeros((100,50))
all_nm_naff=np.zeros((100,50))
    
for bear in range(100):
    sp_wh=weights_dict['Control'][bear][ward]['wh']
    n_aff=np.count_nonzero(sp_wh,0)
    all_sp_naff[bear,:]=n_aff
    
    nm_vh=weights_dict['Normed'][bear][ward]['vh']
    nm_s=weights_dict['Normed'][bear][ward]['s']
    n_aff=np.count_nonzero(nm_vh,0)
    mrph_nm=np.tile(1/n_aff,[50,1])
    nm_wh=nm_s*nm_vh*mrph_nm
    nm_wh[np.isnan(nm_wh)]=0
    all_nm_naff[bear,:]=n_aff
    
    
BINS=range(51);
sp_wh=all_sp_naff.reshape(5000)
nm_wh=all_nm_naff.reshape(5000)
[sp_dist,sp_ends]=np.histogram(sp_wh,BINS)
[nm_dist,nm_ends]=np.histogram(nm_wh,BINS) 
sp_mids=0.5+np.arange(0,50)
nm_mids=0.5+np.arange(0,50)
axes[0].bar(sp_mids,sp_dist/np.trapz(sp_dist,sp_mids),color='tab:blue',edgecolor='none')

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_edgecolor(None)
axes[0].spines['bottom'].set_facecolor('black')
axes[0].spines['bottom'].set_linewidth(1)
axes[0].spines['left'].set_edgecolor(None)
axes[0].spines['left'].set_facecolor('black')
axes[0].spines['left'].set_linewidth(1)
axes[0].set_ylim((0,0.2)) 
axes[0].set_yticks=[0,1]
axes[0].set_xlim((0,50))
 #   axes[ward].set_xticks=[100,150,200]
axes[0].tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)


axes[1].bar(nm_mids,nm_dist/np.trapz(nm_dist,nm_mids),color='tab:orange',edgecolor='none')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_edgecolor(None)
axes[1].spines['bottom'].set_facecolor('black')
axes[1].spines['bottom'].set_linewidth(1)
axes[1].spines['left'].set_edgecolor(None)
axes[1].spines['left'].set_facecolor('black')
axes[1].spines['left'].set_linewidth(1)
axes[1].set_ylim((0,0.2)) 
axes[1].set_yticks=[0,1]
axes[1].set_xlim((0,50))
 #   axes[ward].set_xticks=[100,150,200]
axes[1].tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)
plt.savefig('fig/Fig_4_ci.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)




#%%
weights_dict = pickle.load( open('sim_data/Fig_4_rec_weight_dict.pkl', "rb" ) )
res=100 

key_inds=[0,4,9,24,99]

all_sp_wh=np.zeros((100,50,50))
all_nm_wh=np.zeros((100,50,50))

for bear in range(100):
    sp_wh=weights_dict['Control'][bear][4]['wh']
    all_sp_wh[bear,:,:]=sp_wh
    nm_vh=weights_dict['Normed'][bear][4]['vh']
    nm_s=weights_dict['Normed'][bear][4]['s']
    n_aff=np.count_nonzero(nm_vh,0)
    mrph_nm=np.tile(1/n_aff,[50,1])
    nm_wh=nm_s*nm_vh*mrph_nm
    nm_wh[np.isnan(nm_wh)]=0
    all_nm_wh[bear,:,:]=nm_wh

f, axes = plt.subplots(2, 1, figsize=(1.1811,2.3622,), sharex=True , sharey=True)
sns.despine      

BINS=np.linspace(-5,5,res+1);
#sp_wh=all_sp_naff.reshape(5000)
#nm_wh=all_nm_naff.reshape(5000)

[sp_dist,sp_ends]=np.histogram(all_sp_wh[abs(all_sp_wh)>0.1],BINS)
[nm_dist,nm_ends]=np.histogram(all_nm_wh[abs(all_nm_wh)>0.1],BINS) 
sp_mids=(sp_ends[1:(res+1)]+sp_ends[0:res])/2
nm_mids=(nm_ends[1:(res+1)]+nm_ends[0:res])/2
axes[0].bar(sp_mids,sp_dist/np.trapz(sp_dist,sp_mids),color='tab:blue',edgecolor='none',width=0.2)

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_edgecolor(None)
axes[0].spines['bottom'].set_facecolor('black')
axes[0].spines['bottom'].set_linewidth(1)
axes[0].spines['left'].set_edgecolor(None)
axes[0].spines['left'].set_facecolor('black')
axes[0].spines['left'].set_linewidth(1)
axes[0].set_ylim((0,0.75)) 
axes[0].set_yticks=[0,1]
axes[0].set_xlim((-5,5))
 #   axes[ward].set_xticks=[100,150,200]
axes[0].tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)


axes[1].bar(nm_mids,nm_dist/np.trapz(nm_dist,nm_mids),color='tab:orange',edgecolor='none',width=0.2)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_edgecolor(None)
axes[1].spines['bottom'].set_facecolor('black')
axes[1].spines['bottom'].set_linewidth(1)
axes[1].spines['left'].set_edgecolor(None)
axes[1].spines['left'].set_facecolor('black')
axes[1].spines['left'].set_linewidth(1)
axes[1].set_ylim((0,0.75)) 
axes[1].set_yticks=[0,1]
axes[1].set_xlim((-5,5))
 #   axes[ward].set_xticks=[100,150,200]
axes[1].tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)

plt.savefig('fig/Fig_4_cii.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)

#%%
sp_cst=np.load('sim_data/Fig_4_recff_sp_errs.npy')
nm_cst=np.load('sim_data/Fig_4_recff_nm_errs.npy')
sp_acc=np.load('sim_data/Fig_4_recff_sp_accs.npy')
nm_acc=np.load('sim_data/Fig_4_recff_nm_accs.npy')

sp_cst_mn=np.mean(sp_cst,1)
nm_cst_mn=np.mean(nm_cst,1)
sp_cst_sd=np.sqrt(np.var(sp_cst,1))
nm_cst_sd=np.sqrt(np.var(nm_cst,1))


sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 1, figsize=(1.1811,2.3622,), sharex=True)
sns.despine      
    
axes[0].semilogy(range(100),sp_cst_mn,'tab:blue',range(100),nm_cst_mn,'tab:orange',linewidth=0.5)
axes[0].fill_between(range(100),sp_cst_mn-sp_cst_sd,sp_cst_mn+sp_cst_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[0].fill_between(range(100),nm_cst_mn-sp_cst_sd,nm_cst_mn+sp_cst_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_edgecolor('black')
axes[0].spines['bottom'].set_facecolor('black')
axes[0].spines['bottom'].set_linewidth(0.5)
axes[0].spines['left'].set_edgecolor('black')
axes[0].spines['left'].set_facecolor('black')
axes[0].spines['left'].set_linewidth(0.5)
axes[0].set_ylim([100,2500])
axes[0].set_xticks([0,50,100])
axes[0].set_yticks([10,100,1000],[10,100,1000])
axes[0].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=0.5)
 

sp_acc_mn=np.mean(sp_acc,1)
nm_acc_mn=np.mean(nm_acc,1)
sp_acc_sd=np.sqrt(np.var(sp_acc,1))
nm_acc_sd=np.sqrt(np.var(nm_acc,1))

axes[1].plot(range(100),sp_acc_mn,'tab:blue',range(100),nm_acc_mn,'tab:orange',linewidth=0.5)
axes[1].fill_between(range(100),sp_acc_mn-sp_acc_sd,sp_acc_mn+sp_acc_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[1].fill_between(range(100),nm_acc_mn-nm_acc_sd,nm_acc_mn+nm_acc_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_edgecolor('black')
axes[1].spines['bottom'].set_facecolor('black')
axes[1].spines['bottom'].set_linewidth(0.5)
axes[1].spines['left'].set_edgecolor('black')
axes[1].spines['left'].set_facecolor('black')
axes[1].spines['left'].set_linewidth(0.5)
axes[1].set_xticks([0,50,100])
axes[1].set_yticks([0.6,1],[0.6,1])
axes[1].set_ylim([0.6,1.1])
axes[1].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=0.5)
 

plt.savefig('fig/Fig_4_e.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)


#%%
weights_dict = pickle.load( open('sim_data/Fig_4_recff_weight_dict.pkl', "rb" ) )
res=10

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 1, figsize=(1.1811,2.3622,), sharex=True , sharey=True)
sns.despine      

key_inds=[0,4,9,24,99]

all_sp_naff=np.zeros((25,50))
all_nm_naff=np.zeros((25,50))
    
for bear in range(25):
    sp_wh=weights_dict['Control'][bear][4]['wh']
    n_aff=np.count_nonzero(sp_wh,0)
    all_sp_naff[bear,:]=n_aff
    
    nm_vh=weights_dict['Normed'][bear][4]['vh']
    nm_sh=weights_dict['Normed'][bear][4]['sh']
    n_aff=np.count_nonzero(nm_vh,0)
    mrph_nm=np.tile(1/n_aff,[50,1])
    nm_wh=nm_sh*nm_vh*mrph_nm
    nm_wh[np.isnan(nm_wh)]=0
    all_nm_naff[bear,:]=n_aff
    
    
BINS=range(51);

[sp_dist,sp_ends]=np.histogram(all_sp_naff,BINS)
[nm_dist,nm_ends]=np.histogram(all_nm_naff,BINS) 
sp_mids=0.5+np.arange(0,50)
nm_mids=0.5+np.arange(0,50)

all_sp0_naff=np.zeros((25,50))
all_nm0_naff=np.zeros((25,50))
for bear in range(25):
    sp_w0=weights_dict['Control'][bear][4]['w0']
    n_aff=np.count_nonzero(sp_w0,0)
    all_sp0_naff[bear,:]=n_aff
    
    nm_v0=weights_dict['Normed'][bear][4]['v0']
    nm_s0=weights_dict['Normed'][bear][4]['s0']
    n_aff=np.count_nonzero(nm_v0,0)
    mrph_nm=np.tile(1/n_aff,[2,1])
    nm_w0=nm_s0*nm_v0*mrph_nm
    nm_w0[np.isnan(nm_w0)]=0
    all_nm0_naff[bear,:]=n_aff
    
    
BINS=range(51);

[sp0_dist,sp0_ends]=np.histogram(all_sp0_naff,BINS)
[nm0_dist,nm0_ends]=np.histogram(all_nm0_naff,BINS) 
sp0_mids=0.5+np.arange(0,50)
nm0_mids=0.5+np.arange(0,50)



# axes[0].bar(sp0_mids,sp0_dist/np.trapz(sp0_dist,sp0_mids),color='tab:blue',edgecolor='none')
# axes[0].spines['top'].set_visible(False)
# axes[0].spines['right'].set_visible(False)
# axes[0].spines['bottom'].set_edgecolor(None)
# axes[0].spines['bottom'].set_facecolor('black')
# axes[0].spines['bottom'].set_linewidth(1)
# axes[0].spines['left'].set_edgecolor(None)
# axes[0].spines['left'].set_facecolor('black')
# axes[0].spines['left'].set_linewidth(1)
# axes[0].set_ylim((0,0.2)) 
# axes[0].set_yticks=[0,1]
# axes[0].set_xlim((0,50))
#  #   axes[ward].set_xticks=[100,150,200]
# axes[0].tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)

axes[0].bar(sp_mids,sp_dist/np.trapz(sp_dist,sp_mids),color='tab:blue',edgecolor='none')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_edgecolor(None)
axes[0].spines['bottom'].set_facecolor('black')
axes[0].spines['bottom'].set_linewidth(1)
axes[0].spines['left'].set_edgecolor(None)
axes[0].spines['left'].set_facecolor('black')
axes[0].spines['left'].set_linewidth(1)
axes[0].set_ylim((0,0.2)) 
axes[0].set_yticks=[0,1]
axes[0].set_xlim((0,50))
 #   axes[ward].set_xticks=[100,150,200]
axes[0].tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)

# axes[2].bar(nm0_mids,nm0_dist/np.trapz(nm0_dist,nm0_mids),color='tab:orange',edgecolor='none')
# axes[2].spines['top'].set_visible(False)
# axes[2].spines['right'].set_visible(False)
# axes[2].spines['bottom'].set_edgecolor(None)
# axes[2].spines['bottom'].set_facecolor('black')
# axes[2].spines['bottom'].set_linewidth(1)
# axes[2].spines['left'].set_edgecolor(None)
# axes[2].spines['left'].set_facecolor('black')
# axes[2].spines['left'].set_linewidth(1)
# axes[2].set_ylim((0,0.2)) 
# axes[2].set_yticks=[0,1]
# axes[2].set_xlim((0,50))
#  #   axes[ward].set_xticks=[100,150,200]
# axes[2].tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)


axes[1].bar(nm_mids,nm_dist/np.trapz(nm_dist,nm_mids),color='tab:orange',edgecolor='none')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_edgecolor(None)
axes[1].spines['bottom'].set_facecolor('black')
axes[1].spines['bottom'].set_linewidth(1)
axes[1].spines['left'].set_edgecolor(None)
axes[1].spines['left'].set_facecolor('black')
axes[1].spines['left'].set_linewidth(1)
axes[1].set_ylim((0,0.2)) 
axes[1].set_yticks=[0,1]
axes[1].set_xlim((0,50))
 #   axes[ward].set_xticks=[100,150,200]
axes[1].tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)


plt.savefig('fig/Fig_4_fi.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)




#%%
weights_dict = pickle.load( open('sim_data/Fig_4_recff_weight_dict.pkl', "rb" ) )
res=100 

key_inds=[0,4,9,24,99]

all_sp_wh=np.zeros((25,50,50))
all_nm_wh=np.zeros((25,50,50))

for bear in range(25):
    sp_wh=weights_dict['Control'][bear][4]['wh']
    all_sp_wh[bear,:,:]=sp_wh
    nm_vh=weights_dict['Normed'][bear][4]['vh']
    nm_sh=weights_dict['Normed'][bear][4]['sh']
    n_aff=np.count_nonzero(nm_vh,0)
    mrph_nm=np.tile(1/n_aff,[50,1])
    nm_wh=nm_sh*nm_vh*mrph_nm
    nm_wh[np.isnan(nm_wh)]=0
    all_nm_wh[bear,:,:]=nm_wh

all_sp_w0=np.zeros((25,2,50))
all_nm_w0=np.zeros((25,2,50))

for bear in range(25):
    sp_w0=weights_dict['Control'][bear][4]['w0']
    all_sp_w0[bear,:,:]=sp_w0
    nm_v0=weights_dict['Normed'][bear][4]['v0']
    nm_s0=weights_dict['Normed'][bear][4]['s0']
    n_aff=np.count_nonzero(nm_v0,0)
    mrph_nm=np.tile(1/n_aff,[2,1])
    nm_w0=nm_v0/nm_s0
    nm_w0[np.isnan(nm_w0)]=0
    all_nm_w0[bear,:,:]=nm_w0

f, axes = plt.subplots(4, 1, figsize=(1.1811,2.3622,), sharex=True , sharey=True)
sns.despine      

BINS=np.linspace(-5,5,res+1);
#sp_wh=all_sp_naff.reshape(5000)
#nm_wh=all_nm_naff.reshape(5000)

[sp_dist,sp_ends]=np.histogram(all_sp_wh[abs(all_sp_wh)>0.1],BINS)
[nm_dist,nm_ends]=np.histogram(all_nm_wh[abs(all_nm_wh)>0.1],BINS) 
sp_mids=(sp_ends[1:(res+1)]+sp_ends[0:res])/2
nm_mids=(nm_ends[1:(res+1)]+nm_ends[0:res])/2

[sp0_dist,sp0_ends]=np.histogram(all_sp_w0[abs(all_sp_w0)>0.1],BINS)
[nm0_dist,nm0_ends]=np.histogram(all_nm_w0[abs(all_nm_w0)>0.1],BINS) 
sp0_mids=(sp0_ends[1:(res+1)]+sp0_ends[0:res])/2
nm0_mids=(nm0_ends[1:(res+1)]+nm0_ends[0:res])/2

axes[0].bar(sp0_mids,sp0_dist/np.trapz(sp0_dist,sp0_mids),color='tab:blue',edgecolor='none',width=0.2)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_edgecolor(None)
axes[0].spines['bottom'].set_facecolor('black')
axes[0].spines['bottom'].set_linewidth(1)
axes[0].spines['left'].set_edgecolor(None)
axes[0].spines['left'].set_facecolor('black')
axes[0].spines['left'].set_linewidth(1)
axes[0].set_ylim((0,0.75)) 
axes[0].set_yticks=[0,1]
axes[0].set_xlim((-5,5))
 #   axes[ward].set_xticks=[100,150,200]
axes[0].tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)


axes[1].bar(sp_mids,sp_dist/np.trapz(sp_dist,sp_mids),color='tab:blue',edgecolor='none',width=0.2)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_edgecolor(None)
axes[1].spines['bottom'].set_facecolor('black')
axes[1].spines['bottom'].set_linewidth(1)
axes[1].spines['left'].set_edgecolor(None)
axes[1].spines['left'].set_facecolor('black')
axes[1].spines['left'].set_linewidth(1)
axes[1].set_ylim((0,0.75)) 
axes[1].set_yticks=[0,1]
axes[1].set_xlim((-5,5))
 #   axes[ward].set_xticks=[100,150,200]
axes[1].tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)

axes[2].bar(nm0_mids,nm0_dist/np.trapz(nm0_dist,nm0_mids),color='tab:orange',edgecolor='none',width=0.2)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].spines['bottom'].set_edgecolor(None)
axes[2].spines['bottom'].set_facecolor('black')
axes[2].spines['bottom'].set_linewidth(1)
axes[2].spines['left'].set_edgecolor(None)
axes[2].spines['left'].set_facecolor('black')
axes[2].spines['left'].set_linewidth(1)
axes[2].set_ylim((0,0.75)) 
axes[2].set_yticks=[0,1]
axes[2].set_xlim((-5,5))
 #   axes[ward].set_xticks=[100,150,200]
axes[2].tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)

plt.savefig('fig/Fig_4_fii.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)

axes[3].bar(nm_mids,nm_dist/np.trapz(nm_dist,nm_mids),color='tab:orange',edgecolor='none',width=0.2)
axes[3].spines['top'].set_visible(False)
axes[3].spines['right'].set_visible(False)
axes[3].spines['bottom'].set_edgecolor(None)
axes[3].spines['bottom'].set_facecolor('black')
axes[3].spines['bottom'].set_linewidth(1)
axes[3].spines['left'].set_edgecolor(None)
axes[3].spines['left'].set_facecolor('black')
axes[3].spines['left'].set_linewidth(1)
axes[3].set_ylim((0,0.75)) 
axes[3].set_yticks=[0,1]
axes[3].set_xlim((-5,5))
 #   axes[ward].set_xticks=[100,150,200]
axes[3].tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)

plt.savefig('fig/Fig_4_fii.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)
