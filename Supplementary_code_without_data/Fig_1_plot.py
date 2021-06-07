# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:45:18 2019

@author: Alex Bird
"""

#%%
# Panel B Somata etc
ri=100 # Axial resistivity
r=1 # Radius
gl=5e-05 # Membrane conductivity
C=1 # Membrane capacitance
lam=np.sqrt(r/(2*ri*gl)) # Length constant
G_inf=np.pi*r**2/(lam*ri) # Semi-infinite input conductance
L_range=np.linspace(0,2,1000) # Electronic lengths to plot
syn_delt=0.001;
hom_mean=syn_delt/(G_inf*L_range) # Mean voltage plasticity
hom_mean[hom_mean>5]=5
hom_sd=hom_mean*np.sqrt(L_range**2/np.tanh(L_range)+L_range/(np.sinh(L_range)**2)-1)
lb=np.zeros(1000)
ub=np.zeros(1000)
for ward in range(1000):
    if hom_mean[ward]+hom_sd[ward]>5:
         ub[ward]=5
    else:
         ub[ward]=hom_mean[ward]+hom_sd[ward]
ub[np.isnan(ub)]=5

soma_radii=[2.5,5,7.5]
s_inds=np.zeros((len(soma_radii),1000))
for ward in range(len(s_inds)):
    G_s=4*np.pi*soma_radii[ward]**2*gl
    s_inds[ward,:]=syn_delt/(G_inf*L_range) *np.tanh(L_range)/(np.tanh(L_range)+G_s)
s_inds[np.isnan(s_inds)]=5
s_inds[s_inds>5]=5
hom_mean[hom_mean>5]=5

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(3, 1, figsize=(1.8,2.36), sharex=True , sharey=False)
sns.despine      
axes[0].plot(L_range,hom_mean,'k',linewidth=0.5)
axes[0].plot(L_range,s_inds[0,:],'k--',L_range,s_inds[1,:],'k-.',L_range,s_inds[2,:],'k:',linewidth=0.5)
axes[0].fill_between(L_range,lb,ub,facecolor='gainsboro',color=None,alpha=0.2)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_edgecolor('black')
axes[0].spines['bottom'].set_facecolor('black')
axes[0].spines['bottom'].set_linewidth(0.5)
axes[0].spines['left'].set_edgecolor('black')
axes[0].spines['left'].set_facecolor('black')
axes[0].spines['left'].set_linewidth(0.5)
axes[0].set_ylim([0,5])
axes[0].set_xticks([0,1,2])
axes[0].set_yticks([0,2.5,5],[0,2.5,5])
axes[0].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=1)

# Maximum voltage
tF=0.002
tL=0.02
syn_delt=2.5
tS_list=[0.01,0.05,0.1]
all_v_max=[]
all_v_var=[]
for tS in tS_list:
    # Get v_max
    n_cyc=10 # Number of root-finding iterations
    for ward in range(n_cyc):
        if ward==0:
            t_samp=np.linspace(0,0.1,1000)
        else:
            t_samp=np.linspace(t1,t2,1000)
        LHS=(tS-tL)*(tF*np.exp(-t_samp/tL)-tL*np.exp(-t_samp/tF))
        RHS=(tF-tL)*(tS*np.exp(-t_samp/tL)-tL*np.exp(-t_samp/tS))
        signs=np.sign(LHS-RHS)
        if signs[1]==1: # Positive to negative
            poses=np.nonzero(signs==1)
            t1=t_samp[poses[0][-1]]
            t2=t_samp[poses[0][-1]+1]
        elif signs[1]==-1: # Negative to positive
            negs=np.nonzero(signs==-1)
            t1=t_samp[negs[0][-1]]
            t2=t_samp[negs[0][-1]+1]
        t_star=(t1+t2)/2
    
    v_max=syn_delt*tL/(2*L_range*(tF-tS))*((tF/(tF-tL))*(np.exp(-t_star/tF)-np.exp(-t_star/tL))-(tS/(tS-tL))*(np.exp(-t_star/tS)-np.exp(-t_star/tL)))
    v_max_var=np.zeros(np.size(v_max))
    n_max=100 # Number of voltage modes to consider 
    for ward in range(len(v_max)):
        L=L_range[ward]
        v_ward=0
        for n in range(0,n_max):
            tN=tL/(1+(n*np.pi/L)**2)
            v_ward+=(tF*tN/(tF-tN)*(np.exp(-t_star/tF)-np.exp(-t_star/tN))-(tS*tN/(tS-tN))*(np.exp(-t_star/tS)-np.exp(-t_star/tN)))**2
        
        v_max_var[ward]=(syn_delt**2/(4*L**2*(tF-tS)**2))*v_ward
    v_max[np.isnan(v_max)]=5
    v_max[v_max>5]=5
    v_max_var[np.isnan(v_max_var)]=1000
    all_v_max.append(v_max)
    all_v_var.append(v_max_var)
lb=np.zeros(1000)
ub=np.zeros(1000)
for ward in range(1000):
    if (hom_mean[ward]+hom_sd[ward])>5:
         ub[ward]=5
    else:
         ub[ward]=hom_mean[ward]+hom_sd[ward]
ub[np.isnan(ub)]=5
axes[1].plot(L_range,all_v_max[0],'k-.',linewidth=0.5)
axes[1].fill_between(L_range,lb,ub,facecolor='gainsboro',color=None,alpha=0.2)
axes[1].plot(L_range,all_v_max[1],'k',linewidth=0.5)
#axes[1].fill_between(L_range,all_v_max[1]-np.sqrt(all_v_var[1]),all_v_max[1]+np.sqrt(all_v_var[1]),facecolor='gainsboro',color=None,alpha=0.2)
axes[1].plot(L_range,all_v_max[2],'k:',linewidth=0.5)
#axes[1].fill_between(L_range,all_v_max[2]-np.sqrt(all_v_var[2]),all_v_max[2]+np.sqrt(all_v_var[2]),facecolor='gainsboro',color=None,alpha=0.2)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_edgecolor('black')
axes[1].spines['bottom'].set_facecolor('black')
axes[1].spines['bottom'].set_linewidth(0.5)
axes[1].spines['left'].set_edgecolor('black')
axes[1].spines['left'].set_facecolor('black')
axes[1].spines['left'].set_linewidth(0.5)
axes[1].set_ylim([0,5])
axes[1].set_xticks([0,1,2])
axes[1].set_yticks([0,2.5,5],[0,2.5,5])
axes[1].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=1)

# Total voltage 

all_v_tot=[]
all_tot_var=[]
for tS in tS_list:
   # Get v_max
    v_tot=10*syn_delt*tL/(2*L_range)
    v_tot_var=np.zeros(np.size(v_tot))
    n_max=100 # Number of voltage modes to consider 
    for ward in range(len(v_tot)):
        L=L_range[ward]
        pref=100*syn_delt**2/(8*L**2*(tF-tS)**2)
        v_ward=0
        for n in range(n_max):
            tN=tL/(1+(n*np.pi/L)**2)
            war1=(tF*tN)**2/(tF+tN)
            war2=(tS*tN)**2/(tS+tN)
            war3=2*tF*tS*tN**2/((tF-tN)*(tS-tN))*(tF*tS/(tF+tS)-(tF*tN/(tF+tN)-(tS*tN/(tS+tN)+tN/2)))
            v_ward+=war1+war2-war3
        v_tot_var[ward]=pref*abs(v_ward)
    v_tot[np.isnan(v_tot)]=5
    v_tot[v_tot>5]=5
    v_tot_var[np.isnan(v_tot_var)]=1000
    for v_ind in range(1,1000):
        if v_tot_var[v_ind]>=v_tot_var[v_ind-1]:
             v_tot_var[v_ind]=v_tot_var[v_ind-1]
    all_v_tot.append(v_tot)
    all_tot_var.append(v_tot_var)
ub_list=[]
for sooth in range(3):
    ub=np.zeros(1000)
    for ward in range(1000):
        if (all_v_tot[sooth][ward]+np.sqrt(all_tot_var[sooth][ward]))>5:
            ub[ward]=5
        else:
            ub[ward]=(all_v_tot[sooth][ward]+np.sqrt(all_tot_var[sooth][ward]))
    for ward in range(1000):
        if L_range[ward]>1:
            if ub[ward]<ub[ward-1]:
                ub[ward]=ub[ward-1]-0.001
    ub_list.append(ub)

axes[2].plot(L_range,all_v_tot[0],'k-.',linewidth=0.5)
axes[2].fill_between(L_range,lb,ub_list[0],facecolor='gainsboro',color=None,alpha=0.2,lw=0)
axes[2].plot(L_range,all_v_tot[1],'k',linewidth=0.5)
axes[2].fill_between(L_range,lb,ub_list[1],facecolor='gainsboro',color=None,alpha=0.2,lw=0)
axes[2].plot(L_range,all_v_tot[2],'k:',linewidth=0.5)
axes[2].fill_between(L_range,lb,ub_list[2],facecolor='gainsboro',color=None,alpha=0.2,lw=0)
axes[2].spines['right'].set_visible(False)
axes[2].spines['bottom'].set_edgecolor('black')
axes[2].spines['bottom'].set_facecolor('black')
axes[2].spines['bottom'].set_linewidth(0.5)
axes[2].spines['left'].set_edgecolor('black')
axes[2].spines['left'].set_facecolor('black')
axes[2].spines['left'].set_linewidth(0.5)
axes[2].set_ylim([0,5])
axes[2].set_xticks([0,1,2])
axes[2].set_yticks([0,2.5,5],[0,2.5,5])
axes[2].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=1)

plt.savefig('fig/Fig_1_b.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)
#%%
# Panel D_1 30 neurons, digits
sp_cst=np.load('sim_data/Fig_1_30_digits_sp_cst.npy')
sp_acc=np.load('sim_data/Fig_1_30_digits_sp_acc.npy')/10000
nm_cst=np.load('sim_data/Fig_1_30_digits_nm_cst.npy')
nm_acc=np.load('sim_data/Fig_1_30_digits_nm_acc.npy')/10000

n_rep=sp_cst.shape[0]
n_epoch=sp_cst.shape[1]
sp_cst_mn=np.zeros(n_epoch)
sp_cst_sd=np.zeros(n_epoch)
sp_acc_mn=np.zeros(n_epoch)
sp_acc_sd=np.zeros(n_epoch)
nm_cst_mn=np.zeros(n_epoch)
nm_cst_sd=np.zeros(n_epoch)
nm_acc_mn=np.zeros(n_epoch)
nm_acc_sd=np.zeros(n_epoch)

for ep_ind in range(n_epoch):
    vec_sp_cst=sp_cst[:,ep_ind]
    sp_cst_mn[ep_ind]=np.mean(vec_sp_cst)
    sp_cst_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_sp_cst))**0.5
    vec_sp_acc=sp_acc[:,ep_ind]
    sp_acc_mn[ep_ind]=np.mean(vec_sp_acc)
    sp_acc_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_sp_acc))**0.5
    vec_nm_cst=nm_cst[:,ep_ind]
    nm_cst_mn[ep_ind]=np.mean(vec_nm_cst)
    nm_cst_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_nm_cst))**0.5
    vec_nm_acc=nm_acc[:,ep_ind]
    nm_acc_mn[ep_ind]=np.mean(vec_nm_acc)
    nm_acc_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_nm_acc))**0.5




sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 1, figsize=(1.1811,2.36), sharex=True , sharey=False)
sns.despine      
axes[0].plot(range(n_epoch),sp_cst_mn,'tab:blue',range(n_epoch),nm_cst_mn,'tab:orange',linewidth=0.5)
axes[0].fill_between(range(n_epoch),sp_cst_mn-sp_cst_sd,sp_cst_mn+sp_cst_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[0].fill_between(range(n_epoch),nm_cst_mn-nm_cst_sd,nm_cst_mn+nm_cst_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_edgecolor('black')
axes[0].spines['bottom'].set_facecolor('black')
axes[0].spines['bottom'].set_linewidth(0.5)
axes[0].spines['left'].set_edgecolor('black')
axes[0].spines['left'].set_facecolor('black')
axes[0].spines['left'].set_linewidth(0.5)
axes[0].set_xticks([0,50,100])
axes[0].set_yticks([0.2,0.5,0.8],[0.2,0.5,0.8])
axes[0].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=1)
 

axes[1].plot(range(n_epoch),sp_acc_mn,'tab:blue',range(n_epoch),nm_acc_mn,'tab:orange',linewidth=0.5)
axes[1].fill_between(range(n_epoch),sp_acc_mn-sp_acc_sd,sp_acc_mn+sp_acc_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[1].fill_between(range(n_epoch),nm_acc_mn-nm_acc_sd,nm_acc_mn+nm_acc_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_edgecolor('black')
axes[1].spines['bottom'].set_facecolor('black')
axes[1].spines['bottom'].set_linewidth(0.5)
axes[1].spines['left'].set_edgecolor('black')
axes[1].spines['left'].set_facecolor('black')
axes[1].spines['left'].set_linewidth(0.5)
axes[1].set_yticks=[7000,8000,9000]
axes[1].tick_params(direction='out', length=2, width=1, labelsize=6 , pad=1)  

plt.savefig('fig/Fig_1_d_1.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)

#%%
# Panel D_2 30 neurons, fashion
sp_cst=np.load('sim_data/Fig_1_30_fashion_sp_cst.npy')
sp_acc=np.load('sim_data/Fig_1_30_fashion_sp_acc.npy')/10000
nm_cst=np.load('sim_data/Fig_1_30_fashion_nm_cst.npy')
nm_acc=np.load('sim_data/Fig_1_30_fashion_nm_acc.npy')/10000

n_rep=sp_cst.shape[0]
n_epoch=sp_cst.shape[1]
sp_cst_mn=np.zeros(n_epoch)
sp_cst_sd=np.zeros(n_epoch)
sp_acc_mn=np.zeros(n_epoch)
sp_acc_sd=np.zeros(n_epoch)
nm_cst_mn=np.zeros(n_epoch)
nm_cst_sd=np.zeros(n_epoch)
nm_acc_mn=np.zeros(n_epoch)
nm_acc_sd=np.zeros(n_epoch)

for ep_ind in range(n_epoch):
    vec_sp_cst=sp_cst[:,ep_ind]
    sp_cst_mn[ep_ind]=np.mean(vec_sp_cst)
    sp_cst_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_sp_cst))**0.5
    vec_sp_acc=sp_acc[:,ep_ind]
    sp_acc_mn[ep_ind]=np.mean(vec_sp_acc)
    sp_acc_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_sp_acc))**0.5
    vec_nm_cst=nm_cst[:,ep_ind]
    nm_cst_mn[ep_ind]=np.mean(vec_nm_cst)
    nm_cst_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_nm_cst))**0.5
    vec_nm_acc=nm_acc[:,ep_ind]
    nm_acc_mn[ep_ind]=np.mean(vec_nm_acc)
    nm_acc_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_nm_acc))**0.5


sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 1, figsize=(1.1811,2.36), sharex=True , sharey=False)
sns.despine      
axes[0].plot(range(n_epoch),sp_cst_mn,'tab:blue',range(n_epoch),nm_cst_mn,'tab:orange',linewidth=0.5)
axes[0].fill_between(range(n_epoch),sp_cst_mn-sp_cst_sd,sp_cst_mn+sp_cst_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[0].fill_between(range(n_epoch),nm_cst_mn-nm_cst_sd,nm_cst_mn+nm_cst_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_edgecolor('black')
axes[0].spines['bottom'].set_facecolor('black')
axes[0].spines['bottom'].set_linewidth(0.5)
axes[0].spines['left'].set_edgecolor('black')
axes[0].spines['left'].set_facecolor('black')
axes[0].spines['left'].set_linewidth(0.5)
axes[0].set_xticks([0,50,100])
axes[0].set_yticks([0.2,0.5,0.8],[0.2,0.5,0.8])
axes[0].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=1)
 

axes[1].plot(range(n_epoch),sp_acc_mn,'tab:blue',range(n_epoch),nm_acc_mn,'tab:orange',linewidth=0.5)
axes[1].fill_between(range(n_epoch),sp_acc_mn-sp_acc_sd,sp_acc_mn+sp_acc_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[1].fill_between(range(n_epoch),nm_acc_mn-nm_acc_sd,nm_acc_mn+nm_acc_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_edgecolor('black')
axes[1].spines['bottom'].set_facecolor('black')
axes[1].spines['bottom'].set_linewidth(0.5)
axes[1].spines['left'].set_edgecolor('black')
axes[1].spines['left'].set_facecolor('black')
axes[1].spines['left'].set_linewidth(0.5)
axes[1].set_yticks=[7000,8000,9000]
axes[1].tick_params(direction='out', length=2, width=1, labelsize=6 , pad=1)  

plt.savefig('fig/Fig_1_d_2.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)


#%%
# Panel D_3 100 neurons, digits
sp_cst=np.load('sim_data/Fig_1_100_digits_sp_cst.npy')
sp_acc=np.load('sim_data/Fig_1_100_digits_sp_acc.npy')/10000
nm_cst=np.load('sim_data/Fig_1_100_digits_nm_cst.npy')
nm_acc=np.load('sim_data/Fig_1_100_digits_nm_acc.npy')/10000

n_rep=sp_cst.shape[0]
n_epoch=sp_cst.shape[1]
sp_cst_mn=np.zeros(n_epoch)
sp_cst_sd=np.zeros(n_epoch)
sp_acc_mn=np.zeros(n_epoch)
sp_acc_sd=np.zeros(n_epoch)
nm_cst_mn=np.zeros(n_epoch)
nm_cst_sd=np.zeros(n_epoch)
nm_acc_mn=np.zeros(n_epoch)
nm_acc_sd=np.zeros(n_epoch)

for ep_ind in range(n_epoch):
    vec_sp_cst=sp_cst[:,ep_ind]
    sp_cst_mn[ep_ind]=np.mean(vec_sp_cst)
    sp_cst_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_sp_cst))**0.5
    vec_sp_acc=sp_acc[:,ep_ind]
    sp_acc_mn[ep_ind]=np.mean(vec_sp_acc)
    sp_acc_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_sp_acc))**0.5
    vec_nm_cst=nm_cst[:,ep_ind]
    nm_cst_mn[ep_ind]=np.mean(vec_nm_cst)
    nm_cst_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_nm_cst))**0.5
    vec_nm_acc=nm_acc[:,ep_ind]
    nm_acc_mn[ep_ind]=np.mean(vec_nm_acc)
    nm_acc_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_nm_acc))**0.5




sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 1, figsize=(1.1811,2.36), sharex=True , sharey=False)
sns.despine      
axes[0].plot(range(n_epoch),sp_cst_mn,'tab:blue',range(n_epoch),nm_cst_mn,'tab:orange',linewidth=0.5)
axes[0].fill_between(range(n_epoch),sp_cst_mn-sp_cst_sd,sp_cst_mn+sp_cst_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[0].fill_between(range(n_epoch),nm_cst_mn-nm_cst_sd,nm_cst_mn+nm_cst_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_edgecolor('black')
axes[0].spines['bottom'].set_facecolor('black')
axes[0].spines['bottom'].set_linewidth(0.5)
axes[0].spines['left'].set_edgecolor('black')
axes[0].spines['left'].set_facecolor('black')
axes[0].spines['left'].set_linewidth(0.5)
axes[0].set_xticks([0,50,100])
axes[0].set_yticks([0.2,0.5,0.8],[0.2,0.5,0.8])
axes[0].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=1)
 

axes[1].plot(range(n_epoch),sp_acc_mn,'tab:blue',range(n_epoch),nm_acc_mn,'tab:orange',linewidth=0.5)
axes[1].fill_between(range(n_epoch),sp_acc_mn-sp_acc_sd,sp_acc_mn+sp_acc_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[1].fill_between(range(n_epoch),nm_acc_mn-nm_acc_sd,nm_acc_mn+nm_acc_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_edgecolor('black')
axes[1].spines['bottom'].set_facecolor('black')
axes[1].spines['bottom'].set_linewidth(0.5)
axes[1].spines['left'].set_edgecolor('black')
axes[1].spines['left'].set_facecolor('black')
axes[1].spines['left'].set_linewidth(0.5)
axes[1].set_yticks=[7000,8000,9000]
axes[1].tick_params(direction='out', length=2, width=1, labelsize=6 , pad=1)  

plt.savefig('fig/Fig_1_d_3.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)

#%%
# Panel D_4 100 neurons, fashion
sp_cst=np.load('sim_data/Fig_1_100_fashion_sp_cst.npy')
sp_acc=np.load('sim_data/Fig_1_100_fashion_sp_acc.npy')/10000
nm_cst=np.load('sim_data/Fig_1_100_fashion_nm_cst.npy')
nm_acc=np.load('sim_data/Fig_1_100_fashion_nm_acc.npy')/10000

n_rep=sp_cst.shape[0]
n_epoch=sp_cst.shape[1]
sp_cst_mn=np.zeros(n_epoch)
sp_cst_sd=np.zeros(n_epoch)
sp_acc_mn=np.zeros(n_epoch)
sp_acc_sd=np.zeros(n_epoch)
nm_cst_mn=np.zeros(n_epoch)
nm_cst_sd=np.zeros(n_epoch)
nm_acc_mn=np.zeros(n_epoch)
nm_acc_sd=np.zeros(n_epoch)

for ep_ind in range(n_epoch):
    vec_sp_cst=sp_cst[:,ep_ind]
    sp_cst_mn[ep_ind]=np.mean(vec_sp_cst)
    sp_cst_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_sp_cst))**0.5
    vec_sp_acc=sp_acc[:,ep_ind]
    sp_acc_mn[ep_ind]=np.mean(vec_sp_acc)
    sp_acc_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_sp_acc))**0.5
    vec_nm_cst=nm_cst[:,ep_ind]
    nm_cst_mn[ep_ind]=np.mean(vec_nm_cst)
    nm_cst_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_nm_cst))**0.5
    vec_nm_acc=nm_acc[:,ep_ind]
    nm_acc_mn[ep_ind]=np.mean(vec_nm_acc)
    nm_acc_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_nm_acc))**0.5




sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 1, figsize=(1.1811,2.36), sharex=True , sharey=False)
sns.despine      
axes[0].plot(range(n_epoch),sp_cst_mn,'tab:blue',range(n_epoch),nm_cst_mn,'tab:orange',linewidth=0.5)
axes[0].fill_between(range(n_epoch),sp_cst_mn-sp_cst_sd,sp_cst_mn+sp_cst_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[0].fill_between(range(n_epoch),nm_cst_mn-nm_cst_sd,nm_cst_mn+nm_cst_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_edgecolor('black')
axes[0].spines['bottom'].set_facecolor('black')
axes[0].spines['bottom'].set_linewidth(0.5)
axes[0].spines['left'].set_edgecolor('black')
axes[0].spines['left'].set_facecolor('black')
axes[0].spines['left'].set_linewidth(0.5)
axes[0].set_xticks([0,50,100])
axes[0].set_yticks([0.2,0.5,0.8],[0.2,0.5,0.8])
axes[0].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=1)
 

axes[1].plot(range(n_epoch),sp_acc_mn,'tab:blue',range(n_epoch),nm_acc_mn,'tab:orange',linewidth=0.5)
axes[1].fill_between(range(n_epoch),sp_acc_mn-sp_acc_sd,sp_acc_mn+sp_acc_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[1].fill_between(range(n_epoch),nm_acc_mn-nm_acc_sd,nm_acc_mn+nm_acc_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_edgecolor('black')
axes[1].spines['bottom'].set_facecolor('black')
axes[1].spines['bottom'].set_linewidth(0.5)
axes[1].spines['left'].set_edgecolor('black')
axes[1].spines['left'].set_facecolor('black')
axes[1].spines['left'].set_linewidth(0.5)
axes[1].set_yticks=[7000,8000,9000]
axes[1].tick_params(direction='out', length=2, width=1, labelsize=6 , pad=1)  

plt.savefig('fig/Fig_1_d_4.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)


#%%
# Panel D_5 300 neurons, digits
sp_cst=np.load('sim_data/Fig_1_300_digits_sp_cst.npy')
sp_acc=np.load('sim_data/Fig_1_300_digits_sp_acc.npy')/10000
nm_cst=np.load('sim_data/Fig_1_300_digits_nm_cst.npy')
nm_acc=np.load('sim_data/Fig_1_300_digits_nm_acc.npy')/10000

n_rep=sp_cst.shape[0]
n_epoch=sp_cst.shape[1]
sp_cst_mn=np.zeros(n_epoch)
sp_cst_sd=np.zeros(n_epoch)
sp_acc_mn=np.zeros(n_epoch)
sp_acc_sd=np.zeros(n_epoch)
nm_cst_mn=np.zeros(n_epoch)
nm_cst_sd=np.zeros(n_epoch)
nm_acc_mn=np.zeros(n_epoch)
nm_acc_sd=np.zeros(n_epoch)

for ep_ind in range(n_epoch):
    vec_sp_cst=sp_cst[:,ep_ind]
    sp_cst_mn[ep_ind]=np.mean(vec_sp_cst)
    sp_cst_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_sp_cst))**0.5
    vec_sp_acc=sp_acc[:,ep_ind]
    sp_acc_mn[ep_ind]=np.mean(vec_sp_acc)
    sp_acc_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_sp_acc))**0.5
    vec_nm_cst=nm_cst[:,ep_ind]
    nm_cst_mn[ep_ind]=np.mean(vec_nm_cst)
    nm_cst_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_nm_cst))**0.5
    vec_nm_acc=nm_acc[:,ep_ind]
    nm_acc_mn[ep_ind]=np.mean(vec_nm_acc)
    nm_acc_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_nm_acc))**0.5




sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 1, figsize=(1.1811,2.36), sharex=True , sharey=False)
sns.despine      
axes[0].plot(range(n_epoch),sp_cst_mn,'tab:blue',range(n_epoch),nm_cst_mn,'tab:orange',linewidth=0.5)
axes[0].fill_between(range(n_epoch),sp_cst_mn-sp_cst_sd,sp_cst_mn+sp_cst_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[0].fill_between(range(n_epoch),nm_cst_mn-nm_cst_sd,nm_cst_mn+nm_cst_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_edgecolor('black')
axes[0].spines['bottom'].set_facecolor('black')
axes[0].spines['bottom'].set_linewidth(0.5)
axes[0].spines['left'].set_edgecolor('black')
axes[0].spines['left'].set_facecolor('black')
axes[0].spines['left'].set_linewidth(0.5)
axes[0].set_xticks([0,50,100])
axes[0].set_yticks([0.2,0.5,0.8],[0.2,0.5,0.8])
axes[0].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=1)
 

axes[1].plot(range(n_epoch),sp_acc_mn,'tab:blue',range(n_epoch),nm_acc_mn,'tab:orange',linewidth=0.5)
axes[1].fill_between(range(n_epoch),sp_acc_mn-sp_acc_sd,sp_acc_mn+sp_acc_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[1].fill_between(range(n_epoch),nm_acc_mn-nm_acc_sd,nm_acc_mn+nm_acc_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_edgecolor('black')
axes[1].spines['bottom'].set_facecolor('black')
axes[1].spines['bottom'].set_linewidth(0.5)
axes[1].spines['left'].set_edgecolor('black')
axes[1].spines['left'].set_facecolor('black')
axes[1].spines['left'].set_linewidth(0.5)
axes[1].set_yticks=[7000,8000,9000]
axes[1].tick_params(direction='out', length=2, width=1, labelsize=6 , pad=1)  

plt.savefig('fig/Fig_1_d_5.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)

#%%
# Panel D_6 300 neurons, fashion
sp_cst=np.load('sim_data/Fig_1_300_fashion_sp_cst.npy')
sp_acc=np.load('sim_data/Fig_1_300_fashion_sp_acc.npy')/10000
nm_cst=np.load('sim_data/Fig_1_300_fashion_nm_cst.npy')
nm_acc=np.load('sim_data/Fig_1_300_fashion_nm_acc.npy')/10000

n_rep=sp_cst.shape[0]
n_epoch=sp_cst.shape[1]
sp_cst_mn=np.zeros(n_epoch)
sp_cst_sd=np.zeros(n_epoch)
sp_acc_mn=np.zeros(n_epoch)
sp_acc_sd=np.zeros(n_epoch)
nm_cst_mn=np.zeros(n_epoch)
nm_cst_sd=np.zeros(n_epoch)
nm_acc_mn=np.zeros(n_epoch)
nm_acc_sd=np.zeros(n_epoch)

for ep_ind in range(n_epoch):
    vec_sp_cst=sp_cst[:,ep_ind]
    sp_cst_mn[ep_ind]=np.mean(vec_sp_cst)
    sp_cst_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_sp_cst))**0.5
    vec_sp_acc=sp_acc[:,ep_ind]
    sp_acc_mn[ep_ind]=np.mean(vec_sp_acc)
    sp_acc_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_sp_acc))**0.5
    vec_nm_cst=nm_cst[:,ep_ind]
    nm_cst_mn[ep_ind]=np.mean(vec_nm_cst)
    nm_cst_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_nm_cst))**0.5
    vec_nm_acc=nm_acc[:,ep_ind]
    nm_acc_mn[ep_ind]=np.mean(vec_nm_acc)
    nm_acc_sd[ep_ind]=(n_rep/(n_rep-1)*np.var(vec_nm_acc))**0.5




sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 1, figsize=(1.1811,2.36), sharex=True , sharey=False)
sns.despine      
axes[0].plot(range(n_epoch),sp_cst_mn,'tab:blue',range(n_epoch),nm_cst_mn,'tab:orange',linewidth=0.5)
axes[0].fill_between(range(n_epoch),sp_cst_mn-sp_cst_sd,sp_cst_mn+sp_cst_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[0].fill_between(range(n_epoch),nm_cst_mn-nm_cst_sd,nm_cst_mn+nm_cst_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_edgecolor('black')
axes[0].spines['bottom'].set_facecolor('black')
axes[0].spines['bottom'].set_linewidth(0.5)
axes[0].spines['left'].set_edgecolor('black')
axes[0].spines['left'].set_facecolor('black')
axes[0].spines['left'].set_linewidth(0.5)
axes[0].set_xticks([0,50,100])
axes[0].set_yticks([0.2,0.5,0.8],[0.2,0.5,0.8])
axes[0].tick_params(direction='out', length=2, width=1 , labelsize=6 , pad=1)
 

axes[1].plot(range(n_epoch),sp_acc_mn,'tab:blue',range(n_epoch),nm_acc_mn,'tab:orange',linewidth=0.5)
axes[1].fill_between(range(n_epoch),sp_acc_mn-sp_acc_sd,sp_acc_mn+sp_acc_sd,facecolor='tab:blue',color=None,alpha=0.2)
axes[1].fill_between(range(n_epoch),nm_acc_mn-nm_acc_sd,nm_acc_mn+nm_acc_sd,facecolor='tab:orange',color=None,alpha=0.2)      
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_edgecolor('black')
axes[1].spines['bottom'].set_facecolor('black')
axes[1].spines['bottom'].set_linewidth(0.5)
axes[1].spines['left'].set_edgecolor('black')
axes[1].spines['left'].set_facecolor('black')
axes[1].spines['left'].set_linewidth(0.5)
axes[1].set_yticks=[7000,8000,9000]
axes[1].tick_params(direction='out', length=2, width=1, labelsize=6 , pad=1)  

plt.savefig('fig/Fig_1_d_6.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)

