#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 21:31:45 2021

@author: Alex
"""
import scipy.io as spio
mat1 = spio.loadmat('/Users/Alex/Downloads/trees_learn_init.mat', squeeze_me=True)
mat2 = spio.loadmat('/Users/Alex/Downloads/trees_learn_fin.mat', squeeze_me=True)
proc_data=mat2['proc_data']
proc_init=mat1['proc_init']

res0=50
res1=50

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 1, figsize=(1.1811,1.1811,))
sns.despine      

A=proc_data['lengths'].mean()
B=proc_init['lengths'].mean()
w=np.mean(A)
v=np.mean(B)
A*=(v/w)
    
BINS0=np.linspace(0.9*min(A),max(A),res0+1);
BINS1=np.linspace(0.9*min(A),max(A),res1+1);
[A_dist,A_ends]=np.histogram(A,BINS0)
[B_dist,B_ends]=np.histogram(B,BINS1) 
A_mids=(A_ends[1:(res0+1)]+A_ends[0:res0])/2
B_mids=(B_ends[1:(res1+1)]+B_ends[0:res1])/2


axes[0].plot(A_mids,smooth(A_dist/np.trapz(A_dist,A_mids),5),'darkgreen',B_mids,B_dist/np.trapz(B_dist,B_mids),'lightgreen')#,edgecolor='none',width=0.2)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].spines['bottom'].set_edgecolor(None)
axes[0].spines['bottom'].set_facecolor('black')
axes[0].spines['bottom'].set_linewidth(1)
axes[0].spines['left'].set_edgecolor(None)
axes[0].spines['left'].set_facecolor('black')
axes[0].spines['left'].set_linewidth(1)
#axes[0].set_ylim((0,0.2)) 
#axes[0].set_yticks=[0,1]
#axes[0].set_xlim((0,50))
 #   axes[ward].set_xticks=[100,150,200]
axes[0].tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)

res0=50
res1=50

B=np.zeros(100000)
for ward in range(1000):
    grid=np.zeros((10000))
    inds=np.random.choice(10000,3000,replace=False)
    grid[inds]=1
    grid=grid.reshape((100,100))
    B[(ward*100):((ward+1)*100)]=np.count_nonzero(grid,1)
    
A=proc_data['degs'].mean()

    
BINS0=np.linspace(0,75,res0+1);
BINS1=np.linspace(0,50,res1+1);
[A_dist,A_ends]=np.histogram(A,BINS0)
[B_dist,B_ends]=np.histogram(B,BINS1) 
A_mids=(A_ends[1:(res0+1)]+A_ends[0:res0])/2
B_mids=(B_ends[1:(res1+1)]+B_ends[0:res1])/2


axes[1].plot(A_mids,smooth(A_dist/np.trapz(A_dist,A_mids),5),'darkgreen',B_mids,B_dist/np.trapz(B_dist,B_mids),'lightgreen')#,edgecolor='none',width=0.2)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].spines['bottom'].set_edgecolor(None)
axes[1].spines['bottom'].set_facecolor('black')
axes[1].spines['bottom'].set_linewidth(1)
axes[1].spines['left'].set_edgecolor(None)
axes[1].spines['left'].set_facecolor('black')
axes[1].spines['left'].set_linewidth(1)
axes[1].set_ylim((0,0.1)) 
#axes[0].set_yticks=[0,1]
axes[1].set_xlim((0,100))
 #   axes[ward].set_xticks=[100,150,200]
axes[1].tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)
plt.savefig('fig/Fig_5_e.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)



plt.savefig('fig/Fig_5_b.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)




#%%
res0=40
res1=40

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(1.1811,1.1811,), sharex=True , sharey=True)
sns.despine      

A=proc_data['locws'].mean()
B=proc_init['locws'].mean()
    
BINS0=np.linspace(0,4,res0+1);
BINS1=np.linspace(0,4,res1+1);
[A_dist,A_ends]=np.histogram(A,BINS0)
[B_dist,B_ends]=np.histogram(B,BINS1) 
A_mids=(A_ends[1:(res0+1)]+A_ends[0:res0])/2
B_mids=(B_ends[1:(res1+1)]+B_ends[0:res1])/2


axes.plot(A_mids,smooth(A_dist/np.trapz(A_dist,A_mids),5),'darkgreen',B_mids,B_dist/np.trapz(B_dist,B_mids),'lightgreen')#,edgecolor='none',width=0.2)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_edgecolor(None)
axes.spines['bottom'].set_facecolor('black')
axes.spines['bottom'].set_linewidth(1)
axes.spines['left'].set_edgecolor(None)
axes.spines['left'].set_facecolor('black')
axes.spines['left'].set_linewidth(1)
axes.set_ylim((0,4)) 
#axes[0].set_yticks=[0,1]
axes.set_xlim((0,5))
 #   axes[ward].set_xticks=[100,150,200]
axes.tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)
plt.savefig('fig/Fig_5_c.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)

#%%
res0=50
res1=50

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(1.1811,1.1811,), sharex=True , sharey=True)
sns.despine      

A=proc_data['somws'].mean()
B=proc_init['somws'].mean()

    
BINS0=np.linspace(0,50,res0+1);
BINS1=np.linspace(0,50,res1+1);
[A_dist,A_ends]=np.histogram(A,BINS0)
[B_dist,B_ends]=np.histogram(B,BINS1) 
A_mids=(A_ends[1:(res0+1)]+A_ends[0:res0])/2
B_mids=(B_ends[1:(res1+1)]+B_ends[0:res1])/2


axes.plot(A_mids,A_dist/np.trapz(A_dist,A_mids),'darkgreen',B_mids,B_dist/np.trapz(B_dist,B_mids),'lightgreen')#,edgecolor='none',width=0.2)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_edgecolor(None)
axes.spines['bottom'].set_facecolor('black')
axes.spines['bottom'].set_linewidth(1)
axes.spines['left'].set_edgecolor(None)
axes.spines['left'].set_facecolor('black')
axes.spines['left'].set_linewidth(1)
axes.set_ylim((0,0.2)) 
#axes[0].set_yticks=[0,1]
axes.set_xlim((0,50))
 #   axes[ward].set_xticks=[100,150,200]
axes.tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)
plt.savefig('fig/Fig_5_d.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)



#%%
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(1, 1, figsize=(1.1811,1.1811,), sharex=True , sharey=True)
sns.despine      

#errors=np.load('sim_data/Fig_5_errs',errors)
accs=1-errors

means=np.mean(accs,1)
sds=np.sqrt(np.var(accs,1))


axes.errorbar(range(5),means,sds,color='darkgreen',lw=0.5,marker="o",ms=1)#,edgecolor='none',width=0.2)
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
axes.spines['bottom'].set_edgecolor(None)
axes.spines['bottom'].set_facecolor('black')
axes.spines['bottom'].set_linewidth(1)
axes.spines['left'].set_edgecolor(None)
axes.spines['left'].set_facecolor('black')
axes.spines['left'].set_linewidth(1)
axes.set_ylim((0,1.1)) 
#axes[0].set_yticks=[0,1]
axes.set_xlim((-0.5,5.5))
 #   axes[ward].set_xticks=[100,150,200]
axes.tick_params(direction='out', length=2, width=0.5 , labelsize=6 , pad=1)
plt.savefig('fig/Fig_5_e.pdf', format='pdf',bbox_inches='tight',pad_inches=0.25)