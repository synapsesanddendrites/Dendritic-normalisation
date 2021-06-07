#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:36:30 2021

@author: Alex
"""
def model_plot(trues_lst,outs,model,index=0):
    
    T=np.zeros(1000)
    P=np.zeros((1000,2))
    run_ind=0
    for ward in range(1000):
        U=trues_lst[ward][index]
        Q1=outs[model][ward][index,0]
        Q2=outs[model][ward][index,0]
        if Q1>0:
            T[run_ind]=U
            P[run_ind,0]=Q1
            P[run_ind,1]=Q2
            run_ind+=1
    plt.scatter(T[0:run_ind],P[0:run_ind,0])
    