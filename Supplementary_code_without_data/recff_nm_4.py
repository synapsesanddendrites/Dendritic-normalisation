#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:38:40 2020

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
import copy


lr = 0.25
epochs=100


def random_nm_weights():
    BIN_DIM = 50
    INPUT_DIM = 2
    HIDDEN_DIM = 50
    OUTPUT_DIM = 1
    sparsity=0.3
    con_delt=0.15
    s0=100
    sh=10
    train_per_epoch=1000
    test_per_epoch=1000
    largest = pow(2, BIN_DIM)

    w1 = np.random.normal(0, 1, [HIDDEN_DIM, OUTPUT_DIM])
    wh_mask=np.zeros(HIDDEN_DIM**2)
    w0_mask=np.zeros(INPUT_DIM*HIDDEN_DIM)
    wh0_allowed=np.random.choice((HIDDEN_DIM+INPUT_DIM)*HIDDEN_DIM,round(sparsity*(HIDDEN_DIM+INPUT_DIM)*HIDDEN_DIM),replace=False)
    w0_mask[wh0_allowed[wh0_allowed<(INPUT_DIM*HIDDEN_DIM)]]=1
    wh_mask[wh0_allowed[(wh0_allowed>=(HIDDEN_DIM*INPUT_DIM))]-(HIDDEN_DIM*INPUT_DIM)]=1
    w0_mask=w0_mask.reshape([INPUT_DIM, HIDDEN_DIM])
    wh_mask=wh_mask.reshape([HIDDEN_DIM, HIDDEN_DIM])
    
    v0 =w0_mask*np.random.randn(INPUT_DIM, HIDDEN_DIM)
    vh = wh_mask*np.random.randn(HIDDEN_DIM, HIDDEN_DIM)

    n_aff=np.count_nonzero(vh,0)+np.count_nonzero(v0,0)
    mrph_nm_0=np.tile(1/n_aff,[INPUT_DIM,1])
    mrph_nm_h=np.tile(1/n_aff,[HIDDEN_DIM,1])

    w0=s0*v0*mrph_nm_0
    w0[np.isnan(w0)]=0
    wh=sh*vh*mrph_nm_h
    wh[np.isnan(wh)]=0

    in_state=dict()
    in_state['v0']=v0#nm_init['v0']
    in_state['vh']=vh#nm_init['vh']
    in_state['w1']=w1#nm_init['w1']
    in_state['s0']=s0#nm_init['s0']
    in_state['sh']=sh#nm_init['sh']
    in_state['train_per_epoch']=train_per_epoch
    in_state['test_per_epoch']=test_per_epoch   
    in_state['BIN_DIM']=BIN_DIM
    in_state['INPUT_DIM']=INPUT_DIM
    in_state['HIDDEN_DIM']=HIDDEN_DIM
    in_state['OUTPUT_DIM']=OUTPUT_DIM
    in_state['sparsity']=sparsity
    in_state['con_delt']=con_delt
    in_state['lr']=lr
    in_state['largest']=largest
    in_state['not_last']=True
    return in_state

errs_nm = list()
accs_nm = list()
weights_nm=list()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(out):
    return out * (1 - out)

def binary(input_num,BIN_DIM):
    string = np.binary_repr(input_num)
    out_vec=np.zeros(BIN_DIM)
    ind=BIN_DIM-1
    for ward in string[::-1]:
        out_vec[ind]=int(ward)
        ind-=1
    return out_vec

def bin2dec(b):
    out = 0
    for i, x in enumerate(b[::-1]):
        out += x * pow(2, i)
    return out


def train_one_epoch(in_state,epoch=0,n_rep=0):
    BIN_DIM=in_state['BIN_DIM']
    INPUT_DIM=in_state['INPUT_DIM']
    HIDDEN_DIM=in_state['HIDDEN_DIM']
    OUTPUT_DIM=in_state['OUTPUT_DIM']
    sparsity=in_state['sparsity']
    con_delt=in_state['con_delt']
    lr=in_state['lr']
    largest=in_state['largest']
    
    vh=in_state['vh']
    v0=in_state['v0']
    w1=in_state['w1']
    s0=in_state['s0']
    sh=in_state['sh']
    
    w0_mask=np.zeros([INPUT_DIM,HIDDEN_DIM])
    wh_mask=np.zeros([HIDDEN_DIM,HIDDEN_DIM])
    w0_mask[v0!=0]=1
    wh_mask[vh!=0]=1
    
    train_per_epoch=in_state['train_per_epoch']
    test_per_epoch=in_state['test_per_epoch']
    
    n_aff=np.count_nonzero(vh,0)+np.count_nonzero(v0,0)
    mrph_nm_0=np.tile(1/n_aff,[INPUT_DIM,1])
    mrph_nm_h=np.tile(1/n_aff,[HIDDEN_DIM,1])

    w0=s0*v0*mrph_nm_0
    w0[np.isnan(w0)]=0
    wh=sh*vh*mrph_nm_h
    wh[np.isnan(wh)]=0


    if n_rep>0:
        new_inds_0=in_state['new_inds_0']
        new_inds_h=in_state['new_inds_h']
        
        w0=w0.reshape(INPUT_DIM*HIDDEN_DIM)
        w0[new_inds_0]=0
        w0_mask=w0_mask.reshape(INPUT_DIM*HIDDEN_DIM)
        w0_mask[new_inds_0]=0
        wh=wh.reshape(HIDDEN_DIM**2)
        wh[new_inds_h]=0
        wh_mask=wh_mask.reshape(HIDDEN_DIM**2)
        wh_mask[new_inds_h]=0
        
        n_change=len(new_inds_h)+len(new_inds_0)
        pos_locs_h=np.nonzero(wh_mask==0)
        pos_locs_0=np.nonzero(w0_mask==0)
        
        new_inds_all=np.random.choice(np.size(pos_locs_h)+np.size(pos_locs_0),n_change,replace=False)
        new_inds_h=new_inds_all[new_inds_all<np.size(pos_locs_h)]
        new_inds_0=new_inds_all[new_inds_all>=np.size(pos_locs_h)]-np.size(pos_locs_h)

        wh[new_inds_h]=np.random.randn(len(new_inds_h))/sh                                                                    
        wh_mask[new_inds_h]=1
        
        w0[new_inds_0]=np.random.randn(len(new_inds_0))/s0                                                                      
        w0_mask[new_inds_0]=1
        
        w0_mask=w0_mask.reshape([INPUT_DIM, HIDDEN_DIM])
        w0=w0.reshape([INPUT_DIM, HIDDEN_DIM])
        wh_mask=wh_mask.reshape([HIDDEN_DIM, HIDDEN_DIM])
        wh=wh.reshape([HIDDEN_DIM, HIDDEN_DIM])
        
        n_aff=np.count_nonzero(vh,0)+np.count_nonzero(v0,0)
        mrph_nm_0=np.tile(1/n_aff,[INPUT_DIM,1])
        mrph_nm_h=np.tile(1/n_aff,[HIDDEN_DIM,1])

        v0=w0/(mrph_nm_0*s0)
        v0[np.isnan(w0)]=0
        vh=wh/(mrph_nm_h*sh)
        vh[np.isnan(wh)]=0
    
    d0_raw = np.zeros_like(w0)
    d1 = np.zeros_like(w1)
    dh_raw = np.zeros_like(wh)
    error=0
    for ward in range(train_per_epoch):
        a_dec = np.random.randint(largest / 2)
        b_dec = np.random.randint(largest / 2)
        c_dec = a_dec + b_dec
    
        a_bin = binary(a_dec,BIN_DIM)
        b_bin = binary(b_dec,BIN_DIM)
        c_bin = binary(c_dec,BIN_DIM)
    
        pred = np.zeros_like(c_bin)
    
        overall_err = 0 # total error in the whole calculation process.
    
        output_deltas = list()
        hidden_values = list()
        hidden_values.append(np.zeros(HIDDEN_DIM))
    
        future_delta = np.zeros(HIDDEN_DIM)
    
        # forward propagation
        for pos in range(BIN_DIM)[::-1]:
            X = np.array([[a_bin[pos], b_bin[pos]]]) # shape=(1, 2)
            Y = np.array([[c_bin[pos]]]) # shape=(1, 1)
        
            hidden = sigmoid(np.dot(X, w0) + np.dot(hidden_values[-1], wh))
            output = sigmoid(np.dot(hidden, w1))
        
            pred[pos] = np.round(output[0][0])
        
            # squared mean error
            output_err = Y - output
            output_deltas.append(output_err * deriv_sigmoid(output))
            hidden_values.append(hidden)
        
            overall_err += np.abs(output_err[0])
    
        # backpropagation through time
        for pos in range(BIN_DIM):
            X = np.array([[a_bin[pos], b_bin[pos]]])
        
            hidden = hidden_values[-(pos + 1)]
            prev_hidden = hidden_values[-(pos + 2)]
        
            output_delta = output_deltas[-(pos + 1)]
            hidden_delta = (np.dot(future_delta, wh.T) + np.dot(output_delta, w1.T)) * deriv_sigmoid(hidden)
        
            d1 += np.dot(np.atleast_2d(hidden).T, output_delta)
            dh_raw += np.dot(np.atleast_2d(prev_hidden).T, hidden_delta)
            d0_raw += np.dot(X.T, hidden_delta)
            
            future_delta = hidden_delta 
         
        d0_raw=d0_raw * w0_mask 
        dh_raw=dh_raw * wh_mask 
        n_aff=np.count_nonzero(wh_mask,0)+np.count_nonzero(w0_mask,0)
        mrph_nm_0=np.tile(1/n_aff,[INPUT_DIM,1])
        mrph_nm_h=np.tile(1/n_aff,[HIDDEN_DIM,1])
        d0=s0*d0_raw*mrph_nm_0
        d0[np.isnan(d0)]=0
        dh=sh*dh_raw*mrph_nm_h
        dh[np.isnan(wh)]=0
        
        dsh=np.sum(np.dot(vh,dh.T))/sh
        ds0=np.sum(np.dot(v0,d0.T))/s0
        
        w1 += lr * d1
        v0 += lr * d0
        vh += lr * dh
        s0 += lr * ds0
        sh += lr * dsh
        
        #s0=100
        #V0=v0*sh/100
        
        n_aff=np.count_nonzero(vh,0)+np.count_nonzero(v0,0)
        mrph_nm_0=np.tile(1/n_aff,[INPUT_DIM,1])
        mrph_nm_h=np.tile(1/n_aff,[HIDDEN_DIM,1])
        w0=s0*v0*mrph_nm_0
        w0[np.isnan(w0)]=0
        wh=sh*vh*mrph_nm_h
        wh[np.isnan(wh)]=0
        
        d1 *= 0
        d0 *= 0
        dh_raw *= 0
    
        error += overall_err
    accuracy=0
    for ward in range(test_per_epoch):
        a_dec = np.random.randint(largest / 2)
        b_dec = np.random.randint(largest / 2)
        c_dec = a_dec + b_dec

        a_bin = binary(a_dec,BIN_DIM)
        b_bin = binary(b_dec,BIN_DIM)
        c_bin = binary(c_dec,BIN_DIM)

        pred = np.zeros_like(c_bin)

        overall_err = 0 # total error in the whole calculation process.

        output_deltas = list()
        hidden_values = list()
        hidden_values.append(np.zeros(HIDDEN_DIM))

        future_delta = np.zeros(HIDDEN_DIM)

        # forward propagation
        for pos in range(BIN_DIM)[::-1]:
            X = np.array([[a_bin[pos], b_bin[pos]]]) # shape=(1, 2)
            Y = np.array([[c_bin[pos]]]) # shape=(1, 1)
            
            hidden = sigmoid(np.dot(X, w0) + np.dot(hidden_values[-1], wh))
            output = sigmoid(np.dot(hidden, w1))
    
            pred[pos] = np.round(output[0][0])
    
        # squared mean error
            output_err = Y - output
            output_deltas.append(output_err * deriv_sigmoid(output))
            hidden_values.append(hidden)
    
            overall_err += np.abs(output_err[0])
        if bin2dec(pred) == c_dec:
            accuracy+=1
    
    print('Epoch: '+str(epoch)+', Validation accuracy: '+str(accuracy/test_per_epoch)+', Error: '+str(error))
    output_error=error
    output_accuracy=(accuracy/test_per_epoch)
            
    # Do SET 
    n_con=np.count_nonzero(wh)+np.count_nonzero(w0)
    n_change=round(n_con*con_delt)
    
    w_vals=np.append(wh[np.nonzero(wh)],w0[np.nonzero(w0)])
    w_abs=np.sort(np.abs(w_vals))
    w_thresh=w_abs[n_change]
        
    wh[np.abs(wh)<w_thresh]=0
    wh_mask[np.abs(wh)<w_thresh]=0
    w0[np.abs(w0)<w_thresh]=0
    w0_mask[np.abs(w0)<w_thresh]=0
        
                                
    if epoch<=epochs: # Add random new weights if not last run
        n_con_2=np.count_nonzero(wh)+np.count_nonzero(w0)
        n_change=np.round(sparsity*HIDDEN_DIM*(HIDDEN_DIM+INPUT_DIM))-n_con_2
                    
        pos_locs_h=np.nonzero(wh_mask==0)
        pos_locs_0=np.nonzero(w0_mask==0)
        
        pos_rws_h=pos_locs_h[0]
        pos_cls_h=pos_locs_h[1]
        pos_rws_0=pos_locs_0[0]
        pos_cls_0=pos_locs_0[1]
                
        
        new_inds=np.random.choice(np.size(pos_rws_h)+np.size(pos_rws_0),int(n_change),replace=False)
        new_inds_h=new_inds[new_inds<np.size(pos_rws_h)]
        new_inds_0=new_inds[new_inds>=np.size(pos_rws_h)]-np.size(pos_rws_h)

        wh[pos_rws_h[new_inds_h],pos_cls_h[new_inds_h]]=np.random.randn(len(new_inds_h))/sh                                                                  
        wh_mask[pos_rws_h[new_inds_h],pos_cls_h[new_inds_h]]=1
        
        w0[pos_rws_0[new_inds_0],pos_cls_0[new_inds_0]]=np.random.randn(len(new_inds_0))/s0                                                                
        w0_mask[pos_rws_0[new_inds_0],pos_cls_0[new_inds_0]]=1
        
        n_aff=np.count_nonzero(vh,0)+np.count_nonzero(v0,0)
        mrph_nm_0=np.tile(1/n_aff,[INPUT_DIM,1])
        mrph_nm_h=np.tile(1/n_aff,[HIDDEN_DIM,1])

        v0=w0/(mrph_nm_0*s0)
        v0[np.isnan(w0)]=0
        vh=wh/(mrph_nm_h*sh)
        vh[np.isnan(wh)]=0
    
    
    out_state=copy.deepcopy(in_state)
    out_state['vh']=vh
    out_state['v0']=v0
    out_state['w1']=w1
    out_state['s0']=s0
    out_state['sh']=sh
    out_state['new_inds_0']=new_inds_0
    out_state['new_inds_h']=new_inds_h
    out_list=[output_accuracy,output_error,out_state]
    return out_list

bad_start=True
while bad_start:
    in_state=random_nm_weights()
    out_list=train_one_epoch(in_state,0,0)
    if out_list[0]>0.6:
        errs_nm.append(out_list[1])
        accs_nm.append(out_list[0])
        weights_nm.append(out_list[2])
        in_state=out_list[2]
        bad_start=False

nm_fallback = pickle.load( open('sim_data/Fig_4_recff_nm_fallback.pkl', "rb" ) )
n_comp=len(nm_fallback)
epoch=1
while epoch<=epochs:
    not_better=True
    n_rep=0
    n_walk=0
    while not_better:
        out_list=train_one_epoch(in_state,epoch,n_rep)
        if out_list[0]>=(0.9*accs_nm[-1]):
            not_better=False
            epoch+=1
        else:
            n_rep+=1
            if n_rep<2:
                in_state=nm_fallback[int(np.floor(n_comp*np.random.rand()))][epoch]
            else:
                try:
                    sprd=int(np.sign(np.random.rand()-0.5)*np.floor(n_rep*np.random.rand()))
                    in_state=nm_fallback[int(np.floor(n_comp*np.random.rand()))][epoch+sprd]
                except Exception as error:
                    print(error)
                    
            
    
    errs_nm.append(out_list[1])
    accs_nm.append(out_list[0])
    if out_list[0]>0.8:
        weights_nm.append(out_list[2])
        in_state=out_list[2]
    else:
        weights_nm.append(nm_fallback[int(np.floor(n_comp*np.random.rand()))][epoch])
        in_state=nm_fallback[int(np.floor(n_comp*np.random.rand()))][epoch]
nm_fallback.append(weights_nm)
with open('sim_data/Fig_4_recff_nm_fallback.pkl', 'wb') as f:
    pickle.dump(nm_fallback, f)

