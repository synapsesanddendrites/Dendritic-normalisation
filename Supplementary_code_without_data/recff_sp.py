#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:38:40 2020

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt

BIN_DIM=1
INPUT_DIM = 28**2
HIDDEN_DIM = 60
OUTPUT_DIM = 1
sparsity=0.5
con_delt=0.1

lr = 0.001

epochs=5
train_per_epoch=10
test_per_epoch=1000


w1 = np.random.normal(0, 1, [HIDDEN_DIM, OUTPUT_DIM])
w0_mask=np.zeros((INPUT_DIM*HIDDEN_DIM))
wh_mask=np.zeros(HIDDEN_DIM*HIDDEN_DIM)
wh0_allowed=np.random.choice((INPUT_DIM+HIDDEN_DIM)*HIDDEN_DIM,round(sparsity*(INPUT_DIM+HIDDEN_DIM)*HIDDEN_DIM))
w0_mask[wh0_allowed[wh0_allowed<(INPUT_DIM*HIDDEN_DIM)]]=1
wh_mask[wh0_allowed[(wh0_allowed>=(INPUT_DIM*HIDDEN_DIM))]-(INPUT_DIM*HIDDEN_DIM)]=1
w0_mask=w0_mask.reshape([INPUT_DIM, HIDDEN_DIM])
wh_mask=wh_mask.reshape([HIDDEN_DIM, HIDDEN_DIM])
w0 = w0_mask*np.random.randn(INPUT_DIM, HIDDEN_DIM)
wh = wh_mask*np.random.randn(HIDDEN_DIM, HIDDEN_DIM)

d0 = np.zeros_like(w0)
d1 = np.zeros_like(w1)
dh = np.zeros_like(wh)

errs = list()
accs = list()


def sigmoid(x):
    x[x<0]=0
    return x

def deriv_sigmoid(out):
    out[out>=0]=1
    out[out<0]=0
    return out


for epoch in range(epochs):
    error=0
    for ward in range(train_per_epoch):
        inds=np.random.choice(60000,BIN_DIM)
        
        this_input=tr_images[inds,:]/256
        this_output=(tr_labels[inds]).cumsum()
        pred = np.zeros_like(this_output)
        
        overall_err = 0 # total error in the whole calculation process.
    
        output_deltas = list()
        hidden_values = list()
        hidden_values.append(np.zeros(HIDDEN_DIM))
    
        future_delta = np.zeros(HIDDEN_DIM)
        
        # forward propagation
        for pos in range(BIN_DIM)[::-1]:
            X = this_input[pos,:]
            Y = this_output[pos]
           
            hidden = sigmoid(np.dot(X, w0) + np.dot(hidden_values[-1], wh))
            output = np.dot(hidden,w1)
            
            pred[-pos]=np.round(output[0])
            #log-loss
            output_err = abs(Y-output[0])
            output_deltas.append(output_err)
            hidden_values.append(hidden)
        
            overall_err += np.abs(output_err)
    
        # backpropagation through time
        for pos in range(BIN_DIM):
            X = this_input[pos,:]
        
            hidden = hidden_values[-(pos + 1)]
            prev_hidden = hidden_values[-(pos + 2)]
        
            output_delta = output_deltas[-(pos + 1)]
            hidden_delta = (np.dot(future_delta, wh.T) + np.dot(output_delta, w1.T)) * deriv_sigmoid(hidden)
        
            d1 += np.dot(np.atleast_2d(hidden).T, output_delta.reshape(1,OUTPUT_DIM))
            dh += np.dot(np.atleast_2d(prev_hidden).T, hidden_delta.reshape(1,HIDDEN_DIM))
            d0 += np.dot(X.reshape(INPUT_DIM,1), hidden_delta.reshape(1,HIDDEN_DIM))

            future_delta = hidden_delta 
    
        w1 += lr * d1
        w0 += lr * w0_mask * d0
        wh += lr * wh_mask * dh
    
        d1 *= 0
        d0 *= 0
        dh *= 0
    
        error += overall_err
        
    accuracy=0
    error=0
    for ward in range(test_per_epoch):
        inds=np.random.choice(60000,BIN_DIM)
        
        this_input=tr_images[inds,:]/256
        this_output=(tr_labels[inds]).cumsum()
        pred = np.cumsum(this_output)
    
        hidden_values = list()
        hidden_values.append(np.zeros(HIDDEN_DIM))
        # forward propagation
        for pos in range(BIN_DIM)[::-1]:
            X = this_input[pos,:]
            Y = this_output[pos]
           
            hidden = sigmoid(np.dot(X, w0) + np.dot(hidden_values[-1], wh))
            output = np.dot(hidden,w1)
            
            pred[pos]=np.round(output[0])
            #log-loss
            output_err = abs(Y-output[0])
            output_deltas.append(output_err)
            hidden_values.append(hidden)

        if this_output[BIN_DIM-1] == pred[0]:
            accuracy+=1
            error+=output_err
    print('Epoch: '+str(epoch)+', Validation accuracy: '+str(accuracy/test_per_epoch)+', Error: '+str(error))
    errs.append(error)
    accs.append(accuracy/test_per_epoch)
    
    # Do SET 
    totsz=np.size(wh)+np.size(w0)
    n_con=np.count_nonzero(wh)+np.count_nonzero(w0)
    n_change=round(n_con*con_delt)
    w_vals=np.append(w0[np.nonzero(w0)],wh[np.nonzero(wh)])
    w_abs=np.sort(np.abs(w_vals))
    w_thresh=w_abs[n_change]
    wh[np.abs(wh)<w_thresh]=0
    wh_mask[np.abs(wh)<w_thresh]=0
         
                            
    if epoch<epochs: # Add random new weights if not last run
        n_con_2=np.count_nonzero(wh)+np.count_nonzero(w0)
        n_change=n_con-n_con_2
                    
        pos_locsh=np.nonzero(wh==0)
        posh_rws=pos_locsh[0]
        posh_cls=pos_locsh[1]
        
        pos_locs0=np.nonzero(w0==0)
        pos0_rws=pos_locs0[0]
        pos0_cls=pos_locs0[1]
        
        new_ws=np.random.randn(n_change)
        new_inds=np.random.choice(np.size(posh_rws)+np.size(pos0_rws),n_change,replace=False)
                
        wh[posh_rws[new_inds[new_inds<np.size(posh_rws)]],posh_cls[new_inds[new_inds<np.size(posh_rws)]]]=new_ws[new_inds<np.size(posh_rws)]
        wh_mask[posh_rws[new_inds[new_inds<np.size(posh_rws)]],posh_cls[new_inds[new_inds<np.size(posh_rws)]]]=1
        
        w0[pos0_rws[new_inds[new_inds>=np.size(posh_rws)]-np.size(posh_rws)],pos0_cls[new_inds[new_inds>=np.size(posh_rws)]-np.size(posh_rws)]]=new_ws[new_inds>=np.size(posh_rws)]
        w0_mask[pos0_rws[new_inds[new_inds>=np.size(posh_rws)]-np.size(posh_rws)],pos0_cls[new_inds[new_inds>=np.size(posh_rws)]-np.size(posh_rws)]]=1
        