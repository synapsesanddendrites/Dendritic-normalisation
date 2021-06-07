#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:38:40 2020

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt


BIN_DIM=8
INPUT_DIM = 2*28**2
HIDDEN_DIM = 30
RECURRENT_DIM = 50
OUTPUT_DIM = 1
sparsity=0.3
con_delt=0.15


w2 = np.random.normal(0, 1, [RECURRENT_DIM, OUTPUT_DIM])
w0 = np.random.normal(0, 1, [INPUT_DIM, HIDDEN_DIM])
w1_mask=np.zeros((HIDDEN_DIM*RECURRENT_DIM))
wh_mask=np.zeros((RECURRENT_DIM*RECURRENT_DIM))

wh1_allowed=np.random.choice((HIDDEN_DIM+RECURRENT_DIM)*RECURRENT_DIM,round(sparsity*(HIDDEN_DIM+RECURRENT_DIM)*RECURRENT_DIM))
w1_mask[wh1_allowed[wh1_allowed<(HIDDEN_DIM*RECURRENT_DIM)]]=1
wh_mask[wh1_allowed[(wh1_allowed>=(HIDDEN_DIM*RECURRENT_DIM))]-(HIDDEN_DIM*RECURRENT_DIM)]=1
w1_mask=w1_mask.reshape([HIDDEN_DIM, RECURRENT_DIM])
wh_mask=wh_mask.reshape([RECURRENT_DIM, RECURRENT_DIM])
wh = wh_mask*np.random.randn(RECURRENT_DIM, RECURRENT_DIM)
w1 = w1_mask*np.random.randn(HIDDEN_DIM, RECURRENT_DIM)

lr = 0.1

epochs=100
train_per_epoch=20000
test_per_epoch=5000


largest = pow(2, BIN_DIM)
decimal = np.array([range(largest)]).astype(np.uint8).T
binary = np.unpackbits(decimal.view(np.uint8),axis=1)

d0 = np.zeros_like(w0)
d1 = np.zeros_like(w1)
dh = np.zeros_like(wh)
d2 = np.zeros_like(w2)

errs = list()
accs = list()

r=np.linspace(0,59999,60000)
zero_inds=r[tr_labels==0]
one_inds=r[tr_labels==1]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(out):
    return out * (1 - out)

def bin2dec(b):
    out = 0
    for i, x in enumerate(b[::-1]):
        out += x * pow(2, i)
    
    return out


for epoch in range(epochs):
    error=0
    for ward in range(train_per_epoch):
        a_dec = np.random.randint(largest / 2)
        b_dec = np.random.randint(largest / 2)
        c_dec = a_dec + b_dec
    
        a_bin = binary[a_dec]
        b_bin = binary[b_dec]
        c_bin = binary[c_dec]
        
        a_input=np.zeros([28**2,BIN_DIM])
        b_input=np.zeros([28**2,BIN_DIM])
        for sooth in range(BIN_DIM):
            if a_bin[sooth]==0:
                this_ind=int(np.random.choice(zero_inds))
                a_input[:,sooth]=tr_images[this_ind,:]/255
            else:
                this_ind=int(np.random.choice(one_inds))
                a_input[:,sooth]=tr_images[this_ind,:]/255
            if b_bin[sooth]==0:
                this_ind=int(np.random.choice(zero_inds))
                b_input[:,sooth]=tr_images[this_ind,:]/255
            else:
                this_ind=int(np.random.choice(one_inds))
                b_input[:,sooth]=tr_images[this_ind,:]/255
            
        pred = np.zeros_like(c_bin)
    
        overall_err = 0 # total error in the whole calculation process.
    
        output_deltas = list()
        hidden_values = list()
        recurrent_values = list()
        recurrent_values.append(np.zeros(RECURRENT_DIM))
    
        future_delta = np.zeros((1,RECURRENT_DIM))
    
        # forward propagation
        for pos in range(BIN_DIM)[::-1]:
            X = np.append(a_input[:,pos], b_input[:,pos]).reshape(1,2*28**2) # shape=(1, 2)
            Y = np.array([[c_bin[pos]]]) # shape=(1, 1)
        
            hidden = sigmoid(np.dot(X, w0))
            recurrent = sigmoid(np.dot(hidden, w1) + np.dot(recurrent_values[-1], wh))
            output = sigmoid(np.dot(recurrent, w2))
        
            pred[pos] = np.round(output[0][0])
        
            # squared mean error
            output_err = Y - output
            output_deltas.append(output_err * deriv_sigmoid(output))
            hidden_values.append(hidden)
            recurrent_values.append(recurrent)
            overall_err += np.abs(output_err[0])
    
        # backpropagation through time
        for pos in range(BIN_DIM):
            X = np.append(a_input[:,pos], b_input[:,pos]).reshape(1,2*28**2) # shape=(1, 2)
        
            recurrent = recurrent_values[-(pos + 1)]
            hidden = hidden_values[-(pos + 1)]
            prev_recurrent = recurrent_values[-(pos + 2)]
        
            output_delta = output_deltas[-(pos + 1)]
            recurrent_delta = (np.dot(future_delta, wh.T) + np.dot(output_delta, w2.T)) * deriv_sigmoid(recurrent)
            hidden_delta = (np.dot(recurrent_delta, w1.T)) * deriv_sigmoid(hidden)
            input_delta = (np.dot(hidden_delta, w0.T)) * deriv_sigmoid(X)
        
            d2 += np.dot(np.atleast_2d(recurrent).T, output_delta)
            dh += np.dot(np.atleast_2d(prev_recurrent).T, recurrent_delta)
            d1 += np.dot(np.atleast_2d(hidden).T, recurrent_delta)
            d0 += np.dot(X.T, hidden_delta)

            future_delta = recurrent_delta 

        w2 += lr * d2
        w1 += lr * d1
        w0 += lr * d0
        wh += lr * wh_mask * dh
        
        d2 += 0
        d1 *= 0
        d0 *= 0
        dh *= 0
    
        error += overall_err
        
    accuracy=0
    for ward in range(test_per_epoch):
        a_dec = np.random.randint(largest / 2)
        b_dec = np.random.randint(largest / 2)
        c_dec = a_dec + b_dec

        a_bin = binary[a_dec]
        b_bin = binary[b_dec]
        c_bin = binary[c_dec]

        pred = np.zeros_like(c_bin)

        overall_err = 0 # total error in the whole calculation process.

        output_deltas = list()
        hidden_values = list()
        recurrent_values = list()
        recurrent_values.append(np.zeros(RECURRENT_DIM))

        future_delta = np.zeros(RECURRENT_DIM)

        # forward propagation
        for pos in range(BIN_DIM)[::-1]:
            X = np.append(a_input[:,pos], b_input[:,pos]).reshape(1,2*28**2) # shape=(1, 2)
            Y = np.array([[c_bin[pos]]]) # shape=(1, 1)
            
            hidden = sigmoid(np.dot(X, w0))
            recurrent = sigmoid(np.dot(hidden, w1) + np.dot(recurrent_values[-1], wh))
            output = sigmoid(np.dot(recurrent, w2))
        
            pred[pos] = np.round(output[0][0])
        
            # squared mean error
            output_err = Y - output
            output_deltas.append(output_err * deriv_sigmoid(output))
            hidden_values.append(hidden)
            recurrent_values.append(recurrent)
            overall_err += np.abs(output_err[0])
        if bin2dec(pred) == c_dec:
            accuracy+=1
    print('Epoch: '+str(epoch)+', Validation accuracy: '+str(accuracy/test_per_epoch)+', Error: '+str(error))
    errs.append(error)
    accs.append(accuracy/test_per_epoch)
    
    # Do SET 
    totsz=np.size(wh)
    n_con=np.count_nonzero(wh)
    n_change=round(n_con*con_delt)
    w_vals=wh[np.nonzero(wh)]
    w_abs=np.sort(np.abs(w_vals))
    w_thresh=w_abs[n_change]
    wh[np.abs(wh)<w_thresh]=0
    wh_mask[np.abs(wh)<w_thresh]=0
         
                            
    if epoch<epochs: # Add random new weights if not last run
        n_con_2=np.count_nonzero(wh)
        n_change=n_con-n_con_2
                    
        pos_locs=np.nonzero(wh==0)
        pos_rws=pos_locs[0]
        pos_cls=pos_locs[1]
                
        new_ws=np.random.randn(n_change)
        new_inds=np.random.choice(np.size(pos_rws),n_change,replace=False)
                
        wh[pos_rws[new_inds],pos_cls[new_inds]]=new_ws
        wh_mask[pos_rws[new_inds],pos_cls[new_inds]]=1
        