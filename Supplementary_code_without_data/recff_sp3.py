#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:38:40 2020

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import optimizers




(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape(-1,28, 28, 1)   #Reshape for CNN -  should work!!
x_test = x_test.reshape(-1,28, 28, 1)

tr_all_inds=np.arange(0,60000)
tr_zero_inds=tr_all_inds[y_train==0]
tr_one_inds=tr_all_inds[y_train==1]

tst_all_inds=np.arange(0,10000)
tst_zero_inds=tst_all_inds[y_test==0]
tst_one_inds=tst_all_inds[y_test==1]

tr_all=np.concatenate((tr_one_inds,tr_zero_inds),0)
tst_all=np.concatenate((tst_one_inds,tst_zero_inds),0)

x_train=x_train[tr_all]
y_train=y_train[tr_all]
x_test=x_test[tst_all]
y_test=y_test[tst_all]

pre_model = Sequential()
pre_model.add(Flatten(input_shape=(28,28,1)))
pre_model.add(Dense(30,activation='sigmoid'))
pre_model.add(Dense(2, activation='softmax'))
sgd = optimizers.SGD(0.05, momentum=0,nesterov=False)
pre_model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist=pre_model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=10,verbose=0,epochs=100)

W=pre_model.get_weights()

#%%
BIN_DIM = 8
INPUT_DIM = 30
HIDDEN_DIM = 50
OUTPUT_DIM = 1
sparsity=0.3
con_delt=0.15

lr = 0.25

epochs=100
train_per_epoch=1000
test_per_epoch=1000


largest = pow(2, BIN_DIM)
decimal = np.array([range(largest)]).astype(np.uint8).T
binary = np.unpackbits(decimal.view(np.uint8),axis=1)


w0_allowed_a = np.random.choice(HIDDEN_DIM,int(2*np.floor(sparsity*HIDDEN_DIM/2)))
w0_a = np.zeros([30,HIDDEN_DIM])
for ward in range(int(np.floor(sparsity*HIDDEN_DIM/2))):
    w0_a[:,w0_allowed_a[ward]]=W[2][:,0]
    w0_a[:,w0_allowed_a[ward+int(np.floor(sparsity*HIDDEN_DIM/2))]]=W[2][:,1]

w0_allowed_b = np.random.choice(HIDDEN_DIM,int(2*np.floor(sparsity*HIDDEN_DIM/2)))
w0_b = np.zeros([30,HIDDEN_DIM])
for ward in range(int(np.floor(sparsity*HIDDEN_DIM/2))):
    w0_b[:,w0_allowed_a[ward]]=W[2][:,0]
    w0_b[:,w0_allowed_a[ward+int(np.floor(sparsity*HIDDEN_DIM/2))]]=W[2][:,1]

              
w1 = np.random.normal(0, 1, [HIDDEN_DIM, OUTPUT_DIM])
wh_mask=np.zeros(HIDDEN_DIM**2)
wh_allowed=np.random.choice(HIDDEN_DIM**2,round(sparsity*HIDDEN_DIM**2))
wh_mask[wh_allowed]=1
wh_mask=wh_mask.reshape([HIDDEN_DIM, HIDDEN_DIM])
wh = wh_mask*np.random.randn(HIDDEN_DIM, HIDDEN_DIM)

d0a = np.zeros_like(w0_a)
d0b = np.zeros_like(w0_b)
d1 = np.zeros_like(w1)
dh = np.zeros_like(wh)

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
        
        a_input=np.zeros([30,BIN_DIM])
        b_input=np.zeros([30,BIN_DIM])
        for sooth in range(BIN_DIM):
            if a_bin[sooth]==0:
                this_ind=int(np.random.choice(zero_inds))
                a_input[:,sooth]=sigmoid(np.dot(tr_images[this_ind,:]/255,W[0])+W[1])
            else:
                this_ind=int(np.random.choice(one_inds))
                a_input[:,sooth]=sigmoid(np.dot(tr_images[this_ind,:]/255,W[0])+W[1])
            if b_bin[sooth]==0:
                this_ind=int(np.random.choice(zero_inds))
                b_input[:,sooth]=sigmoid(np.dot(tr_images[this_ind,:]/255,W[0])+W[1])
            else:
                this_ind=int(np.random.choice(one_inds))
                b_input[:,sooth]=sigmoid(np.dot(tr_images[this_ind,:]/255,W[0])+W[1])
            
        pred = np.zeros_like(c_bin)
    
        overall_err = 0 # total error in the whole calculation process.
    
        output_deltas = list()
        hidden_values = list()
        hidden_values.append(np.zeros(HIDDEN_DIM))
    
        future_delta = np.zeros((1,HIDDEN_DIM))
    
        # forward propagation
        for pos in range(BIN_DIM)[::-1]:
            Xa = a_input[:,pos]
            Xb = b_input[:,pos]
            Y = np.array([[c_bin[pos]]]) # shape=(1, 1)
        
            hidden = sigmoid(np.dot(Xa, w0_a) + np.dot(Xb, w0_b) + np.dot(hidden_values[-1], wh))
            output = sigmoid(np.dot(hidden, w1))
        
            pred[pos] = np.round(output)
        
            # squared mean error
            output_err = Y - output
            output_deltas.append(output_err * deriv_sigmoid(output))
            hidden_values.append(hidden)
            overall_err += np.abs(output_err[0])
    
        # backpropagation through time
        for pos in range(BIN_DIM):
            Xa = a_input[:,pos]
            Xb = b_input[:,pos]
            
            hidden = hidden_values[-(pos + 1)]
            prev_hidden = hidden_values[-(pos + 2)]
        
            output_delta = output_deltas[-(pos + 1)]
            hidden_delta = (np.dot(future_delta, wh.T) + np.dot(output_delta, w1.T)) * deriv_sigmoid(hidden)
            input_a_delta = (np.dot(hidden_delta, w0_a.T)) * deriv_sigmoid(Xa)
            input_b_delta = (np.dot(hidden_delta, w0_b.T)) * deriv_sigmoid(Xb)
            
            d1 += np.dot(np.atleast_2d(hidden).T, output_delta)
            dh += np.dot(np.atleast_2d(prev_hidden).T, hidden_delta)
           
            d0a += np.dot(Xa.T, input_a_delta.T)
            d0b += np.dot(Xb.T, input_b_delta.T)
            
            future_delta = hidden_delta 

        w1 += lr * d1
        w0_a += lr * d0a
        w0_b += lr * d0b
        wh += lr * wh_mask * dh
        
        d1 *= 0
        d0a *= 0
        d0b *= 0
        dh *= 0
    
        error += overall_err
        
    accuracy=0
    for ward in range(test_per_epoch):
        hidden_values = list()
        hidden_values.append(np.zeros(HIDDEN_DIM))
    
        future_delta = np.zeros((1,HIDDEN_DIM))
    
        # forward propagation
        for pos in range(BIN_DIM)[::-1]:
            Xa = a_input[:,pos]
            Xb = b_input[:,pos]
            Y = np.array([[c_bin[pos]]]) # shape=(1, 1)
        
            hidden = sigmoid(np.dot(Xa, w0_a) + np.dot(Xb, w0_b) + np.dot(hidden_values[-1], wh))
            output = sigmoid(np.dot(hidden, w1))
        
            pred[pos] = np.round(output)
        
            # squared mean error
            output_err = Y - output
            output_deltas.append(output_err * deriv_sigmoid(output))
            hidden_values.append(hidden)
            overall_err += np.abs(output_err[0])
            
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
        