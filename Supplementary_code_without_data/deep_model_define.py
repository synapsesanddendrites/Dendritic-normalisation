#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:37:31 2020

@author: Alex
"""
if len(sp_masks)==0:
    for ward in range(3):
        sp_masks.append([])
#non_sparse=[]
model_nm = Sequential()
#model_nm.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(32,32,3)))        
#model_nm.add(MaxPooling2D(pool_size=(2, 2)))
#model_nm.add(Dropout(0.3))
#model_nm.add(Conv2D(filters=64, kernel_size=3, activation='relu'))     
#model_nm.add(MaxPooling2D(pool_size=(2, 2)))
#model_nm.add(Dropout(0.3))
#model_nm.add(Conv2D(filters=128, kernel_size=3, activation='relu'))    
#model_nm.add(MaxPooling2D(pool_size=(2, 2)))
#model_nm.add(Dropout(0.3))
model_nm.add(Flatten(input_shape=(32,32,3)))
#model_nm.add(Dropout(0.3)) 
model_nm.add(Morph_layer(1000,kernel_constraint=MaskWeights(sp_masks[0]),activation='relu'))
model_nm.add(Morph_layer(1000,kernel_constraint=MaskWeights(sp_masks[1]),activation='relu'))
model_nm.add(Morph_layer(1000,kernel_constraint=MaskWeights(sp_masks[2]),activation='relu'))
model_nm.add(Dense(10, activation='softmax'))