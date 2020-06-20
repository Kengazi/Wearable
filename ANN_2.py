#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 08:23:59 2019

@author: kendalljohnson
"""

# Using Artificial Nueral Networks to Determine if there was a stimulas or not
print('Using ANN')

# Imports :: 

import numpy as np
np.random.seed(7)
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Data  ::

df = pd.read_excel('Full_Train.xlsx') 
HR = df.values[:,0:7]
Stim = df.values[:,8]

# Varibles :: 

inputs = HR
output = Stim

# ANN model ::

model = Sequential()
model.add(Dense(10, input_dim=7,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='softmax'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) # other optimizer : adam and sgd
model.fit(inputs,output, epochs=1, batch_size=16)
print(model.summary())



#model.train_on_batch(x_batch, y_batch)

#classes = model.predict(x_test, batch_size=128)


#model.fit(x_train, y_train, epochs=5)

# Predictions
#xtest.shape
#model.predict(x_test)
#np.argmax(yp[]) # amount of index
#model.evaluate(x_test,y_test)  # [loss , accuracy ]

