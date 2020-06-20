#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:41:21 2019

@author: kendalljohnson
"""
# SVM gra

# Imports ::

#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import math as m
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# Files ::
df = pd.read_excel('X_data.xlsx')
cnt = df.values[:,1]
HR = df.values[:,2]                 # Heart Rate in beats per minute
IBI = df.values[:,3]                # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus  
Pat_size = HR.size

# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

HR_df = pd.DataFrame(HR)
IBI_df = pd.DataFrame(IBI)

# Putting frames together 

frames = [HR_df, IBI_df]
Com_df = pd.concat(frames, axis=1)
df = pd.read_csv('IBI1.csv')               # Heart Rate in beats per minute
time2 = df.values[:,0]                # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus 
time2 = time2[:278] 
IBI = df.values[:,1]
IBI = IBI*1000
IBI = IBI[:278] 
#Pat_size = HR.size

# Clip data

Stim1t = time2[:112]
Stim1IBI = IBI[:112]
Stim2t = time2[113:156]
Stim2IBI = IBI[113:156]
Stim3t = time2[157:278]
Stim3IBI = IBI[157:278]



# Machince Learning :: Support Vector Machine  


X_train, X_test,y_train, y_test = train_test_split(inputs,df.Stim,test_size = 0.2)
model = SVC()
#model.fit(inputs,Stim) #,criterion='4')

model.fit(X_train,y_train)
model.score(X_test,y_test)#,C = '1'gamma='auto',kernal='rbf')


# Plotting 
plt.ylabel('Heart Rate in BPM')                         # Plot x axis name
plt.xlabel('Time in sec')                                      # Plot y axis label
plt.grid()                                              # Plot grid lines 
plt.scatter(t2,IBI,label = 'New Data')            # plot Best fit line
plt.legend() 