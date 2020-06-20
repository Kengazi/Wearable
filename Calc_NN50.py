#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:32:07 2019

@author: kendalljohnson
"""

# Calculating NN50

# Imports :: modules used to create code

import pandas as pd                 # DataFrame Tool
import numpy as np                  # Basically a great calculator for now
import matplotlib.pyplot as plt     # For 2-D Graphing
from sklearn import linear_model    # For our Linear Regression Model // Sklearn is a M.L. modules
import scipy.optimize as so         # Basically a great calculator for now
#from sklearn.cross_validation import train_test_split

# Data (training data)  :: physical numbers we are analysing 

df = pd.read_excel('HR.xlsx')
HR = df.values[:,0]                 # Heart Rate in beats per minute
PR = df.values[:,1]                 # Persons Rating of how strong was the stimulus (neg) in the training
NN50 = df.values[:,2]               # NN50 is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus 
noise = np.random.normal(0,1,572)   # Created noise to move data around 

# Data (real patient data) :: missing NN50 and ratings

df1 = pd.read_csv('HR2.csv')
Pat_data = df1.values[:,0]
Pat_data = Pat_data[1:]
Pat_data = Pat_data[:30]

# Varables from DataFrames for example the mean, median, standard dev

count = 30#Pat_data.size
mean = 85.182178
std   = 11.667138
min  =   56.970000
q1  =   78.900000
med  =   88.550000
q3  =   94.750000
max  =  100.180000

# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

HR1 = pd.DataFrame(HR)
PR1 = pd.DataFrame(PR)
NN501 = pd.DataFrame(NN50)
Pat_data1 = pd.DataFrame(Pat_data)

# Lin Regression Model :: 

reg = linear_model.LinearRegression()# Linear Reg model
reg.fit(df[['HR']],df.NN50)          # Adding varibles to model 

H = 80
i = 300 #ms
i = i/1000
t = 50 #ms 
t = t / 1000

def IBI(H):
    IBI = 60000/H
    return IBI # in miliseconds

New_HR = Pat_data

Real_IBI = []                     
for i in range(count):    
   C = New_HR[i] + noise[i]
   A = IBI(C)
   Real_IBI.append(A)

#D = Pred_IBI.size
#for i in range(count):
   #C = New_HR[i] + noise[i]
   #pred = reg.predict([[C]])
   #pred1 = pred[0] 
   #Pred_IBI.append(pred1)
#A = IBI(HR)
#print('The Inter_beat Interval is {} miliseconds'.format(A))
#plt.scatter(New_HR,Real_IBI,label = 'Training Data')            # Plot training Data
#plt.title("Linear Reg of Heart rate data")              # Plot title
#plt.xlabel('Heart Rate in BPM')                         # Plot x axis name
#plt.ylabel('IBI')                                      # Plot y axis label
#plt.grid()                                              # Plot grid lines 
#plt.legend()                                            # Plot legend
#plt.scatter(HR,IBI(HR),label = 'New Data')            # plot Best fit line
#plt.scatter(HR,NN50,'r',label = ' Best-fit line')            # plot Best fit line

   # New NN50 frame of real data  
Real_IBI1 = pd.DataFrame(Real_IBI)
# Putting frames together 
frames = [Pat_data1, Real_IBI1]
result = pd.concat(frames, axis=1)


# Save as CSV
#result.to_excel(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/P_data_1.xlsx')