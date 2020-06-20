#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 08:45:03 2019

@author: kendalljohnson
"""

# Making accurate Heart rate to IBI excel for Patient X using linear regression

# Imports :: modules used to create code

import pandas as pd                 # DataFrame Tool
import numpy as np                  # Basically a great calculator for now
import matplotlib.pyplot as plt     # For 2-D Graphing
from sklearn import linear_model    # For our Linear Regression Model // Sklearn is a M.L. modules
import scipy.optimize as so         # Basically a great calculator for now
from sklearn.linear_model import LogisticRegression  # For our Linear Regression Model // Sklearn is a M.L. modules
from sklearn.model_selection import train_test_split # Label Encoder

# Data (HR data)  :: physical numbers we are analysing 

df = pd.read_excel('Pat_X.xlsx')
t1 = df.values[:,0]
HR = df.values[:,2]                 # Heart Rate in beats per minute
t2 = df.values[:,3]
IBI = df.values[:,4] 
Dates = df.values[:,5]                 # Heart Rate in beats per minute

HR_size = len(HR)
t1_size = len(t1)
t2_size = len(t2)
IBI_size = len(IBI)

# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

t1 = pd.DataFrame(t1)
Dates = pd.DataFrame(Dates)
HR = pd.DataFrame(HR)
t2 = pd.DataFrame(t2)
IBI = pd.DataFrame(IBI)
# Putting frames together 
frames = [t1,HR,t2, IBI,Dates]
result = pd.concat(frames, axis=1)

X1 = t1
y =  HR

# Lin Regression Model :: 

modelHR = linear_model.LinearRegression()# Linear Reg model
modelHR.fit(df[['in1']],df.HR)          # Adding varibles to model 

G = modelHR.score(X1,y)
m1 = modelHR.coef_
m1 = m1[0]
b1 = modelHR.intercept_


#modelIBI = linear_model.LinearRegression()# Linear Reg model
#modelIBI.fit(df[['in2']],df.IBI)          # Adding varibles to model 
#M = modelIBI.score
#m2 = modelIBI.coef_
#m2 = m2[0]
#b2 = modelIBI.intercept_

print("The HR model score is {0}".format(G))
def Line(x,m,b):
    y = m*x + b
    return y

x = t1
y = m1 * t1 + b1
#D = Pred_IBI.size
#for i in range(count):
   #C = New_HR[i] + noise[i]
   #pred = reg.predict([[C]])
   #pred1 = pred[0] 
   #Pred_IBI.append(pred1)
#A = IBI(HR)
#print('The Inter_beat Interval is {} miliseconds'.format(A))
#plt.scatter(t1,HR,label = 'Training Data')            # Plot training Data
#plt.title("Linear Reg of Heart rate data")              # Plot title
plt.ylabel('Heart Rate in BPM')                         # Plot x axis name
plt.xlabel('Time in in')                                      # Plot y axis label
plt.grid()                                              # Plot grid lines 
                                            # Plot legend
plt.scatter(t1,HR,label = 'HR Data')            # plot Best fit line
plt.plot(x,y,color='red',label = 'Best fit line')
plt.legend()
#plt.scatter(HR,NN50,'r',label = ' Best-fit line')            # plot Best fit line

   # New NN50 frame of real data  


# Save as excel
#result.to_excel(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/Pat_X.xlsx')