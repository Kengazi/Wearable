#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 08:33:22 2019

@author: kendalljohnson
"""

# Making accurate Heart rate to IBI excel for Patient X using linear regression

# Imports :: modules used to create code

import pandas as pd                 # DataFrame Tool
import numpy as np                  # Basically a great calculator for now
import matplotlib.pyplot as plt     # For 2-D Graphing
from sklearn import linear_model    # For our Linear Regression Model // Sklearn is a M.L. modules
import scipy.optimize as so         # Basically a great calculator for now
#from sklearn.cross_validation import train_test_split

# Data (HR data)  :: physical numbers we are analysing 

df = pd.read_csv('HRx.csv')
t1 = df.values[:,1]
t1 = t1[18:]
HR = df.values[:,0]                 # Heart Rate in beats per minute
HR = HR[18:]
Dates = df.values[:,3]                 # Heart Rate in beats per minute
Dates = Dates[18:]
HR_size = len(HR)
t1_size = len(t1)
#noise = np.random.normal(0,1,572)   # Created noise to move data around 

# Data (real patient data) :: missing NN50 and ratings

df1 = pd.read_csv('IBIx.csv')
t2 = df1.values[:,0]
t2 = t2[:-21]
IBI = df1.values[:,1]
IBI = IBI * 1000
IBI = IBI[:-21]

t2_size = len(t2)
IBI_size = len(IBI)

# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

t1 = pd.DataFrame(t1)
Dates = pd.DataFrame(Dates)
HR = pd.DataFrame(HR)
t2 = pd.DataFrame(t2)
IBI1000 = IBI/1000
IBI1000 = pd.DataFrame(IBI1000)
IBI = pd.DataFrame(IBI)



# Putting frames together 
frames = [t1,HR,t2, IBI,Dates,IBI1000]
result = pd.concat(frames, axis=1)


# Lin Regression Model :: 

#reg = linear_model.LinearRegression()# Linear Reg model
#reg.fit(df[['HR']],df.NN50)          # Adding varibles to model 



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
plt.xlabel('Time in sec')                                      # Plot y axis label
plt.grid()                                              # Plot grid lines 
                                           # Plot legend
plt.scatter(t2,IBI,label = 'New Data')            # plot Best fit line
#plt.scatter(HR,NN50,'r',label = ' Best-fit line')            # plot Best fit line
plt.legend() 
   # New NN50 frame of real data  


# Save as excel
result.to_excel(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/Pat_X.xlsx')