#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:23:23 2019

@author: kendalljohnson
"""

# Raw Data Conversion Part 1 

# Slicing raw data from Dr Holly particepents the HR and IBI and creating excel file

# Imports :: modules used to create code

import pandas as pd                  # DataFrame Tool
#import numpy as np                  # Basically a great calculator for now
import matplotlib.pyplot as plt      # For 2-D Graphing
#import scipy.optimize as so

# Data (HR data)  :: physical numbers we are analysing 

df = pd.read_csv('HR100.csv')
HR = df.values[:,0]
HR = HR[1:]
#HR = df.values[:,2]                 # Heart Rate in beats per minute
#DateHR = df.values[:,2]                 # Heart Rate in beats per minute

# Size of Arrays
HR_size = len(HR)
t1 = []
for i in range(HR_size):
    i
    t1.append(i)
    
    
t1_size = len(t1)
#DateHR_size = len(DateHR)

# Data (IBI data) :: physical numbers we are analysing

df1 = pd.read_csv('IBI100.csv')
t2 = df1.values[:,0]
IBI = df1.values[:,1]
IBI = IBI * 1000
#DateIBI = df1.values[:,3]                 # Heart Rate in beats per minute

# Size of Arrays
t2_size = len(t2)
IBI_size = len(IBI)
#DateHR_size = len(DateHR)

# Data (Temperture data) :: 

df2 = pd.read_csv('TEMP100.csv')
t3 = df2.values[:,0]
#t3 = t3[:-21]
Temp = df2.values[:,0]
#DateIBI = df2.values[:,3]                 # Heart Rate in beats per minute
#DateIBI = DateIBI[18:]

# Size of Arrays
t3_size = len(t3)
Temp_size = len(Temp)
#DateTemp_size = len(Temp)

# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

# Heart Rate
t1 = pd.DataFrame({'Time1': t1})
HR = pd.DataFrame({'HR': HR})
#DateHR = pd.DataFrame({'DatesHR': DateHR})
# IBI
t2 = pd.DataFrame(({'Time2':t2}))
IBI = pd.DataFrame({'IBI': IBI})
#IBI1000 = IBI/1000
#IBI1000 = pd.DataFrame({'IBIsec': IBI1000})
#DateIBI = pd.DataFrame({'DatesIBI': DateIBI})
# Temperture
#t3 = pd.DataFrame({'Time3': t3})
#Temp = pd.DataFrame({'Temp': Temp})
#DateTemp = pd.DataFrame({'DatesTemp': DateTemp})


# Putting frames together 
frames = [t1,HR,t2, IBI]
result = pd.concat(frames, axis=1)

# Defining a line of Best fit
def line(x,m,b):                  # Name of function and ending with :
    y = x*m+b                     # Actual equation (Linear)
    return y     

# Scipy  Matrix :: easy way to get varibles in code using scipy.optimize
    
#popt0,pcov0 = so.curve_fit(line,t2,IBI)

#m0 = popt0[0]                           # slope value
#b0 = popt0[1]                           # Intercept value

#y1 = m0 * t2 + b0

# Plotting
plt.title('Fig 1 \n IBI(time)')
plt.ylabel('IBI in ms')                         # Plot x axis name
plt.xlabel('Time')                                      # Plot y axis label
plt.grid()                                              # Plot grid lines 
plt.scatter(t2,IBI,label = 'IBI New Data')            # plot Best fit line
#plt.plot(t2, y1,'k',label = ' Best-fit line') 
plt.legend() 