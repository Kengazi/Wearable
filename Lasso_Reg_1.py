#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:20:07 2019

@author: kendalljohnson
"""

# Making accurate Heart rate to IBI excel for Patient X using linear regression

# Imports :: modules used to create code

import pandas as pd                 # DataFrame Tool
#import numpy as np                  # Basically a great calculator for now
import matplotlib.pyplot as plt     # For 2-D Graphing
from sklearn import linear_model    # For our Linear Regression Model // Sklearn is a M.L. modules
import scipy.optimize as so         # Basically a great calculator for now
from sklearn.linear_model import Lasso # Reducing features
#from sklearn.linear_model  # For our Linear Regression Model // Sklearn is a M.L. modules
from sklearn.model_selection import train_test_split # Label Encoder
from sklearn.preprocessing import StandardScaler
# Data (HR data)  :: physical numbers we are analysing 

df = pd.read_excel('X_data.xlsx')
#t1 = df.values[:,0]
HR = df.values[:,2]                 # Heart Rate in beats per minute
#t2 = df.values[:,3]
IBI = df.values[:,3] 
#Dates = df.values[:,5]                 # Heart Rate in beats per minute

HR_size = len(HR)
#t1_size = len(t1)
#t2_size = len(t2)
IBI_size = len(IBI)

# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

HR = pd.DataFrame(HR)
IBI = pd.DataFrame(IBI)


# Create Standardized Features
scaler = StandardScaler()
HR_stand = scaler.fit_transform(HR)


X1 = HR_stand
y =  IBI

# Lin Regression Model :: 
regression = Lasso(alpha=0.1)
modelHR = regression.fit(HR_stand,df.IBI)# Adding varibles to model 

G = modelHR.score(X1,y)
m1 = modelHR.coef_
m1 = m1[0]
b1 = modelHR.intercept_

print("The HR/IBI model score is {0}".format(G))
def Line(x,m,b):
    y = m*x + b
    return y

#x = X1
#yy = Line(x,m1,b1)
#X1 = pd.DataFrame(X1)
popt0,pcov0 = so.curve_fit(Line,HR,y)

m0 = popt0[0]                           # slope value
b0 = popt0[1]                           # Intercept value

yyy = Line(HR,m0,b0)


plt.title("Lasso Reg of Heart rate data")              # Plot title
plt.xlabel('Heart Rate in BPM')                         # Plot x axis name
plt.ylabel('IBI in milliseconds')                                      # Plot y axis label
plt.grid()                                              # Plot grid lines 
plt.plot(HR,yyy,color='red',label = 'Scipy Best fit line')                 
plt.scatter(HR,IBI,label = 'Train_Data')            # plot Best fit line
plt.plot(HR,yy,color='green',label = 'M.L. Best fit line')
plt.legend()                                        # Plot legend


# Save as excel
#result.to_excel(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/Pat_X.xlsx')