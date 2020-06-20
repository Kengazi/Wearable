#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:19:14 2019

@author: kendalljohnson
"""

# Linear Regression of the IBI

print('Linear Regression of the IBI(HR)') # Unneccery

# Imports :: modules used to create code

import pandas as pd                 # DataFrame Tool
import numpy as np                  # Basically a great calculator for now
import matplotlib.pyplot as plt     # For 2-D Graphing
from sklearn import linear_model    # For our Linear Regression Model // Sklearn is a M.L. modules
#import scipy.optimize as so         # Basically a great calculator for now
from sklearn.model_selection import train_test_split # Label Encoder
#from sklearn.cross_validation import train_test_split

# Data (real/training data)  :: physical numbers we are analysing 

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

# Log Regression Model :: 

model = linear_model.LinearRegression()# Linear Reg model
X_train, X_test,y_train, y_test = train_test_split(df[['HR']],df.IBI,test_size=0.1)
model.fit(X_train,y_train)
HH = model.score(X_test,y_test)
HH = HH * 100

X = model.predict(X_test)
#P = model.predict_proba(X_test)

# Varables from real IBI DataFrame for example the mean, median, standard dev
   
Max = IBI_df.max()
Std = IBI_df.std() # SDNN
Min = IBI_df.min()
Mean = IBI_df.mean()
Sum = IBI_df.sum()

# Important Heart Data and Equations ::
IBI # AVNN
Std = IBI_df.std() # SDNN
Std = Std[0]
preNN50 = []
preNN20 = []
for i in range(Pat_size-1):
    A = IBI[i+1]-IBI[i]
    preNN50.append(A)
    preNN20.append(A)

NN50 = []
for i in range(Pat_size-1):
    if np.abs(preNN50[i])>50:
        NN50.append(preNN50[i])
        
pNN50 = (len(NN50))/(len(preNN50))*100

NN20 = []
for i in range(Pat_size-1):
    if np.abs(preNN20[i])>20:
        NN20.append(preNN20[i])
        
pNN20 = (len(NN20))/(len(preNN20))*100


if pNN50 > 1 and pNN20 > 1:
    print('The relaspe occured')
else: 
    print('The relaspe has not occured')
    
print('The SDNN is {}'.format(Std))
print('The NN50 is {}'.format(len(NN50)))
print('The pNN50 is {}'.format(pNN50))
print('The NN20 is {}'.format(len(NN20)))
print('The pNN20 is {}'.format(pNN20))
print('The Model score is {}'.format(HH))


# Predicted data

Pred_IBI = []

for i in range(Pat_size):
   C = HR[i]
   pred = model.predict([[C]])
   pred1 = pred[0] 
   Pred_IBI.append(pred1)
   
   
# For Manual Machine Learning

# the differece between Real and Predicted for Cost function
Diff_IBI = []
for j in range(Pat_size):
   D = IBI[j]-Pred_IBI[j]
   Diff_IBI.append(D)
 
# the Cost fucntion root mean square 
Cost_IBI = []
for j in range(Pat_size):
   D = IBI[j]-Pred_IBI[j]
   E = D**2
   Cost_IBI.append(E)
   
Sum_C = sum(Cost_IBI)
Cost = Sum_C / Pat_size



# Optimization curve fit line

def Parab(x, a, b, c):
    return (x**2)*a+x*b+c

# Optimization parameters
    
#popt1, pcov1 = so.curve_fit(Parab,,y1)
#y01=Parab(x1,popt1[0],popt1[1],popt1[2])


# Plotting 
   
#print('The Inter_beat Interval is {} miliseconds'.format(A))
plt.scatter(HR,IBI,color = "red",label = 'Training Data')            # Plot training Data
plt.title("Linear Reg of Heart rate data")              # Plot title
plt.xlabel('Heart Rate in BPM')                         # Plot x axis name
plt.ylabel('IBI')                                      # Plot y axis label
plt.grid()                                              # Plot grid lines 
plt.scatter(HR,Pred_IBI,color = "blue",label = 'Predicted Data')            # plot Best fit line
plt.legend()                                            # Plot legend
#plt.scatter(HR,NN50,'r',label = ' Best-fit line')            # plot Best fit line



# Save as CSV

#result.to_csv(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/Pat_IBI_1.csv')