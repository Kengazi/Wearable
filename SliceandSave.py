#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 08:33:22 2019

@author: kendalljohnson
"""

# Raw Data Conversion Part 1 

# All together

# Imports :: modules used to create code

import pandas as pd                  # DataFrame Tool
import numpy as np                  # Basically a great calculator for now
import matplotlib.pyplot as plt      # For 2-D Graphing
#import scipy.optimize as so
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Data (Incoming sensor data)  :: physical numbers we are analysing 

df = pd.read_csv('Heart1.csv') # Reading the active data
HRa = df.values[:,0]
HRa = HRa[1:]
IBIa = df.values[:,1]                 # Heart Rate in beats per minute
IBIa = IBIa[1:]
#Tempa = df.values[:,1]                 # Heart Rate in beats per minute
#Tempa = Tempa[1:]

# Size of Arrays
HRa_size = len(HRa)


time = []
for i in range(HRa_size):
    i
    time.append(i)
       
time_size = len(time)   # Amount of (time /2) in sec

# Data (Training data) :: physical numbers we are analysing

df1 = pd.read_csv('IBI3.csv')    # Made training set with HR,IBI,Temp,Std
HRt = df1.values[:,0]
HRt = HRt[1:]
IBIt = df1.values[:,1]                 # Heart Rate in beats per minute
IBIt = IBIt[1:]
Tempt = df1.values[:,2]                 # Heart Rate in beats per minute
Tempt = Tempt[1:]

# Size of Arrays
HRt_size = len(HRt)

# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

# Create DataFrame
HRa = pd.DataFrame(HRa)
IBIa = pd.DataFrame(IBIa)
#Tempa = pd.DataFrame(Tempa)

HRt = pd.DataFrame(HRt)
IBIt = pd.DataFrame(IBIt)
#Tempt = pd.DataFrame(Tempt)

# Data (Temperture data) :: 

#df2 = pd.read_csv('TEMP2.csv')
#t3 = df1.values[:,1]
#t3 = t2[:-21]
#Temp = df2.values[:,2]
#DateIBI = df2.values[:,3]                 # Heart Rate in beats per minute
#DateIBI = DateIBI[18:]

# Size of Arrays
#t3_size = len(t3)
#Temp_size = len(Temp)
#DateTemp_size = len(Temp)

# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

# Heart Rate
#IBI1000 = IBI/1000
#IBI1000 = pd.DataFrame({'IBIsec': IBI1000})
#DateIBI = pd.DataFrame({'DatesIBI': DateIBI})
# Temperture
#t3 = pd.DataFrame({'Time3': t3})
#Temp = pd.DataFrame({'Temp': Temp})
#DateTemp = pd.DataFrame({'DatesTemp': DateTemp})


# Putting frames together 
framea = [HRa,IBIa,Tempa]
resulta = pd.concat(framea, axis=1)

framet = [HRt,IBIt]#,Tempt]
resultt = pd.concat(framet, axis=1)
# Varables :: Getting varables from real IBI DataFrame for example the mean, median, standard dev

# Important Heart Data and Equations ::

Std = IBIa.std() # SDNN
Std = Std[0]
preNN50 = []
preNN20 = []
for i in range(HRa_size-1):
    A = IBIa[i+1]-IBIa[i]
    preNN50.append(A)
    preNN20.append(A)

NN50 = []
for i in range(HRa_size-1):
    if np.abs(preNN50[i])>50:
        NN50.append(preNN50[i])
        
pNN50 = (len(NN50))/(len(preNN50))*100

NN20 = []
for i in range(HRa_size-1):
    if np.abs(preNN20[i])>20:
        NN20.append(preNN20[i])
        
pNN20 = (len(NN20))/(len(preNN20))*100
    
print('The SDNN is {}'.format(Std))
print('The NN50 is {}'.format(len(NN50)))
print('The pNN50 is {}'.format(pNN50))
print('The NN20 is {}'.format(len(NN20)))
print('The pNN20 is {}'.format(pNN20))


Std = IBI_df.std() # SDNN
Mean = IBI_df.mean()


Std = np.array([Std])
Std_df = pd.DataFrame([Std])
pNN20 = np.array([pNN20])
pNN20_df = pd.DataFrame([pNN20])
pNN50 = np.array([pNN50])
pNN50_df = pd.DataFrame([pNN50])

# Putting frames together 

frames = [HR_df,IBI_df,Std_df,pNN20_df,pNN50_df]
result = pd.concat(frames, axis=1)

# Equations for Binary indicators  
HRi_df = pd.DataFrame(HRi)
Mean_HRi = HRi_df.mean()                    # Mean for DataFrames
Mean_HRi = Mean_HRi[0]                   # Getting number out of array 
HRt_df = pd.DataFrame(HRt)
Mean_HRt = HRi_df.mean()                    # Mean for DataFrames
Mean_HRt = Mean_HRt[0]                   # Getting number out of array 

IBIi_df = pd.DataFrame(IBIi)
Mean_IBIi = IBIi_df.mean()                    # Mean for DataFrames
Mean_IBIi = Mean_IBIi[0]                   # Getting number out of array 
IBIt_df = pd.DataFrame(IBIt)
Mean_IBIt = IBIi_df.mean()                    # Mean for DataFrames
Mean_IBIt = Mean_IBIt[0]                   # Getting number out of array 



# Heart rate :: if 1 then they are working out 
HRa = []
if Mean_HRi < (Mean_HRt)*1.2:
    HRa.append(1)
else:
    HRa.append(0)

# IBI
IBIa = []
if Mean_IBIi > Mean_IBIt:
    IBIa.append(1)
else:
    IBIa.append(0)

# SDNN
SDa = []
if SDi > SDt:
    SDa.append(1)
else:
    SDa.append(0)

# pNN20 
pNN20a = []
if pNN20i > pNN20t:
    pNN20a.append(1)
else:
    pNN20a.append(0)

# pNN50 
pNN50a = []
if pNN50i > pNN50t:
    pNN50a.append(1)
else:
    pNN50a.append(0)
    
HRa = np.array([HRa])
HR_df = pd.DataFrame([HRa])    
IBIa = np.array([IBIa])
IBI_df = pd.DataFrame([IBIa])
SDa = np.array([SDa])
SDa_df = pd.DataFrame([SDa])
pNN20a = np.array([pNN20a])
pNN20_df = pd.DataFrame([pNN20a])
pNN50a = np.array([pNN50a])
pNN50_df = pd.DataFrame([pNN50a])

frames = [HR_df,IBI_df,SDa_df,pNN20_df,pNN50_df]   # completly binary Need to add a place for target stimulus
result = pd.concat(frames, axis=1)

# Create DataFrame
HRa = pd.DataFrame(HRa)
IBIa = pd.DataFrame(IBIa)
Stda = pd.DataFrame(Stda)
pNN20a = pd.DataFrame(pNN20a)
pNN50a = pd.DataFrame(pNN50a)
Stima = pd.DataFrame(Stima)

# Putting frames together 
frames = [HRt,IBIt,Stdt, pNN20t,pNN50t]
inputs = pd.concat(frames, axis=1)

# Machince Learning :: Decision Tree 

#model = tree.DecisionTreeClassifier()
X_train, X_test,y_train, y_test = train_test_split(inputs,df.Stim,test_size = 0.2)
#model.fit(inputs,Stim) #,criterion='4')
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train,y_train)
model.score(X_test,y_test)

# Predicting Data

# if HRa > HRt, IBIa < IBIt, Stda > Stdt, pNN20a > pNN20t, pNN50a > pNN50t 

# Does the tree predict a stimulas ?

Pred = model.predict([[1,0,0,0,1]])
Pred = Pred[0]
if Pred == 1:
    print('Stimulus has occured')
else:
    print('No Stimulus has occured')
    
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
plt.title('IBI(time)')
plt.ylabel('IBI in ms')                         # Plot x axis name
plt.xlabel('Time')                                      # Plot y axis label
plt.grid()                                              # Plot grid lines 
plt.scatter(t2,IBI,label = 'New Data')            # plot Best fit line
#plt.plot(t2, y1,'k',label = ' Best-fit line') 
plt.legend() 