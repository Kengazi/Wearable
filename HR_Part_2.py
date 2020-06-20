#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:12:20 2019

@author: kendalljohnson
"""
# Make training data into excel with all values plus stim
print('Training data into excel with all values plus stimulas')


import pandas as pd                  # DataFrame Tool
import numpy as np                   # Basically a great calculator for now

# Data (HR data)  :: physical numbers we are analysing 

df = pd.read_csv('Heart1.csv') # Reading the active data
HRa = df.values[:,0]
HRa = HRa[1:]
IBIa = df.values[:,1]                 # Heart Rate in beats per minute
IBIa = IBIa[1:]
#Tempa = df.values[:,2]                 # Heart Rate in beats per minute
#Tempa = Tempa[1:]



# Data (real patient data) :: missing NN50 and ratings

df1 = pd.read_csv('IBI1.csv')
t2 = df1.values[:,0]
t2 = t2[:-21]
IBI = df1.values[:,1]
IBI = IBI * 1000
IBI = IBI[:-21]

t2_size = len(t2)
IBI_size = len(IBI)

# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

# Create DataFrame
#time1 = pd.DataFrame({'time1': t1})
HR = pd.DataFrame({'HR': HR})
time2 = pd.DataFrame({'time2': t2})
IBIa = pd.DataFrame({'IBI': IBI})
#Tempa = pd.DataFrame(Tempa)


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

# Varables :: Getting varables from real IBI DataFrame for example the mean, median, standard dev

# Important Heart Data and Equations ::

Std = IBIa.std() # SDNN
Std = Std[0]
preNN50 = []
preNN20 = []
for i in range(t2_size-1):
    A = IBI[i+1]-IBI[i]
    preNN50.append(A)
    preNN20.append(A)

NN50 = []
for i in range(t2_size-1):
    if np.abs(preNN50[i])>50:
        NN50.append(preNN50[i])
        
pNN50 = (len(NN50))/(len(preNN50))*100

NN20 = []
for i in range(t2_size-1):
    if np.abs(preNN20[i])>20:
        NN20.append(preNN20[i])
        
pNN20 = (len(NN20))/(len(preNN20))*100
    
print('The SDNN for incoming data is {:.4}'.format(Std))
print('The NN50 for incoming data is {}'.format(len(NN50)))
print('The pNN50 for incoming data is {:.5}%'.format(pNN50))
print('The NN20 for incoming data is {}'.format(len(NN20)))
print('The pNN20 for incoming data is {:.5}%'.format(pNN20))

# Making data frames

Std = np.array([Std])
Std_df = pd.DataFrame({'SDNN':Std})
pNN20 = np.array([pNN20])
pNN20_df = pd.DataFrame({'pNN20':pNN20})
pNN50 = np.array([pNN50])
pNN50_df = pd.DataFrame({'pNN50':pNN50})

# Stimulus
Stim = 1
Stim = np.array([Stim])
Stimu = Stim[0]
Stim_df = pd.DataFrame({'Stim':Stim})
if Stimu == 1:
    print('This is a stimulated event')
else:
    print('This is not a stimulated event')
# Putting frames together 
framea = [HR,time2,IBIa]#,Tempa]
resulta = pd.concat(framea, axis=1)

frame = [resulta,Std_df,pNN20_df,pNN50_df,Stim_df]
result = pd.concat(frame, axis=1)

# Save as excel
result.to_excel(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/Train_1.xlsx')