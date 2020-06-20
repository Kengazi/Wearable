#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 07:23:59 2019

@author: kendalljohnson
"""

# Linear Regression of the IBI

print('Part 2_1')

# Imports :: modules used to create code

import pandas as pd                 # DataFrame Tool
import numpy as np                  # Basically a great calculator for now


# Data (real/training data)  :: physical numbers we are analysing 

df = pd.read_excel('X_data.xlsx') # Heart1p1.csv
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

Std = np.array([Std])
Std_df = pd.DataFrame([Std])
pNN20 = np.array([pNN20])
pNN20_df = pd.DataFrame([pNN20])
pNN50 = np.array([pNN50])
pNN50_df = pd.DataFrame([pNN50])

frames = [HR_df,IBI_df,Std_df,pNN20_df,pNN50_df]
result = pd.concat(frames, axis=1)
# Save as excel
#result.to_excel(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/Heart1p2.xlsx')