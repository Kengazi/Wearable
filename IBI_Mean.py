#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:47:17 2019

@author: kendalljohnson
"""

# Gettimng sliced average IBI values 

# Imports ::

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import math as m
#from sklearn.cluster import KMeans

# Files ::

df = pd.read_csv('IBI104.csv')               # Heart Rate in beats per minute
time2 = df.values[:,0]                # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus 
#time2 = time2[:278] 
IBI = df.values[:,1]
IBI = IBI*1000
#IBI = IBI[:278] 
#Pat_size = HR.size

# Clip data 100
#Values = 360 - 1
#A = 35#           End of first 
#A_ = A + 1      # Begining of 2nd    # 82
#B = 185          # End of Second
#B_ = B + 1      # Begining of 3rd
#C = 245         # End of 3rd 
#C_ = C + 1       # begining of 4th
#D = 359         # End of 4th / full amount of entrees

# Clip data 101
#Values = 360 - 1
#A = 4#           End of first 
#A_ = A + 1      # Begining of 2nd    # 82
#B = 29          # End of Second


# Clip data 102
#Values = 360 - 1
#A = 223#           End of first 
#A_ = A + 1      # Begining of 2nd    # 82
#B = 287          # End of Second
#B_ = B + 1      # Begining of 3rd
#C = 327         # End of 3rd 
#C_ = C + 1       # begining of 4th
#D = 509         # End of 4th / full amount of entrees

# Clip data 103
#Values = 360 - 1
#A = 70#           End of first 
#A_ = A + 1      # Begining of 2nd    # 82
#B = 102          # End of Second
#B_ = B + 1      # Begining of 3rd
#C = 134         # End of 3rd 
#C_ = C + 1       # begining of 4th
#D = 179         # End of 4th / full amount of entrees

# Clip data 104
#Values = 209 - 1
A = 40#           End of first 
A_ = A + 1      # Begining of 2nd    # 82
B = 209          # End of Second
#B_ = B + 1      # Begining of 3rd


Stim1t = time2[:A]
Stim1IBI = IBI[:A]

Stim2t = time2[A_:B]
Stim2IBI = IBI[A_:B]

Stim3t = time2[B_:C]
Stim3IBI = IBI[B_:C]

Stim4t = time2[C_:D]
Stim4IBI = IBI[C_:D]


# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

T_df = pd.DataFrame(time2)
#IBI = pd.DataFrame(IBI)



IBI1_df = pd.DataFrame(Stim1IBI)
Mean1 = IBI1_df.mean()
#print(Mean1)

IBI2_df = pd.DataFrame(Stim2IBI)
Mean2 = IBI2_df.mean()
#print(Mean2)

IBI3_df = pd.DataFrame(Stim3IBI)
Mean3 = IBI3_df.mean()
#print(Mean3)

IBI4_df = pd.DataFrame(Stim1IBI)
Mean4 = IBI4_df.mean()
#print(Mean4)
# Varables :: Getting varables from real IBI DataFrame for example the mean, median, standard dev

# Important Heart Data and Equations ::

t2_size = len(Stim2IBI)
IBI_size = len(IBI)

Std = IBI2_df.std() # SDNN
Std = Std[0]
preNN50 = []
preNN20 = []
for i in range(t2_size-1):
    A = Stim2IBI[i+1]-Stim2IBI[i]
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