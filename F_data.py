#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 20:02:21 2019

@author: kendalljohnson
"""
# Imports ::

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#import math as m
#rom sklearn.cluster import KMeans


# Raw data

IBI_A = [734,698,694,710,753,843,655,666,678,681,673,661,669,677,711,703]

IBI_K = [738,656.95,697,718,651,849,663.7,666.69,673.77,680,716,630,702,660,706,702]

SDNN = [73.6,47.1,36.65,43.05,209,51.7,52.6,42.48,34,47.75,54.8,42.42,48.74,50,40.85,45.98]

pNN20 = [64.7,67.5,62,58.9,100,73.9,65.7,66.13,52.6,64.5,75.36,73.33,83.3,69.7,64.1,61.6]

pNN50 = [47.06,32.4,25.9,22.3,100,39,35,24.2,18,25.8,37.7,43.3,30,39,25.6,29.3]

Stim = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]

def HR(IBI):
    HR = 60000/IBI
    return HR

HR_A = []
HR_K = []
for i in range(len(IBI_A)):
    HR_a = HR(IBI_A[i])
    HR_k = HR(IBI_K[i])
    HR_A.append(HR_a)
    HR_K.append(HR_k)
    
# Making data frames
    
HR_A = np.array(HR_A)
HR_A_df = pd.DataFrame({'HR_A':HR_A})
HR_K = np.array(HR_K)
HR_K_df = pd.DataFrame({'HR_A':HR_K})
IBI_A = np.array(IBI_A)
IBI_A_df = pd.DataFrame({'IBI_A':IBI_A})
IBI_K = np.array(IBI_K)
IBI_K_df = pd.DataFrame({'IBI_A':IBI_K})
SDNN = np.array(SDNN)
SDNN_df = pd.DataFrame({'SDNN':SDNN})
pNN20 = np.array(pNN20)
pNN20_df = pd.DataFrame({'pNN20':pNN20})
pNN50 = np.array(pNN50)
pNN50_df = pd.DataFrame({'pNN50':pNN50})

# Stimulus
#Stim = 1
Stim = np.array(Stim)
Stim_df = pd.DataFrame({'Stim':Stim})
   
# Putting frames together 


frame = [HR_A_df,HR_K_df, IBI_A_df, IBI_K_df,SDNN_df,pNN20_df,pNN50_df,Stim_df]
result = pd.concat(frame, axis=1)

# Save as excel
result.to_excel(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/Full_Train.xlsx')