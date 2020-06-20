#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:11:15 2019

@author: kendalljohnson
"""


# Imports :: modules used to create code
import math as m
import pandas as pd                 # DataFrame Tool
import numpy as np                  # Basically a great calculator for now
import matplotlib.pyplot as plt     # For 2-D Graphing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Data (HR data)  :: physical numbers we are analysing 

df = pd.read_excel('P_1.xlsx')
t1 = df.values[:,0]
HR = df.values[:,2]                 # Heart Rate in beats per minute
t22 = df.values[:,3]
t22 = t22[:278]

t2 = []                 # Because time is a float
for i in range(len(t22)):
    C = m.floor(t22[i])
    D = m.ceil(t22[i])
    B = t22[i] - C 
    if B < .5:
        t2.append(C)
    else:
        t2.append(D)
    
IBI = df.values[:,4] 
IBI = IBI[:278]

# Clip data

Stimt = t2[:112]
StimIBI = IBI[:112]

UnStimt = t2[113:156]
UnStimIBI = IBI[113:156]


#Stim3t = time2[157:278]
#Stim3IBI = IBI[157:278]


t1_size = len(t1)
t2_size = len(UnStimt)
t3_size = len(Stimt)

# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

IBIstim = []
IBIunstim = []
time1 = []
time2 = []
HRstim = []
HRunstim = []
for i in range(t1_size):
    for j in range(t3_size):
        if t1[i] == t2[j]:
            time1.append(t1[i])
            HRstim.append(HR[i])

            IBIstim.append(StimIBI[j])
        else:
            'do nothing'

for i in range(t1_size):
    for j in range(t2_size):
        if t1[i] == t2[j]:
            time2.append(t1[i])
            HRunstim.append(HR[i])
            IBIunstim.append(UnStimIBI[j])
        else:  
            'do nothing'
            
x = [81,76]    
y = [580,950]        
# Plotting
 
plt.xlabel('Heart Rate in BPM')                         # Plot x axis name
plt.ylabel('Interbeat Intervals')                                      # Plot y axis label
plt.title('Support Vector Machine of Stim vs Unstim')
plt.grid()                                              # Plot grid lines 
plt.scatter(HRstim,IBIstim,marker = '+',label = 'Stim Data')            # plot Best fit line
plt.scatter(HRunstim,IBIunstim,marker = '.',label = 'Unstim Data')            # plot Best fit line
plt.plot(x,y,color='red',label='SVM Divider')
plt.legend() 



 # DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev          
            
HRstim = np.array(HRstim)
HRS_df = pd.DataFrame({'StimHR':HRstim})
HRunstim = np.array(HRunstim)
HRU_df = pd.DataFrame({'UntimHR':HRunstim})

IBIstim = np.array(IBIstim)
IBIS_df = pd.DataFrame({'StimIBI':IBIstim})
IBIunstim = np.array(IBIunstim)
IBIU_df = pd.DataFrame({'UnstimIBI':IBIunstim})

s1 = IBIstim.size
s2 = IBIunstim.size

#ss1 = np.linspace(1,s1,s1)
#ss2 = np.linspace(1,s2,s2)
ss1 = np.ones(s1)
ss2 = np.zeros(s2)

Stim_df = pd.DataFrame({'Stimulation ':ss1})
Unstim_df = pd.DataFrame({'Stimulation ':ss2})
# Putting frames together 
frame1 = [HRS_df,IBIS_df,Stim_df] # add a frame with the event determination 
result1 = pd.concat(frame1, axis=1)

frame2 = [HRU_df,IBIU_df,Unstim_df] # add a frame with the event determination 
result2 = pd.concat(frame2, axis=1)

result = pd.concat([result1,result2],axis=0)
# Machince Learning :: Support Vector Machine  

#X_train, X_test,y_train, y_test = train_test_split(inputs,df.Stim,test_size = 0.2)
#model = SVC()
#model.fit(inputs,Stim) #,criterion='4')

#model.fit(X_train,y_train)
#model.score(X_test,y_test)#,C = '1'gamma='auto',kernal='rbf')

# Save as excel
#result.to_excel(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/SVM_prep.xlsx')