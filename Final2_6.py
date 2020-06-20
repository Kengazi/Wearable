#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:16:40 2019

@author: kendalljohnson
"""

# Best Best Algorthem
# Using an Artificial Neural Network to Determine if there was a Stressor present or not in opiod rehad patients

print('Uses Sensor data with training data to be calculated using machine learning for the Heart Rate of an opiod rehab patient\n')


# All together

# Imports :: modules used to create code

import pandas as pd                               # DataFrame Tool
import numpy as np                                # Basically a great calculator for now
#import matplotlib.pyplot as plt                  # For 2-D Graphing
#import scipy.optimize as so
from sklearn.neural_network import MLPClassifier  # For Artificial Neural Network
from sklearn.preprocessing import StandardScaler
np.random.seed(5)                                 # For Artificial Neural Network

# Data (Incoming sensor data)  :: physical numbers we are analysing 

df = pd.read_csv('Heart2.csv') # Reading the active data

HR = df.values[:,0]                               # Heart Rate in beats per minute
HR = HR[1:]
IBI = df.values[:,2]                              # Inter beat interval in ms
IBI = IBI[1:]
Temp = df.values[:,1]                             # Temp in F
Temp = Temp[1:]
Time = df.values[:,3]                             # Time in date 
Time = Time[1:]
HRs_size = len(IBI)

# Varables :: Getting varables from real IBI DataFrame for example the mean, median, standard dev from incoming sensor data

# Getting NN values or IBI[2]-IBI[1]
preNN50 = []
preNN20 = []
for i in range(HRs_size-1):
    A = IBI[i+1]-IBI[i]                            # Getting all of the NN values into arrays 
    preNN50.append(A)
    preNN20.append(A)
    
# Gettin NN50
NN50 = []
for i in range(HRs_size-1):
    if np.abs(preNN50[i])>50:
        NN50.append(preNN50[i])
        
# Getting pNN50     
pNN50 = (len(NN50))/(len(preNN50))*100

# Gettin NN20
NN20 = []
for i in range(HRs_size-1):
    if np.abs(preNN20[i])>20:
        NN20.append(preNN20[i])
        
# Getting pNN20
pNN20 = (len(NN20))/(len(preNN20))*100


# Getting Sensor Heart Rate mean
HR = np.array(HR)                                  # put into Numpy Array
HR = pd.DataFrame(HR)                              # put into DataFrame
mean1 = HR.mean()                                  # Getting the mean using dataframe commands
HRmean = mean1[0]
# Getting Sensor IBI mean
IBI = np.array(IBI)                                # put into Numpy Array
IBI = pd.DataFrame(IBI)                            # put into DataFrame
mean2 = IBI.mean()                                 # Getting the mean using dataframe commands
IBImean = mean2[0]
# Getting Sensor IBI Standard dev
SDNN = IBI.std()                                   # Getting the standard dev using dataframe commands
SDNN = SDNN[0]

# Printing Incoming Sensor Values
print('The Average Heart Rate is for incoming data is {:.4}'.format(HRmean))
print('The Average Interbeat Interval is for incoming data is {:.4}'.format(IBImean))
print('The SDNN for incoming data is {:.4}'.format(SDNN))
print('The NN50 for incoming data is {}'.format(len(NN50)))
print('The pNN50 for incoming data is {:.5}%'.format(pNN50))
print('The NN20 for incoming data is {}'.format(len(NN20)))
print('The pNN20 for incoming data is {:.5}%'.format(pNN20))


# Putting arrays together 

s_frames = [HRmean,HRmean,IBImean,IBImean,SDNN,pNN20,pNN50]

# Data (Training data)  :: Sectioned data that is a positive stimulus 

dft = pd.read_excel('Full_Train.xlsx') # Pateint data from Dr. Hollys readings 
countt = dft.values[:,0]
HR_A = dft.values[:,1]                 # Heart Rate in beats per minute
HR_K = dft.values[:,2]                 # Heart Rate in beats per minute
IBI_A = dft.values[:,3]                # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus  # Ignore first 10 secs
IBI_K = dft.values[:,4]                # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus
SDNNt = dft.values[:,5]
pNN20t = dft.values[:,6]
pNN50t = dft.values[:,7]
Stim = dft.values[:,8]
HRt_size = len(countt)

# Turning arrays to list

#HR_A = pd.DataFrame(HR_A)
HR_K = list(HR_K)
#IBI_A = pd.DataFrame(IBI_A)
IBI_K = list(IBI_K)
SDNNt = list(SDNNt)
pNN20t = list(pNN20t)
pNN50t = list(pNN50t)
countt = pd.DataFrame(countt)
Stim = list(Stim)

# Machine learning codes ::
scaler = StandardScaler()
x = dft.values[:,1:8]
scaler.fit(x) # Normalize
x_norm = scaler.transform(x)# Normalize
y = Stim
y = list(y)
#y_norm = scaler.fit(y).transform(y)
#clf = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(10,10,10,9), random_state=1) # 93.5
#mlp = MLPClassifier(solver='lbfgs', alpha=1e-4 ,max_iter=10,hidden_layer_sizes=(6,12,21,18,18), random_state=10) # 100
mlp = MLPClassifier(activation='relu',solver='lbfgs',max_iter=1, alpha=1e-4, hidden_layer_sizes=(17,12,29), random_state=1) # 100
#clf = MLPClassifier(activation='relu',solver='adam', alpha=1e-4, hidden_layer_sizes=(10,10), random_state=1)
# The model score is 81.25% int1 7 int2 14 int3 46 
#The model score is 81.25% int1 8 int2 34 int3 46 
#The model score is 81.25% int1 9 int2 44 int3 4 
#The model score is 81.25% int1 13 int2 25 int3 20 
#The model score is 81.25% int1 13 int2 44 int3 23 
#The model score is 81.25% int1 13 int2 48 int3 31 
#The model score is 81.25% int1 17 int2 10 int3 42 
#The model score is 81.25% int1 17 int2 12 int3 29 
#The model score is 81.25% int1 26 int2 20 int3 8 

mlp.fit(x_norm, y)  
C = (mlp.score(x, y, sample_weight=None))*100
print('\nThe model score is {}% \n'.format(C)) # [loss , accuracy ]
# Predictions
s_frames = np.array(s_frames)# Normalize
s_frames_norm = scaler.transform([s_frames])# Normalize
Pred = mlp.predict([s_frames])
Pred = Pred[0]
prob_results = mlp.predict_proba(s_frames_norm)
prob = prob_results[0]*100
prob_Y = prob[0]
prob_N = prob[1]
print("Probability of Stressor occured: Yes {0:.3} % and No {1:.3}%".format(prob_N,prob_Y))

if Pred == 1:
    print('\nThe model predicts that a Stressor is Present')
else:
    print('\nThe model predicts that a Stressor is Not Present')