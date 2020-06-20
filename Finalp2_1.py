#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 21:15:06 2019

@author: kendalljohnson
"""

# THe file that uses Sensor data with training data to be calc Using machine learning for the Heart of an opionid 
print('Training data into excel with all values plus stimulas')


# All together

# Imports :: modules used to create code

import pandas as pd                  # DataFrame Tool
import numpy as np                  # Basically a great calculator for now
#import matplotlib.pyplot as plt      # For 2-D Graphing
#import scipy.optimize as so
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# Data (Incoming sensor data)  :: physical numbers we are analysing 

df = pd.read_csv('Heart2.csv') # Reading the active data

HR = df.values[:,0]
HR = HR[1:]

IBI = df.values[:,2]                 # Heart Rate in beats per minute
IBI = IBI[1:]

Temp = df.values[:,1]                 # Heart Rate in beats per minute
Temp = Temp[1:]

Time = df.values[:,3]                 # Heart Rate in beats per minute
Time = Time[1:]

HRs_size = len(IBI)

# Varables :: Getting varables from real IBI DataFrame for example the mean, median, standard dev from incoming sensor data

preNN50 = []
preNN20 = []
for i in range(HRs_size-1):
    A = IBI[i+1]-IBI[i]
    preNN50.append(A)
    preNN20.append(A)

NN50 = []
for i in range(HRs_size-1):
    if np.abs(preNN50[i])>50:
        NN50.append(preNN50[i])
        
pNN50 = (len(NN50))/(len(preNN50))*100

NN20 = []
for i in range(HRs_size-1):
    if np.abs(preNN20[i])>20:
        NN20.append(preNN20[i])
        
pNN20 = (len(NN20))/(len(preNN20))*100


IBI = np.array(IBI)
IBI = pd.DataFrame(IBI)

SDNN = IBI.std() # SDNN
SDNN = SDNN[0]
    
print('The SDNN for incoming data is {:.4}'.format(SDNN))
print('The NN50 for incoming data is {}'.format(len(NN50)))
print('The pNN50 for incoming data is {:.5}%'.format(pNN50))
print('The NN20 for incoming data is {}'.format(len(NN20)))
print('The pNN20 for incoming data is {:.5}%'.format(pNN20))

# Making data frames

HR = np.array(HR)
HR = pd.DataFrame(HR)    
IBI = np.array(IBI)
IBI = pd.DataFrame(IBI)
SDNN = np.array(SDNN)
SDNN = pd.DataFrame([SDNN])
pNN20 = np.array(pNN20)
pNN20 = pd.DataFrame([pNN20])
pNN50 = np.array(pNN50)
pNN50 = pd.DataFrame([pNN50])

# Putting Frames together 

s_frames = [HR,IBI,SDNN,pNN20,pNN50]   # completly binary Need to add a place for target stimulus
s_result = pd.concat(s_frames, axis=1)

# Data (Training data)  :: Sectioned data that is a positive stimulus 

dft = pd.read_excel('Full_Train.xlsx')    # Pateint data from Dr. Hollys readings 
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

# Making DataFrames

HR_A = pd.DataFrame(HR_A)
HR_K = pd.DataFrame(HR_K)
IBI_A = pd.DataFrame(IBI_A)
IBI_K = pd.DataFrame(IBI_K)
SDNNt = pd.DataFrame(SDNNt)
pNN20t = pd.DataFrame(pNN20t)
pNN50t = pd.DataFrame(pNN50t)
countt = pd.DataFrame(countt)
Stim = pd.DataFrame(Stim)



# Putting Frames together

frame1 = [HR_A,HR_K,IBI_A,IBI_K,SDNNt,pNN20t,pNN50t]
frame2 = [HR_K,IBI_K,SDNNt,pNN20t,pNN50t]
result1 = pd.concat(frame1, axis=1)
result2 = pd.concat(frame2, axis=1)

# Machine learning codes ::

# Machince Learning :: Linear Regression  

Lin_Reg = LinearRegression()
Lin_Reg.fit(result2,Stim)
LinPred = Lin_Reg.predict(s_result)
LinPred = LinPred[0]
print('The stim is {:}'.format(LinPred))
print("The Linear Regression Score is {}".format(Lin_Reg.score(result2,Stim)))

# Machince Learning :: Logistic Regression
Log_Reg = LogisticRegression()
Log_Reg.fit(result2,Stim)
LogPred = Log_Reg.predict(s_result)
LogPred = LogPred[0]
print('The stim is {:}'.format(LogPred))
print("The  Logistic Regression Score is {}".format(Log_Reg.score(result2,Stim)))

# Machince Learning :: Support Vector Machine 
SVM = SVC()
SVM.fit(result2,Stim) #,criterion='4')
SVMPred = SVM.predict(s_result)
SVMPred = SVMPred[0]
print('The stim is {:}'.format(SVMPred))
print("The  Support Vector Machine Score is {}".format(SVM.score(result2,Stim)))

# Machince Learning :: Decision Tree
Tree = tree.DecisionTreeClassifier()
Tree.fit(result2,Stim) #,criterion='4')
TreePred = Tree.predict(s_result)
TreePred = TreePred[0]
print('The stim is {:}'.format(TreePred))
print("The  Decision Tree Score is {}".format(Tree.score(result2,Stim)))

# Machince Learning :: Random Forest with 10 epochs
Forest10 = RandomForestClassifier(n_estimators=10)
Forest10.fit(result2,Stim) #,criterion='4')
Forest10Pred = Forest10.predict(s_result)
Forest10Pred = Forest10Pred[0]
print('The stim is {:}'.format(Forest10Pred))
print("The  Random Forest Score is {} with 10 epochs".format(Forest10.score(result2,Stim)))

# Machince Learning :: Random Forest with 20 epochs
Forest20 = RandomForestClassifier(n_estimators=20)
Forest20.fit(result2,Stim) #,criterion='4')
Forest20Pred = Forest20.predict(s_result)
Forest20Pred = Forest20Pred[0]
print('The stim is {:}'.format(Forest20Pred))
print("The  Random Forest Score is {} with 20 epochs".format(Forest20.score(result2,Stim)))

# Machince Learning :: Random Forest with 30 epochs
Forest30 = RandomForestClassifier(n_estimators=30)
Forest30.fit(result2,Stim) #,criterion='4')
Forest30Pred = Forest30.predict(s_result)
Forest30Pred = Forest30Pred[0]
print('The stim is {:}'.format(Forest30Pred))
print("The  Random Forest Score is {} with 30 epochs".format(Forest30.score(result2,Stim)))

# Machince Learning :: Random Forest with 40 epochs
Forest40 = RandomForestClassifier(n_estimators=40)
Forest40.fit(result2,Stim) #,criterion='4')
Forest40Pred = Forest40.predict(s_result)
Forest40Pred = Forest40Pred[0]
print('The stim is {:}'.format(Forest40Pred))
print("The  Random Forest Score is {} with 40 epochs".format(Forest40.score(result2,Stim)))


# Stimulus
#Stim = 1
Stim = np.array([Stim])
Stimu = Stim[0]
Stim_df = pd.DataFrame({'Stim':Stim})

if Stimu == 1:
    print('This is a stimulated event')
else:
    print('This is not a stimulated event')
    

# Save as excel
#result.to_excel(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/Train_33.xlsx')