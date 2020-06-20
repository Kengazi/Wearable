#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:36:04 2019

@author: kendalljohnson
"""

# Whole Thing 9
# We hope to demonstaight the Heart rate being read and having the M.L. algorhthem activated and determine if the indiviual has been stimulated or not

print('Whole thing 9')

# Imports :: modules used to create code

import pandas as pd                 # DataFrame Tool
import numpy as np                  # Basically a great calculator for now
#import matplotlib.pyplot as plt     # For 2-D Graphing
from sklearn import linear_model    # For our Linear Regression Model // Sklearn is a M.L. modules
#import scipy.optimize as so         # Basically a great calculator for now
#import statistics as st
#from sklearn.cross_validation import train_test_split

# Data (real/training data)  :: physical numbers we are analysing 

df = pd.read_excel('X_data.xlsx')
count = df.values[:,0]
HR_t = df.values[:,1]                 # Heart Rate in beats per minute
IBI_t = df.values[:,2]                # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus  # Ignore first 10 secs
HR_size = HR_t.size

# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

HR_df = pd.DataFrame(HR_t)
IBI_df = pd.DataFrame(IBI_t)

# Data (different patient data) :: 

df1 = pd.read_csv('HR4.csv') #df1 = pd.read_csv('Heart1.csv')
HR_in = df1.values[:,0]
IBI_in = df1.values[:,0]
#Pat_data = Pat_data[1:]              # slicing of unusable values
Pat_size1 = HR_in.size            # gettting the size of Data point we are using 
cont = np.linspace(0,Pat_size1,Pat_size1)

# Putting frames together 

frames = [HR_df, IBI_df]
Com_df = pd.concat(frames, axis=1)





# Lin Regression Model :: 

reg = linear_model.LinearRegression()# Linear Reg model
reg.fit(df[['HR']],df.IBI)           # Adding varibles to model 
A = reg.coef_
B = reg.intercept_



def Line(HR,m,b):
    y = m*HR + b
    return y

def Exer(T_avg,in_avg):
    A = in_avg / T_avg
    if A > 1.2 or A < 0.8:
        return(A)
#Y = A*Pat_data + B
    


############################## Main Body of Code ##############################
    
# Empty Arrays to store info
    
New_HR = Pat_data 
SC = reg.score(HR_df,IBI_df)
Real_IBI = []
Pred_IBI = []
Neww_HR = []
preNN50 = []
preNN20 = []
NN50 = []
NN20 = []
Cost_IBI = []

# Getting Sig Heart rate

for i in range(Pat_size1):
    A = abs(New_HR[i+1] - New_HR[i]) # the difference in Heart rates
    Per1 = .3 # Perameter for Size of Difference of Heart rates to start code # the starting heart rate is New_HR[i] // above .5
    Per2 = 1.05 # Perameter for size of ratio between starting heart rate New_HR[i] and average of next 17 seconds of heart rate
    Per2a = .95 # Perameter for size of ratio between starting heart rate New_HR[i] and average of next 17 seconds of heart rate
    Per3 = .5 # Size of a unhealthy pNN50 score in percentage
    Per4 = 30 # Size of a unhealthy SDNN score in standard deviation score
    Per5 = .5 # Size of a unhealthy pNN20 score in percentage
    if A > Per1:  # Getting Sig Heart rate
        F = len(Neww_HR)
        Neww_HR.append(New_HR[i])    # If phyical activity 
        D = New_HR[i]
        if D > Per2 or D < Per2a: # make sure you are not running
            # Gettin Real IBI
            for j in range(F):      # Calculated IBI
                G = Neww_HR[j]
                H = IBI(G)
                Real_IBI.append(H) # putting numbers in python list
            # Using Lin Reg
                I = Neww_HR[j]   
                pred = reg.predict([[I]])   # M.L. predicted value
                pred1 = pred[0]             # Getting number out of array 
                Pred_IBI.append(pred1)      # putting numbers in python list
                
            # Using dataframes to get standard dev
                Pred_IBI_size = len(Pred_IBI)
                Pred_IBI_df = pd.DataFrame(Pred_IBI)
                Std = Pred_IBI_df.std() # SDNN
                SDNN = Std[0]               # Getting number out of array 
            # Cost Function
                for k in range(Pred_IBI_size):
                    S = Real_IBI[k] - Pred_IBI[k]
                    SS = S**2  
                    Cost_IBI.append(SS)     # putting numbers in python list            
                    Sum_C = sum(Cost_IBI)   
                    Cost = Sum_C / Pred_IBI_size
            # Get totals of NN difference
                for s in range(Pred_IBI_size):
                    J = Real_IBI[s]-Real_IBI[s-1]
                    preNN50.append(J)       # putting numbers in python list
            # Get totals of NN difference         
                for r in range(Pred_IBI_size):
                    K = Real_IBI[r]-Real_IBI[r-1]
                    preNN20.append(K)       # putting numbers in python list
            # Get NN50 and pNN50 
                for t in range(Pred_IBI_size):
                    if np.abs(preNN50[t]) > 50: # getting values bigger then 50 ms
                        NN50.append(preNN50[t])   # putting numbers in python list
                        pNN50 = (len(NN50))/(len(preNN50))*100 # Equation to get percentage of NN50
            # Get NN20 and pNN20                     
                for y in range(Pred_IBI_size):
                    if np.abs(preNN20[y]) > 20:# getting values bigger then 20 ms
                        NN20.append(preNN20[y])# putting numbers in python list
                        pNN20 = (len(NN20))/(len(preNN20))*100 # Equation to get percentage of NN20
                    
                        if pNN50 > Per3 and SDNN > Per4 and pNN20 > Per5:
                            print('A relapse has certeinly occured')
                            print('The SDNN is {0},The NN50 is {1},The pNN50 is {2},The NN20 is {3},The pNN20 is {4},The Cost is {5},The score of the model is {6}, this is iteration {7}'.format(SDNN,len(NN50),pNN50,len(NN20),pNN20,Cost,SC,i))
                            print('it1 is {0}, it2 is {1}, it3 is {2}, it4 is {3}, it5 is {4}'.format(j,k,s,t,y))
                        elif pNN50 == 0 and SDNN > Per4 and pNN20 > Per5:
                            print('A relapse has likely occured')
                            print('The SDNN is {0},The NN50 is {1},The pNN50 is {2},The NN20 is {3},The pNN20 is {4},The Cost is {5},The score of the model is {6}, this is iteration {7}'.format(SDNN,len(NN50),pNN50,len(NN20),pNN20,Cost,SC,i))
                            print('it1 is {0}, it2 is {1}, it3 is {2}, it4 is {3}, it5 is {4}'.format(j,k,s,t,y))                        
                        elif pNN50 == 0 and SDNN > Per4 and pNN20 == 0:
                            print('A relapse has possibly occured')
                            print('The SDNN is {0},The NN50 is {1},The pNN50 is {2},The NN20 is {3},The pNN20 is {4},The Cost is {5},The score of the model is {6}, this is iteration {7}'.format(SDNN,len(NN50),pNN50,len(NN20),pNN20,Cost,SC,i))
                            print('it1 is {0}, it2 is {1}, it3 is {2}, it4 is {3}, it5 is {4}'.format(j,k,s,t,y))                        
                        else: 
                            print('A relapse has certeinly not occured')
                            print('The SDNN is {0},The NN50 is {1},The pNN50 is {2},The NN20 is {3},The pNN20 is {4},The Cost is {5},The score of the model is {6}, this is iteration {7}'.format(SDNN,len(NN50),pNN50,len(NN20),pNN20,Cost,SC,i))
                            print('it1 is {0}, it2 is {1}, it3 is {2}, it4 is {3}, it5 is {4}'.format(j,k,s,t,y))
        else:
            print('dont run algo because is due to phyical activity')    
    else:
        print('dont run algo because Heart Rate is not significent')
