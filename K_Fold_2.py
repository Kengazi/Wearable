#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:28:35 2019

@author: kendalljohnson
"""

# Imports :: modules used to create code
#import math as m
import pandas as pd                 # DataFrame Tool
#import numpy as np                  # Basically a great calculator for now
#import matplotlib.pyplot as plt     # For 2-D Graphing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn import tree

# Data
digits = load_digits()

df = pd.read_excel('SVM_prep.xlsx')
HR = df.values[:,0]
IBI = df.values[:,1]                 # Heart Rate in beats per minute
Stim = df.values[:,2]

# Making Data Frames
HR = pd.DataFrame(HR)
IBI = pd.DataFrame(IBI)           
Stim = pd.DataFrame(Stim)       
# Putting frames together 
frame1 = [HR,IBI] # add a frame with the event determination 
Triggers = pd.concat(frame1, axis=1)
# K Fold ::

KFold = KFold(n_splits = '3')
for train_index, test_index in KFold.split():
    print(train_index, test_index)

def GetScore(model,x_train, x_test,y_train, y_test):
    model.fit(x_train,y_train)
    print("The  Logistic Regression Score is {}".format(Log_Reg.score(x_test,y_test)))
    
# Stratified KFold ::
folds = StratifiedKFold(n_splits='3')
for train_index, test_index in KFold.split(Triggers):
    x_train, x_test,y_train, y_test = Triggers[train_index], Triggers[test_index],Stim[train_index],Stim[test_index]
    
print(GetScore(model,x_train, x_test,y_train, y_test))

# Machince Learning :: Linear Regression  

x_train, x_test,y_train, y_test = train_test_split(digits.data,digits.target,test_size = 0.3)
# Machince Learning :: Linear Regression

Lin_Reg = LinearRegression()
Lin_Reg.fit(x_train,y_train)
print("The Linear Regression Score is {}".format(Lin_Reg.score(x_test,y_test)))

# Machince Learning :: Logistic Regression
Log_Reg = LogisticRegression()
Log_Reg.fit(x_train,y_train)
print("The  Logistic Regression Score is {}".format(Log_Reg.score(x_test,y_test)))

# Machince Learning :: Support Vector Machine 
SVM = SVC()
SVM.fit(x_train,y_train) #,criterion='4')
print("The  Support Vector Machine Score is {}".format(SVM.score(x_test,y_test)))

# Machince Learning :: Decision Tree
Tree = tree.DecisionTreeClassifier()
Tree.fit(x_train,y_train) #,criterion='4')
print("The  Decision Tree Score is {}".format(Tree.score(x_test,y_test)))

# Machince Learning :: Random Forest with 10 epochs
Forest = RandomForestClassifier(n_estimators=10)
Forest.fit(x_train,y_train) #,criterion='4')
print("The  Random Forest Score is {} with 10 epochs".format(Forest.score(x_test,y_test)))

# Machince Learning :: Random Forest with 20 epochs
Forest = RandomForestClassifier(n_estimators=20)
Forest.fit(x_train,y_train) #,criterion='4')
print("The  Random Forest Score is {} with 20 epochs".format(Forest.score(x_test,y_test)))

# Machince Learning :: Random Forest with 30 epochs
Forest = RandomForestClassifier(n_estimators=30)
Forest.fit(x_train,y_train) #,criterion='4')
print("The  Random Forest Score is {} with 30 epochs".format(Forest.score(x_test,y_test)))

# Machince Learning :: Random Forest with 40 epochs
Forest = RandomForestClassifier(n_estimators=40)
Forest.fit(x_train,y_train) #,criterion='4')
print("The  Random Forest Score is {} with 40 epochs".format(Forest.score(x_test,y_test)))
# Save as excel
#result.to_excel(r'/Users/kendalljohnson/Desktop/s2019_phys_251_kj/SVM_prep.xlsx')