#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:40:14 2019

@author: kendalljohnson
"""
from sklearn.neural_network import MLPClassifier
import pandas as pd
# Data  ::

df = pd.read_excel('Full_Train.xlsx') 
HR = df.values[:,0:7]
Stim = df.values[:,8]

# Varibles :: 

X = HR
y = Stim

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)  
clf.predict([[2., 2.], [-1., -2.]])