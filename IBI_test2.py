#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 07:13:50 2019

@author: kendalljohnson
"""
# Atempt to use K-means on random file

# Imports ::

#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import math as m
from sklearn.cluster import KMeans

# Files ::

df = pd.read_csv('IBI100.csv')               # Heart Rate in beats per minute
time2 = df.values[:,0]                # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus 
#time2 = time2[:278] 
IBI = df.values[:,1]
IBI = IBI*1000
#IBI = IBI[:278] 
#Pat_size = HR.size

# Clip data

Stim1t = time2[:112]
Stim1IBI = IBI[:112]
Stim2t = time2[113:156]
Stim2IBI = IBI[113:156]
Stim3t = time2[157:278]
Stim3IBI = IBI[157:278]



# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

T_df = pd.DataFrame(time2)
IBI_df = pd.DataFrame(IBI)

# Putting frames together 

frames = [T_df, IBI_df]
Com_df = pd.concat(frames, axis=1)

# Color Map

colmap = {1: 'r',2: 'g',3:'b',4:'y',5:'k',6:'c',7:'w'}

# Machine Learning :: Using K-means

kmeans = KMeans(n_clusters=2)
kmeans.fit(Com_df)

# Main Body
labels = kmeans.predict(Com_df)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(5,5))
colors = map(lambda x: colmap[x+1],labels)
colors1 = list(colors)
plt.scatter(T_df,IBI_df, color=colors1,alpha=0.5,edgecolor='k')
Cent = []
for idx, centroid in enumerate(centroids):
    Cent.append(centroid)
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0,420)
plt.ylim(550,900)
plt.title('K-Means of IBI(time)')
#plt.scatter(Stim1t,Stim1IBI,marker='+',label = 'Part 1')
#plt.scatter(Stim2t,Stim2IBI,marker='^',label = 'Part 2')
#plt.scatter(Stim3t,Stim3IBI,marker='*',label = 'Part 3')
plt.xlabel('Time')
plt.ylabel('Interbeat Intervals in milliseconds')
plt.legend()
plt.grid()

Cent = pd.DataFrame(Cent)

A = Cent.values[:,1]
print('IBI y values:',A)