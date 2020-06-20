#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 07:46:38 2019

@author: kendalljohnson
"""

# Imports ::
#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import math as m
from sklearn.cluster import KMeans


# Files ::
df = pd.read_excel('Pat_X.xlsx')               # Heart Rate in beats per minute
time2 = df.values[:,3]                # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus 
time2 = time2[:278] 
IBI = df.values[:,4]
IBI = IBI[:278] 
#Pat_size = HR.size


# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

T_df = pd.DataFrame(time2)
IBI_df = pd.DataFrame(IBI)

# Putting frames together 

frames = [T_df, IBI_df]
Com_df = pd.concat(frames, axis=1)

# Color Map

colmap = {1: 'r',2: 'g',3:'b'}

# Machine Leanring

kmeans = KMeans(n_clusters=3)
kmeans.fit(Com_df)

# Main Body
labels = kmeans.predict(Com_df)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(5,5))
colors = map(lambda x: colmap[x+1],labels)
colors1 = list(colors)
plt.scatter(T_df,IBI_df, color=colors1,alpha=0.5,edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0,1100)
plt.ylim(500,1000)
plt.title('K-Means of IBI(HR)')
plt.xlabel('Time intervals')
plt.ylabel('Interbeat Intervals in milliseconds')
plt.grid()