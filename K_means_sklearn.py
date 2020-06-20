#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 06:49:57 2019

@author: kendalljohnson
"""

# Sk learn K means
# Imports ::
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Files ::
df = pd.read_excel('X_data.xlsx')
cnt = df.values[:,1]
HR = df.values[:,2]                 # Heart Rate in beats per minute
IBI = df.values[:,3]                # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus  
Pat_size = HR.size

# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev

HR_df = pd.DataFrame(HR)
IBI_df = pd.DataFrame(IBI)

# Putting frames together 

frames = [HR_df, IBI_df]
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
plt.scatter(HR_df,IBI_df, color=colors1,alpha=0.5,edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(65,100)
plt.ylim(500,1000)
plt.title('K-Means of IBI(HR)')
plt.xlabel('Heart rate in BPM')
plt.ylabel('Interbeat Intervals in milliseconds')
plt.grid()
    