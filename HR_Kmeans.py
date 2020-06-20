#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:44:39 2019

@author: kendalljohnson
"""
# Imports ::
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m


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

np.random.seed(200)
k = 3

# Centroid[i] = [x,y]
centroids = {
    i+1: [np.random.randint(0,1100),np.random.randint(500,1100)]
    for i in range(k)
    }

fig = plt.figure(figsize=(5,5))
plt.scatter(T_df,IBI_df,color='k')
colmap = {1: 'r',2: 'g',3:'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,1100)
plt.ylim(500,1100)
plt.title('K-Means of IBI(Time)')
plt.xlabel('Time in a incrament')
plt.ylabel('Interbeat Intervals in milliseconds')
plt.grid()
plt.show()

# The Assignment Stage

def assignment(df, centroids):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = np.sqrt((T_df - centroids[i][0]) **2 + (IBI_df - centroids[i][1])**2)     
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

df = assignment(df, centroids)
print(df.head())

fig = plt.figure(figsize=(5,5))
plt.scatter(T_df,IBI_df, color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,1100)
plt.ylim(500,1100)
plt.title('K-Means of IBI(Time)')
plt.xlabel('Time in a incrament')
plt.ylabel('Interbeat Intervals in milliseconds')
plt.grid()
plt.show()

import copy
old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest']== i ]['HR'])
        centroids[i][1] = np.mean(df[df['closest']== i]['IBI'])
    return k
        
centroids = update(centroids)

fig = plt.figure(figsize=(5,5))
ax = plt.axes()
plt.scatter(T_df,IBI_df, color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,1100)
plt.ylim(500,1100)
plt.title('K-Means of IBI(Time)')
plt.xlabel('Time in a incrament')
plt.ylabel('Interbeat Intervals in milliseconds')
plt.grid()
for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc = colmap[i], ec=colmap[i])   
plt.show()
    

    