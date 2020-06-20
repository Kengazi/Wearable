#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:01:45 2019

@author: kendalljohnson
"""
# By hand K-Means of HR to IBI data // not needed
# Imports ::
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

######################### Plot 1 ###########################
print('Plot 1 : Putting all on grid stage')
np.random.seed(200)
k = 3

# Centroid[i] = [x,y]
centroids = {
    i+1: [np.random.randint(55,120),np.random.randint(500,1000)]
    for i in range(k)
    }

fig = plt.figure(figsize=(5,5))
plt.scatter(HR_df,IBI_df,color='k')
colmap = {1: 'r',2: 'g',3:'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(65,100)
plt.ylim(500,1000)
plt.title('K-Means of IBI(HR)')
plt.xlabel('Heart rate in BPM')
plt.ylabel('Interbeat Intervals in milliseconds')
plt.grid()
plt.show()
######################### Plot 2 ###########################

# The Assignment Stage

def assignment(df, centroids):
    for i in centroids.keys():
        df['distance_from_{}'.format(i)] = np.sqrt((HR_df - centroids[i][0]) **2 + (IBI_df - centroids[i][1])**2)
            
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

df = assignment(df, centroids)
print(df.head())
print('Plot 2: Assignment Stage')
fig = plt.figure(figsize=(5,5))
plt.scatter(HR_df,IBI_df, color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(65,100)
plt.ylim(500,1000)
plt.title('K-Means of IBI(HR)')
plt.xlabel('Heart rate in BPM')
plt.ylabel('Interbeat Intervals in milliseconds')
plt.grid()
plt.show()

######################### Plot 3 ###########################
print('Plot 3: Update Stage')
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
plt.scatter(HR_df,IBI_df, color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(65,100)
plt.ylim(500,1000)
plt.title('K-Means of IBI(HR)')
plt.xlabel('Heart rate in BPM')
plt.ylabel('Interbeat Intervals in milliseconds')
plt.grid()
for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc = colmap[i], ec=colmap[i])   
plt.show()

######################### Plot 4 ####################################

df = assignment(df, centroids)

print(df.head())
print('Plot 4: Re- Assignment Stage')
fig = plt.figure(figsize=(5,5))
plt.scatter(HR_df,IBI_df, color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(65,100)
plt.ylim(500,1000)
plt.title('K-Means of IBI(HR)')
plt.xlabel('Heart rate in BPM')
plt.ylabel('Interbeat Intervals in milliseconds')
plt.grid()
    
######################### Plot 5 ####################################
print('Plot 5: Repeat Assignment Stage')
while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df,centroids)
    if closest_centroids.equals(df['closest']):
        break
    
fig = plt.figure(figsize=(5,5))
plt.scatter(HR_df,IBI_df, color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(65,100)
plt.ylim(500,1000)
plt.title('K-Means of IBI(HR)')
plt.xlabel('Heart rate in BPM')
plt.ylabel('Interbeat Intervals in milliseconds')
plt.grid()
    