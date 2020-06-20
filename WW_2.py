#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:59:29 2019

@author: kendalljohnson
"""

# Heart Rate Monitor 1
print('Wrist wearable 1')

# Imports

from gpiozero import MCP3008, LED, TonalBuzzer      # Gpiozero is used to comm with pins on Raspberry Pi
from gpiozero.tones import Tone                     # Used to make defined sounds
from time import sleep                              # Used to control timing on Pi
import pandas as pd                                 # DataFrame Tool
import numpy as np                                  # Basically a great calculator for now
#from sklearn import linear_model                    # M.L. tool

# Analog inputs
HRM = MCP3008(channel = 0)                          # Heart rate moniter
Temp = MCP3008(channel = 1)                         # Temperture sensor
HRM2 = MCP3008(channel = 2)                         # Possible Photo sensor

# Buzzer
buzzer = TonalBuzzer(17)                            # Buzzer

# LED
red = LED(22)                                       # Red LED 
green = LED(27)                                     # Green LED

# Named Varibles
   
def Noise1():                                       # String of Buzzing 1
    buzzer.play(Tone('A4'))
    sleep(.3)
    buzzer.play(Tone('B4'))
    sleep(.15)
    buzzer.play(Tone('C4'))
    sleep(.15)
    buzzer.play(Tone('D4'))
    sleep(.35)
    
def Noise2():                                       # String of Buzzing 2
    buzzer.play(Tone('A4'))
    sleep(.1)
    buzzer.play(Tone('E4'))
    sleep(.15)

def Noise3():                                       # String of Buzzing 3
    buzzer.play(Tone('E4'))
    sleep(.1)
    buzzer.play(Tone('A4'))
    sleep(.15)
    
def Noise4():                                       # String of Buzzing 4
    buzzer.play(Tone('C4'))
    sleep(.1)
    buzzer.play(Tone('B4'))
    sleep(.15)

def BlinkR():                                       # String of Blinking Red
    red.on
    sleep(.3)
    red.off
    sleep(.3)
    red.on
    sleep(.3)
    red.off
    sleep(.3)
    red.on
    sleep(.3)
    red.off
    
def BlinkG():                                       # String of Blinking Green
    green.on
    sleep(.3)
    green.off
    sleep(.3)
    green.on
    sleep(.3)
    green.off
    sleep(.3)
    green.on
    sleep(.3)
    green.off
    
def BlinkRG():                                       # String of Blinking Red and Green
    green.on
    sleep(.3)
    green.off
    sleep(.1)
    red.on
    sleep(.3)
    red.off
    sleep(.1)
    green.on
    sleep(.3)
    green.off
    sleep(.1)
    red.on
    sleep(.3)
    red.off
    sleep(.1)
    green.on
    sleep(.3)
    green.off
    sleep(.1)
    red.on
    sleep(.3)
    red.off
    sleep(.1)
    green.on
    sleep(.3)
    green.off
    sleep(.1)
    red.on
    sleep(.3)

          ############################## Main Body of Code ##############################  
          
# Start the Heart rate monitor reading 
while True:
    
# Getting Values of the Sensors 
    HR = HRM.value * 160                            # Normalized heart rate multpied by the high of that person
    Tem = Temp.value * 104                          # Normalized Temperture multpied by the high temp
    HR2 = HRM2.value *160                           # Normalized heart rate multpied by the high of that person
    Per1 = 10                                       # Perameter for Size of Difference of Heart rates to start code # the starting heart rate is New_HR[i] // above .5
    Per2 = 1.05                                     # Perameter for size of ratio between starting heart rate New_HR[i] and average of next 17 seconds of heart rate
    Per2a = .95                                     # Perameter for size of ratio between starting heart rate New_HR[i] and average of next 17 seconds of heart rate
    Per3 = .5                                       # Size of a unhealthy pNN50 score in percentage
    Per4 = 30                                       # Size of a unhealthy SDNN score in standard deviation score
    Per5 = .5                                       # Size of a unhealthy pNN20 score in percentage
    
# If the Heart rate is less then 45 bpm we will assume there to be no heart beat
    if HR < 45:
        print('No Heart Rate received')
        Noise2()
        sleep(.5)                                   # wait half a second
    else:
        print('Heart rate in bpm:' ,HR,'Temp in F:' ,Tem,'Heart rate 2 in bpm' ,HR2,)  # printing the values of the body
        
# Putting incoming heart rates in a list
        HR = []                                     # blank list
        HR.append(HR)                               # adding HRs to blank list
        HR_val = len(HR)                            # Number of charicater in list 
        for i in range(HR_val+1):
            A = HR[i]- HR[i-1]
            if abs(A) >  Per1:                 
                Noise3()
                
# Getting the average IBI from calculations
                def IBI(H):                         # define Equation for IBI
                    IBI = 60000/H                   # Actual Equation
                    return IBI                      # in miliseconds
# Putting incoming IBIs in a list              
                Real_IBI = []
                for j in range (HR_val):
                    A = IBI(HR[j])
                    IBI.append(A)
                    IBI_val = len(IBI)
# The Normal - Normal 
                preNN50 = []
                preNN20 = []
                for k in range(IBI_val-1):
                    NN = abs(IBI[i+1]-IBI[i])
                    preNN20.append(NN)
                    preNN50.append(NN)
                    NN_val = len(NN)
                    
# Get NN50 and pNN50
                NN50 = []
                NN20 = []
                if np.abs(preNN50[k]) > 50:         # getting values bigger then 50 ms
                    NN50.append(preNN50[k])         # putting numbers in python list
                    pNN50 = (len(NN50))/(len(preNN50))*100 # Equation to get percentage of NN50
# Get NN20 and pNN20                     
                if np.abs(preNN20[k]) > 20:         # getting values bigger then 20 ms
                    NN20.append(preNN20[k])         # putting numbers in python list
                    pNN20 = (len(NN20))/(len(preNN20))*100 # Equation to get percentage of NN20
                    
# DataFrames  :: Simply will show all basic data on your set of data for example the mean, median, standard dev
                HR_df = pd.DataFrame(HR)
                IBI_df = pd.DataFrame(IBI)
                
# Putting frames together 
                frames = [HR_df, IBI_df]
                df = pd.concat(frames, axis=1)
                
# Varables from real IBI DataFrame for example the mean, median, standard dev
                Max = df.max()                      # Max for DataFrames
                Max_HR = Max[0]                     # Getting number out of array  
                MaxIBI = Max[1]                     # Getting number out of array
                Std = df.std()                      # SD for DataFrames
                SD_HR = Std[0]                      # Getting number out of array  
                SDNN = Std[1]                       # Getting number out of array
                Min = df.min()                      # Min for DataFrames
                Min_HR = Min[0]                     # Getting number out of array  
                MinIBI = Min[1]                     # Getting number out of array
                Mean = df.mean()                    # Mean for DataFrames
                Mean_HR = Mean[0]                   # Getting number out of array  
                MeanIBI = Mean[1]                   # Getting number out of array
                Sum = df.sum()                      # Sum for DataFrames
                Sum_HR = Sum[0]                     # Getting number out of array  
                SumIBI = Sum[1]                     # Getting number out of array
                
                D = Max_HR/Mean_HR                  # Exercising coeffiecent 
                
                print('The SDNN is {0},The NN50 is {1},The pNN50 is {2},The NN20 is {3},The pNN20 is {4},Exercise coeff is {5}, this is iteration {6}'.format(SDNN,len(NN50),pNN50,len(NN20),pNN20,D,i))
                print('it1 is {0}, it2 is {1}, it3 is {2}'.format(i,j,k))
                sleep(.5)                           # wait half a second
                
                
                #reg = linear_model.LinearRegression()# Linear Reg model
                #reg.fit(df[['HR']],df.IBI)         # Adding varibles to model 
                #A = reg.coef_
                #B = reg.intercept_
                                    
                
            else:
                print('dont run algo because Heart Rate change is not significent')
                Noise2()
                sleep(.5)                           # wait half a second
            
        
        
