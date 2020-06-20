#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 22:20:17 2019

@author: kendalljohnson
"""

# Heart Rate Monitor 1

from gpiozero import MCP3008, LED
from gpiozero import TonalBuzzer
from gpiozero.tones import Tone
from time import sleep
import pandas as pd                  # DataFrame Tool
import numpy as np                   # Basically a great calculator for now
from sklearn import linear_model 

# Analog inputs
HRM = MCP3008(channel = 0)  # Heart rate moniter
Temp = MCP3008(channel = 1) # Temperture sensor
HRM2 = MCP3008(channel = 2) # Possible Photo sensor
# left = MCP3008(channel = 3)

# Buzzer
buzzer = TonalBuzzer(17) # Buzzer

# LED
red = LED(22)
green = LED(27)

# Named Varibles
   
def Noise1():
    buzzer.play(Tone('A4'))
    sleep(.3)
    buzzer.play(Tone('B4'))
    sleep(.15)
    buzzer.play(Tone('C4'))
    sleep(.15)
    buzzer.play(Tone('D4'))
    sleep(.35)
    
def Noise2():
    buzzer.play(Tone('A4'))
    sleep(.1)
    buzzer.play(Tone('E4'))
    sleep(.15)

def Noise3():
    buzzer.play(Tone('E4'))
    sleep(.1)
    buzzer.play(Tone('A4'))
    sleep(.15)
    
def Noise4():
    buzzer.play(Tone('C4'))
    sleep(.1)
    buzzer.play(Tone('B4'))
    sleep(.15)

def BlinkR():
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
    
def BlinkG():
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
    
def BlinkRG():
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
    HR = HRM.value * 160    # Normalized heart rate multpied by the high of that person
    Tem = Temp.value * 100 # 
    HR2 = HRM2.value *160 # Normalized heart rate multpied by the high of that person
    if HR < 45:
        print('No Heart Rate received')
        Noise2()
    else:
        print('Heart rate in bpm:' ,HR,'Temp in F:' ,Tem,'Heart rate 2 in bpm' ,HR2,)  # printing the values of the body
        sleep(.5)   # wait half a second
        
        # Putting incoming heart rates in a list
        HR = []   # blank list
        HR.append(HR)  # adding HRs to blank list
        HR_val = len(HR) # Number of charcter in list 
        for i in range(HR_val):
            A = HR[i]- HR[i-1]
            if abs(A) >  10:  
                Noise3()
                def IBI(H):
                    IBI = 60000/H
                    return IBI # in miliseconds
                
                Real_IBI = []
                for i in range (HR_val):
                    A = IBI(HR[i])
                    IBI.append(A)
                    
                
                
            else:
                print('dont run algo because Heart Rate change is not significent')
                Noise2()
            
        
        

    
    










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
            A = New_HR[i+1] - New_HR[i]         # the difference in Heart rates
            Per1 = .3                           # Perameter for Size of Difference of Heart rates to start code # the starting heart rate is New_HR[i] // above .5
            Per2 = 1.05                         # Perameter for size of ratio between starting heart rate New_HR[i] and average of next 17 seconds of heart rate
            Per2a = .95                         # Perameter for size of ratio between starting heart rate New_HR[i] and average of next 17 seconds of heart rate
            Per3 = .5                           # Size of a unhealthy pNN50 score in percentage
            Per4 = 30                           # Size of a unhealthy SDNN score in standard deviation score
            Per5 = .5                           # Size of a unhealthy pNN20 score in percentage
            if abs(A) > Per1:                   # Getting Sig Heart rate
                F = len(Neww_HR)                # Number of designated heart beats
                Neww_HR.append(New_HR[i])       # If phyical activity 
                B = New_HR[i]+ New_HR[i+1]+ New_HR[i+2]+ New_HR[i+3]+ New_HR[i+4]+ New_HR[i+5]+ New_HR[i+6]+ New_HR[i+7]+ New_HR[i+8]+ New_HR[i+9] + New_HR[i+10] + New_HR[i+11] + New_HR[i+12]+ New_HR[i+13]+ New_HR[i+14]+ New_HR[i+15]+ New_HR[i+16] 
                C = B/17 
                D = New_HR[i]
                E = D/C
                if E > Per2 or E < Per2a:       # make sure you are not running
                    # Gettin Real IBI
                    for j in range(F):          # Calculated IBI
                        for k in range(Pred_IBI_size):
                            G = Neww_HR[j]
                            H = IBI(G)
                            Real_IBI.append(H)  # putting numbers in python list
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
                            S = Real_IBI[k] - Pred_IBI[k]
                            SS = S**2  
                            Cost_IBI.append(SS)     # putting numbers in python list            
                            Sum_C = sum(Cost_IBI)   
                            Cost = Sum_C / Pred_IBI_size
                        # Get totals of NN difference
                            J = Real_IBI[k]-Real_IBI[k-1] # N - (N-1)
                            preNN50.append(J)       # putting numbers in python list
                        # Get totals of NN difference         
                            K = Real_IBI[k]-Real_IBI[k-1]# N - (N-1) in mircosecond 
                            preNN20.append(K)       # putting numbers in python list
                    # Get NN50 and pNN50 
                            if np.abs(preNN50[k]) > 50: # getting values bigger then 50 ms
                                NN50.append(preNN50[k])   # putting numbers in python list
                                pNN50 = (len(NN50))/(len(preNN50))*100 # Equation to get percentage of NN50
                    # Get NN20 and pNN20                     
                            if np.abs(preNN20[k]) > 20:# getting values bigger then 20 ms
                                NN20.append(preNN20[k])# putting numbers in python list
                                pNN20 = (len(NN20))/(len(preNN20))*100 # Equation to get percentage of NN20
                            
                                if pNN50 > Per3 and SDNN > Per4 and pNN20 > Per5:
                                    print('A relapse has certeinly occured')
                                    print('The SDNN is {0},The NN50 is {1},The pNN50 is {2},The NN20 is {3},The pNN20 is {4},The Cost is {5},The score of the model is {6}, this is iteration {7}'.format(SDNN,len(NN50),pNN50,len(NN20),pNN20,Cost,SC,i))
                                    print('it1 is {0}, it2 is {1}, it3 is {2}'.format(i,j,k))
                                elif pNN50 == 0 and SDNN > Per4 and pNN20 > Per5:
                                    print('A relapse has likely occured')
                                    print('The SDNN is {0},The NN50 is {1},The pNN50 is {2},The NN20 is {3},The pNN20 is {4},The Cost is {5},The score of the model is {6}, this is iteration {7}'.format(SDNN,len(NN50),pNN50,len(NN20),pNN20,Cost,SC,i))
                                    print('it1 is {0}, it2 is {1}, it3 is {2}'.format(i,j,k))                        
                                elif pNN50 == 0 and SDNN > Per4 and pNN20 == 0:
                                    print('A relapse has possibly occured')
                                    print('The SDNN is {0},The NN50 is {1},The pNN50 is {2},The NN20 is {3},The pNN20 is {4},The Cost is {5},The score of the model is {6}, this is iteration {7}'.format(SDNN,len(NN50),pNN50,len(NN20),pNN20,Cost,SC,i))
                                    print('it1 is {0}, it2 is {1}, it3 is {2}'.format(i,j,k))                        
                                else: 
                                    print('A relapse has certeinly not occured')
                                    print('The SDNN is {0},The NN50 is {1},The pNN50 is {2},The NN20 is {3},The pNN20 is {4},The Cost is {5},The score of the model is {6}, this is iteration {7}'.format(SDNN,len(NN50),pNN50,len(NN20),pNN20,Cost,SC,i))
                                    print('it1 is {0}, it2 is {1}, it3 is {2}'.format(i,j,k))
                else:
                    print('dont run algo because is due to phyical activity')    
            else:
                print('dont run algo because Heart Rate is not significent')
            
        
        

