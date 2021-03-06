#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:54:08 2019

@author: kendalljohnson
"""

# Heart Rate Monitor 
print('Using a Raspberry Pi for find HRV and perform Machine Learning on values to determine if Stressor is present')
# Import 
from gpiozero import MCP3008
from gpiozero import TonalBuzzer
from gpiozero.tones import Tone
from time import sleep
import time
import pandas as pd
import csv

# Analog inputs
HRM = MCP3008(channel = 0)  # Heart rate moniter
Temp = MCP3008(channel = 1) # Temperture sensor

# Buzzer
buzzer = TonalBuzzer(12) # Buzzer

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

# Using training examples from K-means to have predicted HR's and IBI's 

dft = pd.read_excel('Full_Train.xlsx') # Patient data from Dr. Hollys readings 
HR_A = dft.values[:,0]                 # Heart Rate in beats per minute
HR_K = dft.values[:,1]                 # Heart Rate in beats per minute
IBI_A = dft.values[:,2]                # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus  # Ignore first 10 secs
IBI_K = dft.values[:,3]                # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus

# Getting only stimulated values

HR_As = [HR_A[0],HR_A[2],HR_A[4],HR_A[6],HR_A[8],HR_A[10],HR_A[12],HR_A[14]]
HR_Ks = [HR_K[0],HR_K[2],HR_K[4],HR_K[6],HR_K[8],HR_K[10],HR_K[12],HR_K[14]]
IBI_As = [IBI_A[0],IBI_A[2],IBI_A[4],IBI_A[6],IBI_A[8],IBI_A[10],IBI_A[12],IBI_A[14]]
IBI_Ks = [IBI_K[0],IBI_K[2],IBI_K[4],IBI_K[6],IBI_K[8],IBI_K[10],IBI_K[12],IBI_K[14]]

# Putting into dataFrame

HR_As = pd.DataFrame(HR_As)
HR_Ks = pd.DataFrame(HR_Ks)
IBI_As = pd.DataFrame(IBI_As)
IBI_Ks = pd.DataFrame(IBI_Ks)

# Average value

HR_Asm = HR_As.mean() # mean
HR_Asm = HR_Asm[0]

HR_Ksm = HR_Ks.mean() # mean
HR_Ksm = HR_Ksm[0]

IBI_Asm = IBI_As.mean() # mean
IBI_Asm = IBI_Asm[0]

IBI_Ksm = IBI_Ks.mean() # mean
IBI_Ksm = IBI_Ksm[0]

################################# Loop to get real values ##################

csvfile = "Heart2.csv"
while True:
# IBI equation
    
    def IBI(HR):
        IBI = 60000/HR
        return IBI  
      
    HR = HRM.value * 85
    #HR = HRM.value * A + B
    Tem = Temp.value * 180
    IBI = IBI(HR)
    
    if HR < 40:
        print('Heart Rate not read')
        Noise4()
        sleep(.3)

    elif (HR > HR_Ksm and IBI < IBI_Ksm) or (IBI < IBI_Asm and HR > HR_Asm):   # all are stimulated values
        print(" Possible Stressor occured ")
        Noise1()
        print('Your Heart Rate is {0:0.1f} bpm, Your IBI is {1:0.1f} ms, and Temperture is {2:0.1f} F, '.format(HR,IBI,Tem))
        timeC = time.strftime("%I")+':'+time.strftime("%M")+':'+time.strftime("%S")
        data = [HR,IBI,Tem,timeC]
        with open(csvfile,"a")as output:
            writer = csv.writer(output, delimiter=",",lineterminator = '\n')
            writer.writerow(data)
        sleep(.3)
        Noise2()
        import Finalp2_7    
    else: 
        print('Your Heart Rate is {0:0.1f} bpm,Your IBI is {1:0.1f} ms, and Temperture is {2:0.1f} F, '.format(HR,IBI,Tem))
        Noise3()
        sleep(.3)
            