
# Read data save it to a file :: would like to try and get more accurate data

# Heart Rate Monitor 2

from gpiozero import MCP3008, LED
from gpiozero import TonalBuzzer
from gpiozero.tones import Tone
from time import sleep
import time
import pandas as pd
#import sys
import csv

# Analog inputs
HRM = MCP3008(channel = 0)  # Heart rate moniter
Temp = MCP3008(channel = 1) # Temperture sensor

# Buzzer
buzzer = TonalBuzzer(27) # Buzzer


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
# IBI equation
    
def IBI(HR):
    IBI = 60000/HR
    return IBI

# Using training examples from K-means to have predicted HR's and IBI's 

dft = pd.read_excel('Full_Train.xlsx') # Pateint data from Dr. Hollys readings 
HR_A = dft.values[:,0]                 # Heart Rate in beats per minute
HR_K = dft.values[:,1]                 # Heart Rate in beats per minute
IBI_A = dft.values[:,2]                # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus  # Ignore first 10 secs
IBI_K = dft.values[:,3]                # IBI is a varible giving from hearts beat anaylsis that is a great indicator of a person's stimulus

################################# Loop to get real values ##################

csvfile = "Heart1.csv"
while True:
    HR = HRM.value * 95
    #HR = HRM.value * A + B
    Temp = Temp.value * 180
    IBI = IBI(HR)
    
    if HR < 50:
        print('Heart Rate not read')
        Noise4()
        sleep(.3)
    elif HR > 50:
        print('Your Heart Rate is {0:0.1f} bpm, Your IBI is {1:0.f} ms, and Temperture is {2:0.1f} C, '.format(HR,IBI,Temp))
    else:
        if HR > HR_Ks and IBI < IBI_Ks:   # all are stimulaeted values
            print("Possible ")
            timeC = time.strftime("%I")+':'+time.strftime("%M")+':'+time.strftime("%S")
            data = [HR,IBI,Temp,timeC]
            with open(csvfile,"a")as output:
                writer = csv.writer(output, delimiter=",",lineterminator = '\n')
                writer.writerow(data)
        sleep(.5)
            
        
