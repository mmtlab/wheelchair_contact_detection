# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:22:05 2023

@author: Alexk
"""


# Nog te doen:
#     gemiddelde straal van de vier verschillende handpatronen bekijken
#     minimale en maximale gemiddelden bekijken van de handpatronen
#     De threshold tussen die 4 handpatronen bepalen
#     Delay nog in verwerken(GPs zijn script)


import numpy as np
import pandas as pd
#%matplotlib qt
import matplotlib.pyplot as plt
import os
from hcd import xcorrelation
from mpl_toolkits import mplot3d
from hcd import capsleeve
def find_contact(cap_data,contact_threshold=-20):
    contact_data=pd.DataFrame(columns=['Time (s)','Contact','Position'])
    contact_data['Time (s)']=cap_data['Time (s)']
    for j in range(len(cap_data['Time (s)'])):
        position=[]
        for i in range(24):
           if cap_data['el'+str(i+1)][j]<contact_threshold:
               contact_data['Contact'][j]=True
               position.append('el'+str(i+1))
           if position==[]: 
                contact_data['Contact'][j]=False
        if position==[]:
            contact_data['Position'][j]=np.nan
        else:
            contact_data['Position'][j]=position
    # contact_data['Contact']=contact_data['Contact'].astype(float)
    return contact_data
# things to change
fileNameHandPos = 'hand_pattern_arc16_handposition.csv' #don't forget .csv 
fileNameCap = 'hand_pattern_arc16.csv' #don't forget .csv
filenameLed = 'hand_pattern_arc16.csv' #don't forget .csv
#Also change this path!!
cwd = os.chdir(r'G:/.shortcut-targets-by-id/1MjZH9XN1gWNoiyVyaLwpjTjfKx4AsRiw/handrim contact detection/Tests/20230511/01_raw/capacitive sleeve/hand_pattern_arc16/NGIMU - 0035EEA3')
# Make sure its the right capacitydata corresponding to the handposdata, see the other script to change it.
amountOfNonStrokesBefore = 1 #Normally during measurment this value should be 2, corresponding with the amount of touches before the first stroke, what the 3 times up and down is and LED-touch is
amountOfNonStrokesAfter = 1 #Normally during measurment this value should be 1, corresponding with the amount of touches after the last stroke, what the 3 times up and down is
THRESHOLD = 10 # change for the different sensitifty to distinquisch between a stroke and noise
showPlots = True
show3dPlots = False

#part of the loading of sensordata
sensordata = pd.read_csv('sensors.csv')

#Handposition Data and interpoletian
cwd = os.chdir(r'G:/.shortcut-targets-by-id/1MjZH9XN1gWNoiyVyaLwpjTjfKx4AsRiw/handrim contact detection/Tests/20230511/02_preprocessing/realsenseRight/handposition')
handPos=pd.read_csv(fileNameHandPos)
handPos['RadAngle[rad]']=xcorrelation.fillNanWithInterp(handPos['RadAngle[rad]'])
handPos['RadDistance[mm]']=xcorrelation.fillNanWithInterp(handPos['RadDistance[mm]'])

#Capacity Sleave Data 
cwd = os.chdir(r'G:/.shortcut-targets-by-id/1MjZH9XN1gWNoiyVyaLwpjTjfKx4AsRiw/handrim contact detection/Tests/20230511/02_preprocessing/capacitive sleeve')
capSleeveData = pd.read_csv(fileNameCap)
capSleeveData = pd.read_csv(r"C:\Users\Alexk\OneDrive\Documenten\School\RUG\jaar 3 bewegingswetenschappen(2022-2023)\BAP\HAND_PATTERN\hand_pattern_arc16\NGIMU - 0035EEA3\auxserial.csv")
capSleeveData=capsleeve.decode_data(capSleeveData)
capSleeveData=find_contact(capSleeveData,contact_threshold=-15)

cwd = os.chdir(r'G:/.shortcut-targets-by-id/1MjZH9XN1gWNoiyVyaLwpjTjfKx4AsRiw/handrim contact detection/Tests/20230511/02_preprocessing/realsenseRight/ledstatus')
led_data = pd.read_csv(filenameLed)
contact_time=capsleeve.find_first_bump(sensordata)
led_time=led_data["Time(s)"].iloc[-1].astype(float)
csini= led_time-contact_time 

Y = 0 #index doesnt change with the way the code deletes certain parts, so I use this to change the index for the Handpos part
T = 0 #index doesnt change with the way the code deletes certain parts, so I use this to change the index for the Capacity Sleeve part
#Function where the correlation between the camera and capacity sleeve changes the amount of data points
if csini < 0:
    csini = csini * -1
    D = capSleeveData['Time(s)'] > csini
    Q = [i for i, val in enumerate(D) if val]#gives the indexes of the true values
    T = Q[0]
    # capSleeveData = np.array(capSleeveData[T:len(capSleeveData['Time(s)'])+1])
    # capSleeveData = np.delete(capSleeveData, 0 , 1)
    # capSleeveData = pd.DataFrame(data = capSleeveData,columns = ['time(s)','Contact','Position'])
    capSleeveData['time(s)'] = [x - csini for x in capSleeveData['time(s)']]
    handPos = np.array(handPos)
    handPos = pd.DataFrame(data = handPos,columns = ["time","RadDistance[mm]","RadAngle[rad]","NormDistance[mm]"])
    
elif csini > 0:    
    D = handPos['time'] > csini
    Q = [i for i, val in enumerate(D) if val]
    Y = Q[0]
    handPos = np.array(handPos[Y:len(handPos['time'])+1])
    handPos = pd.DataFrame(data = handPos,columns = ["time","RadDistance[mm]","RadAngle[rad]","NormDistance[mm]"])
    handPos['time'] = [x - csini for x in handPos['time']]
    # capSleeveData = np.array(capSleeveData)
    # capSleeveData = np.delete(capSleeveData, 0 , 1)
    # capSleeveData = pd.DataFrame(data = capSleeveData,columns = ['time(s)','Contact','Position'])
# trueIndexCap = np.where(capSleeveData['Contact'])[0] 
# trueIndexCap = (np.where(capSleeveData.iloc[:][1]==True))
trueIndexCap = np.array(capSleeveData.index[capSleeveData['Contact'] == True].tolist())
# trueIndexCap = np.array(trueIndexCap)
differenceBetweenIndexes = [y-x for x, y in zip(trueIndexCap[:-1], trueIndexCap[1:])] # Calcutates the differences between the True-indexes 


startOfStroke = [trueIndexCap[0]]# creates the list 'startOfStroke' with already the first index when there was conatact in it

for i in range (len(differenceBetweenIndexes)):
    if differenceBetweenIndexes[i] >= THRESHOLD: 
        indexOfStroke = int(trueIndexCap[i+1]) 
        startOfStroke.append(indexOfStroke)
    elif differenceBetweenIndexes[i] < THRESHOLD and differenceBetweenIndexes[i] > 1: # replaces the random 'True' with False
        Z = trueIndexCap[i+1]
        capSleeveData['Contact'][Z] = False

startOfStroke = np.array(startOfStroke)
startOfStroke = startOfStroke[amountOfNonStrokesBefore:len(startOfStroke) - amountOfNonStrokesAfter]#This way you exclude the first touches during the correlation things and the last correlation touch


endOfContact = [] 

for i in range (len(differenceBetweenIndexes)):
    if differenceBetweenIndexes[i] >= THRESHOLD and i>0: 
        indexOfEndOfContact = int(trueIndexCap[i]) 
        endOfContact.append(indexOfEndOfContact)

endOfContact = np.array(endOfContact)
endOfContact = endOfContact[amountOfNonStrokesBefore:len(endOfContact) - amountOfNonStrokesAfter]#This way you exclude the first touches during the correlation things and the last correlation touch

nStrokes = len(endOfContact)# amount of strokes

timeStartStroke = capSleeveData['Time (s)'][startOfStroke]
timeEndContact = capSleeveData['Time (s)'][endOfContact]



startStrokeHandPos = []
for i in range(nStrokes+1):
    # calculate the difference array
    difference_Array = np.absolute(handPos['time']-timeStartStroke[startOfStroke[i]])
    # find the index of minimum element from the array
    index = difference_Array.argmin()
    startStrokeHandPos.append(index)


startStrokeHandPos = np.array(startStrokeHandPos)


endContactHandPos = []
for i in range(nStrokes):
    # calculate the difference array
    difference_Array = np.absolute(handPos['time']-timeEndContact[endOfContact[i]])
    # find the index of minimum element from the array
    index = difference_Array.argmin()
    endContactHandPos.append(index)


endContactHandPos = np.array(endContactHandPos)

# alles gaat nog goed

first_Stroke = handPos[startStrokeHandPos[0]:startStrokeHandPos[1]]

#Dit onderdeel is niet relevant voor jou! gaat al over de gemiddelde straal van de verschillende strokes ;)
AA_MeanRadius = np.mean(handPos['RadDistance[mm]'][startStrokeHandPos[0]:startStrokeHandPos[len(startStrokeHandPos)-1]])

#Gemiddelde over alle strokes zonder de Handcontacten
NonContact = []
Average = []
for i in range(nStrokes):
    Z = handPos['RadDistance[mm]'][endContactHandPos[i]:startStrokeHandPos[i+1]]
    NonContact.append(Z)
    Z = np.mean(NonContact[i])
    Average.append(Z)
   
    

AA_Kleinstegemiddelde = min(Average)
AA_Grootstegemiddelde = max(Average)
AA_AverageofAverage = np.mean(Average)


initialContactDegrees=[]
for i in range (nStrokes):
    radDataHandPos = np.array(handPos['RadAngle[rad]'][startStrokeHandPos[i]])
    #change zero point from horizontal on the right to vertical
    radDataHandPos=radDataHandPos-1.5707963268
    #change rad to def
    radDataHandPos=np.rad2deg(radDataHandPos)
    initialContactDegrees.append(radDataHandPos)

handReleaseDegrees=[]
for i in range (nStrokes):
    radDataHandPos = np.array(handPos['RadAngle[rad]'][endContactHandPos[i]])
    #change zero point from horizontal on the right to vertical
    radDataHandPos=radDataHandPos-1.5707963268
    #change rad to def
    radDataHandPos=np.rad2deg(radDataHandPos)
    handReleaseDegrees.append(radDataHandPos)


#functie die van poolcoordinaten naar cartesian coordinaten gaat. 
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

# Here the Cartesian coordinates are added to the variable handpos
handPos['x'],handPos['y'] = pol2cart(handPos['RadDistance[mm]'],handPos['RadAngle[rad]'])


if showPlots == True:
        #for i in range(nStrokes):
         # handPos.iloc[startStrokeHandPos[i]:startStrokeHandPos[i+1]].plot(x ='x', y ='y')
        # index = 0
        handPos.iloc[startStrokeHandPos[0]:startStrokeHandPos[20]].plot(x ='x', y ='y')
        #handPos.iloc[startStrokeHandPos[index]].plot(x ='x', y ='y', marker = 'o')
        #handPos.plot(x ='x', y ='y')
        #plt.plot(handpos['time'],handpos['RadAngle[rad]'])
        #plt.plot(sensordata['Time (s)'],sensordata['Accelerometer Y (g)'])
        plt.show()
    
if show3dPlots == True:
    
    plt.figure('3D figure')
    ax = plt.axes(projection='3d')
    ax.plot3D(handPos['x'], handPos['y'], handPos['NormDistance[mm]'], 'blue')
    plt.xlabel("x")
    plt.ylabel("y")
