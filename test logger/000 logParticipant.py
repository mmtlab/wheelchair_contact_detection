# -*- coding: utf-8 -*-
"""
To record the data of the participant on a csv file
"""

import time
import datetime
import csv


#%% 
CSVfileName = 'test participant data.csv'  


subject = int(input("\nInsert the subject number: "))
start = time.time()
gender = input("Insert the gender: ")
age = int(input("Insert the age: "))
height = int(input("Insert the height [cm]: "))
weight = int(input("Insert the weight [kg]: "))
elbowWrist = int(input("Insert the elbow - wrist distance [cm]: "))
wristMiddleTip = int(input("Insert the wrist - middle finger tip distance [cm]: "))
handedneesScore = input("Insert the handedness score [100 = right; -100 = left]: ")


dateStart = datetime.datetime.fromtimestamp(start).strftime('%d/%m/%Y %H:%M:%S')



comment = input("do you want to add any comment? [don't use commas]: ")

newRow = [subject,  gender, age, height, weight, elbowWrist, wristMiddleTip,handedneesScore,\
          dateStart, comment]
    
  
f = open(CSVfileName, 'a', encoding='UTF8', newline='')
writer = csv.writer(f)
writer.writerow(newRow)
f.close()

print('\nParticipant registered successfully')



#%% to initialize the file
# newRow = ['subject', 'gender', 'age', 'height', 'weight', 'elbow - wrist', \
#           'wrist - middle finger tip','handedness','dateStart', 'comment']
# CSVfileName = 'test participant data head.csv'    
# f = open(CSVfileName, 'w', encoding='UTF8', newline='')
# writer = csv.writer(f)
# writer.writerow(newRow)
# f.close()

