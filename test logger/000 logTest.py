# -*- coding: utf-8 -*-
"""
To log the executed tests on a csv file
"""

import time
import datetime
import csv

#%% 
CSVfileName = 'test recording.csv'  
decision = input("=" * 80 + "\nPress any key to record a new test, [q] to quit... ")
while decision != 'q':
    
    
    _ = input('\nAre the cables of cap sleeve plugged correctly?')
    testNum = int(input("\nInsert the test number: "))
    subject = int(input("Insert the subject number: "))
    speed = int(input("Insert the speed [1/2]: "))
    res = int(input("Insert the resistance [1/2]: "))
    racket = input("Is the test with the racket? [Y/N]: ")
    racket = racket.upper()
    run = input("Insert the run number [first run is 0]: ")
    

    
    _ = input("\nPress any key to get the test code... ")
    testName = "T{:03d}S{}S{}R{}{}R_run{}".format(testNum, subject, speed, res, racket, run)
        
    print('\nThe test code is: ' + testName)
    start = time.time()
    
    _ = input("\nPress any key to end the test...")
    end = time.time()
    
    dateStart = datetime.datetime.fromtimestamp(start).strftime('%d/%m/%Y %H:%M:%S')
    dateEnd = datetime.datetime.fromtimestamp(end).strftime('%d/%m/%Y %H:%M:%S')
    
    
    comment = input("\nDo you want to add any comment? [don't use commas]: ")
    
    newRow = [testName, dateStart, dateEnd, testNum, subject, speed, res, racket, run, comment]
        
      
    f = open(CSVfileName, 'a', encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerow(newRow)
    f.close()
    
    print('\nTest registered successfully')

    
    decision = input("=" * 80 + "\nPress any key to record a new test, [q] to quit... ")

# #%% to initialize the fileS
# newRow = ['testName', 'dateStart', 'dateEnd', 'testNum', 'subject', 'speed', 'res', 'racket',  'comment']
# CSVfileName = 'test recording head.csv'    
# f = open(CSVfileName, 'w', encoding='UTF8', newline='')
# writer = csv.writer(f)
# writer.writerow(newRow)
# f.close()


# %%
