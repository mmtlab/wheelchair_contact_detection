# -*- coding: utf-8 -*-
"""
To create a series of random tests

@author: eferlius
"""
import numpy as np
import csv
import random

CSVfileName = r'G:\Shared drives\Wheelchair Ergometer\handrim contact detection\Software\Python\test logger\test order.csv'

newRow=['t{}'.format(i) for i in range(8)]
newRow.insert(0,'sub')
list_exec = ["S{}R{}{}R".format(speed, resistance, block) for speed in ['1','2'] for resistance in ['1','2'] for block in ['Y','N']]

f = open(CSVfileName, 'w', encoding='UTF8', newline='')
writer = csv.writer(f)
writer.writerow(newRow)
f.close()

for subject_num in range(100):
    
    random.shuffle(list_exec)
    list_exec.insert(0, 'P{}'.format(subject_num))
    newRow = list_exec
    # remove first element, which is the participant number
    list_exec = list_exec[1::]
        
    f = open(CSVfileName, 'a', encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerow(newRow)
    f.close()
    
print('done')
     
    