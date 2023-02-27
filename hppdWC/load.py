"""

"""
#%% imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pyrealsense2 as rs
import mediapipe



import sys
import keyboard
import os

import csv

import time
import datetime
import tqdm

from . import analysis 
from . import bagRS
from . import runMP 
from . import utils

#%% functions
def loadMW(fileCompleteName):
    '''
    Loads the data from the measurement wheel, that can be in .dat or in .data 
    format. 
    The first 11 lines are skipped, since are information like this:
        %  Name/ID : 20220428_MW 
        %  Gender:  male
        %  Age: 25
        %  weight(kg):80
        %  Height(cm):180
        %  Wheel Size :24 in/540mm
        %  Wheel Side :Right 
        %  S/N :PRO00008
        %  Comment:%  Data were taken at mm/dd/yyyy HH:MM:SS PM
        % Comment : 
        %%%%%
    Since the data can be in .dat or in .data format, a control on the format is
    executed before reading the file

    Parameters
    ----------
    fileCompleteName : string
        Complete path to the .dat or .data file

    Returns
    -------
    mw_data : pandas dataframe
        of 10 columns: time PushStart PushEnd Fx Fy Fz Tx Ty Tz Angle

    Angle is expressed in rad

    '''
    # making sure to start always with a dat file
    # if it's a dat file, then pass
    if fileCompleteName[-4:] == '.dat':
        pass
    # if it's a data file, then make it dat
    elif fileCompleteName[-5:] == '.data':
        fileCompleteName = fileCompleteName[:-1]
    # if it has no extension or a different extension, add .dat
    else:
        fileCompleteName = fileCompleteName + '.dat'
    
    try:
    # try to load the .dat file
        try: 
            dataset=[i.strip().split() for i in open(fileCompleteName).readlines()]
        # try to load the .data file   
        except (FileNotFoundError, IOError): 
            dataset=[i.strip().split() for i in open(fileCompleteName+'a').readlines()]
    # if no dat and not even data extensions are valid, no file is available
    except (FileNotFoundError, IOError):
        print('The file is not found')
        return None
        
    names = ['time', 'Push Start', 'Push End', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz', 'Angle deg']
    # skipping the first 12 rows
    mw_data = pd.DataFrame(dataset[12:], columns = names).astype(float)
    # flipping the torque in Z
    mw_data["Tz"] *= -1

    mw_data['AngSpeed deg'] = mw_data['Angle deg'].diff()
    mw_data['AngAcc deg'] = mw_data['AngSpeed deg'].diff()


    
    return mw_data
