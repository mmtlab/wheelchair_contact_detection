# -*- coding: utf-8 -*-
"""
To perform analysis for HPPD project
"""
#%% imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pyrealsense2 as rs
import mediapipe
import scipy




import os
import copy

import csv

import time
import datetime
import tqdm
import logging

import worklab as wl



from . import bagRS
from . import load 
from . import runMP 
from . import utils
from . import geom
from . import plots
#%% functions
def fromAbsTimeStampToRelTime(df):
    '''
    Given a pandas dataframe, computes the difference of all the elements of the 
    first column [:,0] with respect to the first one [0,0].
    Use this to pass from the timestamp to the relative time of execution of the
    test

    Parameters
    ----------
   df : pandas dataframe
       of 1+3*n columns: time x0 y0 z0 x1 y1 z1 ... xn yn zn
       The column time is in timestamp: e.g. 1.65*10^9

    Returns
    -------
    df_copy : pandas dataframe
        of 1+3*n columns: time x0 y0 z0 x1 y1 z1 ... xn yn zn
        The column time is in time: e.g. 0.03 

    '''
    # copy of the original dataframe
    df_copy = df.copy()
    # the first column is substituted with the difference respect to element 0
    df_copy.iloc[:,0] = df.iloc[:,0]-df.iloc[0,0]
    return df_copy

def scaleXYZonImageShape(df, image_width, image_height):
    '''
    Since mediapipe gives landmarks on the relative dimensions of the image
    [0,1], it's necessary to rescale these values on the real images dimension

    Parameters
    ----------
    df : pandas dataframe
        of 1+3*n columns: time x0 y0 z0 x1 y1 z1 ... xn yn zn
        The values of x y z are in the range [0,1]
    image_width : int
        Number of pixels in the x axis
    image_height : int
        Number of pixels in the y axis

    Returns
    -------
    df_copy : df : pandas dataframe
        of 1+3*n columns: time x0 y0 z0 x1 y1 z1 ... xn yn zn
        The values of x and z are in the range [0, image_width]
        The values of y are in the range [0, image_height]

    '''
    # getting the shape
    rows, cols = df.shape
    # how many iterations: taking into account that:
        # - the first column is the time
        # - every 3 columns are one landmark (xyz)
    iterations = int((cols - 1) / 3)
    
    # copy of the original dataframe
    df_copy = df.copy()
    
    # iterate per columns, correcting the values according to image dimension
    for i in range(iterations):
        # for x
        df_copy.iloc[:,i*3+1] = df.iloc[:,i*3+1] * image_width
        # for y
        df_copy.iloc[:,i*3+2] = df.iloc[:,i*3+2] * image_height
        # for z
        df_copy.iloc[:,i*3+3] = df.iloc[:,i*3+3] * image_width 
        
    return df_copy
    
def computeMean(df, listOfLandMarks):
    '''
    Computes the mean value of x, y, z position of the landmarks specified on 
    the list
    
    Being the input dataframe with the following layout:
        time  x00 y00 z00 x01 y01 z01 ... x20 y20 z20
    and declaring for example to compute the mean on the landMarks 1 and 3,
    the values to be considered will be:
        for x: 1, 7
        for y: 2, 8
        for z: 3, 9
    This is obtained multiplying for 3 the number of the landmark and adding
    1 for x, 2 for y and 3 for z
    Parameters
    ----------
    df : pandas dataframe
        of 64 columns: time  x00 y00 z00 x01 y01 z01 ... x20 y20 z20
    listOfLandMarks : List
        Contains the keypoints that will be considered to compute the mean

    Returns
    -------
    df_mean : pandas dataframe 
        of 4 columns: time x y z
        contains the mean values

    '''
    # getting the shape
    rows, cols = df.shape
    # creating an empty dataframe of given dimension
    df_mean = pd.DataFrame(np.zeros((rows, 4)), columns=['time', 'x', 'y','z'])
    # declaring which columns of the dataframe should be considered
    listOfLandMarks = utils.mulIntToList(listOfLandMarks, 3)

    # copying the time
    df_mean.iloc[:,0] = df.iloc[:,0]
    # mean executed on all the rows of the declared columns. axis = 1 because 
    # the mean is row wise
    # for x
    df_mean.iloc[:,1] = np.mean(df.iloc[:, utils.addIntToList(listOfLandMarks,1)], axis = 1)
    # for y
    df_mean.iloc[:,2] = np.mean(df.iloc[:, utils.addIntToList(listOfLandMarks,2)], axis = 1)
    # for z
    df_mean.iloc[:,3] = np.mean(df.iloc[:, utils.addIntToList(listOfLandMarks,3)], axis = 1)
    
    return df_mean

def handCenterPosition(df):
    '''
    Calls the function computeMean on the keypoint 0, 1, 5, 9, 13, 17

    Parameters
    ----------
    df : pandas dataframe
        of 64 columns: time  x00 y00 z00 x01 y01 z01 ... x20 y20 z20
        contains the coordinates of all the keypoints

    Returns
    -------
    df_centre : pandas dataframe 
        of 4 columns: time x y z
        contains the mean values

    '''
    df_centre = computeMean(df, [0,1,5,9,13,17])
    
    return df_centre

def drawRefFrameOnImage(image, x_wheel_centre, y_wheel_centre, color = (255, 0, 0)):
    '''
    Given an image and the coordinates of the centre of the ref frame, draws it
    x pointing right
    y pointing up
    
    
    ^ +y
    |
    |
    |-------> +x
    

    Parameters
    ----------
    image : image (matrix h*w*3)
        Input image where the ref frame should be drawn
    x_wheel_centre : int
        x position of the centre of the wheel
    y_wheel_centre : int
        y position of the centre of the wheel
    color : array of 3 integer from 0 to 255, optional
        To select the color of drawing. The default is (255, 0, 0).
    

    Returns
    -------
    img : image (matrix h*w*3)
        Output image with the ref frame drawn on it

    '''
    img = image.copy()
    image_height, image_width, _ = img.shape
    
    # renaming with shorter names
    h = image_height
    w = image_width
    xwc = x_wheel_centre
    ywc = y_wheel_centre
    
    # horizontal axis
    cv2.arrowedLine(img,(0,ywc),(w,ywc),color,2) 
    cv2.putText(img, '+x', (int(w*0.9),ywc), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2, cv2.LINE_AA)
    
    # vertical axis
    cv2.arrowedLine(img,(xwc,h),(xwc,0),color,2) 
    cv2.putText(img, '+y', (xwc,0+int(h*0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color,2, cv2.LINE_AA)
    
    return img

def angle3d(df, listOfKeyPoints):
    '''
    Given a set of 3 keypoints, in order [A, B, C], computes the direction of 
    the lines connecting AB and BC and computes the angle between the two
    

    Parameters
    ----------
    df : pandas dataframe
        of 64 columns: time  x00 y00 z00 x01 y01 z01 ... x20 y20 z20
        contains the coordinates of all the keypoints
    listOfKeyPoints : list
        of the interested keypoints (specified as integers).

    Returns
    -------
    values : numpy array of float 64
        array containing the cosine of the angle

    '''
    
    if len(listOfKeyPoints) == 3: 
        # get the column names
        columns = utils.correspondingColumns(listOfKeyPoints, letters = ['x','y','z'])
        
        # get the relative columns from the dataframe
        pointA = df[columns[0]]
        pointB = df[columns[1]]
        pointC = df[columns[2]]
    
        # extracting from the 3 column dataframe the column relative to x,y,z
        pointAx = pointA.iloc[:,0]
        pointAy = pointA.iloc[:,1]
        pointAz = pointA.iloc[:,2]
    
        pointBx = pointB.iloc[:,0]
        pointBy = pointB.iloc[:,1]
        pointBz = pointB.iloc[:,2]
    
        pointCx = pointC.iloc[:,0]
        pointCy = pointC.iloc[:,1]
        pointCz = pointC.iloc[:,2]
    
        # keeping sure that the variables are numpy.arrays
        pointAx, _ = utils.toFloatNumpyArray(pointAx)
        pointAy, _ = utils.toFloatNumpyArray(pointAy)
        pointAz, _ = utils.toFloatNumpyArray(pointAz)
    
        pointBx, _ = utils.toFloatNumpyArray(pointBx)
        pointBy, _ = utils.toFloatNumpyArray(pointBy)
        pointBz, _ = utils.toFloatNumpyArray(pointBz)
    
        pointCx, _ = utils.toFloatNumpyArray(pointCx)
        pointCy, _ = utils.toFloatNumpyArray(pointCy)
        pointCz, _ = utils.toFloatNumpyArray(pointCz)
    
        # checking that the length is the same
        if len(pointAx) == len(pointBx) and len(pointAx) == len(pointCx): 
            cos_angle_array = []
            for i in range(len(pointAx)):
                # create the 3d points
                pA = geom.Point3d(pointAx[i],pointAy[i],pointAz[i])
                pB = geom.Point3d(pointBx[i],pointBy[i],pointBz[i])
                pC = geom.Point3d(pointCx[i],pointCy[i],pointCz[i])
                
                # create the two lines
                lineAB = geom.Line3d(pA, pB)
                lineBC = geom.Line3d(pB, pC)
                
                # computing the cosine of the angle between the two lines
                cos_angle = lineAB.cosAngle(lineBC)
                cos_angle_array.append(cos_angle)
        
        # conversion to numpyArray
        cos_angle_array, _ = utils.toFloatNumpyArray(cos_angle_array)
        return cos_angle_array
    else:
        logging.error('requested to compute the 3d angle but {:d} instead of 3 points were given'.format(len(listOfKeyPoints)))
        return None
    
def computeJointAngles(df):
    '''
    Given a dataframe containing the xyz coordinates of each one of the 21 
    keypoints, computes the relative joint angles for the five fingers.
    In total 15 angles are given: 5 fingers * proximal, medial and distal
    The names of the columns are:
    - first letter: (p)roximal, (m)iddle, (d)istal
    - second letter: (t)humb, (i)ndex, (m)iddle, (r)ing, (p)inky

    Parameters
    ----------
   df : pandas dataframe
       of 64 columns: time  x00 y00 z00 x01 y01 z01 ... x20 y20 z20
       contains the coordinates of all the keypoints

    Returns
    -------
    jointAnglesDf : pandas dataframe
        of 16 columns: time thumbProx thumbMed thumbDist indexProx ...
    The names of the columns are:
    - first letter: (p)roximal, (m)iddle, (d)istal
    - second letter: (t)humb, (i)ndex, (m)iddle, (r)ing, (p)inky

    '''
    # creating the dictionary with the angles to be computed
    # first letter: (p)roximal, (m)iddle, (d)istal
    # second letter: (t)humb, (i)ndex, (m)iddle, (r)ing, (p)inky
    # the 3 int are the indexes of the corresponding landmarks
    anglesToCompute = {'pt': [ 0, 1, 2],
                       'mt': [ 1, 2, 3],
                       'dt': [ 2, 3, 4],
                       'pi': [ 0, 5, 6],
                       'mi': [ 5, 6, 7],
                       'di': [ 6, 7, 8],
                       'pm': [ 0, 9,10],
                       'mm': [ 9,10,11],
                       'dm': [10,11,12],
                       'pr': [ 0,13,14],
                       'mr': [13,14,15],
                       'dr': [14,15,16],
                       'pp': [ 0,17,18],
                       'mp': [17,18,19],
                       'dp': [18,19,20]}
    # initialize the lists
    # for the angular values
    list_of_cos_angle_array = []
    # for the column names
    list_of_angle_names = []
    
    for key in anglesToCompute:
        # get the landmark keypoints specified in the dictionary
        keypoints = anglesToCompute[key]
        
        # compute the cosine of the angle
        cos_angle_array = angle3d(df, keypoints) 
        # append the array to the list 
        list_of_cos_angle_array.append(cos_angle_array)
        
        # append the string name
        list_of_angle_names.append(key)
        # stringa = ('{:02d}{:02d}{:02d}'.format(keypoints[0],keypoints[1],keypoints[2]))
        # # append the string name
        # list_of_angle_names.append(key+stringa)
    
    # transpose the list
    list_of_cos_angle_array = np.array(list_of_cos_angle_array).T.tolist()
    
    # create the dataframe
    jointAnglesDf = pd.DataFrame(list_of_cos_angle_array, columns = list_of_angle_names)
    
    # convert the cosine of the angles in angles
    jointAnglesDf = np.rad2deg(np.arccos(jointAnglesDf))
    
    # add the same time reference of the original dataframe
    jointAnglesDf.insert(0, 'time', df.reset_index()['time'])
    
    # give the same indexes of the input dataframe
    jointAnglesDf.set_index(df.index)

    return jointAnglesDf

            
def computeAnglesWrtRefFrame(df, x_wheel_centre, y_wheel_centre):
    '''
    Given x, y, z coordinates of the point, computes the angle with respect to the
    centre of the wheel. z value is ignored
    

    Parameters
    ----------
    df_centre : pandas dataframe 
        of 1+3*n columns: time x0 y0 z0 ... xn yn zn
        contains the coordinates of the points
     x_wheel_centre : int 
         pixel x coordinate of the centre of the wheel
     y_wheel_centre : int 
         pixel y coordinate of the centre of the wheel

    Returns
    -------
    df_angle : pandas dataframe
        of 1+n columns: time angle0 ... anglen
        contains the angle corresponding to each triplet of points given

    '''
    # renaming with shorter names
    xwc = x_wheel_centre
    ywc = y_wheel_centre
    
    # getting the shape of the input dataframe
    rows, cols = df.shape
    
    # the number of angles to be computed depends on how many columns the 
    # dataframe has. Assuming this layout:
    # time x0 y0 z0 x1 y1 z1 ... xn yn zn
    # then the number of angles to be computed is
    howManyAngles = int((cols-1)/3) # -1 for the time, /3 because x y z
    
    # creating the header for the pandas dataframe
    header = ['time']
    for number in range(howManyAngles):
        name = 'angle ' + str(number)
        header.append(name)
     
    # creating an empty dataframe of given dimension
    df_angle = pd.DataFrame(np.zeros((rows, howManyAngles+1)), columns=header)
    
    # copying the time
    df_angle.iloc[:,0] = df.iloc[:,0]
    
    # for each triplet x y z the angle is computed considering x and y coord
    for index in range(howManyAngles):
        # values with respect to the reference frame
        # *rf = * with respect to ref frame
        xrf = pd.Series.to_numpy(   df.iloc[:,index*3+1] - xwc)
        yrf = pd.Series.to_numpy( - df.iloc[:,index*3+2] + ywc)
    
        angles = np.arctan2(yrf,xrf)*180/np.pi
        
        # df_angle.iloc[:,0] is the time. Index starts from 0 so should be increm
        df_angle.iloc[:,index+1] = angles
        
    return df_angle

def landmarksDfToCentreHandAngleDf(df, img_width, img_height, x_wheel_centre, y_wheel_centre):
    '''
    Resumes the operation of 
    - handcentrePosition
    - fromAbsTimeStampToRelTime
    - scaleXYZonImageShape
    - computeAnglesWrtRefFrame
    
    See documentation of the above mentioned functions for more

    Parameters
    ----------
    df : pandas dataframe
        of 64 columns: time  x00 y00 z00 x01 y01 z01 ... x20 y20 z20
        contains the coordinates of all the keypoints
    img_widht : int
        image width.
    img_height : int
        image height.
    x_wheel_centre : int 
        pixel x coordinate of the centre of the wheel
    y_wheel_centre : int 
        pixel y coordinate of the centre of the wheel

    Returns
    -------
    df_angle : pandas dataframe
        of 1+n columns: time angle0
        contains the angle corresponding to the centre of the hand

    '''
    # create a new dataframe containing the hand centre coordinates
    df_centre = handCenterPosition(df)

    # using the relative time instead of the absolute one
    df_centre = fromAbsTimeStampToRelTime(df_centre)

    # scaling the dimension on the image
    df_centre = scaleXYZonImageShape(df_centre, img_width, img_height)

    # computing the angular value with respect to the centre of the wheel
    df_angle = computeAnglesWrtRefFrame(df_centre, x_wheel_centre, y_wheel_centre)
    
    return df_angle

def cutInTime(x, y, interval):
    '''
    Given x (time array), y (values), and interval, erases all the x,y pairs whose x is 
    before interval[0] and after interval[1]
    

    Parameters
    ----------
    x : np.array or list
        time array.
    y : np.array or list
        correspoding values.
    interval : list of 2 values [start finish]
        DESCRIPTION.

    Returns
    -------
    x : np.array or list
        only the values after start and before stop.
    y : np.array or list
        only the corresponding values to x.

    '''
    start = interval[0]
    stop = interval[1]
    if start < stop:
        # first cut signal and then time, time gives the condition
        # cut the tails            
        y = y[x<=stop]
        x = x[x<=stop]
        
        # cut the heads
        y = y[x>=start]
        x = x[x>=start]
        
        # reset the time
        x = x - x[0]
    else:
        logging.warning('not cutting the arrays since stop is before start')
    return x, y


def syncXcorr(signal1, signal2, time1, time2, step = 0.01, \
              interval1 = [0, 0], interval2 = [0, 0], \
              showPlot = False, device1 = 'device 1', device2 = 'device 2', userTitle = '', col1 = 'C0', col2 = 'C1'):
    '''
    Computes the delay of signal2 with respect to signal1 using cross correlation.
    
    To do so, a similar pattern should be present in both signals.
    
    "time1" and "time2" contain the relative time of the recording and should: 
        - be in the same measurement unit (eg: seconds)
        - start both from 0
    The returned value "delay" will be in the same measurement unit.
    
    "signal1" is the one that gives the t=0, while the advance/delay in the 
    starting of the recording of "signal2" is computed.
    The returned value is "delay", which expresses:
        - the timing delay of signal2 wrt to signal1, 
        - the OPPOSITE (minus sign) of the timing delay in the recording
        
    If the recording of 2 starts before 1, when plotting the two signals,
    you see the event happening in 1 first and then in 2.
    
    To graphically synchronize them, it's necessary to move 2 towards right
    To timewise synchronize them, it's necessary to cut the first frames of 2 
    (the ones when 2 was already recording and 1 wasn't) and to reset the timer of 2
    
    If "delay" is *POSITIVE*, then signal2 started to be recorded AFTER "delay" time. 
    To synchronize the two signals, it's necessary to add values in the head of
    signal2
    NOT SYNC SIGNALS
    -----------****------- signal1
    
    --------****------- signal2
    
    delay = 3 -> signal2 started to be recorded 3 after 
    SYNC SIGNALS
    -----------****------- signal1
    
    add--------****------- signal2
    
    If "delay" is *NEGATIVE*, then signal2 started to be recorded BEFORE "delay" time. 
    To synchronize the two signals, it's necessary to cut values from the head of
    signal2
    NOT SYNC SIGNALS
    -----------****------- signal1
    
    --------------****------- signal2
    
    delay = -3 -> signal2 started to be recorded 3 before
    SYNC SIGNALS
    -----------****------- signal1
    
    -----------****------- signal2

    Parameters
    ----------
    signal1 : array
        Contains the y value of signal 1
    signal2 : array
        Contains the y value of signal 2
    time1 : array
        Contains the x value of signal 1
    time2 : array
        Contains the x value of signal 2
    step : int, optional
        To perform cross correlation, both signals should be at the same 
        frequency, it's necessary to resample them. The step should be in the 
        same measurement units of time1 and time2
        The default is 0.01.
    interval1 : list of 2 values: [startTime endTime], optional
        Part of the signal1 that should be considered when executing the xcorr. 
        The default is [0, 0], which means the whole signal.
    interval2 : list of 2 values: [startTime endTime], optional
        Part of the signal2 that should be considered when executing the xcorr. 
        The default is [0, 0], which means the whole signal.
    showPlot : bool, optional
        If the function should display a plot regarding the execution. 
        The default is False.
    device1 : string, optional
        Name of device 1 in the plot. 
        The default is 'device 1'.
    device2 : string, optional
        Name of device 2 in the plot. 
        The default is 'device 2'.
    userTitle : string, optional
        To be added in the title
        The default is ''.
        

    Returns
    -------
    delay : float
        Delay in the same temporal measurement unit of the two signals
        If POSITIVE, signal2 started to be recorded AFTER signal1
        If NEGATIVE, signal2 started to be recorded BEFORE signal1
    maxError : float
        maxError = step / 2 

    '''
    # keeping sure that the variables are numpy.arrays
    signal1, _ = utils.toFloatNumpyArray(signal1)
    signal2, _ = utils.toFloatNumpyArray(signal2) 
    time1, _ = utils.toFloatNumpyArray(time1) 
    time2, _ = utils.toFloatNumpyArray(time2)
    
    signal1 = fillNanWithInterp(signal1, time1)
    signal2 = fillNanWithInterp(signal2, time2)

    # eventually cutting the signal1
    if interval1 != [0, 0]:
        time1, signal1 = cutInTime(time1, signal1, interval1)
            
    # eventually cutting the signal2
    if interval2 != [0, 0]:
        time2, signal2 = cutInTime(time2, signal2, interval2)
            
    # user delay
    # since the xcorrelation works on the y values only, the cutting of the 
    # signals should be taken into account as an additional delay
    userDelay = interval1[0] - interval2[0]
    
    # resampling both signals on the same frequency
    y1, x1, _ = resampleWithInterp(signal1, time1, step, 'time step')
    y2, x2, _ = resampleWithInterp(signal2, time2, step, 'time step')
    
    # putting the values around 0
    y1 = y1 - np.mean(y1)
    y2 = y2 - np.mean(y2)
    
    # normalizing from -1 to 1
    y1 = y1 / np.max(np.abs(y1))
    y2 = y2 / np.max(np.abs(y2))
    
    # compute correlation
    corr = scipy.signal.correlate(y1, y2)
    lags = scipy.signal.correlation_lags(len(y1), len(y2))
    # where there is max correlation
    index = np.argmax(corr)
    
    delay =  lags[index]*step
    # adding the userDelay to the one computed on the signals
    delay = delay + userDelay
    maxError = step/2
    
    if showPlot:
        plots.syncXcorr(x1, y1, interval1, device1, x2, y2, interval2, device2, delay, lags, step, userDelay, maxError, corr, index, userTitle, col1 = col1, col2 = col2)
        # plots.syncXcorrOld(x1, y1, interval1, device1, x2, y2, interval2, device2, delay, lags, step, userDelay, maxError, corr, index, userTitle = '', col1 = 'C0', col2 = 'C1')

    return delay, maxError 

def fillNanWithInterp(y, x = 0, mode = 'linear'):
    '''
    Given an  array containing nans, fills it with the method specified in mode.
    If x is given, the y values returned are the one corresponding to the x specified
    If x is not given, y is assumed to be sampled at a fixed frequency

    Parameters
    ----------
    y : np.array
        original array of values containing nans to be corrected
    x : np.array, optional
        time array of acquisition of signal y. 
        The default is 0, which assumes that y is sampled at a fixed frequency
    mode : string, optional
        kind of interpolation to be performed, passed to scipy.interpolate.interp1d(kind = )
        Please refer to documentation 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html 
        The default is 'linear'.

    Returns
    -------
    yinterp : np.array
        contains the data with nan replaced from interpolated value

    '''
    # keeping sure that the variables are numpy.arrays
    x, _ = utils.toFloatNumpyArray(x)
    y, _ = utils.toFloatNumpyArray(y) 
    
    # if x is not given, it's assumed that the y array is equally spaced
    if np.array_equal(0, x):
        x = np.arange(0, len(y), 1)
        
    # find the indexes where y is not nan
    notNanIndexes = ~np.isnan(y)
    
    # if the first or the last value of y are nan, copy the closest value
    if notNanIndexes[0] == False:
        y[0] = y[notNanIndexes][0]
    if notNanIndexes[-1] == False:
        y[-1] = y[notNanIndexes][-1] 
    
    # find again the indexes where y is not nan
    # now the first and the last value are not nan, and they're the extremes of 
    # the interpolation
    notNanIndexes = ~np.isnan(y)
      
    # considering only the not nan value
    yClean = y[notNanIndexes]
    xClean = x[notNanIndexes]
    
    # feeding the interpolator with only the not nan values and obtaining a function
    finterp = scipy.interpolate.interp1d(xClean, yClean, mode)
    
    # computing the values of function on the original x
    yinterp = finterp(x)
    
    return yinterp

def resampleWithInterp(y, x = 0, xparam = 0.01, param = 'time step', mode = 'linear'):
    '''
    Given a signal y and his time array x, resamples it using interpolation
    the three modes to use this function are:
    - specifying the time *step*: 
        the output is resampled with the given step
    - specifying the *frequency*:
        the output is resampled with the given frequency
    - specifying the *time array*:
        the output is resampled on the given time array
    If signal y has contains nan, they are filled with the function fillNanWithInterp()
    
    Parameters
    ----------
    y : np.array
        original array of values
    x : np.array, optional
        time array of acquisition of signal y. 
        The default is 0, which assumes that y is sampled at a fixed frequency
    xparam : float, integer or array, optional
        if param == 'time step'
            specifies the time step
        if param == 'frequency'
            specifies the frequency
        if param == 'time array'
            is equal to the time array where the resampling should be done. 
        The default is 0.01 and goes with 'time step' specified in param
    param : string, optional
        To specify if the resampling should be done on a signal computed on the 
        given time step, frequency or on the given time array. 
        The default is 'time step' and goes with '0.001' specified in xparam
    mode : string, optional
        kind of interpolation to be performed, passed to scipy.interpolate.interp1d(kind = )
        Please refer to documentation 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html 
        The default is 'linear'.

    Returns
    -------
    yinterp : np.array
        Values of the resampled signal
    xinterp : np.array
        Time array of the resampled signal
    finterp : function
        Interpolator function, only works between the extremities of x 

    '''
    # keeping sure that the variables are numpy.arrays
    x, _ = utils.toFloatNumpyArray(x)
    y, _ = utils.toFloatNumpyArray(y) 
    xparam, _ = utils.toFloatNumpyArray(xparam) 
    
    # if x is not given, it's assumed that the y array is equally spaced
    if np.array_equal(0, x):
        if mode != 'time array':
            x = np.arange(0, len(y), 1)
        else:
            logging.error('asking to resample on a given time array but not \
                  specifiying the input time array')
            return None             
        
    # if y contains at least one nan, fill the space
    if np.isnan(y).any():
        logging.warning('nan values detected, filling them with ' + mode + ' method')
        y = fillNanWithInterp(y, x, mode)
        
    # the three modes to use this function are:
    # - specifying the time *step*
    # - specifying the *frequency*
    # - specifying the *time array*
    validParams = ['time step', 'frequency', 'time array']
    
    if param == validParams[0]: # given step
        step = xparam
        xinterp = np.arange(np.min(x), np.max(x), step)  
    elif param == validParams[1]: # given freq
        freq = xparam
        step = 1/freq
        xinterp = np.arange(np.min(x), np.max(x), step)
    elif param == validParams[2]: # given time array
        xinterp = xparam
        # # eventually cutting the time array 
        # xinterp = xinterp[xinterp<=np.max(x)]
        # xinterp = xinterp[xinterp>=np.min(x)]
        # warning the user if the time array specified exceeds the limits
        if (xinterp[0] < np.min(x) or xinterp[-1] > np.max(x)):
            logging.warning('Using extrapolation: ' + \
                  '\nInterpolator has values between {:.2f} and {:.2f}'\
                      .format(np.min(x), np.max(x)) + \
                  ' and computation between {:.2f} and {:.2f} is asked.'\
                      .format(xparam[0], xparam[-1]))
    else:
        logging.error('not valid param. Valid params are: ' + str(validParams))
        return None
    
    # feeding the interpolator with the input values and obtaining a function
    finterp = scipy.interpolate.interp1d(x, y, kind = mode, fill_value = 'extrapolate')
    
    # computing the values of the function on the xinterp
    yinterp = finterp(xinterp)
    
    return yinterp, xinterp, finterp

def mov_avg(y, samples_before, samples_after = -1):
    '''
    If nan are contained, calls mov_avg_loop()
    If not nans, then calls mov_avg_no_nan()
    
    Please refer to their documentation

    Parameters
    ----------
    y : array
        to be filtered.
    samples_before : int
        how many samples before the current one should be considered for the 
        sliding window.
    samples_after : int, optional
        how many samples after the current one should be considered for the 
        sliding window. The default is -1, which makes samples_after = samples_before.

    Returns
    -------
    y_filt : array, same type of y
        filtered array.

    '''
    if np.isnan(y).any: # slow but handles nan
        return mov_avg_loop(y, samples_before, samples_after = -1)
    else: # fast, it's possible to use it since no nans in the array
        return mov_avg_no_nan(y, samples_before, samples_after = -1)

def mov_avg_loop(y, samples_before, samples_after = -1):
    '''
    NB: slow but doesn't handles nan
    Computes the moving average on the y array with a sliding window centered
    in the current element and with an amplitude of samples_before and samples_after

    if samples_before = samples_after = 0, returns the same array considering 
    only the current element
    if samples_before = samples_after = 1, computes the mean on a sliding window 
    of amplitude = 3 considering the current element and the two adjacent
    if samples_before = samples_after = 2, computes the mean on a sliding window 
    of amplitude = 5 considering the current element and the two before and the two after
    if samples_before = 2 and samples_after = 0, computes the mean on a sliding window 
    of amplitude = 3 considering the current element and the two before
    if samples_before = 0 and samples_after = 2, computes the mean on a sliding window 
    of amplitude = 3 considering the current element and the two after

    The duration of execution is proportional to the length of y

    Parameters
    ----------
    y : array
        to be filtered.
    samples_before : int
        how many samples before the current one should be considered for the 
        sliding window.
    samples_after : int, optional
        how many samples after the current one should be considered for the 
        sliding window. The default is -1, which makes samples_after = samples_before

    Returns
    -------
    y_filt : array, same type of y
        filtered array.
    '''

    # if samples_after it's not specified
    if samples_after == -1:
        samples_after = samples_before

    y_filt = y.copy()
    for i in range(len(y)):
        y_filt[i] = np.nanmean(y[np.maximum(i-samples_before,0):np.minimum(i+samples_after,len(y))+1])
    return y_filt

def mov_avg_no_nan(y, samples_before, samples_after = -1):
    '''
    NB: fast but doesn't handle nan
    Computes the moving average on the y array with a sliding window centered
    in the current element and with an amplitude of samples_before and samples_after

    if samples_before = samples_after = 0, returns the same array considering 
    only the current element
    if samples_before = samples_after = 1, computes the mean on a sliding window 
    of amplitude = 3 considering the current element and the two adjacent
    if samples_before = samples_after = 2, computes the mean on a sliding window 
    of amplitude = 5 considering the current element and the two before and the two after
    if samples_before = 2 and samples_after = 0, computes the mean on a sliding window 
    of amplitude = 3 considering the current element and the two before
    if samples_before = 0 and samples_after = 2, computes the mean on a sliding window 
    of amplitude = 3 considering the current element and the two after

    Equivalent to mov_avg_loop but much faster, the duration of execution is not
    heavily affected from the length of y

    Parameters
    ----------
    y : array
        to be filtered.
    samples_before : int
        how many samples before the current one should be considered for the 
        sliding window.
    samples_after : int, optional
        how many samples after the current one should be considered for the 
        sliding window. The default is -1, which makes samples_after = samples_before.

    Returns
    -------
    y_filt : array, same type of y
        filtered array.
    '''

    # if samples_after it's not specified
    if samples_after == -1:
        samples_after = samples_before

    # amplitude of the window
    size = samples_before + samples_after + 1
    # execution of the filtering
    y_filt = scipy.ndimage.uniform_filter1d(y, size = size, origin = 0, mode = 'reflect')

    # since only size is specified, how to translate the array?
    shift = int(np.ceil((samples_after-samples_before)/2))

    if shift > 0:
        # translation backward
        y_filt[:-shift] = y_filt[shift:]
        y_filt[-shift:] = np.nan*shift
    if shift < 0:
        # translation foreward
        y_filt[-shift:] = y_filt[:shift]
        y_filt[:-shift] = np.nan*abs(shift)

    # fix the first samples
    for i in range(0, min(samples_before + abs(shift), len(y)), 1):
        y_filt[i] = np.nanmean(y[np.maximum(i-samples_before,0):np.minimum(i+samples_after,len(y))+1])
    # fix the last samples
    for i in range(len(y)-1, max(0, len(y) - samples_after - abs(shift) - 1), -1):
        y_filt[i] = np.nanmean(y[np.maximum(i-samples_before,0):np.minimum(i+samples_after,len(y))+1])

    return y_filt

def mov_avg_time(y, freq, time_units_before, time_units_after = -1):
    '''
    Computes the moving average on the y array with a sliding window centered
    in the current element and with an amplitude depending on the corrispondent 
    moment, specified in x array. 

    NB: This function assumes constant sampling frequency.
    If the sampling freq is not constant, use mov_avg_time_variable_freq

    Parameters
    ----------
    y : array
        to be filtered.
    freq : float
        frequency of acquisition of the y array, assumed constant.
    time_units_before : float
        how large is the window on the left. In the same meas units of x array
    time_units_after : float, optional
        how large is the window on the right. In the same meas units of x array
        The default is -1, which makes time_units_after = time_units_before.


    Returns
    -------
    y_filt : array, same type of y
        filtered array.
    '''
    samples_before = int(time_units_before * freq)
    if time_units_after != -1:
        samples_after = int(time_units_after * freq)
    else:
        samples_after = -1

    y_filt = mov_avg(y, samples_before, samples_after)

    return y_filt

def mov_avg_time_variable_freq(x, listYarrays, time_units_before, time_units_after = -1):
    '''
    Computes the moving average on the y array with a sliding window centered
    in the current element and with an amplitude depending on the corrispondent 
    moment, specified in x array. 

    NB: Use this function when necessary, (data acquired at not constant freq), 
    use mov_avg_time instead which is much faster.

    Parameters
    ----------
    x : array
        contains the moment of acquisition of each frame of y.
    y : array
        to be filtered.
    time_units_before : float
        how large is the window on the left. In the same meas units of x array
    time_units_after : float, optional
        how large is the window on the right. In the same meas units of x array
        The default is -1, which makes time_units_after = time_units_before.

    Returns
    -------
    y_filt : array, same type of y
        filtered array.
    '''
    oneArrayFlag = False
    # add a dimension
    if utils.containsScalars(listYarrays):
        oneArrayFlag = True
        listYarrays = [listYarrays]

    # if time_units_after it's not specified
    if time_units_after == -1:
        time_units_after = time_units_before
    # conversion to float in order to be able to add nan
    x = np.array(x).astype("float")

    arrayYarrays = np.array(listYarrays).astype("float")
    arrayYarrays_filt = arrayYarrays.copy()
    for i in range(len(x)):
        this_moment = x[i]
        timing_window = x.copy()
        # give nan value to all the samples before
        timing_window[timing_window<this_moment-time_units_before] = np.nan
        # give nan values to all the samples after
        timing_window[timing_window>this_moment+time_units_after ] = np.nan
        # get indexes where timing_window is not nan
        indexes = np.argwhere(~np.isnan(timing_window))
        # consider y only where timing_window is not nan
        arrayYarrays_filt[:,i] = np.squeeze(np.nanmean(arrayYarrays[:,indexes], axis = 1))
    if oneArrayFlag: # bring back to single dimension
        return np.squeeze(arrayYarrays_filt)
    return arrayYarrays_filt


def findTransitionTime(array, time, timeFlag = 'middle'):
    '''
    Given an array of states and the relative time, returns a dataframe 
    containing three columns:
        - initial state
        - final state
        - time of transition, that can be computed
            - as the time of the last element of initial state
            - as the time of the first element of final state
            - as the avergage between the two

    Parameters
    ----------
    array : array or list
        contains the state.
    time : array or list
        time array.
    timeFlag : list, optional
        how to compute the transition time. The default is 'middle'.

    Returns
    -------
    df : pandas dataframe
    three columns:
        - initial state
        - final state
        - time of transition, that can be computed
            - as the time of the last element of initial state
            - as the time of the first element of final state
            - as the avergage between the two

    '''
    assert len(array) == len(time), f"array and time have difference length: len(array) = {len(array)}, len(time) = {len(time)}"
    assert timeFlag in ['last', 'first', 'middle']
    startState = []
    endState = []
    transitionTime = []
    difference = np.diff(array)
    for i in range(len(difference)):
        if difference[i] != 0:
            startState.append(array[i])
            endState.append(array[i+1])
            if timeFlag == 'last':
                transitionTime.append((time[i]))
            if timeFlag == 'first':
                transitionTime.append((time[i+1]))
            if timeFlag == 'middle':
                transitionTime.append((time[i]+time[i+1])/2)
    df = pd.DataFrame(list(zip(startState,endState, transitionTime)), columns = ['startState', 'endState', 'time'])
    return df





def bisection(array,value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j] and array[j+1]. 
    ``array`` must be monotonic increasing. 
    j=-1 or j=len(array) is returned to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl


def closestDifference (array1, array2):
    '''
    For every element in array1, computes the difference to the closest value contained in array2

    the computation is done: a1 - a2
    which means that if a1 smaller than a2, a negative result is provided

    Parameters
    ----------
    array1 : TYPE
        DESCRIPTION.
    array2 : TYPE
        DESCRIPTION.

    Returns
    -------
    minDiffArray : TYPE
        DESCRIPTION.

    '''
    minDiffArray = []
    for value in array1:
        closestIndex = bisection(array2, value)
        if closestIndex == -1:
            minDiff = array2[0]-value
        elif closestIndex == len(array2) or closestIndex == len(array2)-1:
            minDiff = array2[-1]-value
        # if value is closer to the element on the left
        elif np.abs(array2[closestIndex]-value) < np.abs(array2[closestIndex+1]-value):
            minDiff = array2[closestIndex]-value
        else: # if value is closer to the element on the right
            minDiff = array2[closestIndex+1]-value
        minDiffArray.append(minDiff)
    return minDiffArray



def checkTiming(time_array, nominal_param = 0.01, param = 'time step', mainTitle = '', showPlot = False):
    '''
    Given a time array and a nominal param (that can be either time step or frequency),
    compares the timing of the array with respect to the nominal param.
    If showPlot is True, an histogram is shown 

    Parameters
    ----------
    time_array : array of double
        array containing the time of recording of each frame.
    nominal_param : double, optional
        theorical value, can be the frequency or the time step. 
        The default is 0.01.
    param : string, optional
        to specify if nominal_param is the frequency or the time step. 
        can be either ['time step', 'frequency']
        The default is 'time step'.
    mainTitle : string, optional
        Title of the figure. The default is ''.
    showPlot : bool, optional
        If the histogram should be shown. The default is False.

    Returns
    -------
    dt_mean : double
        mean value of time difference between successive time frames.
    dt_std : double
        standard deviation of time difference between successive time frames

    '''
    # keeping sure that the variables are numpy.arrays
    time_array, _ = utils.toFloatNumpyArray(time_array)        

    # compute the time difference between each pair of samples
    d_time_array = np.diff(time_array)
    
    dt_mean = np.mean(d_time_array)
    dt_std = np.std(d_time_array)
    
    # the user can either specify the time step of the frequency
    validParams = ['time step', 'frequency']
    if param == validParams[0]: # given step
        step = nominal_param
        freq = 1/step
    elif param == validParams[1]: # given freq
        freq = nominal_param
        step = 1/freq
    else:
        logging.error('not valid param. Valid params are: ' + str(validParams))
        return None
    
    if showPlot:
        if mainTitle == '':
            mainTitle = 'timing of acquisition'
        mainTitle = r"{}: mean: {:.4f} - std: {:.4f} s. Nominal value: {:.4f} s [{:.2f} Hz]".\
        format(mainTitle, dt_mean, dt_std, step, freq)

        plots.checkTiming(d_time_array, step, mainTitle)

    return dt_mean, dt_std

def checkTimingMultipleDevices(time_arrays, nominal_params, params, devicesName = None, mainTitle = '', showPlot = False):

    # get the number of devices
    # check if nominalFreqs is a list
    if isinstance(nominal_params, list):
        nDevices = len(nominal_params)
    else:
        nDevices = 1
    
    # if only one device, calls the function for only one time_array
    if nDevices == 1:
        logging.info('only one device was specified, calling checkTiming instead')
        dt_mean, dt_std = checkTiming(time_arrays, nominal_params, params, mainTitle, showPlot)
        return dt_mean, dt_std
       
    else:
        # initialize the array containing the result
        dt_all = []
        dt_mean_all = []
        dt_std_all = []
        step_all = []
        freq_all = []

        # loop for each device
        for time_array, nominal_param, param, deviceName in zip(time_arrays, nominal_params, params, devicesName):
            # keeping sure that the variables are numpy.arrays
            time_array, _ = utils.toFloatNumpyArray(time_array)        

            # compute the time difference between each pair of samples
            d_time_array = np.diff(time_array)
            
            dt_mean = np.mean(d_time_array)
            dt_std = np.std(d_time_array)

            # the user can either specify the time step of the frequency
            validParams = ['time step', 'frequency']
            if param == validParams[0]: # given step
                step = nominal_param
                freq = 1 / step
            elif param == validParams[1]: # given freq
                freq = nominal_param
                step = 1 / freq
            else:
                logging.error('not valid param. Valid params are: ' + str(validParams))
                return None

            dt_all.append(d_time_array)
            dt_mean_all.append(dt_mean)
            dt_std_all.append(dt_std)
            step_all.append(step)
            freq_all.append(freq)

        if showPlot:
            plots.checkTimingMultipleDevices(dt_all, step_all, freq_all, dt_mean_all, dt_std_all, mainTitle = mainTitle, devicesName = devicesName)

        return dt_mean_all, dt_std_all
    
def speedToAngleDeg(speed, time, radius, initial = 0):
    '''
    Converts the speed [m/s] in angular velocity [deg/s] knowing the radius [mm]
    using scipy.integrate.cumtrapz, then 

    Parameters
    ----------
    speed : array [m/s]
        speed for every sample of recording.
    time : array [s]
        samples of recording.
    radius : float [m]
        radius of the circumference.
    initial : float, optional
        initial value for the integration. The default is 0.

    Returns
    -------
    angle_deg : array [deg]
        equivalent angular elapsed distance corresponding to every sample of recording

    '''
    # keeping sure that the variables are numpy.arrays
    time, _ = utils.toFloatNumpyArray(time) 
    speed, _ = utils.toFloatNumpyArray(speed) 

    # from speed [m/s] to elapsed distance
    distance_m = scipy.integrate.cumtrapz(speed, time, initial)
    
    # adding the last element to respect the dimension
    distance_m = np.append(distance_m, distance_m[-1])

    # number of rounds of the wheel: circumference = 2 * pi * radius
    rounds = distance_m / (2*np.pi*radius) 

    # conversion from rounds to rad: 1 round = 2 * pi [rad]
    angle_rad = rounds * 2 * np.pi

    # conversion from rad to degress: [deg] = 180 / pi * [rad]
    angle_deg =  180 / np.pi * angle_rad
    
    return angle_deg

def elaborateErgo(ergo_data_original, pbpvariable, wheelsize, rimsize, minpeak, mindist, cutoff):
    
    ergo_data = copy.deepcopy(ergo_data_original)
    # ELABORATION OF ERGO DATA with WORKLAB
    ergo_data = wl.kin.filter_ergo(ergo_data)
    ergo_data = wl.kin.process_ergo(ergo_data, wheelsize = wheelsize, rimsize = rimsize)
    # not cutting the data, it's preferable to have the 0 time of recording and not of the test
    # ergo_data = wl.ana.cut_data(ergo_data, startEndErgoTime[0], startEndErgoTime[1])
    # to get the speed in km/h
    for side in ergo_data:
        ergo_data[side]['speed'] = ergo_data[side]['speed'] * 3.6

    # Calculate push by push data
    ergo_data_pbp = wl.kin.push_by_push_ergo(ergo_data, variable = pbpvariable, minpeak = minpeak, mindist = mindist, cutoff = cutoff, verbose = False)

    # extract the right side
    ergo_data = ergo_data['right']
    ergo_data_pbp = ergo_data_pbp['right']
    
    return ergo_data, ergo_data_pbp
    
def elaborateRightCam(cam_data_original, ergoCamDelay, imgHeight, imgWidth, x_wheel_centre, y_wheel_centre):
    
    cam_data = cam_data_original.copy()
    # rename the delay
    delay = ergoCamDelay
    # SYNCHRONIZATION wrt ERGO TIME
    cam_data_sync = cam_data.copy()
    cam_data_sync['time'] = cam_data_sync['time'] + delay
    cam_data_sync = cam_data_sync[cam_data_sync['time'] > 0]
    
    # ELABORATION OF CAM DATA
    # computing the joint relative angles
    jointAngles = computeJointAngles(cam_data_sync)

    # create a new dataframe containing the hand center coordinates [from 0 to 1 since in the scale of mediapipe]
    handCenter01 = handCenterPosition(cam_data_sync)
    # scaling the dimension on the image
    handCenterImg = scaleXYZonImageShape(handCenter01, imgWidth, imgHeight)
    # computing the angular value with respect to the center of the wheel
    handAnglewrtRefFrame = computeAnglesWrtRefFrame(handCenterImg, x_wheel_centre, y_wheel_centre)
    
    return handAnglewrtRefFrame, jointAngles

def getROI(cam_data):
    '''
    Computes the region of interest according to the hand position

    Parameters
    ----------
    cam_data : pandas dataframe
        of 64 columns: time  x00 y00 z00 x01 y01 z01 ... x20 y20 z20
        contains the coordinates of all the keypoints
        Can also be a different dataframe, but should have x y z columns

    Returns
    -------
    roi_hand : pandas dataframe
        of 7 columns: time xmin xmax ymin ymax zmin zmax [between 0 and 1]

    '''
    # consider only x, y, z
    cam_data_x = cam_data.filter(regex = 'x')
    cam_data_y = cam_data.filter(regex = 'y')
    cam_data_z = cam_data.filter(regex = 'z')

    # row wise min and max
    x_min = cam_data_x.min(axis=1)
    x_max = cam_data_x.max(axis=1)
    y_min = cam_data_y.min(axis=1)
    y_max = cam_data_y.max(axis=1)
    z_min = cam_data_z.min(axis=1)
    z_max = cam_data_z.max(axis=1)

    # creation of the dataframe
    d = {'time' : cam_data['time'], \
         'x min': x_min, \
         'x max': x_max, \
         'y min': y_min, \
         'y max': y_max, \
         'z min': z_min, \
         'z max': z_max}
    
    roi_hand = pd.DataFrame(data=d)

    return roi_hand

def getROIimg(cam_data, imgHeight, imgWidth, recastToInt = True, cropInsideImg = True):
    '''
    Calls the fuction getROI on cam_data and then scales the dimension of xmin,
    xmax, ymin, ymax, zmin and zmax according to imgHeight and imgWidth, 
    eventually reduces their values in order to fit them on the image

    Parameters
    ----------
    cam_data : pandas dataframe
        of 64 columns: time  x00 y00 z00 x01 y01 z01 ... x20 y20 z20
        contains the coordinates of all the keypoints
        Can also be a different dataframe, but should have x y z columns
    imgHeight : int
        image height.
    imgWidth : int
        image width.
    cropInsideImg : bool, optional
        if the values of the coordinates of x and y should be inside the range of imgHeight and imgWidth

    Returns
    -------
    roi_hand_img : pandas dataframe
        of 7 columns: time xmin xmax ymin ymax zmin zmax 
        [between 0 and imgH and 0 and imgW]

    '''

    # call roi_hand
    roi_hand = getROI(cam_data)

    # copy the dataframe
    roi_hand_img = roi_hand.copy()

    roi_hand_img['x min'] = (roi_hand_img['x min'] * imgWidth-1)
    roi_hand_img['x max'] = (roi_hand_img['x max'] * imgWidth-1)
    roi_hand_img['y min'] = (roi_hand_img['y min'] * imgHeight-1)
    roi_hand_img['y max'] = (roi_hand_img['y max'] * imgHeight-1)
    roi_hand_img['z min'] = (roi_hand_img['z min'] * imgWidth-1)
    roi_hand_img['z max'] = (roi_hand_img['z max'] * imgWidth-1)

    if cropInsideImg:
        # to avoid overflow out of the image
        roi_hand_img['x min'][roi_hand_img['x min']<=0] = 0
        roi_hand_img['x max'][roi_hand_img['x max']>=imgWidth-1] = imgWidth-1
        roi_hand_img['y min'][roi_hand_img['y min']<=0] = 0
        roi_hand_img['y min'][roi_hand_img['y max']>=imgHeight-1] = imgHeight-1

    if recastToInt:
        # not modifying the column 0, contains the time
        roi_hand_img.iloc[:,1:] = roi_hand_img.iloc[:,1:][~roi_hand_img['x min'].isnull()].astype('Int64')

    return roi_hand_img

def getROIimgBackup(cam_data, imgHeight, imgWidth):
    '''
    Calls the fuction getROI on cam_data and then scales the dimension of xmin,
    xmax, ymin, ymax, zmin and zmax according to imgHeight and imgWidth, 
    eventually reduces their values in order to fit them on the image

    Parameters
    ----------
    cam_data : pandas dataframe
        of 64 columns: time  x00 y00 z00 x01 y01 z01 ... x20 y20 z20
        contains the coordinates of all the keypoints
        Can also be a different dataframe, but should have x y z columns
    imgHeight : int
        image height.
    imgWidth : int
        image width.

    Returns
    -------
    roi_hand_img : pandas dataframe
        of 7 columns: time xmin xmax ymin ymax zmin zmax 
        [between 0 and imgH and 0 and imgW]

    '''

    # call roi_hand
    roi_hand = getROI(cam_data)

    # copy the dataframe
    roi_hand_img = roi_hand.copy()

    # scale on image dimension recasting to int and avoid overflow
    roi_hand_img['x min'] = np.floor(roi_hand_img['x min'] * imgWidth)
    roi_hand_img['x max'] =  np.ceil(roi_hand_img['x max'] * imgWidth)
    roi_hand_img['y min'] = np.floor(roi_hand_img['y min'] * imgHeight)
    roi_hand_img['y max'] =  np.ceil(roi_hand_img['y max'] * imgHeight)
    roi_hand_img['z min'] = np.floor(roi_hand_img['z min'] * imgWidth)
    roi_hand_img['z max'] =  np.ceil(roi_hand_img['z max'] * imgWidth)


    # to avoid overflow out of the image
    roi_hand_img['x min'][roi_hand_img['x min']<=0] = 0
    roi_hand_img['x max'][roi_hand_img['x max']>=imgWidth-1] = imgWidth-1
    roi_hand_img['y min'][roi_hand_img['y min']<=0] = 0
    roi_hand_img['y min'][roi_hand_img['y max']>=imgHeight-1] = imgHeight-1

    # not modifying the column 0, contains the time
    roi_hand_img.iloc[:,1:] = roi_hand_img.iloc[:,1:][~roi_hand_img['x min'].isnull()].astype('Int64')

    return roi_hand_img

def findCirclesOnImage(img, minDist, param1, param2, minRadius, maxRadius):
    '''
    find circles on the img, doesn't matter if RGB or BGR since it's converted 
    in gray scale. All the other parameters are the one of cv2.HoughCircles:
        https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Parameters
    ----------
    img : image
        where the circles should be detected.
    minDist : int
        between centers.
    param1 : int
        refer to documentation.
    param2 : int
        refer to documentation. In this case, param2 is recursively decreased 
        until when at least one circle is detected.
    minRadius : int
        minimum radius of the detected circles.
    maxRadius : int
        maximum radius of the detected circles.

    Returns
    -------
    circles : structure in the following shape:
        array([[[xc, yc, r]],
               [[xc, yc, r]],
               ...
               [[xc, yc, r]]])
        containing the coordinates of centre and radius defining a circle

    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = None
    coeff = 1
    # while loop changing param2: make the detection less picky till at least one circle is detected
    while circles is None:
        if coeff < 0:
            return None
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist = minDist,\
        param1 = param1, param2 = param2 * coeff, minRadius = minRadius, maxRadius = maxRadius)
        coeff = coeff - 0.1/coeff

    return circles

def findLinesOnImage(img, edge_low_thresh, edge_high_thresh, rho, theta, threshold, min_line_length, max_line_gap):
    '''
    find lines on the img, doesn't matter if RGB or BGR since it's converted 
    in gray scale. All the other parameters are the one of cv2.HoughLinesP:
        https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb
    

    Parameters
    ----------
    img : image
        where the circles should be detected.
    edge_low_thresh : int
        for canny edge detection
    edge_high_thresh : int
        for canny edge detection
    rho : double
        distance resolution in pixels of the Hough grid.
    theta : double
       angular resolution in radians of the Hough grid.
    threshold : int
        minimum number of votes (intersections in Hough grid cell).
        In this case, threshold is recursively decreased until when at least 
        one line is detected.
    min_line_length : double
        minimum number of pixels making up a line.
    max_line_gap : double
        maximum gap in pixels between connectable line segments.

    Returns
    -------
    lines :  array of arrays
        array([[[x1, y1, x2, y2]],
               [[x1, y1, x2, y2]],
               ...
               [[x1, y1, x2, y2]]])
        contains the coordinates of each pair of points defining a line.

    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, edge_low_thresh, edge_low_thresh)
    # plt.figure()
    # plt.title('detected edges')
    # plt.imshow(edges)
    lines = None
    coeff = 1
    # while loop changing threshold: make the detection less picky till at least one circle is detected
    while lines is None:
        if coeff < 0:
            return None
        lines = cv2.HoughLinesP(edges, rho, theta, threshold * coeff, None, min_line_length, max_line_gap)
        coeff = coeff - 0.1/coeff
    return lines

def colorsOnTheLineImage(img, lines, w = 3, npoints = 100):
    '''
    Given an image and a set of lines, moves along the pixels of the line and 
    collects the color of the 3 channels (RGB or BGR) of each pixel and the around area
    of amplitude w

    Parameters
    ----------
    img : image
        where lines are detected.
    lines : array of arrays
        array([[[x1, y1, x2, y2]],
               [[x1, y1, x2, y2]],
               ...
               [[x1, y1, x2, y2]]])
        contains the coordinates of each pair of points defining a line.
    w : int, optional
        semi amplitude of the padding (in pixels) around the pixel examined. 
        The default is 3.
    npointsOnLine : int, optional
        number of points examined along a line. 
        The default is 100.

    Returns
    -------
    linesColors : array
        of n lines containing for each line 3 columns (corresponding to the RGB or BGR channels) of npoints rows.

    '''

    linesColors = []
    for line in lines:
        tmp = np.empty([1,3], dtype='uint8')
        for x1,y1,x2,y2 in line:
            p1 = geom.Point2d(x1,y1)
            p2 = geom.Point2d(x2,y2)
            myLine = geom.Line2d(p1 = p1, p2 = p2)
            # if it is not a vertical line
            if not np.isinf(myLine.m):
                xarray = np.linspace(x1, x2, num = npoints)
                for x in xarray:
                    tmp_this_pixel = np.empty([1,3], dtype='uint8')
                    # find the corresponding y
                    y = myLine.findY(x)
                    x = int(round(x))
                    y = int(round(y))
                    # consider all the pixels around the given one
                    for i in range(-w,w+1,1):
                        xi = x + i
                        for j in range(-w,w+1,1):
                            yj = y+j
                            try: # could give index error if the pixel is on the boundaries
                                color = img[yj,xi]
                            except:
                                pass
                            else:
                                tmp_this_pixel = np.append(tmp_this_pixel, [color], axis=0)
                    # excluding the first value, uninitialized when creating the empty array
                    tmp_this_pixel = tmp_this_pixel[1:]
                    color = np.mean(tmp_this_pixel, axis=0)
                    tmp = np.append(tmp, [color], axis=0)
            # if it is a vertical line:
            else:
                yarray = np.linspace(y1, y2, num = npoints)
                for y in yarray:
                    tmp_this_pixel = np.empty([1,3], dtype='uint8')
                    x = int(round(x1)) # if it's a vertical line, x is constant
                    y = int(round(y))
                    # consider all the pixels around the given one
                    for i in range(-w,w+1,1):
                        xi = x + i
                        for j in range(-w,w+1,1):
                            yj = y+j
                            try: # could give index error if the pixel is on the boundaries
                                color = img[yj,xi]
                            except:
                                pass
                            else:
                                tmp_this_pixel = np.append(tmp_this_pixel, [color], axis=0)
                    # excluding the first value, uninitialized when creating the empty array
                    tmp_this_pixel = tmp_this_pixel[1:]
                    color = np.mean(tmp_this_pixel, axis=0)
                    tmp = np.append(tmp, [color], axis=0)
            # excluding the first value, uninitialized when creating the empty array
            tmp = tmp[1:]
        linesColors.append([tmp])

    return linesColors

def fromLinesToLinesDF(lines):
    '''
    Given lines, structure in the following shape:
    array([[[x1, y1, x2, y2]],
           [[x1, y1, x2, y2]],
           ...
           [[x1, y1, x2, y2]]])
    containing the coordinates of each pair of points defining a line,
    Creates a dataframe with the following columns:
        x1  y1  x2  y2   distance      slope


    Parameters
    ----------
    lines : array of arrays
        array([[[x1, y1, x2, y2]],
               [[x1, y1, x2, y2]],
               ...
               [[x1, y1, x2, y2]]])
        contains the coordinates of each pair of points defining a line

    Returns
    -------
    lines_df : pandas dataframe
        with the following columns:
    x1  y1  x2  y2   distance      slope

    '''
    lines = np.squeeze(lines)
    x1 = lines[:,0]
    y1 = lines[:,1]
    x2 = lines[:,2]
    y2 = lines[:,3]

    # creation of the dataframe
    d = {'x1': x1.flatten(), \
         'y1': y1.flatten(), \
         'x2': x2.flatten(), \
         'y2': y2.flatten()}

    lines_df = pd.DataFrame(data=d)

    lines_df['distance'] = np.sqrt((lines_df['x1']-lines_df['x2'])**2+(lines_df['y1']-lines_df['y2'])**2)

    lines_df['slope'] = (lines_df['y1']-lines_df['y2'])/(lines_df['x1']-lines_df['x2'])

    return lines_df

def fromCirclesToCirclesDF(circles):
    '''
    Given circles, structure in the following shape:
    array([[[xc, yc, r]],
           [[xc, yc, r]],
           ...
           [[xc, yc, r]]])
    containing the coordinates of centre and radius defining a circle,
    Creates a dataframe with the following columns:
       xc  yc  r


    Parameters
    ----------
    circles : array of arrays
        array([[[xc, yc, r]],
               [[xc, yc, r]],
               ...
               [[xc, yc, r]]])
        the coordinates of centre and radius defining a circle

    Returns
    -------
    circles_df : pandas dataframe
        with the following columns:
    xc  yc  r

    '''
    circles = np.squeeze(circles)
    xc = circles[:,0]
    yc = circles[:,1]
    r  = circles[:,2]

    # creation of the dataframe
    d = {'xc': xc.flatten(), \
         'yc': yc.flatten(), \
         'r' : r.flatten()}

    circles_df = pd.DataFrame(data=d)

    return circles_df

def fromLinesColorsToDF(linesColors):
    # compute mean and std for each line
    means = np.squeeze(np.mean(linesColors,axis = 2))
    stdDevs = np.squeeze(np.std(linesColors,axis = 2))
    # group the data
    data = np.concatenate((means, stdDevs), axis = 1)
    # create the dataframe
    df = pd.DataFrame(data, columns = [string + str(x) for string in ['mean ch', 'std ch'] for x in range(3)])
    # compute mean of all the channels
    df['mean of mean ch'] = sum(df['mean ch'+str(ch_index)] for ch_index in range(3))/3
    df['mean of std ch'] = sum(df['std ch'+str(ch_index)] for ch_index in range(3))/3

    return df


def splitDFProperty(df, column, threshold):
    '''
    Splits a dataframe according to the threshold applied on a column
    Doesn't reset the indexes

    Parameters
    ----------
    df : pandas dataframe
        dataframe that has to be splitted.
    column : string
        name of the column whose property defines the splitting.
    threshold : value
        threshold value for splitting.

    Returns
    -------
    df_above_threshold : pandas dataframe
        contains all the rows whose column is EQUAL or GREATER than the treshold.
    df_below_threshold : pandas dataframe
        contains all the rows whose column is LOWER than the treshold.

    '''
    df_above_threshold = df[df[column] >= threshold]
    df_below_threshold = df[df[column] < threshold]

    return df_above_threshold, df_below_threshold

def pickDFProperty(df, column, nofelements = 1, ascending = False):
    '''
    Given a dataframe, returns only rows whose specified columns expresses the GREATEST (if ascending == False) or LOWEST (if ascending == True) values

    Parameters
    ----------
   df : pandas dataframe
       dataframe that has to be filtered.
   column : string
       name of the column whose property defines the splitting.
    nofelements : int, optional
        number of elements to be picked.
        The default is 1.
    ascending : bool, optional
        if False, the GREATEST values are returned 
        if True, the LOWEST. 
        The default is False.

    Returns
    -------
    df_filtered : pandas dataframe
        dataframe containing only the chosen row(s).
    indexes : array of one or more int
        indexes of the chosen rows of the original dataframe.

    '''
    # sort according to column
    df_sorted = df.sort_values(by = column, ascending = ascending)
    # pick only the first nofelements lines
    df_filtered = df_sorted.iloc[0:nofelements,:]

    # df_filtered is a dataframe -> look for index
    if nofelements>1:
        indexes =  df_filtered.index
    # df_filtered is a series -> look for name corresponding to the index of the dataframe where it was picked
    if nofelements==1:
        indexes = [df_filtered.index[0]]

    return df_filtered, indexes

def findWheelCentreAndHandrim(rgb_img, dep_img, ppx, ppy, fx, fy,dep_threshold_plane_detection=1300,
                              fitPlaneMethod = 'RANSAC',
                              minDist = 1, param1 = 50, param2 =60,
                              minRadius = 120, maxRadius = 140,
                              div = 4, edge_low_thresh = 50, edge_high_thresh = 150,
                              rho = 1, theta = np.pi/180, threshold = 30,
                              min_line_length_coeff = 0 , max_line_gap = 20,
                              w = 5, tolerance = 5000,
                              maxMinDepthHandrimPlaneDetection = np.nan,
                              showPlot = False, mainTitle = ''):
    '''
    Given an rgb image and corrispondent depth image with instrinsic camera coordinates, 
    performs circles and lines detection to find the handrim and the centre of the wheel. 
    Returns the centre of the wheel coordinates both in image and metric coordinates, 
    thecentre of the handrim coordinates both in image (+ the radius of the handrim in the image) and metric coordinates, 
    the plane where the handrim lays and the coordinates of the points of the handrim
    
    NB: slows down the execution

    Parameters
    ----------
    rgb_img : matrix M*N*3
        contains RGB or BGR information for every pixel.
    dep_img : matrix M*N*1
        contains DEP information for every pixel.
    ppx : float
        x centre of the metric camera on image
    ppy : float
        y centre of the metric camera on image
    fx : float
        focal distance on x
    fy : float
        focal distance on y
    minDist : int
        between centers. The default is 1.
    param1 : int
        refer to documentation. The default is 50.
    param2 : int
        refer to documentation. The default is 60.
    minRadius : int
        minimum radius of the detected circles. The default is 180.
    maxRadius : int
        maximum radius of the detected circles. The default is 200.
    div : float, optional
        once found the possible handrims, the image considered for centre 
        detection is cropped in a square from the centre till radius / div. 
        The default is 4.
    edge_low_thresh : int
        for canny edge detection.
        The default is 50.
    edge_high_thresh : int
        for canny edge detection.
        The default is 150.
    rho : double
        distance resolution in pixels of the Hough grid.
        The default is 1.
    theta : double
       angular resolution in radians of the Hough grid.
       The default is np.pi/180.
    threshold : int
        minimum number of votes (intersections in Hough grid cell).
        The default is 30.
    min_line_length : double
        minimum number of pixels making up a line.
        The default is 0.
    max_line_gap : double
        maximum gap in pixels between connectable line segments.
        The default is 20.
    w : int, optional
        amplitude of the padding when computing mean and std of the colors 
        crossed by the line. 
        The default is 5.
    tolerance : double, optional
        amplitude of the circular crown. The default is 5000.
    maxDepthHandrimPlaneDetection : double, optional
        during LSTSQ computation to detect the handrim, values more far away 
        than maxDepthHandrimPlaneDetection are ignored. 
        The default is -1, which doesn't delete any value
    minDepth : double, optional
        during LSTSQ computation to detect the handrim, values closer 
        than minDepth are ignored. 
        The default is 0.
    maxX : double, optional
        during LSTSQ computation to detect the handrim, values of x bigger than 
        maxX are ignored
        The default is -1, which doesn't delete any value
    minX : double, optional
        during LSTSQ computation to detect the handrim, values of x smaller than 
        maxY are ignored are ignored. 
        The default is 0.
    maxY : double, optional
        during LSTSQ computation to detect the handrim, values of y bigger than 
        maxY are ignored
        The default is -1, which doesn't delete any value
    minY : double, optional
        during LSTSQ computation to detect the handrim, values of y smaller than 
        maxY are ignored are ignored. 
        The default is 0.
    showPlot : boolean, optional
        if plots are shown. The default is False
    mainTitle : string, optional
        title to be added to the plots
    
    Returns
    -------
    wc_img : array 1*2
        x, y coordinates of the wheel on image 
    hrc_img : array 1*3
        x, y coordinates of the handrim on image + radius
    centre_metric : array 1*3
        x, y, z coordinates of the wheel on metric 
    handrimPlane : hppdWC.geom.Plane3d object
        plane where the handrim is laying.
    dataHandrim : array n*3
        contains x y z coordinates of the handrim, modeled as a circular crown

    '''
    assert fitPlaneMethod == 'LSTSQ' or fitPlaneMethod == 'RANSAC', \
        f"fitPlaneMethod not valid, possible are LSTSQ or RANSAC, got: {fitPlaneMethod}"
    dep_img = dep_img.astype('float')

# =============================================================================
#     #%%0.0 all possible handrims detection on rgb image
# =============================================================================
    circles = findCirclesOnImage(rgb_img, minDist = minDist,\
    param1 = param1, param2 = param2, minRadius = minRadius, maxRadius = maxRadius)

    # if only one circle is found, add another one so the structure is mantained
    if len(circles)==1:
        circles = np.append(circles, circles, axis=1)

    circles_df = fromCirclesToCirclesDF(circles)
    xc_hr, yc_hr, rhr_img = circles_df.mean(axis=0)

    if showPlot:
        plt.figure()
        plt.grid()
        plt.title(mainTitle + ' - detected handrim')
        plt.imshow(plots.circlesOnImage(rgb_img, circles))

# =============================================================================
#     #%%1 wheel centre detection on rgb image
# =============================================================================
    image = rgb_img.copy()
    image_h, image_w, _ = image.shape
# =============================================================================
#     #%%1.0 find lines on cropped image
# =============================================================================
    # initial guess of the area of interest according to the found handrims
    # get the boundaries for crop
    xmin = int(np.maximum(0, xc_hr - rhr_img/div))
    xmax = int(np.minimum(xc_hr + rhr_img/div, image_w-1))
    ymin = int(np.maximum(0, yc_hr - rhr_img/div))
    ymax = int(np.minimum(yc_hr + rhr_img/div, image_h-1))

    # crop the image
    img = image[ymin : ymax + 1, xmin : xmax + 1]
    img_h, img_w, _ = img.shape

    # find lines on the image
    lines = findLinesOnImage(img, \
    edge_low_thresh = edge_low_thresh, edge_high_thresh = edge_high_thresh, \
    rho = rho, theta = theta, threshold = threshold,\
    min_line_length = np.minimum(img_h,img_w) * min_line_length_coeff ,\
    max_line_gap = max_line_gap)

    if showPlot:
        plots.linesAndLinesOnImage(lines,img, mainTitle = mainTitle + ' - all the found lines')

# =============================================================================
#     # %%1.1 three longest lines with m>0 and m<0
# =============================================================================
    # convert lines into dataframe
    lines_df = fromLinesToLinesDF(lines)

    # cancel the lines with slope == 0 or inf
    lines_df_000_slope = lines_df[lines_df['slope'] == 0]
    lines_df_inf_slope = lines_df[lines_df['slope'] == np.inf]

    # if there are both vertical lines and horizontal lines:
    if not lines_df_000_slope.empty and not lines_df_inf_slope.empty:
        lines_df_pos_slope = lines_df_inf_slope.copy()
        lines_df_neg_slope = lines_df_000_slope.copy()
    else:
        # erase lines that are horizontal or vertical, they're on the edge between m>0 and m<0
        # lines_df = lines_df[lines_df['slope'] != 0]
        # lines_df = lines_df[lines_df['slope'] != np.inf]
        # SPLIT THE DF into lines with pos and neg slope
        lines_df_pos_slope, lines_df_neg_slope = splitDFProperty(lines_df, 'slope', 0.000001)

    # pick the 3 longest lines of each df
    lines_df_pos_slope, index_pos_slope = pickDFProperty(lines_df_pos_slope, 'distance', 3)
    lines_df_neg_slope, index_neg_slope = pickDFProperty(lines_df_neg_slope, 'distance', 3)
    #0.000001 instead of 0 so horizontal lines are considered as neg slope and vertical lines, whoSe slope is inf, are considered as pos slope

    # considering only the chosen lines
    indexes = np.append(index_pos_slope, index_neg_slope)
    validLines = lines[:][indexes]
    # updating the dataframe as well
    validLines_df = lines_df.loc[indexes.tolist(), :].reset_index(drop=True)

    # renaming
    lines = validLines
    lines_df = validLines_df.copy()

    if showPlot:
        plots.linesAndLinesOnImage(lines,img, mainTitle = mainTitle + ' - 3 longest lines with m>0 and m<0')

# =============================================================================
#     #%%1.2 only the line with highest red mean value with m>0 and m<0
# =============================================================================
    # find which lines are covering more red areas
    linesColors = colorsOnTheLineImage(img, lines, w = w)
    linesColors_df = fromLinesColorsToDF(linesColors)

    lines_df = pd.concat([lines_df, linesColors_df], axis = 1)

    if showPlot:
        plots.colorsAlongTheLine(linesColors, mainTitle = mainTitle + ' - corresponding area to pixel w = '+ str(w))

        plots.gaussColorsAlongTheLine(lines_df, mainTitle = mainTitle + ' - mean and std deviation w = '+ str(w))

    # line with smallest std dev of channels with m>0 and m<0
    # split into lines with pos and neg slope
    lines_df_pos_slope, lines_df_neg_slope = splitDFProperty(lines_df, 'slope', 0.00001)
    #0.000001 instead of 0 so horizontal lines are considered as neg slope and vertical lines, whose slope is inf, are considered as pos slope

    # pick the one with highest red value of each df
    lines_df_pos_slope, index_pos_slope = pickDFProperty(lines_df_pos_slope, 'mean ch0', 1, ascending = False)
    lines_df_neg_slope, index_neg_slope = pickDFProperty(lines_df_neg_slope, 'mean ch0', 1, ascending = False)

    # pick the one with smallest std dev of each df
    # lines_df_pos_slope, index_pos_slope = pickDFProperty(lines_df_pos_slope, 'mean of std ch', 1, ascending = True)
    # lines_df_neg_slope, index_neg_slope = pickDFProperty(lines_df_neg_slope, 'mean of std ch', 1, ascending = True)

    # considering only the chosen lines
    indexes = np.append(index_pos_slope, index_neg_slope)
    validLines = lines[:][indexes]
    # updating the dataframe as well
    validLines_df = lines_df.loc[indexes.tolist(), :].reset_index(drop=True)

    # renaming
    lines = validLines
    lines_df = validLines_df.copy()

# =============================================================================
#     #%%1.3 use the class geom to compute the intersection of each pair of lines
# =============================================================================
    # creation of the lines
    validLines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            p1 = geom.Point2d(x1, y1)
            p2 = geom.Point2d(x2, y2)
            myLine = geom.Line2d(p1 = p1, p2 = p2)
            validLines.append(myLine)

    # in case of multiple lines
    # intersection_x = []
    # intersection_y = []        
    # for line1 in validLines:
    #     for line2 in validLines:
    #         # not computing the intersection for lines going in the same direction
    #         if line1.m * line2.m < 0:
    #             x, y = line1.intersection(line2)
    #             intersection_x.append(x)
    #             intersection_y.append(y)

    # only two lines
    intersection_x, intersection_y = validLines[0].intersection(validLines[1])
# =============================================================================
#      #%%1.4 add to intersection the crop of the original image
# =============================================================================
    x_centre = np.mean(intersection_x)
    y_centre = np.mean(intersection_y)

    # value on the complete image
    xwc_img = xmin + x_centre
    ywc_img = ymin + y_centre

    if showPlot:
        fig, ax = plots.linesAndLinesOnImage(lines,img, mainTitle = mainTitle + ' - only the two chosen lines', indexes = indexes)

        fig = plots.detectedCentreOfWheel(image, xwc_img, ywc_img, mainTitle = mainTitle +  ' - detected centre of the wheel', lines = lines, xmin = xmin, ymin = ymin, indexes = indexes)
        plt.plot(xwc_img, ywc_img, 'r*', markersize = 10)

# =============================================================================
#     #%%2.0 handrim detection on rgb image
# =============================================================================
    # pick the handrims whose x distance is minimum with respect to the centre of the wheel
    circles_df_min_dist = circles_df[abs(circles_df['xc'] - xwc_img) == min(abs(circles_df['xc'] - xwc_img))]

    # among, them pick the handrim with the greatest radius
    circles_df_min_dist_biggest_radius = circles_df_min_dist[circles_df_min_dist['r'] == max(circles_df_min_dist['r'])]

    # get parameters of the chosen one
    xhrc_img = circles_df_min_dist_biggest_radius['xc'].iloc[0]
    yhrc_img = circles_df_min_dist_biggest_radius['yc'].iloc[0]
    rhrc_img = circles_df_min_dist_biggest_radius['r'].iloc[0]

# =============================================================================
#     #%%3 handrim plane detection on dep image
# =============================================================================
    image_h, image_w = dep_img.shape
    xmask, ymask = np.meshgrid(np.arange(0, image_w, 1), np.arange(0, image_h, 1))
# =============================================================================
#     #%%3.0 extract 3D points for LSTSQ
# =============================================================================
    dep_image = dep_img.copy()
    # remove invalid values
    dep_image[dep_image <= 0] = np.nan
    dep_image[dep_image>dep_threshold_plane_detection] = np.nan
    # creating a mask of the area of interest (circular crown)
    maskValidValues = abs(((xmask-xhrc_img)**2+(ymask-yhrc_img)**2-(rhrc_img)**2))<tolerance
    # give nan values to all the pixel outside of the area of interest
    dep_image[~maskValidValues] = np.nan

    # find points of the mask in the real 3D world
    pc = geom.convert_depth_frame_to_pointcloud_pp_ff(dep_image, ppx, ppy, fx, fy)
    x,y,z = pc
    data = np.transpose([x,y,z])
    # # remove points outside the range
    if not np.isnan(maxMinDepthHandrimPlaneDetection).all():
        data[:,2][data[:,2]>maxMinDepthHandrimPlaneDetection[0]] = np.nan
        data[:,2][data[:,2]<maxMinDepthHandrimPlaneDetection[1]] = np.nan
    # if maxX > 0:
    #     data[:,2][data[:,0]>maxX-ppx] = np.nan
    # data[:,2][data[:,0]<minX-ppx] = np.nan
    # if maxY > 0:
    #     data[:,2][data[:,1]>maxY-ppy] = np.nan
    # data[:,2][data[:,1]<minY-ppy] = np.nan
    #remove nan rows
    data = data[~np.isnan(data).any(axis=1)]

    if showPlot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:,0],data[:,1],data[:,2],c=data[:,2], marker = '.')
        ax.view_init(elev=-90, azim=-90)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    if showPlot:
        highlightedImage = plots.highlightPartOfImage(image, maskValidValues, coeff = 0.7, colorNotInterest = [255, 255, 255])
        handrim = circles[:,circles_df_min_dist_biggest_radius.index]
        plt.figure()
        plt.grid()
        plt.title(mainTitle + ' - detected handrim')
        plt.imshow(plots.circlesOnImage(highlightedImage, handrim))
        plt.axvline(xwc_img, color = 'r')
        plt.axhline(ywc_img, color = 'r')
        plt.axhline(handrim[:,0,1], color = (0,1,0))

        plots.orthogonalProjectionRCamView(data, flag = 'xyzdata', mainTitle = mainTitle + ' - available data')

# =============================================================================
#     #%%3.1 fit plane with least squares
# =============================================================================
    # data are expressed in the depth camera ref, not on top left
    if fitPlaneMethod == 'LSTSQ':
        coeffX, coeffY, constant, normal = geom.fitPlaneLSTSQ(data)
    if fitPlaneMethod == 'RANSAC':
        coeffX, coeffY, constant, normal = geom.fitPlaneRANSAC(data)
    handrimPlane = geom.Plane3d(coeffX = coeffX, coeffY = coeffY, constant = constant)

# =============================================================================
#     #%%3.2 3D coordinates of the handrim, modeled as a 2D circular crown
# =============================================================================
    dep_image = dep_img.copy()
    # creating a mask of the area of interest (circular crown)
    maskValidValues = abs(((xmask-xhrc_img)**2+(ymask-yhrc_img)**2-(rhrc_img)**2))<tolerance
    # give nan values to all the pixel outside of the area of interest
    dep_image[~maskValidValues] = np.nan

    # find dataHandrim in the real 3D world
    pc = geom.convert_depth_frame_to_pointcloud_pp_ff(dep_image, ppx, ppy, fx, fy)
    x,y,z = pc
    dataHandrim = np.transpose([x,y,z])
    #remove nans
    dataHandrim = dataHandrim[~np.isnan(dataHandrim).any(axis=1)]
    X = dataHandrim[:,0]
    Y = dataHandrim[:,1]
    Z = handrimPlane.findZ(X,Y)
    dataHandrim = np.transpose(np.row_stack((X,Y,Z)))

# =============================================================================
#     #%%4.0 3D coordinates of the centres of the wheel and of the handrim
# =============================================================================
    # laying on the handrimPlane
    x_centre_metric, y_centre_metric  = geom.convert_pixel_coord_to_metric_coordinate_pp_ff(xwc_img, ywc_img, ppx, ppy, fx, fy)
    z_centre_metric = handrimPlane.findZ(x_centre_metric, y_centre_metric)

# =============================================================================
#     #%%5.0 pack outputs
# =============================================================================
    # 2D coordinates of wheel centre
    wc_img = [xwc_img, ywc_img]
    # 2D coordinates of handrim on the image + radius
    hrc_img = [xhrc_img, yhrc_img, rhr_img]
    # 3D coordinates of wheel centre in the real world
    centre_metric = [x_centre_metric, y_centre_metric, z_centre_metric]


    # the values of these two should be close to the real handrim in m
    # np.nanmean(np.sqrt((dataHandrim[:,0]-centre_metric[0])**2+(dataHandrim[:,1]-centre_metric[1])**2))
    # np.nanmean(np.sqrt((dataHandrim[:,0]-centre_metric[0])**2+(dataHandrim[:,1]-centre_metric[1])**2+(dataHandrim[:,2]-centre_metric[2])**2))

    # to plot available data, the detected plane and the handrim circular crown
    xmin = np.min(X)
    xmax = np.max(X)
    ymin = np.min(Y)
    ymax = np.max(Y)

    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
    #xx, yy = geom.convert_pixel_coord_to_metric_coordinate_pp_ff(xx, yy, ppx, ppy, fx, fy)
    Zplane = handrimPlane.findZ(xx, yy)
    dataPlane = np.transpose(np.row_stack((xx.flatten(),yy.flatten(), Zplane.flatten())))
    if showPlot:
        fig, ax = plots.orthogonalProjectionRCamView([data, dataPlane, dataHandrim, np.array([centre_metric])], \
        flag = 'xyzdata', mainTitle = mainTitle + ' - final solution', alpha = 0.1, \
        color_list = ['', 'k', '', 'r'])


    return wc_img, hrc_img, centre_metric, handrimPlane, dataPlane

def findHandrim(rgb_img, dep_img, ppx, ppy, fx, fy,xwc_img,ywc_img,dep_threshold_plane_detection=1300,
                              fitPlaneMethod = 'RANSAC',
                              minDist = 1, param1 = 50, param2 =60,
                              minRadius = 120, maxRadius = 140,
                              div = 4, edge_low_thresh = 50, edge_high_thresh = 150,
                              rho = 1, theta = np.pi/180, threshold = 30,
                              min_line_length_coeff = 0 , max_line_gap = 20,
                              w = 5, tolerance = 5000,
                              maxMinDepthHandrimPlaneDetection = np.nan,
                              showPlot = False, mainTitle = ''):
    '''
    Given an rgb image and corrispondent depth image with instrinsic camera coordinates, 
    performs circles and lines detection to find the handrim and the centre of the wheel. 
    Returns the centre of the wheel coordinates both in image and metric coordinates, 
    thecentre of the handrim coordinates both in image (+ the radius of the handrim in the image) and metric coordinates, 
    the plane where the handrim lays and the coordinates of the points of the handrim
    
    NB: slows down the execution

    Parameters
    ----------
    rgb_img : matrix M*N*3
        contains RGB or BGR information for every pixel.
    dep_img : matrix M*N*1
        contains DEP information for every pixel.
    ppx : float
        x centre of the metric camera on image
    ppy : float
        y centre of the metric camera on image
    fx : float
        focal distance on x
    fy : float
        focal distance on y
    minDist : int
        between centers. The default is 1.
    param1 : int
        refer to documentation. The default is 50.
    param2 : int
        refer to documentation. The default is 60.
    minRadius : int
        minimum radius of the detected circles. The default is 180.
    maxRadius : int
        maximum radius of the detected circles. The default is 200.
    div : float, optional
        once found the possible handrims, the image considered for centre 
        detection is cropped in a square from the centre till radius / div. 
        The default is 4.
    edge_low_thresh : int
        for canny edge detection.
        The default is 50.
    edge_high_thresh : int
        for canny edge detection.
        The default is 150.
    rho : double
        distance resolution in pixels of the Hough grid.
        The default is 1.
    theta : double
       angular resolution in radians of the Hough grid.
       The default is np.pi/180.
    threshold : int
        minimum number of votes (intersections in Hough grid cell).
        The default is 30.
    min_line_length : double
        minimum number of pixels making up a line.
        The default is 0.
    max_line_gap : double
        maximum gap in pixels between connectable line segments.
        The default is 20.
    w : int, optional
        amplitude of the padding when computing mean and std of the colors 
        crossed by the line. 
        The default is 5.
    tolerance : double, optional
        amplitude of the circular crown. The default is 5000.
    maxDepthHandrimPlaneDetection : double, optional
        during LSTSQ computation to detect the handrim, values more far away 
        than maxDepthHandrimPlaneDetection are ignored. 
        The default is -1, which doesn't delete any value
    minDepth : double, optional
        during LSTSQ computation to detect the handrim, values closer 
        than minDepth are ignored. 
        The default is 0.
    maxX : double, optional
        during LSTSQ computation to detect the handrim, values of x bigger than 
        maxX are ignored
        The default is -1, which doesn't delete any value
    minX : double, optional
        during LSTSQ computation to detect the handrim, values of x smaller than 
        maxY are ignored are ignored. 
        The default is 0.
    maxY : double, optional
        during LSTSQ computation to detect the handrim, values of y bigger than 
        maxY are ignored
        The default is -1, which doesn't delete any value
    minY : double, optional
        during LSTSQ computation to detect the handrim, values of y smaller than 
        maxY are ignored are ignored. 
        The default is 0.
    showPlot : boolean, optional
        if plots are shown. The default is False
    mainTitle : string, optional
        title to be added to the plots
    
    Returns
    -------
    wc_img : array 1*2
        x, y coordinates of the wheel on image 
    hrc_img : array 1*3
        x, y coordinates of the handrim on image + radius
    centre_metric : array 1*3
        x, y, z coordinates of the wheel on metric 
    handrimPlane : hppdWC.geom.Plane3d object
        plane where the handrim is laying.
    dataHandrim : array n*3
        contains x y z coordinates of the handrim, modeled as a circular crown

    '''
    assert fitPlaneMethod == 'LSTSQ' or fitPlaneMethod == 'RANSAC', \
        f"fitPlaneMethod not valid, possible are LSTSQ or RANSAC, got: {fitPlaneMethod}"
    dep_img = dep_img.astype('float')

# =============================================================================
#     #%%0.0 all possible handrims detection on rgb image
# =============================================================================
    circles = findCirclesOnImage(rgb_img, minDist = minDist,\
    param1 = param1, param2 = param2, minRadius = minRadius, maxRadius = maxRadius)

    # if only one circle is found, add another one so the structure is mantained
    if len(circles)==1:
        circles = np.append(circles, circles, axis=1)

    circles_df = fromCirclesToCirclesDF(circles)
    xc_hr, yc_hr, rhr_img = circles_df.mean(axis=0)

    if showPlot:
        plt.figure()
        plt.grid()
        plt.title(mainTitle + ' - detected handrim')
        plt.imshow(plots.circlesOnImage(rgb_img, circles))

# =============================================================================
#     #%%1 wheel centre detection on rgb image
# =============================================================================
    image = rgb_img.copy()
    image_h, image_w, _ = image.shape
# =============================================================================
#     #%%1.0 find lines on cropped image
# =============================================================================
    # initial guess of the area of interest according to the found handrims
    # get the boundaries for crop
    xmin = int(np.maximum(0, xc_hr - rhr_img/div))
    xmax = int(np.minimum(xc_hr + rhr_img/div-10, image_w-1))
    ymin = int(np.maximum(0, yc_hr - rhr_img/div))
    ymax = int(np.minimum(yc_hr + rhr_img/div -20, image_h-1))

    # crop the image
    img = image[ymin : ymax + 1, xmin : xmax + 1]
    img_h, img_w, _ = img.shape


# =============================================================================
#     #%%2.0 handrim detection on rgb image
# =============================================================================
    # pick the handrims whose x distance is minimum with respect to the centre of the wheel
    circles_df_min_dist = circles_df[abs(circles_df['xc'] - xwc_img) == min(abs(circles_df['xc'] - xwc_img))]

    # among, them pick the handrim with the greatest radius
    circles_df_min_dist_biggest_radius = circles_df_min_dist[circles_df_min_dist['r'] == max(circles_df_min_dist['r'])]

    # get parameters of the chosen one
    xhrc_img = circles_df_min_dist_biggest_radius['xc'].iloc[0]
    yhrc_img = circles_df_min_dist_biggest_radius['yc'].iloc[0]
    rhrc_img = circles_df_min_dist_biggest_radius['r'].iloc[0]

# =============================================================================
#     #%%3 handrim plane detection on dep image
# =============================================================================
    image_h, image_w = dep_img.shape
    xmask, ymask = np.meshgrid(np.arange(0, image_w, 1), np.arange(0, image_h, 1))
# =============================================================================
#     #%%3.0 extract 3D points for LSTSQ
# =============================================================================
    dep_image = dep_img.copy()
    # remove invalid values
    dep_image[dep_image <= 0] = np.nan
    dep_image[dep_image>dep_threshold_plane_detection] = np.nan
    # creating a mask of the area of interest (circular crown)
    maskValidValues = abs(((xmask-xhrc_img)**2+(ymask-yhrc_img)**2-(rhrc_img)**2))<tolerance
    # give nan values to all the pixel outside of the area of interest
    dep_image[~maskValidValues] = np.nan

    # find points of the mask in the real 3D world
    pc = geom.convert_depth_frame_to_pointcloud_pp_ff(dep_image, ppx, ppy, fx, fy)
    x,y,z = pc
    data = np.transpose([x,y,z])
    # # remove points outside the range
    if not np.isnan(maxMinDepthHandrimPlaneDetection).all():
        data[:,2][data[:,2]>maxMinDepthHandrimPlaneDetection[0]] = np.nan
        data[:,2][data[:,2]<maxMinDepthHandrimPlaneDetection[1]] = np.nan
    # if maxX > 0:
    #     data[:,2][data[:,0]>maxX-ppx] = np.nan
    # data[:,2][data[:,0]<minX-ppx] = np.nan
    # if maxY > 0:
    #     data[:,2][data[:,1]>maxY-ppy] = np.nan
    # data[:,2][data[:,1]<minY-ppy] = np.nan
    #remove nan rows
    data = data[~np.isnan(data).any(axis=1)]

    if showPlot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:,0],data[:,1],data[:,2],c=data[:,2], marker = '.')
        ax.view_init(elev=-90, azim=-90)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    if showPlot:
        highlightedImage = plots.highlightPartOfImage(image, maskValidValues, coeff = 0.7, colorNotInterest = [255, 255, 255])
        handrim = circles[:,circles_df_min_dist_biggest_radius.index]
        plt.figure()
        plt.grid()
        plt.title(mainTitle + ' - detected handrim')
        plt.imshow(plots.circlesOnImage(highlightedImage, handrim))
        plt.axvline(xwc_img, color = 'r')
        plt.axhline(ywc_img, color = 'r')
        plt.axhline(handrim[:,0,1], color = (0,1,0))

        plots.orthogonalProjectionRCamView(data, flag = 'xyzdata', mainTitle = mainTitle + ' - available data')

# =============================================================================
#     #%%3.1 fit plane with least squares
# =============================================================================
    # data are expressed in the depth camera ref, not on top left
    if fitPlaneMethod == 'LSTSQ':
        coeffX, coeffY, constant, normal = geom.fitPlaneLSTSQ(data)
    if fitPlaneMethod == 'RANSAC':
        coeffX, coeffY, constant, normal = geom.fitPlaneRANSAC(data)
    handrimPlane = geom.Plane3d(coeffX = coeffX, coeffY = coeffY, constant = constant)

# =============================================================================
#     #%%3.2 3D coordinates of the handrim, modeled as a 2D circular crown
# =============================================================================
    dep_image = dep_img.copy()
    # creating a mask of the area of interest (circular crown)
    maskValidValues = abs(((xmask-xhrc_img)**2+(ymask-yhrc_img)**2-(rhrc_img)**2))<tolerance
    # give nan values to all the pixel outside of the area of interest
    dep_image[~maskValidValues] = np.nan

    # find dataHandrim in the real 3D world
    pc = geom.convert_depth_frame_to_pointcloud_pp_ff(dep_image, ppx, ppy, fx, fy)
    x,y,z = pc
    dataHandrim = np.transpose([x,y,z])
    #remove nans
    dataHandrim = dataHandrim[~np.isnan(dataHandrim).any(axis=1)]
    X = dataHandrim[:,0]
    Y = dataHandrim[:,1]
    Z = handrimPlane.findZ(X,Y)
    dataHandrim = np.transpose(np.row_stack((X,Y,Z)))

# =============================================================================
#     #%%4.0 3D coordinates of the centres of the wheel and of the handrim
# =============================================================================
    # laying on the handrimPlane
    x_centre_metric, y_centre_metric  = geom.convert_pixel_coord_to_metric_coordinate_pp_ff(xwc_img, ywc_img, ppx, ppy, fx, fy)
    z_centre_metric = handrimPlane.findZ(x_centre_metric, y_centre_metric)

# =============================================================================
#     #%%5.0 pack outputs
# =============================================================================
    # 2D coordinates of wheel centre
    wc_img = [xwc_img, ywc_img]
    # 2D coordinates of handrim on the image + radius
    hrc_img = [xhrc_img, yhrc_img, rhr_img]
    # 3D coordinates of wheel centre in the real world
    centre_metric = [x_centre_metric, y_centre_metric, z_centre_metric]


    # the values of these two should be close to the real handrim in m
    # np.nanmean(np.sqrt((dataHandrim[:,0]-centre_metric[0])**2+(dataHandrim[:,1]-centre_metric[1])**2))
    # np.nanmean(np.sqrt((dataHandrim[:,0]-centre_metric[0])**2+(dataHandrim[:,1]-centre_metric[1])**2+(dataHandrim[:,2]-centre_metric[2])**2))

    # to plot available data, the detected plane and the handrim circular crown
    xmin = np.min(X)
    xmax = np.max(X)
    ymin = np.min(Y)
    ymax = np.max(Y)

    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
    #xx, yy = geom.convert_pixel_coord_to_metric_coordinate_pp_ff(xx, yy, ppx, ppy, fx, fy)
    Zplane = handrimPlane.findZ(xx, yy)
    dataPlane = np.transpose(np.row_stack((xx.flatten(),yy.flatten(), Zplane.flatten())))
    if showPlot:
        fig, ax = plots.orthogonalProjectionRCamView([data, dataPlane, dataHandrim, np.array([centre_metric])], \
        flag = 'xyzdata', mainTitle = mainTitle + ' - final solution', alpha = 0.1, \
        color_list = ['', 'k', '', 'r'])


    return wc_img, hrc_img, centre_metric, handrimPlane, dataPlane
def computeStatFeatures(array):
    '''
    Given an array, computes statistical features and returns them in a list

    Parameters
    ----------
    array : array
        on which the statistical features are computed.

    Returns
    -------
    results : list of double
        statistical values of each variable.
    res_str : list of str
        name of the variable corresponding to each element of results.

    '''
    desc = scipy.stats.describe(array)
    # minimum = desc.minmax[0]
    # maximum = desc.minmax[1]
    mean = desc.mean
    stdD = np.std(array)
    skew = desc.skewness
    kurt = desc.kurtosis
    # p_01 = scipy.stats.scoreatpercentile(array,  1)
    p_05 = scipy.stats.scoreatpercentile(array,  5)
    # p_15 = scipy.stats.scoreatpercentile(array, 15)
    p_25 = scipy.stats.scoreatpercentile(array, 25)
    # p_35 = scipy.stats.scoreatpercentile(array, 35)
    # p_45 = scipy.stats.scoreatpercentile(array, 45)
    p_50 = scipy.stats.scoreatpercentile(array,50)
    # p_55 = scipy.stats.scoreatpercentile(array, 55)
    # p_65 = scipy.stats.scoreatpercentile(array, 65)
    p_75 = scipy.stats.scoreatpercentile(array, 75)
    #p_85 = scipy.stats.scoreatpercentile(array, 85)
    p_95 = scipy.stats.scoreatpercentile(array, 95)
    #p_99 = scipy.stats.scoreatpercentile(array, 99)

    results = [ mean,   stdD,   skew,   kurt,   p_05,   p_25,   p_50,   p_75,   p_95]
    res_str = ['mean', 'stdD', 'skew', 'kurt', 'p_05', 'p_25', 'p_50', 'p_75', 'p_95']
    return results, res_str

def statOnXYZpointcloud(XYZ, showPlot = False):
    # if only one point is passed, transform it into array
    if XYZ.ndim == 1:
        XYZ = np.expand_dims(XYZ, axis = 0)

    # NEXT TIME SAVE ALSO X AND Y RAW, YOU NEVER KNOW!
    x = XYZ[:,0]
    y = XYZ[:,1]
    z = XYZ[:,2]

    xfeat, res_str = computeStatFeatures(x)
    yfeat, _ = computeStatFeatures(y)
    zfeat, _ = computeStatFeatures(z)

    # concatenate the list
    results = xfeat + yfeat + zfeat
    res_str_total = [varName + feat for varName in ['x-','y-','z-'] for feat in res_str]

    if showPlot:
        plots.histograms([x,y,z], nrows= 1, listOfTitles = ['x', 'y', 'z'])

    return results, res_str_total


def computeHandDepFeatures(XYZ, rHandrim, showPlot = False):
    '''
    Given XYZ, point cloud of the hand expressed in the wheel centered ref frame,
    for every point computes:
        - x:
        - y: 
        - vDist: vertical distance on the plane (z coordinate)
        - hDistHR: horizontal distance on the plane divided by the radius 
        of the handrim
        - vAngle: angle on the vertical
        - hAngle: angle on the horizontal plane
    Of each array of values, computes the statistical features (like mean, stdDev,
    skewness, kurtosis, percxx) with the function computeStatFeatures


    Parameters
    ----------
    XYZ : array n*3
        contains XYZ coordinates of each point.
    rHandrim : double
        radius of the handrim in the same measurement unit of x and y coordinates.
    showPlot : bool, optional
        If true, shows histograms of the computed variable . The default is False.

    Returns
    -------
    results : list of double
        statistical values of each variable.
    res_str_total : list of str
        name of the variable corresponding to each element of results.
        

    '''
    # if only one point is passed, transform it into array
    if XYZ.ndim == 1:
        XYZ = np.expand_dims(XYZ, axis = 0)

    # NEXT TIME SAVE ALSO X AND Y RAW, YOU NEVER KNOW!
    x = XYZ[:,0]
    y = XYZ[:,1]
    z = XYZ[:,2]

    # vertical distance wrt the plane xy (height on axis z)
    vDist = z
    # horizontal distance on the plane xy from the centre
    hDist = np.sqrt(x**2+y**2)
    # horizontal distance S with respect to the handrim
    hDistHR = hDist / rHandrim
    # vertical angle according to the two distances
    vAngle = np.rad2deg(np.arctan2(vDist,hDist))
    # horizontal angle on the plane xy from the centre
    hAngle = np.rad2deg(np.arctan2(y, x))

    vDist_stat, res_str = computeStatFeatures(vDist)
    hDistHR_stat, _ = computeStatFeatures(hDistHR)
    vAngle_stat, _ = computeStatFeatures(vAngle)
    hAngle_stat, _ = computeStatFeatures(hAngle)

    # concatenate the list
    results = vDist_stat + hDistHR_stat + vAngle_stat + hAngle_stat
    res_str_total = [varName + x for varName in ['vd-','hd-','va-', 'ha-'] for x in res_str]

    if showPlot:
        plots.histograms([vDist, hDistHR, vAngle, hAngle], nrows= 2, listOfTitles = ['vertical dist', 'horiz dist S on handrim', 'elevation angle', 'horizontal angle'])

    return results, res_str_total


def cos_func(times, amplitude, frequency, phase, offset):
    return amplitude * np.cos(frequency * (times - phase)) + offset

def angleFit(time, listYarrays, filter_amp, window_amp, step_estimation = 1, fill_gap_step_estimation = True, in_guess = None):
    '''
    Given time of recording and array(s) y, slices the array(s) and tries to fit a sinusoidal function, cos = A*cos(omega(t-phi))+B.
    The arrays are filtered with an amplitude of filter_amp (please refer to mov_avg_time_variable_freq()) and then, for every timing window of amplitude window_amp, the filtered signal is considered and a sinusoid is fitted on it.
    The outputs are:
        A - amplitude
        omega - freq
        phi - phase
        B - offset
    At each iteration, these the found parameters at the previous step are used as initial guess.

    Parameters
    ----------
    time : array
        timing of recording.
    listYarrays : array or list of array(s)
        signal(s).
    filter_amp : float
        WHOLE amplitude of the filter for mov_avg_time_variable_freq().
    window_amp : float
        WHOLE amplitude of the timing window of the considered signal where the sinusoidal function is fitted.
    step_estimation : int >= 1, optional
        every how many iterations update the model. The default is 1.
    fill_gap_step_estimation : bool, optional
        the gaps are felt with repetition or nan are left. The default is True.
    in_guess : list of list(s) of four elements, optional
        contains the initial guess for every array. The four elements are amplitude, freq, phase, offset. The default is None.

    Returns
    -------
    popt_array : array 4*Nsamples*Narrays
        contains amplitude, freq, phase, offset for each sample for each array.
    to obtain the values of the sinusoid, pick set of 4 parameters and use
        cos_func(times, amplitude, frequency, phase, offset)
    arrayYarrays_filt : array
        filtered array(s) where the sinusoidal function is estimated 
    '''

    oneArrayFlag = False
    # add a dimension
    if utils.containsScalars(listYarrays):
        oneArrayFlag = True
        listYarrays = [listYarrays]

    nOfSamples = len(time)
    nOfArrays = len(listYarrays)

    # filter the arrays
    arrayYarrays_filt = mov_avg_time_variable_freq(time, listYarrays, filter_amp/2)

    time_units_before = window_amp/2
    time_units_after = window_amp/2

    # will be redefined when curve_fit gives valid output
    popt_array = np.array([[[np.nan]*4]*nOfSamples]*nOfArrays)

    # if in_guess are defined by user, put them on the tail, will be rewritten
    if in_guess:
        popt_array[:,-1] = in_guess
    else:
        for j in np.arange(nOfArrays):
            popt_array[j,-1] = [1.5*np.nanstd(arrayYarrays_filt[j]), 2*np.pi, 0, np.nanmean(arrayYarrays_filt[j])]

    # for every sample
    for i in np.arange(nOfSamples, step = step_estimation):
        this_moment = time[i]
        timing_window = time.copy()
        # give nan value to all the samples before
        timing_window[timing_window<this_moment-time_units_before] = np.nan
        # give nan values to all the samples after
        timing_window[timing_window>this_moment+time_units_after ] = np.nan
        # get indexes where timing_window is not nan
        indexes = np.argwhere(~np.isnan(timing_window))
        # consider x and y only where timing_window is not nan
        this_time = time[indexes]
        this_arrayYarrays_filt = arrayYarrays_filt[:,indexes]

        # for every array
        for this_y, j in zip(this_arrayYarrays_filt, np.arange(nOfArrays)):
            try: # in case no sinusoid is found
                # if popt of previous iteration is defined, use it as start point
                if np.isnan(popt_array[j,i-1]).any():
                    in_guess_amp, in_guess_freq, in_guess_phase, in_guess_offset = popt_array[j,-1]
                else: # if popt isn't -> use initial hints stored in the las element
                    in_guess_amp, in_guess_freq, in_guess_phase, in_guess_offset = popt_array[j,i-1]
    
                tmp, _ = curve_fit(cos_func, np.squeeze(this_time), np.squeeze(fillNanWithInterp(this_y,this_time)), p0=(in_guess_amp, in_guess_freq, in_guess_phase, in_guess_offset))
                popt_array[j,i] = tmp
            except:
                pass
        if fill_gap_step_estimation:
            popt_array[j,i:i+step_estimation] = popt_array[j,i]

    if oneArrayFlag: # bring back to single dimension
        return np.squeeze(popt_array), np.squeeze(arrayYarrays_filt)
    return popt_array, arrayYarrays_filt



def circleFit(time, xcoord, ycoord, filter_amp, window_amp, step_estimation = 1, fill_gap_step_estimation = True):

    nOfSamples = len(time)
    # filter the arrays
    xcoord_filt, ycoord_filt = mov_avg_time_variable_freq(time, [xcoord, ycoord], filter_amp/2)

    time_units_before = window_amp/2
    time_units_after = window_amp/2

    # will be redefined when circle_fit gives valid output
    found_circles_array = np.array([[np.nan]*3]*nOfSamples)

    # for every sample
    for i in np.arange(nOfSamples, step = step_estimation):
        this_moment = time[i]
        timing_window = time.copy()
        # give nan value to all the samples before
        timing_window[timing_window<this_moment-time_units_before] = np.nan
        # give nan values to all the samples after
        timing_window[timing_window>this_moment+time_units_after ] = np.nan
        # get indexes where timing_window is not nan
        indexes = np.argwhere(~np.isnan(timing_window))
        # consider x and y only where timing_window is not nan
        x = xcoord_filt[indexes]
        y = ycoord_filt[indexes]

        try:
            xc,yc,r,_ = cf.least_squares_circle(np.squeeze(np.transpose([x,y])))
            found_circles_array[i] = [xc,yc,r]
        except:
            pass
        if fill_gap_step_estimation:
            found_circles_array[i:i+step_estimation] = found_circles_array[i]

    return found_circles_array

def circleFitDiffFromGiven(nominalCircleXYR, time, xcoord, ycoord, filter_amp, window_amp, step_estimation = 1, fill_gap_step_estimation = True):
    '''
    for every sample, tries to fit a circle according to the coordinates x and y 
    inside the filter amplitude and compares it with the nominal one

    Returns an array of 3 columns:
        - difference of x coordinate of the centre
        - difference of y coordinate of the centre
        - difference of radius

    Parameters
    ----------
    nominalCircleXYR : TYPE
        DESCRIPTION.
    time : TYPE
        DESCRIPTION.
    xcoord : TYPE
        DESCRIPTION.
    ycoord : TYPE
        DESCRIPTION.
    filter_amp : TYPE
        DESCRIPTION.
    window_amp : TYPE
        DESCRIPTION.
    step_estimation : TYPE, optional
        DESCRIPTION. The default is 1.
    fill_gap_step_estimation : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    diffXYR : TYPE
        DESCRIPTION.

    '''
    nOfSamples = len(time)
    foundCircleXYR = circleFit(time, xcoord, ycoord, filter_amp, window_amp, step_estimation = 1, fill_gap_step_estimation = True)

    diffXYR = foundCircleXYR - [utils.makeList(nominalCircleXYR)]*nOfSamples

    return diffXYR

#%% loops
def depROI(fileCompletePath, CSVroiDirectory, CSVOutputDirectory, videoOutputDirectory = '', frequency = 60, nFrames = 20000):
    '''
    Given a bag file and a directory in which the region of interest for every 
    frame of the recording is specified by means of xmin xmax ymin ymax, 
    playbacks the file frame by frame and computes statistical features on the 
    depth of that region:
        minimum, maximum, mean, stdDev, skewness, kurtosis and percentiles 

    EVERYTHING IS EXPRESSED IN THE CAMERA REFERENCE FRAME:
        ----------->X
        |
        |
        |
        v Y
    with Z = 0 correspoding to the camera
    
    A loading bar is provided to the user during the execution. The number of
    frames is estimated consequently the actual number of frames elaborated 
    will be lower.
    
    During the execution, if videOutputDirectory is specified, two videos are
    recorded: 
        - the raw rgb images coming from the bag file with the square of roi drawn on it
        - the colorized dep images with the square of roi drawn on it
    
    A csv file is written at the end of the execution, containing a column for 
    the time and for the statistical features 

    computed time is the absolute time stamp of recording of each frame

    Parameters
    ----------
    fileCompletePath : bag file from realsense recording
        contains the data of rgb and depth images
    CSVroiDirectory : string
        directory where the csv file containing xmin xmax ymin ymax for every 
        test is saved
    CSVOutputDirectory : string
        directory where the csv will be saved.
    videoOutputDirectory : string, optional
        directory where videos will be saved. 
        The default is '', which sets recordVideo to False
    frequency : int, optional
        freq of saving of the videos.
        The default is 60.   
    nFrames : int, optional
        attended number of frames in the recording. The extractor will do 
        nFrames iterations, or, if the extraction is complete, 
        will stop earlier. Better put a larger number than the actual one. 
        Useful to print the loading bar.
        The default is 20000.

    Returns
    -------
    df : pandas dataframe
        of 19 columns: time  minimum maximum mean stdDev skewness kurtosis perc
    time_exec_array : numpy array
        contains the elapsed time for every execution

    '''
    
    recordVideo = True
    if videoOutputDirectory == '':
        recordVideo = False
    
    # eventually adding .bag in the end in case the user forgot it
    fileCompletePath = utils.checkExtension(fileCompletePath, '.bag')

    # get the name of the file
    fileName = os.path.split(fileCompletePath)[1][:-4]
    
    # add to the filename an univoque code    
    thisExecutionDate = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y%m%d%H%M%S')
    fileNameCode = fileName + '-' + thisExecutionDate
    
    if not os.path.isdir(CSVOutputDirectory):
        os.makedirs(CSVOutputDirectory, exist_ok=True)
        logging.info('directory <<' + CSVOutputDirectory + '>> not existing, creating')
    # complete directory for csv file saving
    csvFileCompletePath = os.path.join(CSVOutputDirectory, fileNameCode + '.csv')
    
    if recordVideo:
        if not  os.path.isdir(videoOutputDirectory):
            os.makedirs(videoOutputDirectory, exist_ok=True)
            logging.info('directory <<' + videoOutputDirectory + '>> not existing, creating')
        # complete directory for video saving
        videoRawCompletePath = os.path.join(videoOutputDirectory, fileNameCode + '-raw.avi')
        videoDEPCompletePath = os.path.join(videoOutputDirectory, fileNameCode + '-mp.avi')

    # load the dataframe containing the coordinates of the roi
    roi = pd.read_csv(os.path.join(CSVroiDirectory, fileName + '.csv'))


    logging.info('working on ' + fileCompletePath)

    if recordVideo:
# =============================================================================
#         WRITE ON THE IMAGE PARAMS
# =============================================================================
        font = cv2.FONT_HERSHEY_SIMPLEX
        origin = (20, 20)
        fontScale = 0.8
        color = (255, 255, 255)
        thickness = 1
  
# =============================================================================
#     START THE STREAM OF THE PIPELINE
# =============================================================================
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, fileCompletePath, repeat_playback = False)
    
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 1)  # jet
    aligned_stream = rs.align(rs.stream.color)  # alignment depth -> color
    
# =============================================================================
#    INITIALIZATION
# =============================================================================
    # so at the first executuion becomes 0
    frameCounter = -1 
    # to save the timing execution of each loop (debug)
    time_exec_array = [0] * nFrames
    # time_exec_array = []
    # to save the starting of the execution
    startTime = time.time()
    # at each iteration add a new row containing landMarkArray and timestamp_s
    data = [0] * nFrames
    # data = []

    try:
        for i in tqdm.tqdm(range(nFrames)):
            try:
                frame = pipeline.wait_for_frames()
            except: 
                break

            if i == 0:
                debugFlag = True
# =============================================================================
#             DEBUGGING
# =============================================================================
            frameCounter = frameCounter + 1
            # time frame on the execution of the loop
            now = time.time()
            # time_exec_array = np.append(time_exec_array, now-startTime)
            time_exec_array[frameCounter] = now-startTime
            
# =============================================================================
#             GET THE REQUIRED DATA FROM THE BAG FILE
# =============================================================================
            # alignement of the frames: the obtained resolution is the one of the rgb image
            frame = aligned_stream.process(frame)
            
            # get the depth and color frames
            depth_frame = frame.get_depth_frame()
            color_frame = frame.get_color_frame()

            # get the timestamp in seconds
            timestamp_s = frame.get_timestamp()/1000
            # print(datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f'))
            
            # from frames to images
            # the image saved in the bag file is in rgb format,
            # the one required from mediapipe as well
            color_image_rgb = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()).astype('float')
            # for video recording
            depth_image_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())

# =============================================================================
#             GET THE REQUIRED DATA FROM THE CSV roi FILE
# =============================================================================
            # consider only the row corresponding to the timestamp_s
            thisMoment = roi[abs(roi['time'] - timestamp_s) < 0.00001]

            if thisMoment.shape[0]==0:
                logging.warning('no frame recognized!!!')
            if thisMoment.shape[0]>1:
                logging.warning('more than one available moment!!!')
                logging.warning(thisMoment)
                logging.warning(timestamp_s)

            xmin = thisMoment['x min'].iloc[0]#.astype(int)
            xmax = thisMoment['x max'].iloc[0]#.astype(int)
            ymin = thisMoment['y min'].iloc[0]#.astype(int)
            ymax = thisMoment['y max'].iloc[0]#.astype(int)

            if not np.isnan(xmin):
                # conversion to int
                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)
    # =============================================================================
    #             OPERATIONS ON THE DEP IMAGE
    # =============================================================================
                # in this case, only copying the image
                img = depth_image.copy()

                # for an image, first it's specified the vertical axis and then the horizontal one
                # +1 to consider also the last:
                    # ymin = ymax -> you want to consider a vertical stripe,
                    # but ymin:ymax returns null
                cropped_image = img[ymin:ymax+1, xmin:xmax+1]

                # flattening the matrix into an array
                array = cropped_image.flatten()
    
                # computation of statistical values
                desc = scipy.stats.describe(array)

                minimum = desc.minmax[0]
                maximum = desc.minmax[1]
                mean = desc.mean
                stdDev = np.std(array)
                skewness = desc.skewness
                kurtosis = desc.kurtosis
                perc01 = scipy.stats.scoreatpercentile(array,  1)
                perc05 = scipy.stats.scoreatpercentile(array,  5)
                perc15 = scipy.stats.scoreatpercentile(array, 15)
                perc25 = scipy.stats.scoreatpercentile(array, 25)
                perc35 = scipy.stats.scoreatpercentile(array, 35)
                perc45 = scipy.stats.scoreatpercentile(array, 45)
                perc55 = scipy.stats.scoreatpercentile(array, 55)
                perc65 = scipy.stats.scoreatpercentile(array, 65)
                perc75 = scipy.stats.scoreatpercentile(array, 75)
                perc85 = scipy.stats.scoreatpercentile(array, 85)
                perc95 = scipy.stats.scoreatpercentile(array, 95)
                perc99 = scipy.stats.scoreatpercentile(array, 99)

                newRow = np.array([minimum, maximum, mean, stdDev,skewness, kurtosis, perc01, perc05, perc15, perc25, perc35, perc45, perc55, perc65, perc75, perc85, perc95, perc99])

            else:
                xmin, xmax, ymin, ymax = 0,0,0,0
                newRow = np.array([np.nan]*18)

# =============================================================================
#             CREATE A NEW ROW IN DATA
# =============================================================================
            # create array to be written
            # insert in landMarkArray the element timestamp_s in the position 0
            tmp = np.insert(newRow, 0, timestamp_s)
            # append the row to the data table
            # data.append(tmp)
            data[frameCounter] = tmp
            
            if recordVideo:
# =============================================================================
#                 IMAGE OPERATION
# =============================================================================
                stringForImage = "frame: {:05d} / ".format(frameCounter) + \
                    datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f')

                # cv2 displays images in bgr, they need to be converted
                color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_BGR2RGB)
                # image_for_mp_bgr = cv2.cvtColor(image_for_mp, cv2.COLOR_BGR2RGB)
                # put text on the image
                color_image_bgr = cv2.putText(color_image_bgr, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)
                depth_image_colorized = cv2.putText(depth_image_colorized, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)
                # draw rectangle of roi
                cv2.rectangle(color_image_bgr, (xmin, ymin), (xmax, ymax), (255,255,255), 2)
                cv2.rectangle(depth_image_colorized, (xmin, ymin), (xmax, ymax), (255,255,255), 2)


                if frameCounter == 0:
                    # initialize the video saver
                    image_height, image_width, _ = color_image_bgr.shape
                    videoRawOut = cv2.VideoWriter(videoRawCompletePath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                    videoMPOut = cv2.VideoWriter(videoDEPCompletePath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                        
                videoRawOut.write(color_image_bgr)
                videoMPOut.write(depth_image_colorized)
            
            
    finally:
        # cut the files preallocated with
        data = data[:frameCounter]
        time_exec_array = time_exec_array[:frameCounter]
        
        # logging.debug('EXECUTION TERMINATED')
# =============================================================================
#         CREATION OF THE PANDAS DATAFRAME AND SAVING IN A CSV FILE
# =============================================================================
        # create the header using the function
        header = ['time', 'minimum', 'maximum', 'mean', 'stdDev','skewness', 'kurtosis', \
                  'perc01', 'perc05', 'perc15', 'perc25', 'perc35', 'perc45', \
                  'perc55', 'perc65', 'perc75', 'perc85', 'perc95', 'perc99']
        # create the pandas dataframe
        df = pd.DataFrame(np.vstack(data),  columns=header)
        # saves the pandas dataframe in a csv file
        df.to_csv(csvFileCompletePath, index = False) 
        
# =============================================================================
#         OTHER OPERATIONS
# =============================================================================
        # stop the pipeline
        pipeline.stop()

        elapsedTime = time.time()-startTime
        freqOfExecution = frameCounter/elapsedTime
        
        logging.debug("{} terminated. {:d} frames in {:.2f} seconds ({:.2f} frames per second)"\
              .format(fileName, frameCounter, elapsedTime, freqOfExecution))
            
        return df, time_exec_array


def wheelCentreAndPlane(fileCompletePath, CSVOutputDirectory, nFrames = 20000,  maxMinDepthHandrimPlaneDetection = [1, 0]):
    '''
    

    Parameters
    ----------
    fileCompletePath : TYPE
        DESCRIPTION.
    CSVOutputDirectory : TYPE
        DESCRIPTION.
    nFrames : TYPE, optional
        DESCRIPTION. The default is 20000.
    maxDepthHandrimLSTSQ : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    None.

    '''

    # eventually adding .bag in the end in case the user forgot it
    fileCompletePath = utils.checkExtension(fileCompletePath, '.bag')

    # get the name of the file
    fileName = os.path.split(fileCompletePath)[1][:-4]
    
    # add to the filename an univoque code    
    thisExecutionDate = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y%m%d%H%M%S')
    fileNameCode = fileName + '-' + thisExecutionDate
    
    if not os.path.isdir(CSVOutputDirectory):
        os.makedirs(CSVOutputDirectory, exist_ok=True)
        logging.info('directory <<' + CSVOutputDirectory + '>> not existing, creating')
    # complete directory for csv file saving
    csvFileCompletePath = os.path.join(CSVOutputDirectory, fileNameCode + '.csv')


# =============================================================================
#     START THE STREAM OF THE PIPELINE
# =============================================================================
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, fileCompletePath, repeat_playback = False)

    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 1)  # jet
    aligned_stream = rs.align(rs.stream.color)  # alignment depth -> color

# =============================================================================
#    INITIALIZATION
# =============================================================================
    # so at the first executuion becomes 0
    frameCounter = -1 
    # to save the timing execution of each loop (debug)
    time_exec_array = [0] * nFrames
    # time_exec_array = []
    # to save the starting of the execution
    startTime = time.time()
    # at each iteration add a new row containing landMarkArray and timestamp_s
    data = [0] * nFrames
    # data = []

    try:
        for i in tqdm.tqdm(range(nFrames)):
            try:
                frame = pipeline.wait_for_frames()
            except: 
                break

            # use for debug at a given moment
            if i == 0:
                debugFlag = True
# =============================================================================
#             DEBUGGING
# =============================================================================
            frameCounter = frameCounter + 1
            # time frame on the execution of the loop
            now = time.time()
            # time_exec_array = np.append(time_exec_array, now-startTime)
            time_exec_array[frameCounter] = now-startTime
            
# =============================================================================
#             GET THE REQUIRED DATA FROM THE BAG FILE
# =============================================================================
            # alignement of the frames: the obtained resolution is the one of the rgb image
            frame = aligned_stream.process(frame)
            
            # get the depth and color frames
            depth_frame = frame.get_depth_frame()
            color_frame = frame.get_color_frame()

            # get the timestamp in seconds
            timestamp_s = frame.get_timestamp()/1000
            # print(datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f'))
            
            # from frames to images
            # the image saved in the bag file is in rgb format,
            # the one required from mediapipe as well
            color_image_rgb = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()).astype('float')

            if i == 0:
                # load intrinsic params of camera
                camera_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
                ppx = camera_intrinsics.ppx
                ppy = camera_intrinsics.ppy
                fx = camera_intrinsics.fx
                fy = camera_intrinsics.fy

            # remove invalid values
            depth_image[depth_image <= 0] = np.nan

            output = findWheelCentreAndHandrim(color_image_rgb, depth_image, ppx, ppy, fx, fy,
                                              fitPlaneMethod = 'RANSAC',
                                              minDist = 1, param1 = 50, param2 =60,
                                              minRadius = 180, maxRadius = 200,
                                              div = 4, edge_low_thresh = 50, edge_high_thresh = 150,
                                              rho = 1, theta = np.pi/180, threshold = 30,
                                              min_line_length_coeff = 0 , max_line_gap = 20,
                                              w = 5, tolerance = 5000,
                                              maxMinDepthHandrimPlaneDetection =  maxMinDepthHandrimPlaneDetection,
                                              showPlot = False, mainTitle = '')

            if output is None:
                newRow = np.empty(15).fill(np.nan) # to be done manually when modifying the function
            else:
                wc_img, hrc_img, centre_metric, handrimPlane, dataPlane = output

            # newRow = [wheel_centre[0], wheel_centre[1], wheel_centre[2], handrim[0], handrim[1], handrim[2], handrim[3], handrimPlane.a, handrimPlane.b, handrimPlane.c, handrimPlane.d, handrimPlane.normal[0], handrimPlane.normal[1], handrimPlane.normal[2]]

            # concatenation of lists
            newRow = wc_img + hrc_img + centre_metric + [handrimPlane.a, handrimPlane.b, handrimPlane.c, handrimPlane.d] + handrimPlane.normal.tolist()

# =============================================================================
#             CREATE A NEW ROW IN DATA
# =============================================================================
            # create array to be written
            # insert in landMarkArray the element timestamp_s in the position 0
            tmp = np.insert(newRow, 0, timestamp_s)
            # append the row to the data table
            # data.append(tmp)
            data[frameCounter] = tmp
   
    finally:
        # cut the files preallocated with
        data = data[:frameCounter]
        time_exec_array = time_exec_array[:frameCounter]

        # logging.debug('EXECUTION TERMINATED')
# =============================================================================
#         CREATION OF THE PANDAS DATAFRAME AND SAVING IN A CSV FILE
# =============================================================================
        # create the header using the function
        header = ['time', 'wxc', 'wyc', 'hrxc','hryc', 'hrr', 'xc', 'yc', 'zc',\
                  'a', 'b', 'c', 'd', 'norm x', 'norm y', 'norm z']
        # create the pandas dataframe
        df = pd.DataFrame(np.vstack(data),  columns=header)
        # saves the pandas dataframe in a csv file
        df.to_csv(csvFileCompletePath, index = False) 
        
# =============================================================================
#         OTHER OPERATIONS
# =============================================================================
        # stop the pipeline
        pipeline.stop()

        elapsedTime = time.time()-startTime
        freqOfExecution = frameCounter/elapsedTime
        
        logging.debug("{} terminated. {:d} frames in {:.2f} seconds ({:.2f} frames per second)"\
              .format(fileName, frameCounter, elapsedTime, freqOfExecution))
            
        return df, time_exec_array


def wheelCentreAndPlaneOLD(fileCompletePath, CSVOutputDirectory, nFrames = 20000, maxDepthHandrimLSTSQ = 1000):
    '''
    

    Parameters
    ----------
    fileCompletePath : TYPE
        DESCRIPTION.
    CSVOutputDirectory : TYPE
        DESCRIPTION.
    nFrames : TYPE, optional
        DESCRIPTION. The default is 20000.
    maxDepthHandrimLSTSQ : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    None.

    '''

    # eventually adding .bag in the end in case the user forgot it
    fileCompletePath = utils.checkExtension(fileCompletePath, '.bag')

    # get the name of the file
    fileName = os.path.split(fileCompletePath)[1][:-4]
    
    # add to the filename an univoque code    
    thisExecutionDate = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y%m%d%H%M%S')
    fileNameCode = fileName + '-' + thisExecutionDate
    
    if not os.path.isdir(CSVOutputDirectory):
        os.makedirs(CSVOutputDirectory, exist_ok=True)
        logging.info('directory <<' + CSVOutputDirectory + '>> not existing, creating')
    # complete directory for csv file saving
    csvFileCompletePath = os.path.join(CSVOutputDirectory, fileNameCode + '.csv')


# =============================================================================
#     START THE STREAM OF THE PIPELINE
# =============================================================================
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, fileCompletePath, repeat_playback = False)

    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 1)  # jet
    aligned_stream = rs.align(rs.stream.color)  # alignment depth -> color

# =============================================================================
#    INITIALIZATION
# =============================================================================
    # so at the first executuion becomes 0
    frameCounter = -1 
    # to save the timing execution of each loop (debug)
    time_exec_array = [0] * nFrames
    # time_exec_array = []
    # to save the starting of the execution
    startTime = time.time()
    # at each iteration add a new row containing landMarkArray and timestamp_s
    data = [0] * nFrames
    # data = []

    try:
        for i in tqdm.tqdm(range(nFrames)):
            try:
                frame = pipeline.wait_for_frames()
            except: 
                break

            # use for debug at a given moment
            if i == 68:
                debugFlag = True
# =============================================================================
#             DEBUGGING
# =============================================================================
            frameCounter = frameCounter + 1
            # time frame on the execution of the loop
            now = time.time()
            # time_exec_array = np.append(time_exec_array, now-startTime)
            time_exec_array[frameCounter] = now-startTime
            
# =============================================================================
#             GET THE REQUIRED DATA FROM THE BAG FILE
# =============================================================================
            # alignement of the frames: the obtained resolution is the one of the rgb image
            frame = aligned_stream.process(frame)
            
            # get the depth and color frames
            depth_frame = frame.get_depth_frame()
            color_frame = frame.get_color_frame()

            # get the timestamp in seconds
            timestamp_s = frame.get_timestamp()/1000
            # print(datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f'))
            
            # from frames to images
            # the image saved in the bag file is in rgb format,
            # the one required from mediapipe as well
            color_image_rgb = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()).astype('float')
            # remove invalid values
            depth_image[depth_image <= 0] = np.nan

            output = findWheelCentreAndHandrim(color_image_rgb, depth_image, maxDepthHandrimLSTSQ = maxDepthHandrimLSTSQ, showPlot = False)

            if output is None:
                newRow = np.empty(14).fill(np.nan) # to be done manually when modifying the function
            else:
                wheel_centre, handrim, handrimPlane, dataPlane = output

            # newRow = [wheel_centre[0], wheel_centre[1], wheel_centre[2], handrim[0], handrim[1], handrim[2], handrim[3], handrimPlane.a, handrimPlane.b, handrimPlane.c, handrimPlane.d, handrimPlane.normal[0], handrimPlane.normal[1], handrimPlane.normal[2]]

            # concatenation of lists
            newRow = wheel_centre + handrim + [handrimPlane.a, handrimPlane.b, handrimPlane.c, handrimPlane.d] + handrimPlane.normal

# =============================================================================
#             CREATE A NEW ROW IN DATA
# =============================================================================
            # create array to be written
            # insert in landMarkArray the element timestamp_s in the position 0
            tmp = np.insert(newRow, 0, timestamp_s)
            # append the row to the data table
            # data.append(tmp)
            data[frameCounter] = tmp
   
    finally:
        # cut the files preallocated with
        data = data[:frameCounter]
        time_exec_array = time_exec_array[:frameCounter]

        # logging.debug('EXECUTION TERMINATED')
# =============================================================================
#         CREATION OF THE PANDAS DATAFRAME AND SAVING IN A CSV FILE
# =============================================================================
        # create the header using the function
        header = ['time', 'wxc', 'wyc', 'wzc', 'hrxc','hryc', 'hrzc', 'hrr',\
                  'a', 'b', 'c', 'd', 'norm x', 'norm y', 'norm z']
        # create the pandas dataframe
        df = pd.DataFrame(np.vstack(data),  columns=header)
        # saves the pandas dataframe in a csv file
        df.to_csv(csvFileCompletePath, index = False) 
        
# =============================================================================
#         OTHER OPERATIONS
# =============================================================================
        # stop the pipeline
        pipeline.stop()

        elapsedTime = time.time()-startTime
        freqOfExecution = frameCounter/elapsedTime
        
        logging.debug("{} terminated. {:d} frames in {:.2f} seconds ({:.2f} frames per second)"\
              .format(fileName, frameCounter, elapsedTime, freqOfExecution))
            
        return df, time_exec_array


def handPointCloudFeaturesUpdate(fileCompletePath, CSVroiDirectory, CSVOutputDirectory,\
   rHandrim, videoOutputDirectory = '', frequency = 60, nFrames = 20000, \
   nFramesUpdateRefFrame = 1, maxDepthHandrimLSTSQ = 1000, maxDepthHand = 1000):

    '''
    Given a bag file and a directory in which the region of interest for every 
    frame of the recording is specified by means of xmin xmax ymin ymax, 
    playbacks the file frame by frame and computes statistical features on the 
    depth of that region:
         
    EVERYTHING IS EXPRESSED IN THE HANDRIM PLANE REFERENCE FRAME:
        ^ Y
        |
        |
        |
        |
        ----------->X
    with Z = 0 correspoding to the camera

    The ref frame is computed for every frame on the given couple of rgb and depth image
    
    A loading bar is provided to the user during the execution. The number of
    frames is estimated consequently the actual number of frames elaborated 
    will be lower.
    
    During the execution, if videOutputDirectory is specified, two videos are
    recorded: 
        - the raw rgb images coming from the bag file with the square of roi drawn on it
        - the colorized dep images with the square of roi drawn on it
    To both videos are also added a red cross to show the centre of the wheel, 
    a yellow square to show the centre of the handrim and a green circle to show
    the handrim
    
    Two csv files are written at the end of the execution:
        - one contains a column for the time and the others for the hand features 
        - one contains a column for the time and the others for the parameters of 
        the wheel and of the plane
    
    computed time is the absolute time stamp of recording of each frame
    
    Parameters
    ----------
    fileCompletePath : bag file from realsense recording
        contains the data of rgb and depth images
    CSVroiDirectory : string
        directory where the csv file containing xmin xmax ymin ymax for every 
        test is saved
    CSVOutputDirectory : string
        directory where the csv will be saved.
    videoOutputDirectory : string, optional
        directory where videos will be saved. 
        The default is '', which sets recordVideo to False
    frequency : int, optional
        freq of saving of the videos.
        The default is 60.   
    nFrames : int, optional
       attended number of frames in the recording. The extractor will do 
       nFrames iterations, or, if the extraction is complete, 
       will stop earlier. Better put a larger number than the actual one. 
       Useful to print the loading bar.
       The default is 20000.
    nFramesUpdateRefFrame : int, optional
        how often should the reference frame be updated? 
        The default is 1, which means for every frame
    maxDepthHandrimLSTSQ : double, optional
        When computing the handrim plane using LSTSQ, all the values of depth, 
        expressed in the camera ref frame, bigger than maxDepthHandrimLSTSQ 
        are ignored. See findWheelCentreAndHandrim() for more.
        The default is 1000.
    maxDepthHand : double, optional
        When estimating the pointcloud of the hand, values of depth, expressed in 
        the camera ref frame, are excluded. The default is 1000.
    
    Returns
    -------
    df : pandas dataframe
        of 1+4*N columns: time + statistical variable computed by computeStatFeatures() 
        of the four variables computed by computeHandDepFeatures() in the handrim plane
    dfRefFrame : pandas dataframe
        of 15 columns: time, wxc, wyc, wzc, hrxc, hryc, hrzc, hrr,
                  a, b, c, d, norm x, norm y, norm z
        where:
            - w_c are the coordinates of the wheel centre
            - h_c are the coordinates of the handrim, hrr is the radius
            - a, b, c, d are the parameters of the plane
            - norm x, y, z are the components of the normal vector to the plane in the 
            camera reference frame.
    time_exec_array : numpy array
        contains the elapsed time for every execution
    '''
    
    recordVideo = True
    if videoOutputDirectory == '':
        recordVideo = False
    
    # eventually adding .bag in the end in case the user forgot it
    fileCompletePath = utils.checkExtension(fileCompletePath, '.bag')

    # get the name of the file
    fileName = os.path.split(fileCompletePath)[1][:-4]
    
    # add to the filename an univoque code    
    thisExecutionDate = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y%m%d%H%M%S')
    fileNameCode = fileName + '-' + thisExecutionDate
    
    if not os.path.isdir(CSVOutputDirectory):
        os.makedirs(CSVOutputDirectory, exist_ok=True)
        logging.info('directory <<' + CSVOutputDirectory + '>> not existing, creating')
    # complete directory for csv file saving
    csvFileCompletePath = os.path.join(CSVOutputDirectory, fileNameCode + '.csv')
    
    if recordVideo:
        if not  os.path.isdir(videoOutputDirectory):
            os.makedirs(videoOutputDirectory, exist_ok=True)
            logging.info('directory <<' + videoOutputDirectory + '>> not existing, creating')
        # complete directory for video saving
        videoRawCompletePath = os.path.join(videoOutputDirectory, fileNameCode + '-col.avi')
        videoDEPCompletePath = os.path.join(videoOutputDirectory, fileNameCode + '-dep.avi')

    # load the dataframe containing the coordinates of the roi
    roi = pd.read_csv(os.path.join(CSVroiDirectory, fileName + '.csv'))


    logging.info('working on ' + fileCompletePath)

    if recordVideo:
# =============================================================================
#         WRITE ON THE IMAGE PARAMS
# =============================================================================
        font = cv2.FONT_HERSHEY_SIMPLEX
        origin = (20, 20)
        fontScale = 0.8
        color = (255, 255, 255)
        thickness = 1
  
# =============================================================================
#     START THE STREAM OF THE PIPELINE
# =============================================================================
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, fileCompletePath, repeat_playback = False)
    
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 1)  # jet
    aligned_stream = rs.align(rs.stream.color)  # alignment depth -> color
    
# =============================================================================
#    INITIALIZATION
# =============================================================================
    # so at the first executuion becomes 0
    frameCounter = -1 
    # to save the timing execution of each loop (debug)
    time_exec_array = [0] * nFrames
    # time_exec_array = []
    # to save the starting of the execution
    startTime = time.time()
    # at each iteration add a new row containing
    data = [0] * nFrames
    dataRefFrame = [0] * nFrames
    # data = []
    # to update the detection of the centre of the wheel and the handrim plane
    counterUpdateRefFrame = -1

    try:
        for i in tqdm.tqdm(range(nFrames)):
            try:
                frame = pipeline.wait_for_frames()
            except: 
                break
            counterUpdateRefFrame += 1
            if i == 1132:
                debugFlag = True
# =============================================================================
#             DEBUGGING
# =============================================================================
            frameCounter = frameCounter + 1
            # time frame on the execution of the loop
            now = time.time()
            # time_exec_array = np.append(time_exec_array, now-startTime)
            time_exec_array[frameCounter] = now-startTime
            
# =============================================================================
#             GET THE REQUIRED DATA FROM THE BAG FILE
# =============================================================================
            # alignement of the frames: the obtained resolution is the one of the rgb image
            frame = aligned_stream.process(frame)
            
            # get the depth and color frames
            depth_frame = frame.get_depth_frame()
            color_frame = frame.get_color_frame()

            # get the timestamp in seconds
            timestamp_s = frame.get_timestamp()/1000
            # print(datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f'))
            
            # from frames to images
            # the image saved in the bag file is in rgb format,
            # the one required from mediapipe as well
            color_image_rgb = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()).astype('float')
            # remove invalid values
            depth_image[depth_image <= 0] = np.nan
            # for video recording
            depth_image_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            if frameCounter == 0: # for the first frame, in anycase, find wheel centre and handrim plane
                # find wheel centre and handrimPlane
                output = findWheelCentreAndHandrim(color_image_rgb, depth_image, maxDepthHandrimLSTSQ = maxDepthHandrimLSTSQ, showPlot = False)
                if output is None:
                    newRowRefFrame = np.empty(14).fill(np.nan)
                else:
                    wheel_centre, handrim, handrimPlane, dataPlane = output
                    # update rotation matrix
                    rot, rmsd, sens = geom.rotMatrixToFitPlane(handrimPlane, wheel_centre)

                newRowRefFrame = [wheel_centre[0], wheel_centre[1], wheel_centre[2], handrim[0], handrim[1], handrim[2], handrim[3], handrimPlane.a, handrimPlane.b, handrimPlane.c, handrimPlane.d, handrimPlane.normal[0], handrimPlane.normal[1], handrimPlane.normal[2]]

# =============================================================================
#             GET THE REQUIRED DATA FROM THE CSV roi FILE
# =============================================================================
            # consider only the row corresponding to the timestamp_s
            thisMoment = roi[abs(roi['time'] - timestamp_s) < 0.00001]

            if thisMoment.shape[0]==0:
                logging.warning('no frame recognized!!!')
            if thisMoment.shape[0]>1:
                logging.warning('more than one available moment!!!')
                logging.warning(thisMoment)
                logging.warning(timestamp_s)

            xmin = thisMoment['x min'].iloc[0]#.astype(int)
            xmax = thisMoment['x max'].iloc[0]#.astype(int)
            ymin = thisMoment['y min'].iloc[0]#.astype(int)
            ymax = thisMoment['y max'].iloc[0]#.astype(int)

            if not np.isnan(xmin): # if hand is defined
                # conversion to int
                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)

                if counterUpdateRefFrame == 0 or counterUpdateRefFrame >= nFramesUpdateRefFrame:
                    # reset the counter
                    counterUpdateRefFrame = 0
                    # find wheel centre and handrimPlane
                    output = findWheelCentreAndHandrim(color_image_rgb, depth_image, maxDepthHandrimLSTSQ = maxDepthHandrimLSTSQ, showPlot = False)
                    if output is None:
                        newRowRefFrame = np.empty(14).fill(np.nan)
                    else:
                        wheel_centre, handrim, handrimPlane, dataPlane = output
                        # update rotation matrix
                        rot, rmsd, sens = geom.rotMatrixToFitPlane(handrimPlane, wheel_centre)

                    newRowRefFrame = [wheel_centre[0], wheel_centre[1], wheel_centre[2], handrim[0], handrim[1], handrim[2], handrim[3], handrimPlane.a, handrimPlane.b, handrimPlane.c, handrimPlane.d, handrimPlane.normal[0], handrimPlane.normal[1], handrimPlane.normal[2]]

    # =============================================================================
    #             OPERATIONS ON THE DEP IMAGE
    # =============================================================================
                # extract the region of interest
                valid_image = depth_image.copy()
                valid_image[valid_image <= 0] = np.nan
                valid_image[0:ymin, :] = np.nan
                valid_image[ymax+1:, :] = np.nan
                valid_image[:, 0:xmin] = np.nan
                valid_image[:, xmax+1:] = np.nan
                valid_image[valid_image >= maxDepthHand] = np.nan

                dataHand = utils.depImgToThreeCol(valid_image)
                dataHand = utils.removeNanXYZ(dataHand)

                # since the plane is horizontal, same coefficient for x and y
                pixel2mmCoeff_x = rHandrim / handrim[3]
                pixel2mmCoeff_y = rHandrim / handrim[3]
                # for z the data are already expressed in millimetres

                # bring the pointcloud in the correct reference frame
                dataHandDef = geom.changeRefFrameTRS(dataHand, wheel_centre, rot, pixel2mmCoeff_x, pixel2mmCoeff_y)

                results, res_str_total = computeHandDepFeatures(dataHandDef, rHandrim, showPlot = False)

                newRow = np.array(results)

            else:
                xmin, xmax, ymin, ymax = 0,0,0,0
                newRow = np.array([np.nan]*36)

# =============================================================================
#             CREATE A NEW ROW IN DATA
# =============================================================================
            tmp = np.insert(newRow, 0, timestamp_s)
            data[frameCounter] = tmp

            tmpRefFrame = np.insert(newRowRefFrame, 0, timestamp_s)
            dataRefFrame[frameCounter] = tmpRefFrame

            if recordVideo:
# =============================================================================
#                 IMAGE OPERATION
# =============================================================================
                stringForImage = "frame: {:05d} / ".format(frameCounter) + \
                    datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f')

                # cv2 displays images in bgr, they need to be converted
                color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_BGR2RGB)
                # image_for_mp_bgr = cv2.cvtColor(image_for_mp, cv2.COLOR_BGR2RGB)
                # put text on the image
                color_image_bgr = cv2.putText(color_image_bgr, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)
                depth_image_colorized = cv2.putText(depth_image_colorized, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)

                color_image_bgr = plots.roiWcHrOnImage(color_image_bgr, wheel_centre, handrim, xmin, xmax, ymin, ymax)
                depth_image_colorized = plots.roiWcHrOnImage(depth_image_colorized, wheel_centre, handrim, xmin, xmax, ymin, ymax)
                if frameCounter == 0:
                    # initialize the video saver
                    image_height, image_width, _ = color_image_bgr.shape
                    videoRawOut = cv2.VideoWriter(videoRawCompletePath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                    videoMPOut = cv2.VideoWriter(videoDEPCompletePath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                        
                videoRawOut.write(color_image_bgr)
                videoMPOut.write(depth_image_colorized)
            
            
    finally:
        # cut the files preallocated with
        data = data[:frameCounter]
        dataRefFrame = dataRefFrame[:frameCounter]
        time_exec_array = time_exec_array[:frameCounter]
        
        # logging.debug('EXECUTION TERMINATED')
# =============================================================================
#         CREATION OF THE PANDAS DATAFRAME AND SAVING IN A CSV FILE
# =============================================================================
        # create the header using the function
        headerRefFrame = ['time', 'wxc', 'wyc', 'wzc', 'hrxc','hryc', 'hrzc', 'hrr',\
                  'a', 'b', 'c', 'd', 'norm x', 'norm y', 'norm z']
        header = ['time'] + res_str_total

        # create the pandas dataframe
        df = pd.DataFrame(np.vstack(data),  columns=header)
        # saves the pandas dataframe in a csv file
        df.to_csv(csvFileCompletePath, index = False)

        # create the pandas dataframe
        dfRefFrame = pd.DataFrame(np.vstack(dataRefFrame),  columns=headerRefFrame)
        # saves the pandas dataframe in a csv file
        dfRefFrame.to_csv(csvFileCompletePath.replace('.csv', '-refFrame.csv'), index = False)
        
# =============================================================================
#         OTHER OPERATIONS
# =============================================================================
        # stop the pipeline
        pipeline.stop()

        elapsedTime = time.time()-startTime
        freqOfExecution = frameCounter/elapsedTime
        
        logging.debug("{} terminated. {:d} frames in {:.2f} seconds ({:.2f} frames per second)"\
              .format(fileName, frameCounter, elapsedTime, freqOfExecution))
            
        return df, dfRefFrame, time_exec_array


def handPointCloudFeaturesCSVplnDir(fileCompletePath, CSVroiDirectory, CSVOutputDirectory,\
   CSVplnDirectory, rHandrim, videoOutputDirectory = '', frequency = 60, \
   nFrames = 20000, maxDepthHand = 1000):

    '''
    Given a bag file and a directory in which the region of interest for every 
    frame of the recording is specified by means of xmin xmax ymin ymax, 
    playbacks the file frame by frame and computes statistical features on the 
    depth of that region:
         
    EVERYTHING IS EXPRESSED IN THE HANDRIM PLANE REFERENCE FRAME:
        ^ Y
        |
        |
        |
        |
        ----------->X
    with Z = 0 correspoding to the camera

    The ref frame is computed as the mean value of the ones contained in CSVplnDirectory
    
    A loading bar is provided to the user during the execution. The number of
    frames is estimated consequently the actual number of frames elaborated 
    will be lower.
    
    During the execution, if videOutputDirectory is specified, two videos are
    recorded: 
        - the raw rgb images coming from the bag file with the square of roi drawn on it
        - the colorized dep images with the square of roi drawn on it
    To both videos are also added a red cross to show the centre of the wheel, 
    a yellow square to show the centre of the handrim and a green circle to show
    the handrim
    
    Two csv files are written at the end of the execution:
        - one contains a column for the time and the others for the hand features 
        - one contains a column for the time and the others for the parameters of 
        the wheel and of the plane
    
    computed time is the absolute time stamp of recording of each frame
    
    Parameters
    ----------
    fileCompletePath : bag file from realsense recording
        contains the data of rgb and depth images
    CSVroiDirectory : string
        directory where the csv file containing xmin xmax ymin ymax for every 
        test is saved
    CSVOutputDirectory : string
        directory where the csv will be saved.
    rHandrim : double
        dimension of the handrim in real world, used to scale the image
    CSVplnDirectory : string
        directory where the csv containing the plane data is saved.
        To compute the values of the handrim plane and of the wheel, a mean over
        the whole file is done.
    videoOutputDirectory : string, optional
        directory where videos will be saved. 
        The default is '', which sets recordVideo to False
    frequency : int, optional
        freq of saving of the videos.
        The default is 60.   
    nFrames : int, optional
       attended number of frames in the recording. The extractor will do 
       nFrames iterations, or, if the extraction is complete, 
       will stop earlier. Better put a larger number than the actual one. 
       Useful to print the loading bar.
       The default is 20000.
    maxDepthHand : double, optional
        When estimating the pointcloud of the hand, values of depth, expressed in 
        the camera ref frame, are excluded. The default is 1000.
    
    Returns
    -------
    df : pandas dataframe
        of 1+4*N columns: time + statistical variable computed by computeStatFeatures() 
        of the four variables computed by computeHandDepFeatures() in the handrim plane
    dfRefFrame : pandas dataframe
        of 15 columns: time, wxc, wyc, wzc, hrxc, hryc, hrzc, hrr,
                  a, b, c, d, norm x, norm y, norm z
        where:
            - w_c are the coordinates of the wheel centre
            - h_c are the coordinates of the handrim, hrr is the radius
            - a, b, c, d are the parameters of the plane
            - norm x, y, z are the components of the normal vector to the plane in the 
            camera reference frame.
    time_exec_array : numpy array
        contains the elapsed time for every execution
    '''
    
    recordVideo = True
    if videoOutputDirectory == '':
        recordVideo = False
    
    # eventually adding .bag in the end in case the user forgot it
    fileCompletePath = utils.checkExtension(fileCompletePath, '.bag')

    # get the name of the file
    fileName = os.path.split(fileCompletePath)[1][:-4]
    
    # add to the filename an univoque code    
    thisExecutionDate = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y%m%d%H%M%S')
    fileNameCode = fileName + '-' + thisExecutionDate
    
    if not os.path.isdir(CSVOutputDirectory):
        os.makedirs(CSVOutputDirectory, exist_ok=True)
        logging.info('directory <<' + CSVOutputDirectory + '>> not existing, creating')
    # complete directory for csv file saving
    csvFileCompletePath = os.path.join(CSVOutputDirectory, fileNameCode + '.csv')
    
    if recordVideo:
        if not  os.path.isdir(videoOutputDirectory):
            os.makedirs(videoOutputDirectory, exist_ok=True)
            logging.info('directory <<' + videoOutputDirectory + '>> not existing, creating')
        # complete directory for video saving
        videoRawCompletePath = os.path.join(videoOutputDirectory, fileNameCode + '-col.avi')
        videoDEPCompletePath = os.path.join(videoOutputDirectory, fileNameCode + '-dep.avi')

    # load the dataframe containing the coordinates of the roi
    roi = pd.read_csv(os.path.join(CSVroiDirectory, fileName + '.csv'))
    planeCompletePath = utils.findFileInDirectory(CSVplnDirectory, fileName)
    pln = pd.read_csv(planeCompletePath[0])


    logging.info('working on ' + fileCompletePath)

    if recordVideo:
# =============================================================================
#         WRITE ON THE IMAGE PARAMS
# =============================================================================
        font = cv2.FONT_HERSHEY_SIMPLEX
        origin = (20, 20)
        fontScale = 0.8
        color = (255, 255, 255)
        thickness = 1
  
# =============================================================================
#     START THE STREAM OF THE PIPELINE
# =============================================================================
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, fileCompletePath, repeat_playback = False)
    
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 1)  # jet
    aligned_stream = rs.align(rs.stream.color)  # alignment depth -> color
    
# =============================================================================
#    INITIALIZATION
# =============================================================================
    # so at the first executuion becomes 0
    frameCounter = -1 
    # to save the timing execution of each loop (debug)
    time_exec_array = [0] * nFrames
    # time_exec_array = []
    # to save the starting of the execution
    startTime = time.time()
    # at each iteration add a new row containing
    data = [0] * nFrames
    dataRefFrame = [0] * nFrames
    # data = []
    # to update the detection of the centre of the wheel and the handrim plane
    counterUpdateRefFrame = -1

    # load the plane data
    pln = pln.mean(axis = 0, skipna = True)
    handrimPlane = geom.Plane3d(a = pln['a'], b = pln['b'], c = pln['c'], d = pln['d'])
    wheel_centre = [pln['wxc'], pln['wyc'], pln['wzc']]
    handrim = [pln['hrxc'],	pln['hryc'], pln['hrzc'], pln['hrr']]

    rot, rmsd, sens = geom.rotMatrixToFitPlane(handrimPlane, wheel_centre)

    newRowRefFrame = [wheel_centre[0], wheel_centre[1], wheel_centre[2], handrim[0], handrim[1], handrim[2], handrim[3], handrimPlane.a, handrimPlane.b, handrimPlane.c, handrimPlane.d, handrimPlane.normal[0], handrimPlane.normal[1], handrimPlane.normal[2]]

    try:
        for i in tqdm.tqdm(range(nFrames)):
            try:
                frame = pipeline.wait_for_frames()
            except:
                break
            counterUpdateRefFrame += 1
            if i == 400:# 463:
                debugFlag = True
# =============================================================================
#             DEBUGGING
# =============================================================================
            frameCounter = frameCounter + 1
            # time frame on the execution of the loop
            now = time.time()
            # time_exec_array = np.append(time_exec_array, now-startTime)
            time_exec_array[frameCounter] = now-startTime
            
# =============================================================================
#             GET THE REQUIRED DATA FROM THE BAG FILE
# =============================================================================
            # alignement of the frames: the obtained resolution is the one of the rgb image
            frame = aligned_stream.process(frame)
            
            # get the depth and color frames
            depth_frame = frame.get_depth_frame()
            color_frame = frame.get_color_frame()

            # get the timestamp in seconds
            timestamp_s = frame.get_timestamp()/1000
            # print(datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f'))
            
            # from frames to images
            # the image saved in the bag file is in rgb format,
            # the one required from mediapipe as well
            color_image_rgb = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()).astype('float')
            # remove invalid values
            depth_image[depth_image <= 0] = np.nan
            # for video recording
            depth_image_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())

# =============================================================================
#             GET THE REQUIRED DATA FROM THE CSV roi FILE
# =============================================================================
            # consider only the row corresponding to the timestamp_s
            thisMoment = roi[abs(roi['time'] - timestamp_s) < 0.00001]

            if thisMoment.shape[0]==0:
                logging.warning('no frame recognized!!!')
            if thisMoment.shape[0]>1:
                logging.warning('more than one available moment!!!')
                logging.warning(thisMoment)
                logging.warning(timestamp_s)

            xmin = thisMoment['x min'].iloc[0]#.astype(int)
            xmax = thisMoment['x max'].iloc[0]#.astype(int)
            ymin = thisMoment['y min'].iloc[0]#.astype(int)
            ymax = thisMoment['y max'].iloc[0]#.astype(int)

            if not np.isnan(xmin): # if hand is defined
                # conversion to int
                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)

    # =============================================================================
    #             OPERATIONS ON THE DEP IMAGE
    # =============================================================================
                # extract the region of interest
                valid_image = depth_image.copy()
                valid_image[valid_image <= 0] = np.nan
                valid_image[0:ymin, :] = np.nan
                valid_image[ymax+1:, :] = np.nan
                valid_image[:, 0:xmin] = np.nan
                valid_image[:, xmax+1:] = np.nan
                valid_image[valid_image >= maxDepthHand] = np.nan

                dataHand = utils.depImgToThreeCol(valid_image)
                dataHand = utils.removeNanXYZ(dataHand)

                if not (np.isnan(dataHand).all()): # if there is at least one valid value
    
                    # since the plane is horizontal, same coefficient for x and y
                    pixel2mmCoeff_x = rHandrim / handrim[3]
                    pixel2mmCoeff_y = rHandrim / handrim[3]
                    # for z the data are already expressed in millimetres
    
                    # bring the pointcloud in the correct reference frame
                    dataHandDef = geom.changeRefFrameTRS(dataHand, wheel_centre, rot, pixel2mmCoeff_x, pixel2mmCoeff_y)
    
                    results, res_str_total = computeHandDepFeatures(dataHandDef, rHandrim, showPlot = False)
    
                    newRow = np.array(results)
                else:
                    newRow = np.array([np.nan]*36)

            else:
                xmin, xmax, ymin, ymax = 0,0,0,0
                newRow = np.array([np.nan]*36)

# =============================================================================
#             CREATE A NEW ROW IN DATA
# =============================================================================
            tmp = np.insert(newRow, 0, timestamp_s)
            data[frameCounter] = tmp

            tmpRefFrame = np.insert(newRowRefFrame, 0, timestamp_s)
            dataRefFrame[frameCounter] = tmpRefFrame

            if recordVideo:
# =============================================================================
#                 IMAGE OPERATION
# =============================================================================
                stringForImage = "frame: {:05d} / ".format(frameCounter) + \
                    datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f')

                # cv2 displays images in bgr, they need to be converted
                color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_BGR2RGB)
                # image_for_mp_bgr = cv2.cvtColor(image_for_mp, cv2.COLOR_BGR2RGB)
                # put text on the image
                color_image_bgr = cv2.putText(color_image_bgr, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)
                depth_image_colorized = cv2.putText(depth_image_colorized, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)

                color_image_bgr = plots.roiWcHrOnImage(color_image_bgr, wheel_centre, handrim, xmin, xmax, ymin, ymax)
                depth_image_colorized = plots.roiWcHrOnImage(depth_image_colorized, wheel_centre, handrim, xmin, xmax, ymin, ymax)
                if frameCounter == 0:
                    # initialize the video saver
                    image_height, image_width, _ = color_image_bgr.shape
                    videoRawOut = cv2.VideoWriter(videoRawCompletePath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                    videoMPOut = cv2.VideoWriter(videoDEPCompletePath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                        
                videoRawOut.write(color_image_bgr)
                videoMPOut.write(depth_image_colorized)

    finally:
        # cut the files preallocated with
        data = data[:frameCounter]
        dataRefFrame = dataRefFrame[:frameCounter]
        time_exec_array = time_exec_array[:frameCounter]
        
        # logging.debug('EXECUTION TERMINATED')
# =============================================================================
#         CREATION OF THE PANDAS DATAFRAME AND SAVING IN A CSV FILE
# =============================================================================
        # create the header using the function
        headerRefFrame = ['time', 'wxc', 'wyc', 'wzc', 'hrxc','hryc', 'hrzc', 'hrr',\
                  'a', 'b', 'c', 'd', 'norm x', 'norm y', 'norm z']
        header = ['time'] + res_str_total

        # create the pandas dataframe
        df = pd.DataFrame(np.vstack(data),  columns=header)
        # saves the pandas dataframe in a csv file
        df.to_csv(csvFileCompletePath, index = False)

        # create the pandas dataframe
        dfRefFrame = pd.DataFrame(np.vstack(dataRefFrame),  columns=headerRefFrame)
        # saves the pandas dataframe in a csv file
        dfRefFrame.to_csv(csvFileCompletePath.replace('.csv', '-refFrame.csv'), index = False)
        
# =============================================================================
#         OTHER OPERATIONS
# =============================================================================
        # stop the pipeline
        pipeline.stop()

        elapsedTime = time.time()-startTime
        freqOfExecution = frameCounter/elapsedTime
        
        logging.debug("{} terminated. {:d} frames in {:.2f} seconds ({:.2f} frames per second)"\
              .format(fileName, frameCounter, elapsedTime, freqOfExecution))
            
        return df, dfRefFrame, time_exec_array


def handPointCloudFeaturesPlnData(fileCompletePath, CSVroiDirectory, CSVOutputDirectory,\
   rHandrim, handrimPlane, wheel_centre, handrim, videoOutputDirectory = '',\
   frequency = 60, nFrames = 20000, maxDepthHand = 1000):
    '''
    Given a bag file and a directory in which the region of interest for every 
    frame of the recording is specified by means of xmin xmax ymin ymax, 
    playbacks the file frame by frame and computes statistical features on the 
    depth of that region:
         
    EVERYTHING IS EXPRESSED IN THE HANDRIM PLANE REFERENCE FRAME:
        ^ Y
        |
        |
        |
        |
        ----------->X
    with Z = 0 correspoding to the camera

    The ref frame is the one passed in input.
    
    A loading bar is provided to the user during the execution. The number of
    frames is estimated consequently the actual number of frames elaborated 
    will be lower.
    
    During the execution, if videOutputDirectory is specified, two videos are
    recorded: 
        - the raw rgb images coming from the bag file with the square of roi drawn on it
        - the colorized dep images with the square of roi drawn on it
    To both videos are also added a red cross to show the centre of the wheel, 
    a yellow square to show the centre of the handrim and a green circle to show
    the handrim
    
    Two csv files are written at the end of the execution:
        - one contains a column for the time and the others for the hand features 
        - one contains a column for the time and the others for the parameters of 
        the wheel and of the plane
    
    computed time is the absolute time stamp of recording of each frame
    
    Parameters
    ----------
    fileCompletePath : bag file from realsense recording
        contains the data of rgb and depth images
    CSVroiDirectory : string
        directory where the csv file containing xmin xmax ymin ymax for every 
        test is saved
    CSVOutputDirectory : string
        directory where the csv will be saved.
    rHandrim : double
        dimension of the handrim in real world, used to scale the image
    handrimPlane : hppdWC.geom.Plane3d object
        plane where the handrim is laying.
    wheel_centre : array 1*3
        contains x y z coordinates of the centre of the wheel.
    handrim : array 1*4
        contains x y z coordinates and the radius of the handrim.
    videoOutputDirectory : string, optional
        directory where videos will be saved. 
        The default is '', which sets recordVideo to False
    frequency : int, optional
        freq of saving of the videos.
        The default is 60.   
    nFrames : int, optional
       attended number of frames in the recording. The extractor will do 
       nFrames iterations, or, if the extraction is complete, 
       will stop earlier. Better put a larger number than the actual one. 
       Useful to print the loading bar.
       The default is 20000.
    maxDepthHand : double, optional
        When estimating the pointcloud of the hand, values of depth, expressed in 
        the camera ref frame, are excluded. The default is 1000.
    
    Returns
    -------
    df : pandas dataframe
        of 1+4*N columns: time + statistical variable computed by computeStatFeatures() 
        of the four variables computed by computeHandDepFeatures() in the handrim plane
    dfRefFrame : pandas dataframe
        of 15 columns: time, wxc, wyc, wzc, hrxc, hryc, hrzc, hrr,
                  a, b, c, d, norm x, norm y, norm z
        where:
            - w_c are the coordinates of the wheel centre
            - h_c are the coordinates of the handrim, hrr is the radius
            - a, b, c, d are the parameters of the plane
            - norm x, y, z are the components of the normal vector to the plane in the 
            camera reference frame.
    time_exec_array : numpy array
        contains the elapsed time for every execution
    '''
    
    recordVideo = True
    if videoOutputDirectory == '':
        recordVideo = False
    
    # eventually adding .bag in the end in case the user forgot it
    fileCompletePath = utils.checkExtension(fileCompletePath, '.bag')

    # get the name of the file
    fileName = os.path.split(fileCompletePath)[1][:-4]
    
    # add to the filename an univoque code    
    thisExecutionDate = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y%m%d%H%M%S')
    fileNameCode = fileName + '-' + thisExecutionDate
    
    if not os.path.isdir(CSVOutputDirectory):
        os.makedirs(CSVOutputDirectory, exist_ok=True)
        logging.info('directory <<' + CSVOutputDirectory + '>> not existing, creating')
    # complete directory for csv file saving
    csvFileCompletePath = os.path.join(CSVOutputDirectory, fileNameCode + '.csv')
    
    if recordVideo:
        if not  os.path.isdir(videoOutputDirectory):
            os.makedirs(videoOutputDirectory, exist_ok=True)
            logging.info('directory <<' + videoOutputDirectory + '>> not existing, creating')
        # complete directory for video saving
        videoRawCompletePath = os.path.join(videoOutputDirectory, fileNameCode + '-col.avi')
        videoDEPCompletePath = os.path.join(videoOutputDirectory, fileNameCode + '-dep.avi')

    # load the dataframe containing the coordinates of the roi
    roi = pd.read_csv(os.path.join(CSVroiDirectory, fileName + '.csv'))

    logging.info('working on ' + fileCompletePath)

    if recordVideo:
# =============================================================================
#         WRITE ON THE IMAGE PARAMS
# =============================================================================
        font = cv2.FONT_HERSHEY_SIMPLEX
        origin = (20, 20)
        fontScale = 0.8
        color = (255, 255, 255)
        thickness = 1
  
# =============================================================================
#     START THE STREAM OF THE PIPELINE
# =============================================================================
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, fileCompletePath, repeat_playback = False)
    
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 1)  # jet
    aligned_stream = rs.align(rs.stream.color)  # alignment depth -> color
    
# =============================================================================
#    INITIALIZATION
# =============================================================================
    # so at the first executuion becomes 0
    frameCounter = -1 
    # to save the timing execution of each loop (debug)
    time_exec_array = [0] * nFrames
    # time_exec_array = []
    # to save the starting of the execution
    startTime = time.time()
    # at each iteration add a new row containing
    data = [0] * nFrames
    dataRefFrame = [0] * nFrames
    # data = []
    # to update the detection of the centre of the wheel and the handrim plane
    counterUpdateRefFrame = -1


    rot, rmsd, sens = geom.rotMatrixToFitPlane(handrimPlane, wheel_centre)

    newRowRefFrame = [wheel_centre[0], wheel_centre[1], wheel_centre[2], handrim[0], handrim[1], handrim[2], handrim[3], handrimPlane.a, handrimPlane.b, handrimPlane.c, handrimPlane.d, handrimPlane.normal[0], handrimPlane.normal[1], handrimPlane.normal[2]]

    try:
        for i in tqdm.tqdm(range(nFrames)):
            try:
                frame = pipeline.wait_for_frames()
            except:
                break
            counterUpdateRefFrame += 1
            if i == 8:
                debugFlag = True
# =============================================================================
#             DEBUGGING
# =============================================================================
            frameCounter = frameCounter + 1
            # time frame on the execution of the loop
            now = time.time()
            # time_exec_array = np.append(time_exec_array, now-startTime)
            time_exec_array[frameCounter] = now-startTime
            
# =============================================================================
#             GET THE REQUIRED DATA FROM THE BAG FILE
# =============================================================================
            # alignement of the frames: the obtained resolution is the one of the rgb image
            frame = aligned_stream.process(frame)
            
            # get the depth and color frames
            depth_frame = frame.get_depth_frame()
            color_frame = frame.get_color_frame()

            # get the timestamp in seconds
            timestamp_s = frame.get_timestamp()/1000
            # print(datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f'))
            
            # from frames to images
            # the image saved in the bag file is in rgb format,
            # the one required from mediapipe as well
            color_image_rgb = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()).astype('float')
            # remove invalid values
            depth_image[depth_image <= 0] = np.nan
            # for video recording
            depth_image_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())

# =============================================================================
#             GET THE REQUIRED DATA FROM THE CSV roi FILE
# =============================================================================
            # consider only the row corresponding to the timestamp_s
            thisMoment = roi[abs(roi['time'] - timestamp_s) < 0.00001]

            if thisMoment.shape[0]==0:
                logging.warning('no frame recognized!!!')
            if thisMoment.shape[0]>1:
                logging.warning('more than one available moment!!!')
                logging.warning(thisMoment)
                logging.warning(timestamp_s)

            xmin = thisMoment['x min'].iloc[0]#.astype(int)
            xmax = thisMoment['x max'].iloc[0]#.astype(int)
            ymin = thisMoment['y min'].iloc[0]#.astype(int)
            ymax = thisMoment['y max'].iloc[0]#.astype(int)

            if not np.isnan(xmin): # if hand is defined
                # conversion to int
                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)

    # =============================================================================
    #             OPERATIONS ON THE DEP IMAGE
    # =============================================================================
                # extract the region of interest
                valid_image = depth_image.copy()
                valid_image[valid_image <= 0] = np.nan
                valid_image[0:ymin, :] = np.nan
                valid_image[ymax+1:, :] = np.nan
                valid_image[:, 0:xmin] = np.nan
                valid_image[:, xmax+1:] = np.nan
                valid_image[valid_image >= maxDepthHand] = np.nan

                dataHand = utils.depImgToThreeCol(valid_image)
                dataHand = utils.removeNanXYZ(dataHand)

                if not (np.isnan(dataHand).all()): # if there is at least one valid value
    
                    # since the plane is horizontal, same coefficient for x and y
                    pixel2mmCoeff_x = rHandrim / handrim[3]
                    pixel2mmCoeff_y = rHandrim / handrim[3]
                    # for z the data are already expressed in millimetres
    
                    # bring the pointcloud in the correct reference frame
                    dataHandDef = geom.changeRefFrameTRS(dataHand, wheel_centre, rot, pixel2mmCoeff_x, pixel2mmCoeff_y)
    
                    results, res_str_total = computeHandDepFeatures(dataHandDef, rHandrim, showPlot = False)
    
                    newRow = np.array(results)
                else:
                    newRow = np.array([np.nan]*36)

            else:
                xmin, xmax, ymin, ymax = 0,0,0,0
                newRow = np.array([np.nan]*36)

# =============================================================================
#             CREATE A NEW ROW IN DATA
# =============================================================================
            tmp = np.insert(newRow, 0, timestamp_s)
            data[frameCounter] = tmp

            tmpRefFrame = np.insert(newRowRefFrame, 0, timestamp_s)
            dataRefFrame[frameCounter] = tmpRefFrame

            if recordVideo:
# =============================================================================
#                 IMAGE OPERATION
# =============================================================================
                stringForImage = "frame: {:05d} / ".format(frameCounter) + \
                    datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f')

                # cv2 displays images in bgr, they need to be converted
                color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_BGR2RGB)
                # image_for_mp_bgr = cv2.cvtColor(image_for_mp, cv2.COLOR_BGR2RGB)
                # put text on the image
                color_image_bgr = cv2.putText(color_image_bgr, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)
                depth_image_colorized = cv2.putText(depth_image_colorized, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)

                color_image_bgr = plots.roiWcHrOnImage(color_image_bgr, wheel_centre, handrim, xmin, xmax, ymin, ymax)
                depth_image_colorized = plots.roiWcHrOnImage(depth_image_colorized, wheel_centre, handrim, xmin, xmax, ymin, ymax)
                if frameCounter == 0:
                    # initialize the video saver
                    image_height, image_width, _ = color_image_bgr.shape
                    videoRawOut = cv2.VideoWriter(videoRawCompletePath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                    videoMPOut = cv2.VideoWriter(videoDEPCompletePath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                        
                videoRawOut.write(color_image_bgr)
                videoMPOut.write(depth_image_colorized)

    finally:
        # cut the files preallocated with
        data = data[:frameCounter]
        dataRefFrame = dataRefFrame[:frameCounter]
        time_exec_array = time_exec_array[:frameCounter]
        
        # logging.debug('EXECUTION TERMINATED')
# =============================================================================
#         CREATION OF THE PANDAS DATAFRAME AND SAVING IN A CSV FILE
# =============================================================================
        # create the header using the function
        headerRefFrame = ['time', 'wxc', 'wyc', 'wzc', 'hrxc','hryc', 'hrzc', 'hrr',\
                  'a', 'b', 'c', 'd', 'norm x', 'norm y', 'norm z']
        header = ['time'] + res_str_total

        # create the pandas dataframe
        df = pd.DataFrame(np.vstack(data),  columns=header)
        # saves the pandas dataframe in a csv file
        df.to_csv(csvFileCompletePath, index = False)

        # create the pandas dataframe
        dfRefFrame = pd.DataFrame(np.vstack(dataRefFrame),  columns=headerRefFrame)
        # saves the pandas dataframe in a csv file
        dfRefFrame.to_csv(csvFileCompletePath.replace('.csv', '-refFrame.csv'), index = False)
        
# =============================================================================
#         OTHER OPERATIONS
# =============================================================================
        # stop the pipeline
        pipeline.stop()

        elapsedTime = time.time()-startTime
        freqOfExecution = frameCounter/elapsedTime
        
        logging.debug("{} terminated. {:d} frames in {:.2f} seconds ({:.2f} frames per second)"\
              .format(fileName, frameCounter, elapsedTime, freqOfExecution))
            
        return df, dfRefFrame, time_exec_array

def insideScalar(value, minVal, maxVal):
    '''Returns the value inside the range for a scalar'''
    return np.minimum(np.maximum(value,minVal),maxVal)
def insideArray(value, minVal, maxVal):
    '''Returns the value inside the range for an array'''
    return np.minimum(np.maximum(value,np.array([0]*len(value))),np.array([maxVal]*len(value)))


def handLocationWRTCentreMetric(fileCompletePath, # name of bag file to elaborate
                                handLandMarksDir, # find hand landmarks
                                ROIDir, # find the roi (can be taken from handLandmarks but less effort)
                                CSVxyzHandMetricDir, # save hand landmarks wrt cam ref
                                CSV1PointsDir, # save handLandmarks wrt to wheel ref frame
                                CSV9PointsDir, # save handLandmarks filtered on adjacent point wrt to wheel ref frame
                                CSVroiMetricDir, # save roi wrt to wheel ref frame
                                CSVroiWheelRefDir,
                                CSVPointCloudDir, # save point cloud coord wrt to wheel ref frame
                                centreCoord, # centre coordinates
                                handrimPlane, # plane of the handrim
                                spatialFiltAmp = 2, # amplitude of spatial filter for hand landmarks
                                videoOutputDirectory = '',
                                wheel_centre_img = [], # to be shown in the video
                                handrim_centre_img = [], # to be shown in the video
                                frequency = 60,
                                nFrames = 20000,
                                minDepthHandPointCloud = -0.20,
                                maxDepthHandPointCloud = 1):
    recordVideo = True
    if videoOutputDirectory == '':
        recordVideo = False
    
    # eventually adding .bag in the end in case the user forgot it
    fileCompletePath = utils.checkExtension(fileCompletePath, '.bag')

    # get the name of the file
    fileName = os.path.split(fileCompletePath)[1][:-4]
    
    # add to the filename an univoque code    
    thisExecutionDate = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y%m%d%H%M%S')
    fileNameCode = fileName + '-' + thisExecutionDate

    csvFileCompletePath = []
    for CSVOutputDirectory in [CSVxyzHandMetricDir,
                               CSV1PointsDir,
                               CSV9PointsDir,
                               CSVroiMetricDir,
                               CSVroiWheelRefDir,
                               CSVPointCloudDir]:
        if not os.path.isdir(CSVOutputDirectory):
            os.makedirs(CSVOutputDirectory, exist_ok=True)
            logging.info('directory <<' + CSVOutputDirectory + '>> not existing, creating')
        # complete directory for csv file saving
        csvFileCompletePath.append(os.path.join(CSVOutputDirectory, fileNameCode + '.csv'))
    
    if recordVideo:
        if not  os.path.isdir(videoOutputDirectory):
            os.makedirs(videoOutputDirectory, exist_ok=True)
            logging.info('directory <<' + videoOutputDirectory + '>> not existing, creating')
        # complete directory for video saving
        videoRawCompletePath = os.path.join(videoOutputDirectory, fileNameCode + '-col.avi')
        videoDEPCompletePath = os.path.join(videoOutputDirectory, fileNameCode + '-dep.avi')

    # load the dataframe containing the coordinates of the hand lanmarks
    handLandMarksFile = utils.findFileInDirectory(handLandMarksDir, fileName)[0]
    handLandMarks = pd.read_csv(handLandMarksFile)
    # load the dataframe containing the coordinates of the roi
    ROIFile = utils.findFileInDirectory(ROIDir, fileName)[0]
    ROI = pd.read_csv(ROIFile)

    logging.info('working on ' + fileCompletePath)

    if recordVideo:
# =============================================================================
#         WRITE ON THE IMAGE PARAMS
# =============================================================================
        font = cv2.FONT_HERSHEY_SIMPLEX
        origin = (20, 20)
        fontScale = 0.8
        color = (255, 255, 255)
        thickness = 1
  
# =============================================================================
#     START THE STREAM OF THE PIPELINE
# =============================================================================
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, fileCompletePath, repeat_playback = False)
    
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 1)  # jet
    aligned_stream = rs.align(rs.stream.color)  # alignment depth -> color
    
# =============================================================================
#    INITIALIZATION
# =============================================================================
    # so at the first executuion becomes 0
    frameCounter = -1 
    # to save the timing execution of each loop (debug)
    time_exec_array = [0] * nFrames
    # time_exec_array = []
    # to save the starting of the execution
    startTime = time.time()
    # at each iteration add a new row containing
    dxyzHandMetric = [0] * nFrames
    d1Points = [0] * nFrames
    d9Points = [0] * nFrames
    droiMetric = [0] * nFrames
    droiWheelRef  = [0] * nFrames
    dPointCloud = [0] * nFrames

    # rotation matrix
    rot, rmsd, sens = geom.rotMatrixToFitPlane(handrimPlane, centreCoord)

    try:
        for i in tqdm.tqdm(range(nFrames)):
            try:
                frame = pipeline.wait_for_frames()
            except:
                break

            if i == 0:
                debugFlag = True
# =============================================================================
#             DEBUGGING
# =============================================================================
            frameCounter = frameCounter + 1
            # time frame on the execution of the loop
            now = time.time()
            # time_exec_array = np.append(time_exec_array, now-startTime)
            time_exec_array[frameCounter] = now-startTime
            
# =============================================================================
#             GET THE REQUIRED DATA FROM THE BAG FILE
# =============================================================================
            # alignement of the frames: the obtained resolution is the one of the rgb image
            frame = aligned_stream.process(frame)
            
            # get the depth and color frames
            depth_frame = frame.get_depth_frame()
            color_frame = frame.get_color_frame()

            if i == 0:
                # load intrinsic params of camera
                camera_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

            # get the timestamp in seconds
            timestamp_s = frame.get_timestamp()/1000
            # print(datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f'))
            
            # from frames to images
            # the image saved in the bag file is in rgb format,
            # the one required from mediapipe as well
            color_image_rgb = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()).astype('float')
            img_height, img_width, _ = color_image_rgb.shape
            # remove invalid values
            depth_image[depth_image <= 0] = np.nan
            # for video recording
            depth_image_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())

# =============================================================================
#             GET THE REQUIRED DATA FROM THE CSV hand landmark FILE
# =============================================================================
            try: # if reaching end of execution
                this_roi = ROI.iloc[i,:]
            except:
                break
            xmin = this_roi['x min']#.astype(int)
            xmax = this_roi['x max']#.astype(int)
            ymin = this_roi['y min']#.astype(int)
            ymax = this_roi['y max']#.astype(int)


            if not np.isnan(xmin): # if hand is defined
    # =============================================================================
    #             OPERATIONS ON THE DEP IMAGE
    # =============================================================================
    # 0 hand landmarks in camera frame
                this_landmarks = handLandMarks.iloc[i,:]
                lm_x = this_landmarks.filter(regex = 'x').values*img_width-1
                lm_y = this_landmarks.filter(regex = 'y').values*img_height-1
                lm_z = this_landmarks.filter(regex = 'z')

                lm_x_inside = insideArray(lm_x, 0, img_width-1)
                lm_y_inside = insideArray(lm_y, 0, img_height-1)

                newRowlandmarkMetric = []
                for x_coord, y_coord in zip(lm_x_inside, lm_y_inside):
                    x_metric, y_metric, z_metric = geom.convert_depth_pixel_to_metric_coordinate(depth_image[round(y_coord), round(x_coord)], x_coord, y_coord, camera_intrinsics)
                    newRowlandmarkMetric = newRowlandmarkMetric + [x_metric] + [y_metric] + [z_metric]


    # 1 hand landmarks in wheel ref frame
                landmarkVertical = np.reshape(newRowlandmarkMetric,(21,3), 'C')
                landmarkWheelRefFrame = geom.changeRefFrameTR(landmarkVertical, centreCoord, rot)
                newRowd1Points = landmarkWheelRefFrame.flatten('C')

    # 2 hand landmarks in wheel ref frame, consider 9 points for every landmark
                newRow9Points = []
                for x_coord, y_coord in zip(lm_x_inside, lm_y_inside):
                    tmpx = []
                    tmpy = []
                    tmpz = []
                    for i in range(-spatialFiltAmp,spatialFiltAmp+1,1):
                        for j in range(-spatialFiltAmp,spatialFiltAmp+1,1):
                            x_coord_i = insideScalar(x_coord+i, 0, img_width-1)
                            y_coord_j = insideScalar(y_coord+i, 0, img_height-1)
                            # if no out of range
                            if x_coord_i == x_coord + i and y_coord_j == y_coord + j:
                                x_metric, y_metric, z_metric = geom.convert_depth_pixel_to_metric_coordinate(depth_image[round(y_coord_j), round(x_coord_i)], x_coord_i, y_coord_j, camera_intrinsics)
                                tmpx = tmpx + [x_metric]
                                tmpy = tmpy + [y_metric]
                                tmpz = tmpz + [z_metric]
                    if not np.isnan(tmpx).all():
                        tmpx_mean = np.nanmean(tmpx)
                    else:
                        tmpx_mean = np.nan
                    if not np.isnan(tmpx).all():
                        tmpy_mean = np.nanmean(tmpy)
                    else:
                        tmpy_mean = np.nan
                    if not np.isnan(tmpx).all():
                        tmpz_mean = np.nanmean(tmpz)
                    else:
                        tmpz_mean = np.nan
                    newRow9Points = newRow9Points + [tmpx_mean] + [tmpy_mean] + [tmpz_mean]

                landmark9Vertical = np.reshape(newRow9Points,(21,3), 'C')
                landmark9WheelRefFrame = geom.changeRefFrameTR(landmark9Vertical, centreCoord, rot)
                newRowd9Points = landmark9WheelRefFrame.flatten('C')

                # plt.figure()
                # plt.scatter(landmarkWheelRefFrame[:,0], landmarkWheelRefFrame[:,1])
                # plt.scatter(landmark9WheelRefFrame[:,0], landmark9WheelRefFrame[:,1])
                # plt.plot(landmarkWheelRefFrame[:,0], landmarkWheelRefFrame[:,1])

    # 3 roi coord metric in camera frame
                xmin = this_roi['x min']#.astype(int)
                xmax = this_roi['x max']#.astype(int)
                ymin = this_roi['y min']#.astype(int)
                ymax = this_roi['y max']

                newRowRoiMetric = []
                for y_coord in ([ymin, ymax]):
                    for x_coord in ([xmin, xmax]):
                        x_coord = insideScalar(x_coord, 0, img_width-1)
                        y_coord = insideScalar(y_coord, 0, img_height-1)
                        x_metric, y_metric, z_metric = geom.convert_depth_pixel_to_metric_coordinate(depth_image[round(y_coord), round(x_coord)], x_coord, y_coord, camera_intrinsics)
                        newRowRoiMetric = newRowRoiMetric + [x_metric] + [y_metric] + [z_metric]
    # 4 roi coord in wheel ref frame
                roiMetricVertical = np.reshape(newRowRoiMetric,(4,3), 'C')
                roiWheelRefFrame = geom.changeRefFrameTR(roiMetricVertical, centreCoord, rot)
                newRowRoiWheelRef = roiWheelRefFrame.flatten('C')

                # # conversion in camera ref frame of the roi coord
                # # b = bottom, t = top, l = left, r = right
                # blx, bly = geom.convert_pixel_coord_to_metric_coordinate(xmin, ymin, camera_intrinsics)
                # brx, bry = geom.convert_pixel_coord_to_metric_coordinate(xmax, ymin, camera_intrinsics)
                # trx, tryy = geom.convert_pixel_coord_to_metric_coordinate(xmax, ymax, camera_intrinsics)
                # tlx, tly = geom.convert_pixel_coord_to_metric_coordinate(xmin, ymax, camera_intrinsics)
                # newRowRoiMetric = [blx, bly, brx, bry, trx, tryy, tlx, tly]

    # 4 point cloud in wheel ref frame
                # conversion to int
                xmin = int(xmin)
                xmax = int(xmax)
                ymin = int(ymin)
                ymax = int(ymax)
                # extract the region of interest
                valid_image = depth_image.copy()
                valid_image[valid_image <= 0] = np.nan
                valid_image[0:ymin, :] = np.nan
                valid_image[ymax+1:, :] = np.nan
                valid_image[:, 0:xmin] = np.nan
                valid_image[:, xmax+1:] = np.nan
                # valid_image[valid_image >= maxDepthHand] = np.nan
                # valid_image[valid_image <= minDepthHand] = np.nan

                pc = geom.convert_depth_frame_to_pointcloud(valid_image, camera_intrinsics)
                x,y,z = pc
                dataROI = np.transpose([x,y,z])
                dataROI = utils.removeNanXYZ(dataROI)

                if not (np.isnan(dataROI).all()): # if there is at least one valid value
                    # bring the pointcloud in the correct reference frame
                    dataHandDef = geom.changeRefFrameTR(dataROI, centreCoord, rot)
                    # plt.figure()
                    # plt.plot(dataHandDef,'o')

                    # erase values outside the range
                    dataHandDef[dataHandDef[:,2]>maxDepthHandPointCloud] = [np.nan]*3
                    dataHandDef[dataHandDef[:,2]<minDepthHandPointCloud] = [np.nan]*3
                    dataHandDef = utils.removeNanXYZ(dataHandDef)

                    # plt.plot(dataHandDef,'o')
                    # plt.grid()
                    # plt.legend('all x', 'all y', 'all z', 'x from z in range', 'y from z in range', 'z in range')
                    # plt.suptitle('hand point cloud - removal of values of depth > 1 m and depth < -0.2 m')
                    if not (np.isnan(dataHandDef).all()):
                        results, res_str_total = statOnXYZpointcloud(dataHandDef, showPlot = False)
    
                    newRowRoiPc = np.array(results)
                else:
                    newRowRoiPc = np.array([np.nan]*27)

            else:
                xmin, xmax, ymin, ymax = 0,0,0,0
                newRowlandmarkMetric = np.array([np.nan]*63)
                newRowd1Points = np.array([np.nan]*63)
                newRowd9Points = np.array([np.nan]*63)
                newRowRoiMetric = np.array([np.nan]*12)
                newRowRoiWheelRef = np.array([np.nan]*12)
                newRowRoiPc = np.array([np.nan]*27)


# =============================================================================
#             CREATE A NEW ROW IN DATA
# =============================================================================
            dxyzHandMetric[frameCounter] = np.insert(newRowlandmarkMetric, 0, timestamp_s)
            d1Points[frameCounter] = np.insert(newRowd1Points, 0, timestamp_s)
            d9Points[frameCounter] = np.insert(newRowd9Points, 0, timestamp_s)
            droiMetric[frameCounter] = np.insert(newRowRoiMetric, 0, timestamp_s)
            droiWheelRef[frameCounter] = np.insert(newRowRoiWheelRef, 0, timestamp_s)
            dPointCloud[frameCounter] = np.insert(newRowRoiPc, 0, timestamp_s)

            if recordVideo:
# =============================================================================
#                 IMAGE OPERATION
# =============================================================================
                stringForImage = "frame: {:05d} / ".format(frameCounter) + \
                    datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f')

                # cv2 displays images in bgr, they need to be converted
                color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_BGR2RGB)
                # image_for_mp_bgr = cv2.cvtColor(image_for_mp, cv2.COLOR_BGR2RGB)
                # put text on the image
                color_image_bgr = cv2.putText(color_image_bgr, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)
                depth_image_colorized = cv2.putText(depth_image_colorized, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)

                color_image_bgr = plots.roiWcHrOnImage(color_image_bgr, wheel_centre_img, handrim_centre_img, xmin, xmax, ymin, ymax)
                depth_image_colorized = plots.roiWcHrOnImage(depth_image_colorized, wheel_centre_img,handrim_centre_img, xmin, xmax, ymin, ymax)
                if frameCounter == 0:
                    # initialize the video saver
                    image_height, image_width, _ = color_image_bgr.shape
                    videoRawOut = cv2.VideoWriter(videoRawCompletePath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                    videoMPOut = cv2.VideoWriter(videoDEPCompletePath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                        
                videoRawOut.write(color_image_bgr)
                videoMPOut.write(depth_image_colorized)

    finally:
        # cut the files preallocated
        dxyzHandMetric = dxyzHandMetric[:frameCounter]
        d1Points = d1Points[:frameCounter]
        d9Points = d9Points[:frameCounter]
        droiMetric = droiMetric[:frameCounter]
        droiWheelRef = droiWheelRef[:frameCounter]
        dPointCloud = dPointCloud[:frameCounter]

        time_exec_array = time_exec_array[:frameCounter]
        
        # logging.debug('EXECUTION TERMINATED')
# =============================================================================
#         CREATION OF THE PANDAS DATAFRAME AND SAVING IN A CSV FILE
# =============================================================================
        # create the header using the function
        headerxyzHandMetric = ['time']+['{}{:02d}'.format(letter, num) for num in range(21) for letter in ['x','y', 'z']]
        header1Points = ['time']+['{}{:02d}'.format(letter, num) for num in range(21) for letter in ['x','y','z']]
        header9Points = ['time']+['{}{:02d}'.format(letter, num) for num in range(21) for letter in ['x','y','z']]
        headerRoi = ['time'] + ['blx', 'bly', 'blz',
                                'brx', 'bry', 'brz',
                                'tlx', 'tly', 'tlz',
                                'trx', 'try', 'trz']
        # headerroiMetric = ['time', 'blx', 'bly', 'brx', 'bry', 'trx', 'try', 'tlx', 'tly']
        headerPointCloud = ['time'] + res_str_total

        df = pd.DataFrame(np.vstack(dxyzHandMetric),  columns=headerxyzHandMetric)
        df.to_csv(csvFileCompletePath[0], index = False)
        df = pd.DataFrame(np.vstack(d1Points),  columns=header1Points)
        df.to_csv(csvFileCompletePath[1], index = False)
        df = pd.DataFrame(np.vstack(d9Points),  columns=header9Points)
        df.to_csv(csvFileCompletePath[2], index = False)
        df = pd.DataFrame(np.vstack(droiMetric),  columns=headerRoi)
        df.to_csv(csvFileCompletePath[3], index = False)
        df = pd.DataFrame(np.vstack(droiWheelRef),  columns=headerRoi)
        df.to_csv(csvFileCompletePath[4], index = False)
        df = pd.DataFrame(np.vstack(dPointCloud),  columns=headerPointCloud)
        df.to_csv(csvFileCompletePath[5], index = False)

# =============================================================================
#         OTHER OPERATIONS
# =============================================================================
        # stop the pipeline
        pipeline.stop()

        elapsedTime = time.time()-startTime
        freqOfExecution = frameCounter/elapsedTime
        
        logging.debug("{} terminated. {:d} frames in {:.2f} seconds ({:.2f} frames per second)"\
              .format(fileName, frameCounter, elapsedTime, freqOfExecution))
