# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:45:38 2023

@author: giamp
"""

import mediapipe as mp 
import numpy as np
import csv
import os
import math


def get_hand_landmarks(frame,x_resolution,y_resolution):
    '''
    Given an rgb frame along with the pixel resolution of the image can extract
    the x and y position of each landmark of the hand which are enumerated as you can see at
    the following link https://google.github.io/mediapipe/solutions/hands.html .
    If no hand is detected in the frame then the hand_lm array containing the landmark coordinates
    will be empty.
    The x and y coordinates are given in pixel units where the origin is in the top left corner of the image.

    Parameters
    ----------
    frame : numpy array.
    x_resolution : int
        number of pixel in x direction (from left to right).
    y_resolution : int
        number of pixel in y direction (from up to down).

    Returns
    -------
    hand_lm : array
        contains the landmark coordinates with the following structure[[wrist][thumb][index][middle][ring][pinky]].
        
    '''
    landmarkls=[[0],[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
    handlm=[]
    mphands=mp.solutions.hands
    with mphands.Hands(static_image_mode=True, min_detection_confidence=0.8, min_tracking_confidence=0.5, model_complexity=1) as hands:
        results=hands.process(frame)
        if not results.multi_hand_landmarks:
            pass
        else:
            for finger in landmarkls:
                bufferx=[]
                buffery=[]
                for landmark in finger:
                    x=results.multi_hand_landmarks[0].landmark[mphands.HandLandmark(landmark)].x
                    y=results.multi_hand_landmarks[0].landmark[mphands.HandLandmark(landmark)].y
                    x=x_resolution*x #x resolution for the camera, need rescaling because mediapipe gives 0<x<1
                    y=y_resolution*y #y resolution for the camera, need rescaling because mediapipe gives 0<y<1
                    
                    bufferx.append(x)
                    buffery.append(y)
                handlm.append([bufferx,buffery])
        
    return handlm

def roi_position(handlm,depthframe):
    '''
    Given a set of landmark detect the max and min value in both x and y.
    Finds the center of the region delimited by those value and given a depth map of the pixels
    extracts the z coordinate of the region center.
    The origin is in the top left corner of the image on the camera plane.

    Parameters
    ----------
    handlm : array
        contains the landmark coordinates.
    depthframe : numpy array
        contains the depth value for each pixel.

    Returns
    -------
    xh : int
        x coordinate of the centre of the region in pixel units.
    yh : int
        y coordinate of the centre of the region in pixel units.
    zh : float
        z coordinate of the centre of the region in metres.

    '''
    xminlist=[]
    yminlist=[]
    xmaxlist=[]
    ymaxlist=[]
    for i in handlm:
        d=0
        for j in i:
            if d==0: #firstly extract x
                fingerxmin=min(j)
                fingerxmax=max(j)
                xminlist.append(fingerxmin)
                xmaxlist.append(fingerxmax)
            if d==1: #secondly extract y
                fingerymin=min(j)
                fingerymax=max(j)
                yminlist.append(fingerymin)
                ymaxlist.append(fingerymax)
            d=d+1
    if not handlm==[]:  #if the landmark list is not empty then proceed to find the max,min value of x and y  
       xmin=min(xminlist)
       xmax=max(xmaxlist)
       ymin=min(yminlist)
       ymax=max(ymaxlist)
       xh=int((xmax + xmin)/2)
       yh=int((ymax + ymin)/2)
       zh=int(depthframe[yh][xh])
    else:  #if the landmark list is empty assign NaN to each coordinate
       xh=np.nan
       yh=np.nan
       zh=np.nan    
    return xh,yh,zh 

def multi_array_to_mono_array(hand_lm):
    '''
    Changes the structure of the array to a monodimensional list with the following structure:
    [x0","y0","x1","y1","x2","y2","x3","y3","x4","y4","x5","y5","x6","y6","x7","y7","x8","y8","x9","y9","x10","y10","x11","y11","x12","y12","x13","y13","x14","y14","x15","y15","x16","y16","x17","y17","x18","y18","x19","y19","x20","y20"]

    Parameters
    ----------
    hand_lm : array
        contains the landmark coordinates with the following structure[[wrist][thumb][index][middle][ring][pinky]]..

    Returns
    -------
    row : list
    contains the landmark coordinates with the different structure

    '''
    row=[]
    if not hand_lm==[]:
         for i in range(6): #scan through every sublist of the the hand lm array
             finger_buffer=hand_lm[i]
             x_buffer=finger_buffer[0]
             y_buffer=finger_buffer[1]
             for j in range(len(x_buffer)):
                 row.append(x_buffer[j])
                 row.append(y_buffer[j])
    else: #exception if the hand_lm list is empty because no hand was detcted by mediapipe
        row.append(np.nan)
    return row

def save_multilist_to_CSVfile(filecompletename, multilist, header, dataname, dirpath):
    '''
    Given a list, creates a csv file where each row correspond to a sublist in the input list.

    Parameters
    ----------
    filecompletename : string
        The path of the test we are currently examining.
    multilist : list
        contains the data with the structure [[data1.1,data1.2,etc..],[data2.1,data2.2, etc..],etc..]
    header : list
        contains the head to each column of the CSV file.
    dataname : string
        contains the name of the data we are writing in the CSV.
        i.e. if the data are coordinates: dataname='coordinates'. 
        The CSV file will be named as follow 'testname' + '_dataname' + .csv
    Returns
    -------
    writer : writer object
        
    '''
    filename=os.path.splitext(os.path.split(filecompletename)[1])[0] + '_' + dataname + '.csv'
    filename=os.path.join(dirpath,filename)
    for i in range(len(multilist)):
       multilist[i].insert(0,i) #to add the sub-list index at the beginning of each list
    
    with open(filename,'w',encoding='UTF8', newline='') as f: #initalize the csv file https://docs.python.org/3/library/csv.html
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(multilist)
    return writer

def from_cart_to_cylindrical_coordinates(cart_coord):
    '''
    Transform x,y,z coordinates of a point to dist_rad,theta,z (cylindrical) coordinates.

    Parameters
    ----------
    cart_coord : array n*3
        XYZ coordinates of the point.

    Returns
    -------
    cyl_coord : array n*3
        Cylindrical coordinates of the point.
        Following structure [dist_rad,ang_rad,dist_norm].

    '''
    dist_rad=math.sqrt((cart_coord[0])**2 + (cart_coord[1])**2)
    #since the arctan function in math outputs only an angle between -pi/2 and pi/2 based on x and y position
    #we can adjust the output to express the angle through the 4 quadrants.
    if cart_coord[0]>=0 and cart_coord[1]>=0: #first quadrant
        ang_rad=math.atan(cart_coord[1]/cart_coord[0])
    elif cart_coord[0]<0: #second and third quadrant
        ang_rad=math.atan(cart_coord[1]/cart_coord[0])+math.pi
    else: #fourth quadrant
        ang_rad=math.atan(cart_coord[1]/cart_coord[0])+2*math.pi
    cyl_coord=[dist_rad,ang_rad,cart_coord[2]]
    return cyl_coord

def changeRefFrameTR(data, XYZcentre, r):
    '''
    Applies a rototranslation matrix to the given data.
    First a translation and then a rotation, both expressed in the original ref frame

    Parameters
    ----------
    data : array n*2 or n*3
        XY(Z) coordinates to be rototranslated.
    XYZcentre : array 1*2 or 1*3
        Coordinates of the centre of the new reference frame seen from the original one.
    r : <scipy.spatial.transform._rotation.Rotation>
        Rotation applied to the data.

    Returns
    -------
    data_new_ref : array n*2 or n*3
        Expressed in the new reference frame.

    '''
    # translation
    data_translated = []
    for i in range(len(data)):
        data_translated.append(data[i] - XYZcentre[i])
    # rotation
    data_new_ref = r.apply(data_translated)

    return data_new_ref

