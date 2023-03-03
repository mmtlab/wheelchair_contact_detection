# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:45:38 2023

@author: giamp
"""

import mediapipe as mp 
import cv2
import numpy as np
import sys



def gethandlandmarks(frame,x_resolution,y_resolution):
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

def averagehandposition(handlm,depthframe):
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
       xh='NaN'
       yh='NaN'
       zh='NaN'    
    return xh,yh,zh   

