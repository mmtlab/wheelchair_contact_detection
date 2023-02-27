# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:45:38 2023

@author: giamp
"""

import mediapipe as mp 
import cv2
import numpy as np
import sys



def GetHandLandmarks(frame,x_resolution,y_resolution):
    landmark_ls=[[0],[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
    hand_lm=[]
    mp_hands=mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.8, min_tracking_confidence=0.5, model_complexity=1) as hands:
        results=hands.process(frame)
        if not results.multi_hand_landmarks:
            pass
        else:
            for finger in landmark_ls:
                buffer_x=[]
                buffer_y=[]
                for landmark in finger:
                    x=results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark(landmark)].x
                    y=results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark(landmark)].y
                    x=x_resolution*x #x resolution for the camera, need rescaling because mp gives 0<x<1
                    y=y_resolution*y #y resolution for the camera, need rescaling because mp gives 0<y<1
                    buffer_x.append(x)
                    buffer_y.append(y)
                hand_lm.append([buffer_x,buffer_y])
        
    return hand_lm

def AverageHandPosition(hand_lm,depth_frame):
    x_min_list=[]
    y_min_list=[]
    x_max_list=[]
    y_max_list=[]
    for i in hand_lm:
        d=0
        for j in i:
            if d==0:
                finger_x_min=min(j)
                finger_x_max=max(j)
                x_min_list.append(finger_x_min)
                x_max_list.append(finger_x_max)
            if d==1:
                finger_y_min=min(j)
                finger_y_max=max(j)
                y_min_list.append(finger_y_min)
                y_max_list.append(finger_y_max)
            d=d+1
    if not hand_lm==[]:      
       x_min=min(x_min_list)
       x_max=max(x_max_list)
       y_min=min(y_min_list)
       y_max=max(y_max_list)
       x_h=int((x_max + x_min)/2)
       y_h=int((y_max + y_min)/2)
       z_h=depth_frame[y_h][x_h]
    else:
       x_h='NaN'
       y_h='NaN'
       z_h='NaN'    
    return x_h,y_h,z_h   

