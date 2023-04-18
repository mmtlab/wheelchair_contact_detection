# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:43:56 2023

@author: giamp
"""

import mediapipe as mp 
import numpy as np
import cv2
import os
import math

def write_hand_on_image(img, results):
    if not results.multi_hand_landmarks:
        cv2.putText(img=img, text='NO HANDS DETECTED', org=(50,40), fontFace=cv2.FONT_HERSHEY_PLAIN , fontScale=3, color=(255, 255, 255),thickness=3)  
    else:    
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_hands=mp.solutions.hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img, # image to draw
                hand_landmarks, # model output
                mp_hands.HAND_CONNECTIONS, # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    return img

def mediapipe_process_img(img):
    mp_hands = mp.solutions.hands
    mp_model = mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5)       
    results = mp_model.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return results

def get_hand_landmarks(results,x_resolution,y_resolution):
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
    mp_hands = mp.solutions.hands
    if not results.multi_hand_landmarks:
        pass
    else:
        for finger in landmarkls:
            bufferx=[]
            buffery=[]
            for landmark in finger:
                x=results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark(landmark)].x
                y=results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark(landmark)].y
                x=x_resolution*x #x resolution for the camera, need rescaling because mediapipe gives 0<x<1
                y=y_resolution*y #y resolution for the camera, need rescaling because mediapipe gives 0<y<1
                
                bufferx.append(x)
                buffery.append(y)
            handlm.append([bufferx,buffery])
        
    return handlm

# image=cv2.imread(r'C:\Users\giamp\OneDrive\Pictures\Rullino\WIN_20230329_17_37_28_Pro.jpg')
# image=cv2.imread(r'C:\Users\giamp\OneDrive\Pictures\Rullino\WIN_20230329_16_54_11_Pro.jpg')
# results=mediapipe_process_img(image)
# handlm=get_hand_landmarks(results, 500, 500)
# image=write_hand_on_image(image, results)
# cv2.imshow('a',image)
# cv2.waitKey(0) # waits until a key is pressed
# cv2.destroyAllWindows()



    
    
