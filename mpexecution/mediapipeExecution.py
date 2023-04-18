# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:08:05 2023

@author: eferlius
"""

import cv2
import mediapipe as mp
import basic
import keyboard
import datetime
import time

# source for mediapipe analysis, 
# if 0: camera of the laptop
# if path: path to the avi video
source = 0

HAND = True
POSE = False

SAVE_VIDEO = True

# ----------------------------- PARAMETERS
mp_hands = mp.solutions.hands
SIMh = False # static_image_mode
MNH = 2 # max_num_hands
MDCh = 0.1 # min_detection_confidence
MTCh = 0.5 # min_tracking_confidence

mp_pose = mp.solutions.pose
SIMp = False # static_image_mode
MC = 1 # model_complexity
MDCp = 0.1 # min_detection_confidence
MTCp = 0.5 # min_tracking_confidence

# to write on the image
font = cv2.FONT_HERSHEY_SIMPLEX
org = (10, 50)
fontScale = 1
color = (255, 0, 0) #BGR format
thickness = 2


def draw_mp_hands(img, results):
    annotated_image = img.copy()
    if results.multi_hand_landmarks:
        # loop for every hand found
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(annotated_image, 
                hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style())
    return annotated_image

def draw_mp_pose(img, results):
    annotated_image = img.copy()
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(annotated_image, 
            results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image
    
if SAVE_VIDEO:
    thisMoment = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
    filename = 'test' + thisMoment + '.avi'
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    videoWriter = cv2.VideoWriter(filename, fourcc, 30.0, (640,480))

cap = cv2.VideoCapture(source)

frameCounter = 0
elapsed = []
timer = basic.timer.Timer("timer for loop execution of realsense")

with mp_hands.Hands(static_image_mode = SIMh, max_num_hands=MNH, 
                    min_detection_confidence=MDCh, min_tracking_confidence=MTCh) as hands:
    with mp_pose.Pose(static_image_mode = SIMp, model_complexity = MC, 
                      min_detection_confidence=MDCp, min_tracking_confidence=MTCp) as pose:    
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("frame not received")
                # If loading a video, use 'break' instead of 'continue'.
                if source != 0:
                    break
                else:
                    continue
                
            # flip the image if coming from frontal camera
            if source == 0:
                image = cv2.flip(image, 1)            
             
            image_drawn = image
            if HAND:q
                res_h = hands.process(image)
                image_drawn = draw_mp_hands(image_drawn, res_h)
            if POSE:
                res_p = pose.process(image)
                image_drawn = draw_mp_pose(image_drawn, res_p)
                
            cv2.imshow('result of mediapipe', image_drawn)
            cv2.waitKey(1)  
            
            # timing execution
            # time elapsed with respect to the previous execution
            frameCounter += 1
            lap = timer.lap(printTime = False)
            elapsed.append(lap) 
            
            
            string = ('frame: ' + str(frameCounter) + ' // elapsed: ' + "%06.2f" %(lap*1000) + "[ms]")
            image_drawn = cv2.putText(image_drawn, string, org, font, fontScale, color, thickness, cv2.LINE_AA)
            
            videoWriter.write(image_drawn)

                          
                
            if keyboard.is_pressed("q"):
                # total timing execution
                elapsedTotal = timer.stop(printTime = False)
                break
            
cv2.destroyAllWindows()            
cap.release()

basic.plots.plts([],[elapsed], mainTitle= 'average frequency: {:.2f} hZ'.format(frameCounter/elapsedTotal),
                 listXlabels=['execution number'], listYlabels=['elapsed time [s]'])   
