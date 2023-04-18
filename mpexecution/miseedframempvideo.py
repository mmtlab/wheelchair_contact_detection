# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:48:40 2023

@author: giamp
"""

import mediapipe as mp
import cv2

vid=cv2.VideoCapture(r'D:\VID_20230316_123537_02.mp4')


mp_hands=mp.solutions.hands
missedfr=0
recogfr=0
i=0
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, model_complexity=1) as hands:
    while True:
        i=i+1
        success,image=vid.read()
        if success==False:
            break
        results=hands.process(image)
        if not results.multi_hand_landmarks:
            missedfr=missedfr+1
        else:
            recogfr=recogfr+1
        print(i)

print(str(recogfr/(recogfr+missedfr)*100)+" % of frames have been processed by mediapipe")