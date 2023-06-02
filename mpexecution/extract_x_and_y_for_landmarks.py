import mediapipe as mp 
import cv2
import numpy as np
import sys
vid=cv2.VideoCapture('D:\\002.mp4')

landmark_ls=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
#landmark_ls=[[1]]
#divides the video in frames and stores each frame in a list
def return_frames():
    n_frames=0
    fr_ls=[]
    while True:
        success,image=vid.read()
        if(success==False):
            break
        fr_ls.append(image)
        n_frames=n_frames+1
    return n_frames, fr_ls

#extracts the x and y value for each finger's landmark and stores it with the structure:
# (frame-finger-type_of_coordinate-landmark)

mp_hands=mp.solutions.hands
i=0
lost_fr=[]
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as hands:
    n_frames, fr_ls=return_frames()
    frame_lm=[]
   
    for frame in fr_ls:
        i=i+1
        results=hands.process(frame)
        if not results.multi_hand_landmarks:
            lost_fr.append(i)
        else:
            # cv2.imshow("test",frame)
            # cv2.waitKey(0)
            results=hands.process(frame)
            hand_lm=[]
            for finger in landmark_ls:
                buffer_x=[]
                buffer_y=[]
                for landmark in finger:
                    x=results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark(landmark)].x
                    y=results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark(landmark)].y
                    buffer_x.append(x)
                    buffer_y.append(y)
                hand_lm.append([buffer_x,buffer_y])
            frame_lm.append(hand_lm)
x_hlist=[]
y_hlist=[]
for hand_lm in frame_lm:
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
    x_min=min(x_min_list)
    x_max=max(x_max_list)
    y_min=min(y_min_list)
    y_max=max(y_max_list)
    x_h=(x_max + x_min)/2
    y_h=(y_max + y_min)/2
    x_hlist.append(x_h)
    y_hlist.append(y_h)
    