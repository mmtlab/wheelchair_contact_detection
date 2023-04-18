# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 09:38:53 2023

@author: giamp
"""

import os
import csv
import pyrealsense2 as rs
import numpy as np
import mediapipe as mp
import basic

filecompletename= r'D:\VID_20230316_123537_02.mp4'
number_of_frames=20000
mphands=mp.solutions.hands
missedframes=np.array([0,0])
capturedframes=np.array([0,0])
for j in [0,1]:
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, filecompletename[j], repeat_playback = False)
    
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)
    
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 1)
    aligned_stream = rs.align(rs.stream.color) 
    
    for i in range(number_of_frames):
        try:
            frame = pipeline.wait_for_frames()
        #if no/no more frames are available stops to fetch them.
        except:
            print(i)
            break
        #align the color and depth frame
        frame = aligned_stream.process(frame)
        
        # get the depth and color frames
        depth_frame = frame.get_depth_frame()
        color_frame = frame.get_color_frame()
        color_image_rgb = np.asanyarray(color_frame.get_data())#transform the color frame in a RGB array
        # if j == 0:
        #     color_image_rgb = basic.imagelib.cropImageTLBR(color_image_rgb, [30,154], [136,309])
        # if j == 1:
        #     color_image_rgb = basic.imagelib.cropImageTLBR(color_image_rgb, [70,80], [182,226])
        with mphands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5, model_complexity=1) as hands:
            results=hands.process(color_image_rgb)
            if not results.multi_hand_landmarks:
                missedframes[j]=missedframes[j]+1
                pass
            else:
                capturedframes[j]=capturedframes[j]+1


perc_missedframe=missedframes/(missedframes+capturedframes)
print(perc_missedframe)
        
        
    