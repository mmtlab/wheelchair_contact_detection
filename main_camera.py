# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:24:09 2023

@author: giamp
"""
import numpy as np
import pyrealsense2 as rs
import findregion
import time
import hppdWC
start=time.time()

fileCompleteName=r'D:\01_raw\T001.bag'
NumberOfFrames=10 #usually 20000 for a full acquisition
x_resolution=640
y_resolution=480
rgblist=[]
lm_lst=[]
x_hList=[]
y_hList=[]
#start the pipeline and setup the configuration to read the .bag file
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, fileCompleteName, repeat_playback = False)

profile = pipeline.start(config)
device = profile.get_device()
playback = device.as_playback()
playback.set_real_time(False)

colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 1)  # jet
aligned_stream = rs.align(rs.stream.color) 

for i in range(NumberOfFrames):
    try:
        frame = pipeline.wait_for_frames()
    except:
        print(i)
        break
    frame = aligned_stream.process(frame)
    
    # get the depth and color frames
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()
    color_image_rgb = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    #get the hand landmarks
    hand_lm=findregion.GetHandLandmarks(color_image_rgb,x_resolution,y_resolution)
    lm_lst.append(hand_lm)
    #get the hand position
    x_h,y_h,z_h=findregion.AverageHandPosition(hand_lm,depth_image)
    
    
end=time.time()  
# print(start,end)
pipeline.stop()