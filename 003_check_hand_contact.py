# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:39:56 2023

@author: giamp
"""
import basic
import numpy as np
import pyrealsense2 as rs
from hcd import coordinates

number_of_frames=20000
filecompletename=r'D:\01_raw\T017S03BnrC3r.bag' 
threshold=100
led_status_lst=[]
timestamp=[]
header_led=["time","led status"]
dir_led=r'D:\01_raw'
tl=[0,0]
rb=[100,100]
def led_status (img, tl, br, threshold):
    img=basic.imagelib.cropImageTLBR(img, tl, br)    

    # Calculate average value of each channel
    red_avg = np.mean(img[:,:,0])
    green_avg = np.mean(img[:,:,1])
    blue_avg = np.mean(img[:,:,2])
    
    # Check if the led is on
    if red_avg > threshold:
        led_status=True
    else:
        led_status=False
    
    return led_status

#start the pipeline and setup the configuration to read the .bag file
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, filecompletename, repeat_playback = False)

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
    #get the timestamp of each single frame
    timestamp_s = frame.get_timestamp()/1000
    timestamp.append(timestamp_s)
    # get the depth and color frames
    color_frame = frame.get_color_frame()
    color_image_rgb = np.asanyarray(color_frame.get_data())#transform the color frame in a RGB array
    led_status_lst.append([timestamp_s,led_status(color_image_rgb,tl,rb)])

coordinates.save_multilist_to_CSVfile(filecompletename, led_status_lst, header_led, 'led_status', dir_led)