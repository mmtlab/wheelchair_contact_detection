# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:39:56 2023

@author: giamp
"""
import basic
import os
import numpy as np
import pyrealsense2 as rs
import pandas as pd
from hcd import coordinates

number_of_frames=20000
savecsvfile=True
# filecompletename=r"G:\Download\T129Subject13S2R1YR_run0-001.bag"
# filecompletename=r"C:\bagfiles to process\T032Subject3S1R1NR_run0.bag"
filecompletename=r'C:\Users\Alexk\OneDrive\Documenten\School\RUG\jaar 3 bewegingswetenschappen(2022-2023)\BAP\Bag-files\T050Subject4S1R2NR_run0.bag'
# csv_path=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230519\02_preprocessing\realsenseRight\led status'
csv_path=r'G:\.shortcut-targets-by-id\1MjZH9XN1gWNoiyVyaLwpjTjfKx4AsRiw\handrim contact detection\Tests\20230519\02_preprocessing\realsenseRight\led status'
testname='T050Subject4S1R2NR_run0'
tl=[264,413]
br=[268,418]

timestamp=[]
time=[]
flag=0

csvfullpath=os.path.join(csv_path,testname+'.csv')
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
led_stat=pd.DataFrame(columns=('Time(s)','Red','Green','Blue','Status'))
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
    img=basic.imagelib.cropImageTLBR(color_image_rgb, tl, br)
    red=0
    green=0
    blue=0
    for j in range(br[0]-tl[0]):
        for k in range(br[1]-tl[1]):
            red=red+img[k][j][0]
            green=green+img[k][j][1]
            blue=blue+img[k][j][2]
    if i==0:
        led_stat.at[i,"Time(s)"]=0
    if i!=0:
        led_stat.at[i,"Time(s)"]=timestamp[i]-timestamp[i-1]+led_stat.at[i-1,"Time(s)"]    
    led_stat.at[i,"Red"]= red/((br[0]-tl[0])*(br[1]-tl[1]))
    led_stat.at[i,"Green"]= green/((br[0]-tl[0])*(br[1]-tl[1]))
    led_stat.at[i,"Blue"] = blue/((br[0]-tl[0])*(br[1]-tl[1]))
    if i==60:
       threshold=led_stat.iloc[0:60]["Red"].mean()
       dev=led_stat.iloc[0:60]["Red"].std()
    if i>59 and led_stat.at[i,"Red"]>threshold+3*dev and flag==0:
        delay=led_stat.at[i,"Time(s)"]
        flag=1
        print("The LED has been turned on after "+str(delay)+" seconds")
        
        
      
    
    
    # Check if the led is on
    # if led_stat['Red'][i] > threshold:
    #     led_stat["Status"][i]=True
    # else:
    #     led_stat["Status"][i]=False
    print(i)
led_stat.to_csv(csvfullpath)  
# led_stat["Time(s)"] = pd.to_datetime(led_stat['Time(s)'])    
led_stat.plot(x='Time(s)',y='Red')
pipeline.stop()