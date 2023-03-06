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
import fromcartesiantocylndrical
import transform

start=time.time()

fileCompleteName=r'D:\01_raw\T017S03BnrC3r.bag'
NumberOfFrames=20000 #usually 20000 for a full acquisition
x_resolution=640
y_resolution=480
rgblist=[]
lm_lst=[]
x_hList=[]
y_hList=[]
hand_cyl_coord_lst=[]
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
    #get the camera intrinsic parameters
    if i == 0:
        # load intrinsic params of camera and find the wheel centre as well as the plane on which the wheel stands
        # extract rotation of the new frame compared to the absolute one
        camera_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        ppx = camera_intrinsics.ppx
        ppy = camera_intrinsics.ppy
        fx = camera_intrinsics.fx
        fy = camera_intrinsics.fy
        wc_img, hrc_img, centre_metric, handrimPlane, dataPlane = hppdWC.analysis.findWheelCentreAndHandrim(color_image_rgb,depth_image,ppx,ppy,fx,fy)
        rot, rmsd, sens=hppdWC.geom.rotMatrixToFitPlaneWRTimg(handrimPlane,centre_metric)
    #get the hand landmarks and stores them in lm_lst for each frame with a different configuration for better csv writing
    hand_lm=findregion.gethandlandmarks(color_image_rgb,x_resolution,y_resolution)
    hand_lm_buffer=findregion.changehandlandmarkStructure(hand_lm)
    lm_lst.append(hand_lm_buffer)
    #get the hand position
    xh,yh,zh=findregion.averagehandposition(hand_lm,depth_image)
    hand_coordinates_camera=[xh,yh,zh]
    #transform the hand coordinates to the wheel plane frame
    if hand_coordinates_camera[0] != np.nan:
        # transform
        hand_coordinates_hrplane=transform.changeRefFrameTR(hand_coordinates_camera, centre_metric, rot)
        #transform the hand coordinates from cartesian to cylindrical
        hand_cyl_coordinates=fromcartesiantocylndrical.fromCartToCylindricalCoordinates(hand_coordinates_hrplane)
        hand_cyl_coord_lst.append(hand_cyl_coordinates)
    else:
        hand_cyl_coord_lst.append(np.nan)
        
#saves the landmark list for each frame to a csv file in the current project directory
findregion.savelandmarkstoCSVfile(fileCompleteName, lm_lst)
findregion.savehandcoordinates(fileCompleteName, hand_cyl_coord_lst)
end=time.time()  
print(end-start)
pipeline.stop()