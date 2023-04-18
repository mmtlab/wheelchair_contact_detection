# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 14:43:49 2023

@author: giamp
"""

import numpy as np
import pyrealsense2 as rs
from hcd import coordinates
from hcd import mphands
from hcd import utils
import hppdWC
import basic
import cv2

timer=basic.timer.Timer(name="001_main_camera")

#full path of the file expected
filecompletename=r'D:\01_raw\T017S03BnrC3r.bag' 
#usually 20000 frames for a full acquisition
number_of_frames=20000
x_resolution=848
y_resolution=480
threshold=100
SHOWVIDEO=True
fps=60
distance=1200
led_status_lst=[]
timestamp=[]
rgblist=[]
lm_lst=[]
x_hList=[]
y_hList=[]
hand_cyl_coord_lst=[]
centre_metric_avg=np.array([0,0,0])
handrimPlane_avg=np.array([0,0,0,0])
hand_coordinates_camera_lst=[]
video_full_path=r"D:\01_raw\video.avi"
dir_lm=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230309\02_preprocessing\realsenseRight\landmark'
dir_handposition=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230309\02_preprocessing\realsenseRight\handposition'
# TODO one line loops to write the header
header_lm=["time","x0","y0","x1","y1","x2","y2","x3","y3","x4","y4","x5","y5","x6","y6","x7","y7","x8","y8","x9","y9","x10","y10","x11","y11","x12","y12","x13","y13","x14","y14","x15","y15","x16","y16","x17","y17","x18","y18","x19","y19","x20","y20"]
header_hand_position=["time","RadDistance[m]","RadAngle[rad]","NormDistance[m]"]

video=cv2.VideoWriter(video_full_path, cv2.VideoWriter_fourcc('M','J','P','G'), 60, (x_resolution, y_resolution))
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

timer.lap(lap_name="configuration of bag file")
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
    depth_frame = frame.get_depth_frame()
    color_frame = frame.get_color_frame()
    color_image_rgb = np.asanyarray(color_frame.get_data())#transform the color frame in a RGB array
    depth_image = np.asanyarray(depth_frame.get_data())#transform the depth frame in an array
    #get the camera intrinsic parameters
    if i <100:
        # load intrinsic params of camera and find the wheel centre as well as the plane on which the wheel stands
        # extract rotation of the new frame compared to the absolute one
        camera_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        ppx = camera_intrinsics.ppx
        ppy = camera_intrinsics.ppy
        fx = camera_intrinsics.fx
        fy = camera_intrinsics.fy
        wc_img, hrc_img, centre_metric, handrimPlane, dataPlane = hppdWC.analysis.findWheelCentreAndHandrim(color_image_rgb,depth_image,ppx,ppy,fx,fy)
        
        #update the average of the wheel centre coordinates and of the handrim plane at each iteration
        centre_metric_avg=(i*centre_metric_avg+np.array(centre_metric))/(i+1)
        handrimPlane_avg=(i*handrimPlane_avg+np.array([handrimPlane.a,handrimPlane.b,handrimPlane.c,handrimPlane.d]))/(i+1)
        if i==99:
            handrimPlane_avg=hppdWC.geom.Plane3d(a=handrimPlane_avg[0],b=handrimPlane_avg[1],c=handrimPlane_avg[2],d=handrimPlane_avg[3])
            rot, rmsd, sens=hppdWC.geom.rotMatrixToFitPlaneWRTimg(handrimPlane_avg,centre_metric_avg)
            timer.elap("time to find the handrim plane and centre")
    #turn every pixel further than 1200 to black
    color_image_rgb[depth_image > distance] = [0,0,0]
    #get the hand landmarks and stores them in lm_lst for each frame with a different configuration for better csv writing
    hand_lm=coordinates.get_hand_landmarks(color_image_rgb,x_resolution,y_resolution)       
    hand_lm_buffer=coordinates.multi_array_to_mono_array(hand_lm)
    hand_lm_buffer.insert(0,timestamp_s)
    lm_lst.append(hand_lm_buffer)
    #get the hand position
    xh,yh,zh=coordinates.roi_position(hand_lm,depth_image)
    
    img=color_image_rgb
    centre=(xh,yh)
    if xh is np.nan:
        centre=(0,0)           
    cv2.circle(img,centre, 20, (0,0,255), -1)
    cv2.putText(img=img, text=str(i/60)+ "frame: "+ str(i), org=(50,40), fontFace=cv2.FONT_HERSHEY_PLAIN , fontScale=3, color=(255, 255, 255),thickness=3)
    video.write(img)
        
    #convert the x and y coordinate from pixel to metric
    xh,yh,zh=hppdWC.geom.convert_depth_pixel_to_metric_coordinate_pp_ff(zh, xh, yh, ppx, ppy, fx, fy)
    hand_coordinates_camera=[xh,yh,zh]
    hand_coordinates_camera_lst.append(hand_coordinates_camera)
    print(i)
video.release()