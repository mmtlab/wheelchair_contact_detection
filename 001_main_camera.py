# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:24:09 2023

@author: giamp
"""
import numpy as np
import pyrealsense2 as rs
from hcd import coordinates
import time
import hppdWC

start=time.time()

#full path of the file expected
filecompletename=r'D:\01_raw\T017S03BnrC3r.bag' 
#usually 20000 frames for a full acquisition
number_of_frames=200
x_resolution=640
y_resolution=480
rgblist=[]
lm_lst=[]
x_hList=[]
y_hList=[]
hand_cyl_coord_lst=[]
centre_metric_avg=np.array([0,0,0])
handrimPlane_avg=np.array([0,0,0,0])
hand_coordinates_camera_lst=[]
# TODO one line loops to write the header
dir_analysis=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230309\03_analysis'
header_lm=["frame","x0","y0","x1","y1","x2","y2","x3","y3","x4","y4","x5","y5","x6","y6","x7","y7","x8","y8","x9","y9","x10","y10","x11","y11","x12","y12","x13","y13","x14","y14","x15","y15","x16","y16","x17","y17","x18","y18","x19","y19","x20","y20"]
header_hand_position=["frame","RadDistance[m]","RadAngle[rad]","NormDistance[m]"]

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
    #get the hand landmarks and stores them in lm_lst for each frame with a different configuration for better csv writing
    hand_lm=coordinates.get_hand_landmarks(color_image_rgb,x_resolution,y_resolution)
    hand_lm_buffer=coordinates.multi_array_to_mono_array(hand_lm)
    lm_lst.append(hand_lm_buffer)
    #get the hand position
    xh,yh,zh=coordinates.roi_position(hand_lm,depth_image)
    #convert the x and y coordinate from pixel to metric
    xh,yh,zh=hppdWC.geom.convert_depth_pixel_to_metric_coordinate_pp_ff(zh, xh, yh, ppx, ppy, fx, fy)
    hand_coordinates_camera=[xh,yh,zh]
    hand_coordinates_camera_lst.append(hand_coordinates_camera)
    

#transform the hand coordinates to the wheel plane frame
for j in range(len(hand_coordinates_camera_lst)):
    #check if the current list of coordinates has not nan values
    if hand_coordinates_camera_lst[j][0] != np.nan: 
        # transform
        hand_coordinates_hrplane=coordinates.changeRefFrameTR(hand_coordinates_camera_lst[j], centre_metric_avg, rot)
        #transform the hand coordinates from cartesian to cylindrical
        hand_cyl_coordinates=coordinates.from_cart_to_cylindrical_coordinates(hand_coordinates_hrplane)
        hand_cyl_coord_lst.append(hand_cyl_coordinates)
    else:
        hand_cyl_coord_lst.append(np.nan)
        # TODO check if nan is fine or requires list of three elements
#saves the landmark list for each frame to a csv file in the current project directory
coordinates.save_multilist_to_CSVfile(filecompletename, lm_lst, header_lm, 'landmark',dir_analysis)
#saves the hand coordinates list for each frame to a csv file in the current project directory
coordinates.save_multilist_to_CSVfile(filecompletename, hand_cyl_coord_lst, header_hand_position, 'handposition',dir_analysis)

end=time.time()  
#print execution time in seconds
print(end-start)

pipeline.stop()