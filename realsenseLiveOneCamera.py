# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:13:46 2023

@author: eferlius

shows in real time the stream of one realsense camera, 
if saveInBag is True, saves in a bag file
pressing "s" during the stream, rgb and depth image are saved
depth image might appear completely black: it's not rgb but contains values of 
distance in mm
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import keyboard
import basic
import datetime

# ------------------------------ #
# FLAGS
saveInBag = True
display = False
# ------------------------------ #
# PARAMETERS
# eventual filename for saving
thisMoment = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H-%M-%S')
filename = 'test' + thisMoment + '.bag'
# camera parameters
depth_freq = 60
rgb_freq = 60
w = 848
h = 480
# to write on the image
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2
# ------------------------------ #

#loading the cameras
ctx = rs.context()
devices = ctx.query_devices()

# just to show how many devices are connected
for dev in devices:
    print(dev)
    
serialNumberList = []
for dev in devices:
    # strings are required
    serialNumberList.append(dev.get_info(rs.camera_info.serial_number))

# from now on, only the first device is considered
# create empty lists
pipelines = []
configs = []


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(serialNumberList[0])
aligned_stream = rs.align(rs.stream.color)
config.enable_stream(rs.stream.depth, w, h, rs.format.z16, depth_freq)
config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, rgb_freq)

if saveInBag:
    config.enable_record_to_file(filename)


# Start streaming
pipeline.start(config)

frameCounter = 0
elapsed = []
timer = basic.timer.Timer("timer for loop execution of realsense")

try:
    while True:
        
        frameCounter = frameCounter + 1
        
        # Camera 1
        # Wait for a coherent pair of frames: depth and color
        frame = pipeline.wait_for_frames()
        # alignement of the frames: the obtained resolution is the one of the rgb image
        frame = aligned_stream.process(frame)
        depth_frame = frame.get_depth_frame()
        color_frame = frame.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arraysq
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # timing execution
        # time elapsed with respect to the previous execution
        lap = timer.lap(printTime = False)
        elapsed.append(lap)
        
        if display:
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
    
          
            # Stack all images horizontally
            images = np.vstack((color_image, depth_colormap))
            
            # Show images from both cameras
            cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
            

            # writes on the image a string with timing information
            string = ('frame: ' + str(frameCounter) + ' // elapsed: ' + "%08.2f" %(lap*1000) + "[ms]")
            images = cv2.putText(images, string, org, font, fontScale, color, thickness, cv2.LINE_AA)
            
            # display the image
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)
        
        # Saves images and depth maps from both cameras by pressing 's'
        if keyboard.is_pressed("s"):
            cv2.imwrite("realsenseLive_rgb.png",color_image)
            cv2.imwrite("realsenseLive_dep.png",depth_image)
            print("Save")
        # Quits the loop by pressing 'q'    
        if keyboard.is_pressed("q"):
            # total timing execution
            elapsedTotal = timer.stop(printTime = False)
            cv2.destroyAllWindows()
            break


finally:
    # Stop streaming
    pipeline.stop()


depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)
basic.plots.pltsImg([color_image,depth_image,depth_colormap],ncols = 1, 
                    listTitles=['rgb image', 'depth image', 'rendering of depth image'])
basic.plots.plts([],[elapsed], mainTitle= 'average frequency: {:.2f} hZ'.format(frameCounter/elapsedTotal),
                 listXlabels=['execution number'], listYlabels=['elapsed time [s]'])

    
