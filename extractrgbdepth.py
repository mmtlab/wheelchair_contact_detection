# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:06:31 2023

@author: giamp
"""

import numpy as np
import pyrealsense2 as rs
import cv2
import basicmaster as bm

def extractrgbdepth():
    
         
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()  
    color_image_rgb = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
        
      
    return color_image_rgb,depth_image
file=r'D:\01_raw\T001.bag'
a,b=extractrgbdepth(file)
