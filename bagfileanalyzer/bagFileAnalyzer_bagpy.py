# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:47:07 2023
@author: eferlius
"""

import bagpy
from bagpy import bagreader
import pandas as pd
import cv2 as cv
import numpy as np
# https://jmscslgroup.github.io/bagpy/Reading_bagfiles_from_cloud.html
filename = r"C:\Users\giamp\OneDrive\Desktop\vid\20230510_121219.bag"

b = bagreader(filename)

df=b.topic_table
rgb_MSG=b.message_by_topic('/device_0/sensor_0/Depth_0/image/data')
#print(rgb_MSG)
df_rgb=pd.read_csv(rgb_MSG)
