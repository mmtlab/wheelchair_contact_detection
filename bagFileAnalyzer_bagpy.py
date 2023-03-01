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
filename = r'D:\01_raw\T002.bag'

b = bagreader(filename)

df=b.topic_table
rgb_MSG=b.message_by_topic('/device_0/sensor_1/Color_0/image/data')
df_rgb=pd.read_csv(rgb_MSG)
