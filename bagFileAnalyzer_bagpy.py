# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:47:07 2023

@author: eferlius
"""

import bagpy
from bagpy import bagreader
# https://jmscslgroup.github.io/bagpy/Reading_bagfiles_from_cloud.html
filename = r'C:\Users\eferlius\Desktop\20230217_133916.bag'

b = bagreader(filename)

b.topic_table