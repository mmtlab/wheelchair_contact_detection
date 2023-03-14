# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 09:10:05 2023

@author: giamp
"""

import hppdWC
import worklab as wl
import pandas as pd

testName='T017S03BnrC3'
raw_ergo_dir=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230309\01_raw\ergometer'
raw_mw_dir=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230309\01_raw\measurement wheel'
cam_dir=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230309\02_preprocessing\realsenseRight\handposition'
wheel_radius_mm=500

 # load ergometer data
ergofileCompleteName = hppdWC.utils.findFileInDirectory(raw_ergo_dir, testName)[0]
# simply using the function from worklab
ergo_data = wl.com.load(ergofileCompleteName)
# using only the right side of the ergometer, rename it
ergo_data = ergo_data["right"]
ergo_data['angle deg'] = hppdWC.analysis.speedToAngleDeg(ergo_data['speed'], ergo_data['time'], wheel_radius_mm)

# load measurement wheel data
MWfileCompleteName = hppdWC.utils.findFileInDirectory(raw_mw_dir, testName)[0]
# simply using the function from hppdWC
mw_data = hppdWC.load.loadMW(MWfileCompleteName)

# load realsense camera data
camfileCompleteName = hppdWC.utils.findFileInDirectory(cam_dir, testName)[0]
cam_data = pd.read_csv(camfileCompleteName).astype(float)
cam_data = hppdWC.analysis.fromAbsTimeStampToRelTime(cam_data)


    