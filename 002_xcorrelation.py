# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 09:10:05 2023

@author: giamp
"""
import hppdWC
import worklab as wl
import pandas as pd
import numpy as np
import basic
import matplotlib.pyplot as plt
from hcd import xcorrelation
from hcd import capsleeve

testName='T026Subject2S1R1NR_run0'
# raw_ergo_dir=r'D:\.shortcut-targets-by-id\1MjZH9XN1gWNoiyVyaLwpjTjfKx4AsRiw\handrim contact detection\Tests\20230519\01_raw\ergometer'
raw_ergo_dir=r"G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230519\01_raw\ergometer"
#raw_mw_dir=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230309\01_raw\measurement wheel'
# raw_mw_dir=r'C:/Users/Alexk/OneDrive/Documenten/School/RUG/jaar 3 bewegingswetenschappen(2022-2023)/BAP/Measurement wheel'
# cam_dir=r"D:\.shortcut-targets-by-id\1MjZH9XN1gWNoiyVyaLwpjTjfKx4AsRiw\handrim contact detection\Tests\20230519\02_preprocessing\realsenseRight\handposition"
cam_dir=r"G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230519\02_preprocessing\realsenseRight\handposition"
# cam_dir=r'C:/Users/Alexk/OneDrive/Documenten/School/RUG/jaar 3 bewegingswetenschappen(2022-2023)/BAP/Handposition'
# cap_dir=r'D:\.shortcut-targets-by-id\1MjZH9XN1gWNoiyVyaLwpjTjfKx4AsRiw\handrim contact detection\Tests\20230519\01_raw\capacitive sleeve'
cap_dir=r"G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230519\01_raw\capacitive sleeve"
# led_dir=r'D:\.shortcut-targets-by-id\1MjZH9XN1gWNoiyVyaLwpjTjfKx4AsRiw\handrim contact detection\Tests\20230519\02_preprocessing\realsenseRight\led status'
led_dir=r"G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230519\02_preprocessing\realsenseRight\led status"
wheel_radius_mm=317.5
testType='sub'
calib_ergo_fin_spr=[40, 50]
calib_ergo_fin_sub=[105, 118]
calib_ergo_ini=[0,30]
calib_cam_ini=[0,30]
calib_cap_ini=[0,10]
calib_cap_fin_spr=[50,60]
calib_cap_fin_sub=[100,105]
SHOW_PLOT_XCORR=True
CSVfile=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230519\03_analysis\xcorr delay.csv'
# CSVfile=r'C:\Users\Alexk\OneDrive\Documenten\School\RUG\jaar 3 bewegingswetenschappen(2022-2023)\BAP\Delayfiles\DelayData.csv'
# load ergometer data
ergofileCompleteName = hppdWC.utils.findFileInDirectory(raw_ergo_dir, testName)[0]
# simply using the function from worklab
ergo_data = wl.com.load(ergofileCompleteName)
# using only the right side of the ergometer, rename it
ergo_data = ergo_data["right"]
ergo_data['angle deg'] = hppdWC.analysis.speedToAngleDeg(ergo_data['speed'], ergo_data['time'], wheel_radius_mm)

#load capacitive sleeve data
capsleevefileCompleteName=hppdWC.utils.findFileInDirectory(cap_dir, testName)[0]
cap_data = pd.read_csv(capsleevefileCompleteName + r'\NGIMU - 0035EEA3\sensors.csv')

#load led status data
ledfileCompleteName = hppdWC.utils.findFileInDirectory(led_dir, testName)[0]
led_data=pd.read_csv(ledfileCompleteName)

# load realsense camera data
camfileCompleteName = hppdWC.utils.findFileInDirectory(cam_dir, testName)[0]
cam_data = pd.read_csv(camfileCompleteName).astype(float)
cam_data = hppdWC.analysis.fromAbsTimeStampToRelTime(cam_data)

#%% compute times for xcorr
if testType == 'sprint':
    calib_ergo_fin = calib_ergo_fin_spr
    calib_cap_fin = calib_cap_fin_spr
elif testType == 'sub':
    calib_ergo_fin = calib_ergo_fin_sub
    calib_cap_fin = calib_cap_fin_sub


#%% ergo camera sync

results=xcorrelation.syncXcorr(ergo_data['angle deg'],  -cam_data['RadAngle[rad]'], ergo_data['time'], cam_data['time'], step = 0.01, interval1 = calib_ergo_ini, interval2 = calib_cam_ini)
ecini=results[6]
if SHOW_PLOT_XCORR==True:
    xcorrelation.plot_syncXcorr(results,device1='ergo',device2='cam')
results=xcorrelation.syncXcorr(ergo_data['angle deg'],  - cam_data['RadAngle[rad]'], ergo_data['time'], cam_data['time'], step = 0.01, interval1 = calib_ergo_fin, interval2 = hppdWC.utils.addIntToList(calib_ergo_fin, -ecini))
ecfin=results[6]
if SHOW_PLOT_XCORR==True:
    xcorrelation.plot_syncXcorr(results,device1='ergo',device2='cam')
#%% camera capacitve sleeve sync

csini=xcorrelation.syncCameraCapsleeve(led_data, cap_data)
if SHOW_PLOT_XCORR==True:
    xcorrelation.plotSyncedCameraCapsleeve(cap_data, led_data, csini)

#%% ergo capacitive sleeve sync
esini=ecini-csini
# esfin=ecfin-csfin
#basic.utils.write_row_csv(CSVfile,[testName,ecini,ecfin,csini,esini])
