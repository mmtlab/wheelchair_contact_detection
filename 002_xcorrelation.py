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
from hcd import xcorrelation

testName='T017S03BnrC3'
raw_ergo_dir=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230309\01_raw\ergometer'
raw_mw_dir=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230309\01_raw\measurement wheel'
cam_dir=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230309\02_preprocessing\realsenseRight\handposition'
wheel_radius_mm=320
testType='sprint'
calib_ergo_fin_spr=[75, np.nan]
calib_ergo_fin_sub=[150, np.nan]
calib_ergo_ini=[10,30]
calib_cam_ini=[20,40]
SHOW_PLOT_XCORR=True
CSVfile=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230309\03_analysis\xcorr delay.csv'
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

#%% compute times for xcorr
if testType == 'sprint':
    calib_ergo_fin = calib_ergo_fin_spr
elif testType == 'sub':
    calib_ergo_fin = calib_ergo_fin_sub
calib_ergo_fin[1] = max(ergo_data['time'])

#%% ergo camera sync

results=xcorrelation.syncXcorr(ergo_data['angle deg'], - cam_data['RadAngle[rad]'], ergo_data['time'], cam_data['time'], step = 0.01, interval1 = calib_ergo_ini, interval2 = calib_cam_ini)
ecini=results[6]
if SHOW_PLOT_XCORR==True:
    xcorrelation.plot_syncXcorr(results,device1='ergo',device2='cam')

results=xcorrelation.syncXcorr(ergo_data['angle deg'],  - cam_data['RadAngle[rad]'], ergo_data['time'], cam_data['time'], step = 0.01, interval1 = calib_ergo_fin, interval2 = hppdWC.utils.addIntToList(calib_ergo_fin, -ecini))
ecfin=results[6]
if SHOW_PLOT_XCORR==True:
    xcorrelation.plot_syncXcorr(results,device1='ergo',device2='cam')
#%% ergo mw sync

results=xcorrelation.syncXcorr(ergo_data['angle deg'], mw_data['Angle deg'], ergo_data['time'], mw_data['time'], step = 0.01, interval1 = calib_ergo_ini, interval2 = calib_ergo_ini)
emini=results[6]
if SHOW_PLOT_XCORR==True:
    xcorrelation.plot_syncXcorr(results,device1='ergo',device2='mw')

results=xcorrelation.syncXcorr(ergo_data['angle deg'], mw_data['Angle deg'], ergo_data['time'], mw_data['time'], step = 0.01, interval1 = calib_ergo_fin, interval2 =hppdWC.utils.addIntToList(calib_ergo_fin, -emini))
emfin=results[6]
if SHOW_PLOT_XCORR==True:
    xcorrelation.plot_syncXcorr(results,device1='ergo',device2='mw')
#%% mw camera sync

results=xcorrelation.syncXcorr(mw_data['Angle deg'], - cam_data['RadAngle[rad]'], mw_data['time'], cam_data['time'], step = 0.01, interval1 = calib_ergo_ini, interval2 = calib_cam_ini)
mcini=results[6]
if SHOW_PLOT_XCORR==True:
    xcorrelation.plot_syncXcorr(results,device1='mw',device2='cam')

results=xcorrelation.syncXcorr(mw_data['Angle deg'], - cam_data['RadAngle[rad]'], mw_data['time'], cam_data['time'], step = 0.01, interval1 = calib_ergo_fin, interval2 = hppdWC.utils.addIntToList(calib_ergo_fin, -mcini))
mcfin=results[6]
if SHOW_PLOT_XCORR==True:
    xcorrelation.plot_syncXcorr(results,device1='mw',device2='cam')
basic.utils.write_row_csv(CSVfile,[testName,ecini,ecfin,emini,emfin,mcini,mcfin])
