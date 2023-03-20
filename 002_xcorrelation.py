# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 09:10:05 2023

@author: giamp
"""

import hppdWC
import worklab as wl
import pandas as pd
import numpy as np

testName='T017S03BnrC3'
raw_ergo_dir=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230309\01_raw\ergometer'
raw_mw_dir=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230309\01_raw\measurement wheel'
cam_dir=r'G:\Drive condivisi\Wheelchair Ergometer\handrim contact detection\Tests\20230309\02_preprocessing\realsenseRight\handposition'
#cam_dep_dir=r'G:\Drive condivisi\Wheelchair Ergometer\HPPD\Tests\20220516\03_analysis\realsenseRight\roi dep feat\handPointCloud3dFeatFirst100v1 no deprojection'
wheel_radius_mm=500
testType='sprint'
calib_ergo_fin_spr=[80, np.nan]
calib_ergo_fin_sub=[150, np.nan]
calib_ergo_ini=[10,29]
calib_cam_ini=[20,40]

SHOW_PLOT_XCORR=False

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

# load depth hand information
# camfileDepCompleteName = hppdWC.utils.findFileInDirectory(cam_dep_dir, testName)[0]
# camDep_data = pd.read_csv(camfileDepCompleteName).astype(float)
# camDep_data = hppdWC.analysis.fromAbsTimeStampToRelTime(camDep_data)


#%% compute times for xcorr
if testType == 'sprint':
    calib_ergo_fin = calib_ergo_fin_spr
elif testType == 'sub':
    calib_ergo_fin = calib_ergo_fin_sub
calib_ergo_fin[1] = max(ergo_data['time'])

#%% ergo camera sync
delay, maxError = hppdWC.analysis.syncXcorr(ergo_data['angle deg'],  - cam_data['RadAngle[rad]'], ergo_data['time'], cam_data['time'], step = 0.01, interval1 = calib_ergo_ini, interval2 = calib_cam_ini, showPlot = SHOW_PLOT_XCORR, device1  = 'ergo', device2 = 'camera', userTitle = testName)
ecini = delay

delay, maxError = hppdWC.analysis.syncXcorr(ergo_data['angle deg'],  - cam_data['RadAngle[rad]'], ergo_data['time'], cam_data['time'], step = 0.01, interval1 = calib_ergo_fin, interval2 = hppdWC.utils.addIntToList(calib_ergo_fin, -delay), showPlot = SHOW_PLOT_XCORR, device1  = 'ergo', device2 = 'camera', userTitle = testName)
ecfin = delay

#%% ergo mw sync
delay, maxError = hppdWC.analysis.syncXcorr(ergo_data['angle deg'],  mw_data['Angle deg'], ergo_data['time'], mw_data['time'], step = 0.01, interval1 = calib_ergo_ini, interval2 = calib_ergo_ini, showPlot = SHOW_PLOT_XCORR, device1  = 'ergo', device2 = 'mw', userTitle = testName, col2 = 'C2')
emini = delay

delay, maxError = hppdWC.analysis.syncXcorr(ergo_data['angle deg'],  mw_data['Angle deg'], ergo_data['time'], mw_data['time'], step = 0.01, interval1 = calib_ergo_fin, interval2 = calib_ergo_fin, showPlot = SHOW_PLOT_XCORR, device1  = 'ergo', device2 = 'mw', userTitle = testName, col2 = 'C2')
emfin = delay
#%% mw camera sync
delay, maxError = hppdWC.analysis.syncXcorr(mw_data['Angle deg'],  - cam_data['RadAngle[rad]'], mw_data['time'], cam_data['time'], step = 0.01, interval1 = calib_ergo_ini, interval2 = calib_cam_ini, showPlot = SHOW_PLOT_XCORR, device1  = 'mw', device2 = 'camera', userTitle = testName, col1 = 'C2')
mcini = delay

delay, maxError = hppdWC.analysis.syncXcorr(mw_data['Angle deg'],  - cam_data['RadAngle[rad]'], mw_data['time'], cam_data['time'], step = 0.01, interval1 = calib_ergo_fin, interval2 = hppdWC.utils.addIntToList(calib_ergo_fin, -delay), showPlot = SHOW_PLOT_XCORR, device1  = 'mw', device2 = 'camera', userTitle = testName, col1 = 'C2')
mcfin = delay
