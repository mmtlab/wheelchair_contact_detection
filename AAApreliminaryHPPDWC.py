# -*- coding: utf-8 -*-
"""
copy paste the imports and import the rest using
import AAApreliminaryHPPDWC
"""
#%% imports of all: add here in order to
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import os
import sys
import cv2
import worklab as wl
import scipy
import logging
import csv
from tabulate import tabulate

# if using jupyter lab, uncomment this part
# insert at 1, 0 is the script path (or '' in REPL)
# import sys
# sys.path.insert(1, r'G:\Shared drives\Wheelchair Ergometer\HPPD\Software\Python\HPPD')
import hppdWC

#%% logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(lineno)-4d] %(filename)s \n>>> %(message)s', \
                    datefmt='%Y-%m-%d %H:%M:%S', level = logging.INFO, force = True, filename = 'logger.txt')
logging.info('this is an example')

#%% plot configurations
plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
plt.rcParams["figure.figsize"] = (24,10)

# paper format
plt.rc('axes', titlesize=22) # for each axes title in subplots
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('font', size=22) # for main title in subplots
# plt.rc('legend', fontsize = 22)
plt.rcParams.update({'font.size': 22})

# # poster format
# plt.rc('font', size=26) # for main title in subplots
# plt.rc('axes', titlesize=26) # for each axes title in subplots
# plt.rc('axes', labelsize=22)
# plt.rc('xtick', labelsize=22)
# plt.rc('ytick', labelsize=22)

# presentation format
plt.rc('font', size=28) # for main title in subplots
plt.rc('axes', titlesize=28) # for each axes title in subplots
plt.rc('axes', labelsize=24)
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)


#%% create empty dictionary to store all the variables
DICT = {}
#%% parameters
DICT['VALID TESTS'] = np.concatenate((np.arange(13,37,1), np.arange(43,82,1), np.arange(83,130,1)))

# excluded since the wheel is moving a lot on the ergo and makes the torque oscillate a lot
# 73 wrong center wheel detection
# 81,83 due to wheelchair moving a lot on the ergometer -> torque difficult to detect
DICT['INVALID TESTS'] = [73,81,83]

# division of test according to subject, block and condition
DICT['TESTS'] = {}
DICT['TESTS']['S03'] = [13, 14, 15, 16, 17, 18]
DICT['TESTS']['S04'] = [19, 20, 21, 22, 23, 24]
DICT['TESTS']['S05'] = [25, 26, 27, 28, 29, 30]
DICT['TESTS']['S06'] = [31, 32, 33, 34, 35, 36]
DICT['TESTS']['S08'] = [43, 44, 45, 46, 47, 48]
DICT['TESTS']['S09'] = [49, 50, 51, 52, 53, 54]
DICT['TESTS']['S10'] = [55, 56, 57, 58, 59, 60]
DICT['TESTS']['S11'] = [61, 62, 63, 64, 65, 66]
DICT['TESTS']['S12'] = [67, 68, 69, 70, 71, 72]
DICT['TESTS']['S13'] = [73, 74, 75, 76, 77, 78]
DICT['TESTS']['S14'] = [79, 80, 81, 83, 84]
DICT['TESTS']['S15'] = [85, 86, 87, 88, 89, 90]
DICT['TESTS']['S16'] = [91, 92, 93, 94, 95, 96, 97, 98, 99]
DICT['TESTS']['S17'] = [100, 101, 102, 103, 104, 105]
DICT['TESTS']['S18'] = [106, 107, 108, 109, 110, 111]
DICT['TESTS']['S19'] = [112, 113, 114, 115, 116, 117]
DICT['TESTS']['S21'] = [118, 119, 120, 121, 122, 123]
DICT['TESTS']['S22'] = [124, 125, 126, 127, 128, 129]

DICT['TESTS']['D01'] =    DICT['TESTS']['S03'] \
                        + DICT['TESTS']['S04'] \
                        + DICT['TESTS']['S05'] \
                        + DICT['TESTS']['S06']
DICT['TESTS']['D02'] =    DICT['TESTS']['S08'] \
                        + DICT['TESTS']['S09'] \
                        + DICT['TESTS']['S10']
DICT['TESTS']['D03'] =    DICT['TESTS']['S11'] \
                        + DICT['TESTS']['S12'] \
                        + DICT['TESTS']['S13'] \
                        + DICT['TESTS']['S14']
DICT['TESTS']['D04'] =    DICT['TESTS']['S15'] \
                        + DICT['TESTS']['S16'] \
                        + DICT['TESTS']['S17'] \
                        + DICT['TESTS']['S18']
DICT['TESTS']['D05'] =    DICT['TESTS']['S19'] \
                        + DICT['TESTS']['S21'] \
                        + DICT['TESTS']['S22']
DICT['TESTS']['F'] = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 43, 44, 45, 46, 47, 48, 61, 62, 63, 64, 65, 66, 79, 80, 81, 83, 84, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]
DICT['TESTS']['M'] = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
DICT['TESTS']['Bnr'] = [16, 17, 18, 22, 23, 24, 25, 26, 27, 31, 32, 33, 46, 47, 48, 52, 53, 54, 58, 59, 60, 64, 65, 66, 67, 68, 69, 76, 77, 78, 79, 80, 81, 88, 89, 90, 91, 92, 93, 97, 98, 99, 103, 104, 105, 109, 110, 111, 112, 113, 114, 118, 119, 120, 124, 125, 126]
DICT['TESTS']['Byr'] = [13, 14, 15, 19, 20, 21, 28, 29, 30, 34, 35, 36, 43, 44, 45, 49, 50, 51, 55, 56, 57, 61, 62, 63, 70, 71, 72, 73, 74, 75, 83, 84, 85, 86, 87, 94, 95, 96, 100, 101, 102, 106, 107, 108, 115, 116, 117, 121, 122, 123, 127, 128, 129]
DICT['TESTS']['C1'] = [13, 16, 19, 24, 25, 28, 31, 34, 43, 46, 49, 52, 56, 58, 61, 64, 69, 71, 75, 76, 79, 87, 88, 92, 96, 98, 101, 103, 108, 109, 112, 115, 118, 121, 126, 128]
DICT['TESTS']['C2'] = [15, 18, 20, 22, 27, 30, 32, 35, 44, 48, 51, 54, 55, 59, 63, 65, 67, 70, 73, 78, 80, 83, 85, 90, 91, 94, 97, 100, 104, 106, 111, 113, 117, 120, 123, 124, 127]
DICT['TESTS']['C1C2'] = np.sort(DICT['TESTS']['C1'] + DICT['TESTS']['C2']).tolist()
DICT['TESTS']['C3'] = [14, 17, 21, 23, 26, 29, 33, 36, 45, 47, 50, 53, 57, 60, 62, 66, 68, 72, 74, 77, 81, 84, 86, 89, 93, 95, 99, 102, 105, 107, 110, 114, 116, 119, 122, 125, 129]
DICT['TESTS']['BnrC1'] = [16, 24, 25, 31, 46, 52, 58, 64, 69, 76, 79, 88, 92, 98, 103, 109, 112, 118, 126]
DICT['TESTS']['BnrC2'] = [18, 22, 27, 32, 48, 54, 59, 65, 67, 78, 80, 90, 91, 97, 104, 111, 113, 120, 124]
DICT['TESTS']['BnrC1C2'] = np.sort(DICT['TESTS']['BnrC1'] + DICT['TESTS']['BnrC2']).tolist()
DICT['TESTS']['BnrC3'] = [17, 23, 26, 33, 47, 53, 60, 66, 68, 77, 81, 89, 93, 99, 105, 110, 114, 119, 125]
DICT['TESTS']['ByrC1'] = [13, 19, 28, 34, 43, 49, 56, 61, 71, 75, 87, 96, 101, 108, 115, 121, 128]
DICT['TESTS']['ByrC2'] = [15, 20, 30, 35, 44, 51, 55, 63, 70, 73, 83, 85, 94, 100, 106, 117, 123, 127]
DICT['TESTS']['ByrC1C2'] = np.sort(DICT['TESTS']['ByrC1'] + DICT['TESTS']['ByrC2']).tolist()
DICT['TESTS']['ByrC3'] = [14, 21, 29, 36, 45, 50, 57, 62, 72, 74, 84, 86, 95, 102, 107, 116, 122, 129]

LIST_DATABASE_SPLIT = [DICT['TESTS']['Bnr'],
                       DICT['TESTS']['Byr'],
                       DICT['TESTS']['C1'],
                       DICT['TESTS']['C2'],
                       DICT['TESTS']['C1C2'],
                       DICT['TESTS']['C3'],
                       DICT['TESTS']['BnrC1'],
                       DICT['TESTS']['BnrC2'],
                       DICT['TESTS']['BnrC1C2'],
                       DICT['TESTS']['BnrC3'],
                       DICT['TESTS']['ByrC1'],
                       DICT['TESTS']['ByrC2'],
                       DICT['TESTS']['ByrC1C2'],
                       DICT['TESTS']['ByrC3'],
                       DICT['VALID TESTS'].tolist()]

LIST_DATABASE_SPLIT_NAMES = ['Bnr','Byr','C1','C2','C1C2','C3',
                             'BnrC1','BnrC2','BnrC1C2','BnrC3',
                             'ByrC1','ByrC2','ByrC1C2','ByrC3', 'all']

LIST_DATABASE_SPLIT_DAYS = [DICT['TESTS']['D01'],
                            DICT['TESTS']['D02'],
                            DICT['TESTS']['D03'],
                            DICT['TESTS']['D04'],
                            DICT['TESTS']['D05']]

LIST_DATABASE_SPLIT_DAYS_NAMES = ['D01', 'D02', 'D03', 'D04', 'D05']


DICT['MIN_DET'] = 0.1
minDetString = str("{:02d}".format(int(DICT['MIN_DET']*10)))

DICT['NOM_FREQ'] = {}
DICT['NOM_FREQ']['ergo'] = 100
DICT['NOM_FREQ']['mw'] = 200
DICT['NOM_FREQ']['cam'] = 60

DICT['IMG'] = {}
DICT['IMG']['height'] = 480
DICT['IMG']['width'] = 848

DICT['MEAS'] = {}
DICT['MEAS']['radius handrim m'] = 0.28
DICT['MEAS']['radius wheel m'] = 0.32
DICT['MEAS']['radius handrim mm'] = 280
DICT['MEAS']['radius wheel mm'] = 320

# calibration
DICT['CALI'] = {}
DICT['CALI']['ergo ini'] = [10, 29]
# so you remember to put the max time of the recording
DICT['CALI']['ergo fin sub'] = [150, np.nan]
DICT['CALI']['ergo fin spr'] = [80, np.nan]
DICT['CALI']['cam ini'] = [20, 40]

# test
DICT['TEST'] = {}
DICT['TEST']['spr'] = [30, 40]
DICT['TEST']['sub'] = [30, 120]

#%% directories
DICT['DIR'] = {}
# select where are all the files saved in
DICT['DIR']['ABS DIR'] = r'G:\Shared drives\Wheelchair Ergometer\HPPD\Tests'
# which test session is analyzed?
DICT['DIR']['this test DIR'] = r'20220516'

# directories
DICT['DIR']['raw ergo DIR'] = r'01_raw\ergometer'
DICT['DIR']['raw mw DIR'] = r'01_raw\measurement wheel'
DICT['DIR']['raw cam DIR'] = r'01_raw\realsenseRight\bag files'
DICT['DIR']['png of frames DIR'] = r'02_preprocessing\realsenseRight\bag converted to png'
DICT['DIR']['cam DIR'] = r'02_preprocessing\realsenseRight\hand detection mediapipe\minDet'+minDetString

DICT['DIR']['ref frame DIR'] = r'03_analysis\realsenseRight\wheel centre\first 100 frames'
DICT['DIR']['roi coord DIR'] = r'03_analysis\realsenseRight\roi coord\roi coord '+minDetString
DICT['DIR']['hand xyz metric coord DIR'] = r'03_analysis\realsenseRight\hand position 3D\whole recording\xyz coord metric'
DICT['DIR']['hand roi metric coord DIR'] = r'03_analysis\realsenseRight\hand position 3D\whole recording\roi coord metric'
DICT['DIR']['hand roi wheelframe coord DIR'] = r'03_analysis\realsenseRight\hand position 3D\whole recording\roi coord wheelframe'
DICT['DIR']['landmark single coord DIR'] = r'03_analysis\realsenseRight\hand position 3D\whole recording\landmark single'
DICT['DIR']['landmark aorund DIR'] = r'03_analysis\realsenseRight\hand position 3D\whole recording\landmark around'
DICT['DIR']['point cloud DIR'] = r'03_analysis\realsenseRight\hand position 3D\whole recording\point cloud'

DICT['DIR']['hand xyz metric coord ONLY TEST DIR'] = r'03_analysis\realsenseRight\hand position 3D\only test sync w ergo\xyz coord metric'
DICT['DIR']['hand roi metric coord ONLY TEST DIR'] = r'03_analysis\realsenseRight\hand position 3D\only test sync w ergo\roi coord metric'
DICT['DIR']['hand roi wheelframe coord ONLY TEST DIR'] = r'03_analysis\realsenseRight\hand position 3D\only test sync w ergo\roi coord wheelframe'
DICT['DIR']['landmark single coord ONLY TEST DIR'] = r'03_analysis\realsenseRight\hand position 3D\only test sync w ergo\landmark single'
DICT['DIR']['landmark aorund ONLY TEST DIR'] = r'03_analysis\realsenseRight\hand position 3D\only test sync w ergo\landmark around'
DICT['DIR']['point cloud ONLY TEST DIR'] = r'03_analysis\realsenseRight\hand position 3D\only test sync w ergo\point cloud'
DICT['DIR']['contactYN ONLY TEST DIR'] = r'03_analysis\realsenseRight\hand position 3D\only test sync w ergo\contactYN'

DICT['DIR']['cam depth in roi DIR'] = r'03_analysis\realsenseRight\roi dep feat\handPointCloud3dFeatFirst100v1 no deprojection'
DICT['DIR']['cam depth with ground truth contact'] = r'03_analysis\realsenseRight\roi dep feat\handPointCloud3dFeatFirst100v1 no deprojection with contact'
DICT['DIR']['hand features DIR'] = r'03_analysis\realsenseRight\roi dep feat\handFeatures'
DICT['DIR']['ML databases DIR'] = r'03_analysis\realsenseRight\ML databases' #used before deprojection
DICT['DIR']['hand databases DIR'] = r'03_analysis\realsenseRight\hand position 3D\only test sync w ergo\ML databases'

DICT['DIR']['joint angles DIR'] = r'03_analysis\realsenseRight\joint angles ONLY TEST'
DICT['DIR']['contact time ergo DIR'] = r'03_analysis\ergometer\transition time'

# files
DICT['DIR']['test recording FILE'] = r'00_protocols\test recording.csv'
DICT['DIR']['part recording FILE'] = r'00_protocols\test participant data.csv'
DICT['DIR']['test info FILE'] = r'03_analysis\test information.csv'
DICT['DIR']['sync xcorr FILE'] = r'03_analysis\xcorr values.csv'
DICT['DIR']['contact det param FILE'] = r'03_analysis\ergometer\contact det.csv'
DICT['DIR']['ML models benchmark FILE'] = r'03_analysis\realsenseRight\ML models\modelsBenchMark.csv'
DICT['DIR']['ML database benchmark FILE'] = r'03_analysis\realsenseRight\ML models\databaseBenchMark.csv'
DICT['DIR']['ML database crossed benchmark FILE'] = r'03_analysis\realsenseRight\ML models\databaseCrossedBenchMarkALLDAYS.csv'
DICT['DIR']['ML database benchmark train all FILE'] = r'03_analysis\realsenseRight\ML models\databaseCrossedBenchMarkALLDAYS_TRAINALL.csv'
DICT['DIR']['ML database find outliers FILE'] = r'03_analysis\realsenseRight\ML models\findOutlier.csv'

DICT['DIR']['results post proc FILE'] = r'04_results\post processing and mean error\postProc.csv'
DICT['DIR']['results start end mean FILE'] = r'04_results\post processing and mean error\startDiff_endDiff_means.csv'
DICT['DIR']['results start end FILE'] = r'04_results\post processing and mean error\startDiff_endDiff.csv'

# images
DICT['DIR']['img saved rsr DIR'] = r'03_analysis\realsenseRight\savedImages'
DICT['DIR']['img timing rec DIR'] = r'04_results\timing of recording'
DICT['DIR']['img xcorr ergo cam DIR'] = r'04_results\synchronization\ergo-camera xCorr'
DICT['DIR']['img xcorr ergo mw DIR'] = r'04_results\synchronization\ergo-mw xCorr'
DICT['DIR']['img xcorr mw cam DIR'] = r'04_results\synchronization\mw-camera xCorr'
DICT['DIR']['img xcorr ergo mw complete DIR'] = r'04_results\synchronization\ergo-mw complete'
DICT['DIR']['img xcorr tot DIR'] = r'04_results\synchronization\tot'
DICT['DIR']['img joint angles DIR'] = r'04_results\joint angles'
DICT['DIR']['img handrim ref frame DIR'] = r'04_results\handrim ref frame'
DICT['DIR']['img hand centre ref handrim DIR'] = r'03_analysis\realsenseRight\wheel centre\scatter hand centre on handrim plane'
DICT['DIR']['img ergo contact DIR'] = r'04_results\contact detection on ergo'
# used before deprojection:
DICT['DIR']['img feat ML DIR'] = r'04_results\cam dep feat\features on time'
DICT['DIR']['img feat hist DIR'] = r'04_results\cam dep feat\histograms of features'
DICT['DIR']['img feat scatter DIR'] = r'04_results\cam dep feat\scatter of features'
DICT['DIR']['img raw and filt feat DIR'] = r'04_results\cam dep feat\raw and filtered feat'
DICT['DIR']['img raw feat DIR'] = r'04_results\cam dep feat\raw data'
DICT['DIR']['img cos sin DIR'] = r'04_results\cam angle est feat\features on time'
DICT['DIR']['img circles DIR'] = r'04_results\cam angle est feat\features on time'
# used after deprojection:
DICT['DIR']['img circle time DIR'] = r'04_results\features\circle\time'
DICT['DIR']['img circle hist DIR'] = r'04_results\features\circle\hist'
DICT['DIR']['img cossine time DIR'] = r'04_results\features\cosine and sine\time'
DICT['DIR']['img cossine hist DIR'] = r'04_results\features\cosine and sine\hist'
DICT['DIR']['img hand time DIR'] = r'04_results\features\hand coord\time'
DICT['DIR']['img hand hist DIR'] = r'04_results\features\hand coord\hist'
DICT['DIR']['img hand ortproj DIR'] = r'04_results\features\hand coord\ort proj'
DICT['DIR']['img hand raw and filt DIR'] = r'04_results\features\hand coord\raw and filt comparison'
DICT['DIR']['img hand scat DIR'] = r'04_results\features\hand coord\scatter'
DICT['DIR']['img feat database DIR'] = r'04_results\features\scatter of databases'
DICT['DIR']['img post process filt DIR'] = r'04_results\post processing'
DICT['DIR']['img mean error start end DIR'] = r'04_results\mean error in detection'


# videos
DICT['DIR']['right video raw DIR'] = r'03_analysis\realsenseRight\videos\raw'
DICT['DIR']['rigth video ref frame DIR'] = r'03_analysis\realsenseRight\videos\roi and ref frame first 100'
DICT['DIR']['rigth video min Det 01 DIR'] = r'03_analysis\realsenseRight\videos\minDet01'


# classifiers
DICT['DIR']['class NN DIR'] = r'04_results\classifiers\NN_10_100_10'


#%% automatically creating the absolute path for every path
DICT['DIR']['ABS DIR this test'] = os.path.join(DICT['DIR']['ABS DIR'], DICT['DIR']['this test DIR'])
DICT['DIR'].pop('this test DIR') # remove the non abs directory

# make all the path absolutes
tmp = {}
for path in DICT['DIR']:
    if not 'ABS' in path:
        tmp[path] = os.path.join(DICT['DIR']['ABS DIR this test'], DICT['DIR'][path])
    else:
        tmp[path] = DICT['DIR'][path]
DICT['DIR'] = tmp
#%% camera intrinsic parameters
ppx, ppy, fx, fy = 430.432, 246.571, 612.581, 612.295

#%% functions
def makeList(maybeAList):
    if isinstance(maybeAList, list):
        return maybeAList
    else:
        forSureAList = [maybeAList]
        return forSureAList

def validatefileNumber(testNum, allValidFileNumbers = DICT['VALID TESTS']):
    if testNum == 0:
        validFileNumber = allValidFileNumbers
    else:
        testNumList = makeList(testNum)
        validFileNumber = []
        for testNum in testNumList:
            if testNum in allValidFileNumbers:
                validFileNumber.append(testNum)
    return validFileNumber

def manualCorrections(testNum, mw_data, ergo_data, cam_data_s):
    #TODO
    # correction of mw
    if testNum == 61:
        mw_data = hppdWC.utils.cropDFInTime(mw_data, 40, max(mw_data['time']), resetZero = True)
    if testNum == 100:
        mw_data = hppdWC.utils.cropDFInTime(mw_data, 90, max(mw_data['time']), resetZero = True)

    if testNum == 31:
        pass
    return mw_data, ergo_data, cam_data_s


def loadErgoMWCam(testNum, syncValues = None, testStartStop = None, resetIndex = False):
    '''
    Given test number, load the data of:
        - cam
        - cam dep
        - ergo
        - mw
    If syncValues are given, alignes in time the 4 dataframes with t=0 of ergo
    If testStartStop are given, crops the 4 dataframes only for test
    

    Parameters
    ----------
    testNum : int
        test number.
    syncValues : list of 2(or 3 floats), optional
        delay ergo-cam, delay ergo-mw, delay mw-cam. 
        since synch is made according to ergo time, delay mw-cam is not used
        The default is None -> the arrays are not starting from the same 0 (the one of ergo)
    testStartStop : list of 2 floats, optional
        when the test started and stopped in the ergo time. 
        The default is None -> the arrays are not cropped inside test
    resetIndex : bool, optional
        If true, when cropping, also reset the indexes. The default is False.

    Returns
    -------
    cam data
        DESCRIPTION.
    camDep data
        DESCRIPTION.
    mw data
        DESCRIPTION.
    ergo data (of the right side)
        DESCRIPTION.

    '''
    testName = "T{:03d}".format(testNum)

    raw_ergo_dir  = DICT['DIR']['raw ergo DIR']
    raw_mw_dir  = DICT['DIR']['raw mw DIR']
    cam_dir  = DICT['DIR']['cam DIR']
    cam_dep_dir = DICT['DIR']['cam depth in roi DIR']

    # load ergometer data
    ergofileCompleteName = hppdWC.utils.findFileInDirectory(raw_ergo_dir, testName)[0]
    # simply using the function from worklab
    ergo_data = wl.com.load(ergofileCompleteName)
    ergo_data = wl.kin.filter_ergo(ergo_data)
    ergo_data = wl.kin.process_ergo(ergo_data, wheelsize = DICT['MEAS']['radius wheel m'], rimsize = DICT['MEAS']['radius handrim m'])
    # using only the right side of the ergometer, rename it
    ergo_data = ergo_data["right"]
    ergo_data['angle deg'] = ergo_data['angle'] * 180 / np.pi

    # load measurement wheel data
    MWfileCompleteName = hppdWC.utils.findFileInDirectory(raw_mw_dir, testName)[0]
    # simply using the function from hppdWC
    mw_data = hppdWC.load.loadMW(MWfileCompleteName)
    if testNum == 61:
        mw_data = hppdWC.utils.cropDFInTime(mw_data, 40, resetZero = True)
    if testNum == 100:
        mw_data = hppdWC.utils.cropDFInTime(mw_data, 90, resetZero = True)
    
    # load realsense camera data
    camfileCompleteName = hppdWC.utils.findFileInDirectory(cam_dir, testName)[0]
    cam_data = pd.read_csv(camfileCompleteName).astype(float)
    cam_data = hppdWC.analysis.fromAbsTimeStampToRelTime(cam_data)

    # load depth hand information
    camfileDepCompleteName = hppdWC.utils.findFileInDirectory(cam_dep_dir, testName)[0]
    camDep_data = pd.read_csv(camfileDepCompleteName).astype(float)
    camDep_data = hppdWC.analysis.fromAbsTimeStampToRelTime(camDep_data)

    if syncValues:
        ecDelay = syncValues[0]
        emDelay = syncValues[1]

        cam_data_sync = hppdWC.utils.cropDFInTime(cam_data, -ecDelay, resetZero = True, resetIndex = resetIndex)
        camDep_data_sync = hppdWC.utils.cropDFInTime(camDep_data, -ecDelay, resetZero = True, resetIndex = resetIndex)
        mw_data_sync = hppdWC.utils.cropDFInTime(mw_data, -emDelay, resetZero = True, resetIndex = resetIndex)
        ergo_data_sync = ergo_data.copy()

        if testStartStop:
            start = testStartStop[0]
            stop = testStartStop[1]

            cam_data_test = hppdWC.utils.cropDFInTime(cam_data_sync, start, stop, resetZero = True, resetIndex = resetIndex)
            camDep_data_test = hppdWC.utils.cropDFInTime(camDep_data_sync, start, stop, resetZero = True)
            mw_data_test = hppdWC.utils.cropDFInTime(mw_data_sync, start, stop, resetZero = True, resetIndex = resetIndex)
            ergo_data_test = hppdWC.utils.cropDFInTime(ergo_data_sync, start, stop, resetZero = True, resetIndex = resetIndex)

            return cam_data_test, camDep_data_test, mw_data_test, ergo_data_test
        else:
            return cam_data_sync, camDep_data_sync, mw_data_sync, ergo_data_sync
    else:
        return cam_data, camDep_data, mw_data, ergo_data
