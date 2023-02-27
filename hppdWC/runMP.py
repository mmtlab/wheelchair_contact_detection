# -*- coding: utf-8 -*-
"""
Collection of functions to run mediapipe on a stream from realsense cameras
"""
#%% imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pyrealsense2 as rs
import mediapipe



import sys
import keyboard
import os

import csv

import time
import datetime
import tqdm

import logging

from . import analysis
from . import bagRS
from . import utils
#%% functions runMediaPipe 
def runMediaPipeBase(fileCompletePath, \
                     displayImage = True, recordVideo = False ,\
                     static_image_mode = False, max_num_hands = 1, \
                     min_detection_confidence = 0.5, min_tracking_confidence = 0.5,\
                     numberOfFrames = 20000):
    '''
    Given a bag file, playbacks the file frame by frame and detects the hand 
    using mediapipe hands.
    Since the elaboration is quite long [from 4 to 10 iteration/s, depending
    on the display of the image and the saving of the video], a loading bar is 
    provided to the user during the execution. The number of frames is estimated
    consequently the actual number of frames elaborated will be slightly lower.
    
    The user can interact with the execution pressing:
    [esc] stops the execution
    When displaying the images, three other interactions are available:
    [p] pauses the execution
    [s] saves the images corresponding to that frame
    [enter] resumes the execution
    
    A csv file is written at the end of the execution, containing 64 columns:
    time  x00 y00 z00 x01 y01 z01 ... x20 y20 z20
    time is the absolute time stamp of recording of each frame, useful for 
    synchronization, while the other 63 columns are the coordinates of each 
    landmark expressed with respect to the image dimension [0..1]
    
    assuming the data are organized in the following structure 
    the bag file will come from 01_raw,
    the csv file will be written in 02_preprocessing
    the saved images and the video will be in 03_analysis
    └───YYYYMMDD (first test day date)
        ├───00_protocols
        ├───01_raw
        │   ├───realsense
        │   │   └───nameTest.bag
        │   └───other devices   
        ├───02_preprocessing
        │   ├───realsense
        │   │   └───nameTest.csv
        │   └───other devices   
        ├───03_analysis
        │   ├───realsense
        │   │   ├───nameTest
        │   │   │   └───frameNumber.png
        │   │   └───nameTest.avi
        │   └───other devices 
        └───04_results
    
    Parameters
    ----------
    fileCompletePath : bag file from realsense recording
        contains the data of rgb and depth images
    displayImage : bool, optional
        If during the execution the images are displayed. 
        The default is True.
    recordVideo : bool, optional
        If the video of the result with mediapipe keypoints is saved. 
        The default is False.
    static_image_mode : bool, optional
        It's a mediapipe.hands parameter, check documentation. 
        The default is False.
    max_num_hands : int, optional
        It's a mediapipe.hands parameter, check documentation. 
        The default is 1.
    min_detection_confidence : float, optional
        from 0 to 1. It's a mediapipe.hands parameter, check documentation.
        The default is 0.5.
    min_tracking_confidence : float, optional
        from 0 to 1. It's a mediapipe.hands parameter, check documentation.
        The default is 0.5.
    numberOfFrames : int, optional
        attended number of frames in the recording. The extractor will do 
        numberOfFrames iterations, or, if the extraction is complete, 
        will stop earlier. Better put a larger number than the actual one. 
        Useful to print the loading bar.
        The default is 20000.

    Returns
    -------
    df : pandas dataframe
        of 64 columns: time  x00 y00 z00 x01 y01 z01 ... x20 y20 z20 
    time_exec_array : numpy array
        contains the elapsed time for every execution

    '''
    
    # eventually adding .bag in the end in case the user forgot it
    utils.checkExtension(fileCompletePath, '.bag')
    
# =============================================================================
#     WHERE TO SAVE THE OUTPUTS
# =============================================================================
    # assuming the data are organized in the following structure 
    # the bag file will come from 01_raw,
    # the csv file will be written in 02_preprocessing
    # the saved images and the video will be in 03_analysis
    # └───YYYYMMDD (first test day date)
    #     ├───00_protocols
    #     ├───01_raw
    #     │   ├───realsense
    #     │   │   └───nameTest.bag
    #     │   └───other devices   
    #     ├───02_preprocessing
    #     │   ├───realsense
    #     │   │   └───nameTest.csv
    #     │   └───other devices   
    #     ├───03_analysis
    #     │   ├───realsense
    #     │   │   ├───nameTest
    #     │   │   │   └───frameNumber.png
    #     │   │   └───nameTest.avi
    #     │   └───other devices 
    #     └───04_results
    
    thisExecutionDate = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y%m%d%H%M%S')
    
    # complete directory for csv file saving
    csvFile = fileCompletePath.replace("01_raw", "02_preprocessing")
    csvFile = csvFile[:-4] + '-' + thisExecutionDate + csvFile[-4:]
    csvFile = csvFile.replace(".bag", ".csv")
    
    # folder for images saving
    imagesFolder = fileCompletePath.replace("01_raw", "03_analysis")
    imagesFolder = imagesFolder.replace(".bag", "")
    
    # complete directory for video saving
    videoCompleteName = fileCompletePath.replace("01_raw", "03_analysis")
    videoCompleteName = videoCompleteName[:-4] + '-' + thisExecutionDate + videoCompleteName[-4:]
    videoCompleteName = videoCompleteName.replace(".bag", ".avi")
    
    print('Loading the bag file: \n' + fileCompletePath)

    print("Executing mediapipe ...\n")    
    
# =============================================================================
#     START THE STREAM OF THE PIPELINE
# =============================================================================
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, fileCompletePath, repeat_playback = False)
    
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 1)  # jet
    aligned_stream = rs.align(rs.stream.color)  # alignment depth -> color
    
# =============================================================================
#    INITIALIZATION
# =============================================================================
    # so at the first executuion becomes 0
    frameCounter = -1 
    # to save the timing execution of each loop (debug)
    time_exec_array = [0] * numberOfFrames
    # time_exec_array = []
    # to save the starting of the execution
    startTime = time.time()
    # at each iteration add a new row containing landMarkArray and timestamp_s
    data = [0] * numberOfFrames
    # data = []
    
# =============================================================================
#     RENAMING VARIABLES FOR MEDIAPIPE EXECUTION
# =============================================================================
    mp_hands = mediapipe.solutions.hands
    sim = static_image_mode
    mnh = max_num_hands
    mdc = min_detection_confidence
    mtc = min_tracking_confidence
    
    if displayImage or recordVideo:
# =============================================================================
#         WRITE ON THE IMAGE PARAMS
# =============================================================================
        font = cv2.FONT_HERSHEY_SIMPLEX
        origin = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        
        if displayImage:
# =============================================================================
#             IMAGE DISPLAY DEFINITION
# =============================================================================
            cv2.namedWindow('RealSense - color', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('RealSense - depth (colorized)', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('MediaPipe - input', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('MediaPipe - result', cv2.WINDOW_AUTOSIZE)

    try:
        for i in tqdm.tqdm(range(numberOfFrames)):
            try:
                frame = pipeline.wait_for_frames()
            except: 
                break
# =============================================================================
#             DEBUGGING
# =============================================================================
            frameCounter = frameCounter + 1
            # time frame on the execution of the loop
            now = time.time()
            # time_exec_array = np.append(time_exec_array, now-startTime)
            time_exec_array[frameCounter] = now-startTime
            
# =============================================================================
#             GET THE REQUIRED DATA FROM THE BAG FILE
# =============================================================================
            # alignement of the frames: the obtained resolution is the one of the rgb image
            frame = aligned_stream.process(frame)
            
            # get the depth and color frames
            depth_frame = frame.get_depth_frame()
            color_frame = frame.get_color_frame()

            # get the timestamp in seconds
            timestamp_s = frame.get_timestamp()/1000
            # print(datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f'))
            
            # from frames to images
            # the image saved in the bag file is in rgb format,
            # the one required from mediapipe as well
            color_image_rgb = np.asanyarray(color_frame.get_data())
            
            depth_image = np.asanyarray(depth_frame.get_data()).astype('float')
            depth_image_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            
# =============================================================================
#             OPERATIONS ON THE RGB IMAGE TO BE GIVEN TO MEDIAPIPE
# keep the name image_for_mp for the final image, it will be the one analyzed
# =============================================================================
            # in this case, only copying the image
            image_for_mp = color_image_rgb.copy()
            
# =============================================================================
#            RUN MEDIAPIPE ON THE GIVEN IMAGE
# =============================================================================
            # get the hand keypoints inside the object results
            with mp_hands.Hands(static_image_mode=sim, max_num_hands=mnh, min_detection_confidence=mdc, min_tracking_confidence=mtc) as hands:
                results = hands.process(image_for_mp)
            # convert the results in an array
            landMarkArray = resultsToLandMarkArray(results)
            
# =============================================================================
#             CREATE A NEW ROW IN DATA
# =============================================================================
            # create array to be written
            # insert in landMarkArray the element timestamp_s in the position 0
            tmp = np.insert(landMarkArray, 0, timestamp_s) 
            # append the row to the data table
            # data.append(tmp)
            data[frameCounter] = tmp
            
            if displayImage or recordVideo:
# =============================================================================
#                 DISPLAY IMAGES
# =============================================================================
                stringForImage = "frame: {:05d} / ".format(frameCounter) + \
                    datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f')
                # cv2 displays images in bgr
                color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_BGR2RGB)
                image_for_mp_bgr = cv2.cvtColor(image_for_mp, cv2.COLOR_BGR2RGB)
                depth_image_colorized_bgr = cv2.cvtColor(depth_image_colorized, cv2.COLOR_BGR2RGB)
                # puts text on the image
                color_image_bgr = cv2.putText(color_image_bgr, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)
                # draws the found landmarks on the color_image_rgb
                color_image_bgr_keypoints = drawMPResultsOnImage(color_image_bgr, results)
                
                if displayImage:
                    # shows the images
                    cv2.imshow('RealSense - color', color_image_bgr)
                    cv2.imshow('RealSense - depth (colorized)', depth_image_colorized_bgr)
                    cv2.imshow('MediaPipe - input', image_for_mp_bgr)
                    cv2.imshow('MediaPipe - result', color_image_bgr_keypoints)
                    # to give time to show the images
                    key = cv2.waitKey(1)
        
                    if key == ord('p'):
                        print('\n[p] pressed. Video paused... ')
                        print('press [enter] to continue or [s] to save the images')
            
                        while True:
                            if keyboard.is_pressed("s"):     
                                print('\n[s] pressed, saving images...')
                                # create the folder if it doesn't exist
                                os.makedirs(imagesFolder, exist_ok=True)
                                # create image name
                                imageCompletePath = imagesFolder + '\{:05d}'.format(frameCounter)
                                cv2.imwrite(imageCompletePath+'RSc.png', color_image_bgr)
                                cv2.imwrite(imageCompletePath+'RSd.png', depth_image_colorized_bgr)
                                cv2.imwrite(imageCompletePath+'MPi.png', image_for_mp_bgr)
                                cv2.imwrite(imageCompletePath+'MPo.png', color_image_bgr_keypoints)
                                print('images of frame ' + str(frameCounter) +' saved in ' + \
                                      imagesFolder + ', continuing the execution')
                                break
                            if keyboard.is_pressed("enter"): # it's not possible to use [p] again
                                print('\n[enter] pressed, continuing the execution')
                                break
                            
                if recordVideo:
                    if frameCounter == 0:
                        # create the folder if it doesn't exist
                        os.makedirs(os.path.split(videoCompleteName)[0], exist_ok=True)
                        # initialize the video saver
                        image_height, image_width, _ = color_image_bgr_keypoints.shape
                        videoOut = cv2.VideoWriter(videoCompleteName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                        
                    videoOut.write(color_image_bgr_keypoints)
                   #  key = cv2.waitKey(1)

# =============================================================================
#             TO EXIT THE EXECUTION
# =============================================================================
            # if pressed escape exit program
            if keyboard.is_pressed('esc'):
                print('[esc] pressed, KILLING EXECUTION')
                break
            
            
    finally:
        # cut the files preallocated with
        data = data[:frameCounter]
        time_exec_array = time_exec_array[:frameCounter]
        
        print('EXECUTION TERMINATED')
        print('saving files and closing resources ...')
# =============================================================================
#         CREATION OF THE PANDAS DATAFRAME AND SAVING IN A CSV FILE
# =============================================================================
        # create the folder if it doesn't exist
        os.makedirs(os.path.split(csvFile)[0], exist_ok=True)
        # create the header using the function
        header = defineHeader()
        # create the pandas dataframe
        df = pd.DataFrame(np.vstack(data),  columns=header)
        # saves the pandas dataframe in a csv file
        df.to_csv(csvFile, index = False) 
        
        print('\n [INFO] csv file containg hand keypoints saved in: \n' + csvFile)
        
# =============================================================================
#         OTHER OPERATIONS
# =============================================================================
        # stop the pipeline
        pipeline.stop()
        # close all the windows
        cv2.destroyAllWindows()
        # gives few information to the user
        print('\nALL RESOURCES WERE CLOSED SUCCESSFULLY\n')
        
        elapsedTime = time.time()-startTime
        freqOfExecution = frameCounter/elapsedTime
        
        print("{:d} frames were analyzed in {:.2f} seconds ({:.2f} frames per second)"\
              .format(frameCounter, elapsedTime, freqOfExecution))
            
        return df, time_exec_array

def runMediaPipeBaseFolderSaving(fileCompletePath, CSVOutputDirectory, videoOutputDirectory = '',\
                     static_image_mode = False, max_num_hands = 1, \
                     min_detection_confidence = 0.5, min_tracking_confidence = 0.5,\
                     frequency = 60, depthTreshold = 1000,\
                     numberOfFrames = 20000):
    '''
    Given a bag file, playbacks the file frame by frame and detects the hand 
    using mediapipe hands.
    Since the elaboration is quite long [from 4 to 10 iteration/s], a loading bar is 
    provided to the user during the execution. The number of frames is estimated
    consequently the actual number of frames elaborated will be slightly lower.
    
    During the execution, if videOutputDirectory is specified, two videos are
    recorded: 
        - the raw rgb images coming from the bag file
        - the result of image modification to feed mediapipe and the detected landmarks 
    
    A csv file is written at the end of the execution, containing 64 columns:
    time  x00 y00 z00 x01 y01 z01 ... x20 y20 z20
    time is the absolute time stamp of recording of each frame, useful for 
    synchronization, while the other 63 columns are the coordinates of each 
    landmark expressed with respect to the image dimension [0..1]


    Parameters
    ----------
    fileCompletePath : bag file from realsense recording
        contains the data of rgb and depth images
    CSVOutputDirectory : string
        directory where the csv will be saved.
    videoOutputDirectory : string, optional
        directory where videos will be saved. 
        The default is '', which sets recordVideo to False
    static_image_mode : bool, optional
        It's a mediapipe.hands parameter, check documentation. 
        The default is False.
    max_num_hands : int, optional
        It's a mediapipe.hands parameter, check documentation. 
        The default is 1.
    min_detection_confidence : float, optional
        from 0 to 1. It's a mediapipe.hands parameter, check documentation.
        The default is 0.5.
    min_tracking_confidence : float, optional
        from 0 to 1. It's a mediapipe.hands parameter, check documentation.
        The default is 0.5.
    frequency : int, optional
        frequency of recording of the videos. The default is 60.
    depthTreshold : float, optional
        All the pixels whose depth is bigger than the threshold will be colored
        in black. The default is 1000.
    numberOfFrames : int, optional
        attended number of frames in the recording. The extractor will do 
        numberOfFrames iterations, or, if the extraction is complete, 
        will stop earlier. Better put a larger number than the actual one. 
        Useful to print the loading bar.
        The default is 20000.

    Returns
    -------
    df : pandas dataframe
        of 64 columns: time  x00 y00 z00 x01 y01 z01 ... x20 y20 z20 
    time_exec_array : numpy array
        contains the elapsed time for every execution

    '''
    
    recordVideo = True
    if videoOutputDirectory == '':
        recordVideo = False
    
    # eventually adding .bag in the end in case the user forgot it
    fileCompletePath = utils.checkExtension(fileCompletePath, '.bag')
    
    # get the name of the file
    fileName = os.path.split(fileCompletePath)[1][:-4]
    
    # add to the filename an univoque code    
    thisExecutionDate = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y%m%d%H%M%S')
    fileNameCode = fileName + '-' + thisExecutionDate
    
    if not os.path.isdir(CSVOutputDirectory):
        os.makedirs(CSVOutputDirectory, exist_ok=True)
        logging.info('directory <<' + CSVOutputDirectory + '>> not existing, creating')
    # complete directory for csv file saving
    csvFileCompletePath = os.path.join(CSVOutputDirectory, fileNameCode + '.csv')
    
    if recordVideo:
        if not  os.path.isdir(videoOutputDirectory):
            os.makedirs(videoOutputDirectory, exist_ok=True)
            logging.info('directory <<' + videoOutputDirectory + '>> not existing, creating')
        # complete directory for video saving
        videoRawCompletePath = os.path.join(videoOutputDirectory, fileNameCode + '-raw.avi')
        videoMPCompletePath = os.path.join(videoOutputDirectory, fileNameCode + '-mp.avi')
    

    logging.info('working on ' + fileCompletePath)

    if recordVideo:
# =============================================================================
#         WRITE ON THE IMAGE PARAMS
# =============================================================================
        font = cv2.FONT_HERSHEY_SIMPLEX
        origin = (20, 20)
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
  
# =============================================================================
#     START THE STREAM OF THE PIPELINE
# =============================================================================
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, fileCompletePath, repeat_playback = False)
    
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 1)  # jet
    aligned_stream = rs.align(rs.stream.color)  # alignment depth -> color
    
# =============================================================================
#    INITIALIZATION
# =============================================================================
    # so at the first executuion becomes 0
    frameCounter = -1 
    # to save the timing execution of each loop (debug)
    time_exec_array = [0] * numberOfFrames
    # time_exec_array = []
    # to save the starting of the execution
    startTime = time.time()
    # at each iteration add a new row containing landMarkArray and timestamp_s
    data = [0] * numberOfFrames
    # data = []
    
# =============================================================================
#     RENAMING VARIABLES FOR MEDIAPIPE EXECUTION
# =============================================================================
    mp_hands = mediapipe.solutions.hands
    sim = static_image_mode
    mnh = max_num_hands
    mdc = min_detection_confidence
    mtc = min_tracking_confidence

    try:
        for i in tqdm.tqdm(range(numberOfFrames)):
            try:
                frame = pipeline.wait_for_frames()
            except: 
                break
# =============================================================================
#             DEBUGGING
# =============================================================================
            frameCounter = frameCounter + 1
            # time frame on the execution of the loop
            now = time.time()
            # time_exec_array = np.append(time_exec_array, now-startTime)
            time_exec_array[frameCounter] = now-startTime
            
# =============================================================================
#             GET THE REQUIRED DATA FROM THE BAG FILE
# =============================================================================
            # alignement of the frames: the obtained resolution is the one of the rgb image
            frame = aligned_stream.process(frame)
            
            # get the depth and color frames
            depth_frame = frame.get_depth_frame()
            color_frame = frame.get_color_frame()

            # get the timestamp in seconds
            timestamp_s = frame.get_timestamp()/1000
            # print(datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f'))
            
            # from frames to images
            # the image saved in the bag file is in rgb format,
            # the one required from mediapipe as well
            color_image_rgb = np.asanyarray(color_frame.get_data())
            
            depth_image = np.asanyarray(depth_frame.get_data()).astype('float')
            # depth_image_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            
# =============================================================================
#             OPERATIONS ON THE RGB IMAGE TO BE GIVEN TO MEDIAPIPE
# keep the name image_for_mp for the final image, it will be the one analyzed
# =============================================================================
            # in this case, only copying the image
            image_for_mp = color_image_rgb.copy()
            # thresholding using the depth
            image_for_mp[depth_image > depthTreshold] = [0,0,0]
            
# =============================================================================
#            RUN MEDIAPIPE ON THE GIVEN IMAGE
# =============================================================================
            # get the hand keypoints inside the object results
            with mp_hands.Hands(static_image_mode=sim, max_num_hands=mnh, min_detection_confidence=mdc, min_tracking_confidence=mtc) as hands:
                results = hands.process(image_for_mp)
            # convert the results in an array
            landMarkArray = resultsToLandMarkArray(results)
            
# =============================================================================
#             CREATE A NEW ROW IN DATA
# =============================================================================
            # create array to be written
            # insert in landMarkArray the element timestamp_s in the position 0
            tmp = np.insert(landMarkArray, 0, timestamp_s) 
            # append the row to the data table
            # data.append(tmp)
            data[frameCounter] = tmp
            
            if recordVideo:
# =============================================================================
#                 IMAGE OPERATION
# =============================================================================
                stringForImage = "frame: {:05d} / ".format(frameCounter) + \
                    datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f')
                # cv2 displays images in bgr, they need to be converted
                color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_BGR2RGB)
                image_for_mp_bgr = cv2.cvtColor(image_for_mp, cv2.COLOR_BGR2RGB)
                # put text on the image
                color_image_bgr = cv2.putText(color_image_bgr, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)
                image_for_mp_bgr = cv2.putText(image_for_mp_bgr, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)
                
                # draws the found landmarks on the color_image_rgb
                color_image_bgr_keypoints = drawMPResultsOnImage(image_for_mp_bgr, results)

                if frameCounter == 0:
                    # initialize the video saver
                    image_height, image_width, _ = color_image_bgr_keypoints.shape
                    videoRawOut = cv2.VideoWriter(videoRawCompletePath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                    videoMPOut = cv2.VideoWriter(videoMPCompletePath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                        
                videoRawOut.write(color_image_bgr)
                videoMPOut.write(color_image_bgr_keypoints)
            
            
    finally:
        # cut the files preallocated with
        data = data[:frameCounter]
        time_exec_array = time_exec_array[:frameCounter]
        
        # logging.debug('EXECUTION TERMINATED')
# =============================================================================
#         CREATION OF THE PANDAS DATAFRAME AND SAVING IN A CSV FILE
# =============================================================================
        # create the header using the function
        header = defineHeader()
        # create the pandas dataframe
        df = pd.DataFrame(np.vstack(data),  columns=header)
        # saves the pandas dataframe in a csv file
        df.to_csv(csvFileCompletePath, index = False) 
        
# =============================================================================
#         OTHER OPERATIONS
# =============================================================================
        # stop the pipeline
        pipeline.stop()
        
        elapsedTime = time.time()-startTime
        freqOfExecution = frameCounter/elapsedTime
        
        logging.debug("{} terminated. {:d} frames in {:.2f} seconds ({:.2f} frames per second)"\
              .format(fileName, frameCounter, elapsedTime, freqOfExecution))
            
        return df, time_exec_array

#%% functions other
def resultsToLandMarkArray(results):
    """
    From results = mediapipe.hands.process(image, [params...])
    returns an array containing the values of:
    x00 y00 z00 x01 y01 z01 ... x20 y20 z20    

    Parameters
    ----------
    results : mediapipe.python.solution_base.SolutionOutputs
        Result of image analysis with mediapipe hands

    Returns
    -------
    landMarkArray : numpy array of float64 containing the keypoint coordinates
    dimension of 1 row and 63 columns
    x00 y00 z00 x01 y01 z01 ... x20 y20 z20
        
    """
    landMarkArray = np.full([21*3], np.nan)
    if results.multi_hand_landmarks:
        for number in range(21):
            landMarkArray[number*3] = (results.multi_hand_landmarks[0].landmark[number].x)
            landMarkArray[number*3+1] = (results.multi_hand_landmarks[0].landmark[number].y)
            landMarkArray[number*3+2] = (results.multi_hand_landmarks[0].landmark[number].z)
            # multi_hand_landmarks[0] because interested in only one hand -> put max_num_hands = 1            
    return landMarkArray

def resultsToLandMarkMatrix(results):
    """
    From results = mediapipe.hands.process(image, [params...])
    returns an matrix containing
                   x    y    z
    keypoint00 
    keypoint01 
    keypoint02
    ...
    keypoint20
    

    Parameters
    ----------
    results : mediapipe.python.solution_base.SolutionOutputs
        Result of image analysis with mediapipe hands

    Returns
    -------
    landMarkMatrix : numpy matrix of float64 containing the keypoint coordinates
    dimension of 21 rows and 3 columns
                   x    y    z
    keypoint00 
    keypoint01 
    keypoint02
    ...
    keypoint20

    """
    landMarkMatrix = np.full([21, 3, 1], np.nan)
    if results.multi_hand_landmarks:
        for number in range(21):
            landMarkMatrix[number][0] = results.multi_hand_landmarks[0].landmark[number].x
            landMarkMatrix[number][1] = results.multi_hand_landmarks[0].landmark[number].y
            landMarkMatrix[number][2] = results.multi_hand_landmarks[0].landmark[number].z
            # multi_hand_landmarks[0] because interested in only one hand -> put max_num_hands = 1
    return landMarkMatrix

def defineHeader():
    """
    Creates a list of string to be the first line of the pandas dataframe and
    and of the excel file
    time	x00	y00	z00	x01	y01	z01	...	x19	y19	z19	x20	y20	z20
    
    Returns
    -------
    header : list of strings
    time	x00	y00	z00	x01	y01	z01	...	x19	y19	z19	x20	y20	z20      

    """
    maxNumber = 21;
    letters = ['x','y','z']
    firstColumnTitle = 'time'
    header = [firstColumnTitle]
    for number in range(maxNumber):
        for letter in letters:
            name = letter + "{:02d}".format(number)
            header.append(name)
    return header

def drawMPResultsOnImage(image, results):
    """
    Draws the handkeypoints, as given from mediapipe, on the given image

    Parameters
    ----------
    image : image
        Image where the hand keypoints will be drawn
    results : mediapipe.python.solution_base.SolutionOutputs
        Result of image analysis with mediapipe hands

    Returns
    -------
    annotated_image : image
        Image with keypoints drawn on it

    """
    # =============================================================================
    # MEDIAPIPE ALIASING
    # =============================================================================
    mp_drawing = mediapipe.solutions.drawing_utils
    mp_drawing_styles = mediapipe.solutions.drawing_styles
    mp_hands = mediapipe.solutions.hands
    
    # copy the image and convert it 
    annotated_image = image.copy()
    if results.multi_hand_landmarks:
        image_height, image_width, _ = image.shape
        # for loop for every hand found
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, \
            mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), \
            mp_drawing_styles.get_default_hand_connections_style())
            
    return annotated_image
