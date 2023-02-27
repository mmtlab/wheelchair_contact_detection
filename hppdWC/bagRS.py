# -*- coding: utf-8 -*-
"""
Functions to interact with the realsense recordings for HPPD project

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

import datetime
import time
import tqdm
import logging

from . import utils


#%% functions
def getInfoTopicTable(fileCompleteName):
    '''
    Returns the frequency and the number of frames in a test by means of the
    functions of bagpy, consequently creates a folder in same directory of the 
    bag file analyzed    

    Counts the number of frames in the test loading the bagfile, accessing to
    the topics of image data and getting the value of Message Count
    
    Gets the frequency of execution loading the bagfile, accessing to the topics 
    of image data and getting the value of Frequency

    Parameters
    ----------
    fileCompleteName : .bag file 
        from realsense recording

    Returns
    -------
    frequency : int
        NB: the returned value is an int, the frequencies of acquisition of the two 
        channels may differ and are slightly lower than the nominal value
    numberOfFrames : int
        NB: the returned value is an estimation of the number of paired frames 
        Since the two streams are not paired (the pairing is done with rs.playback)
        the number of frames for the color and depth images can be different and not 
        equal to the number of paired frames that are obtained executing a playback.

    '''
    # reads the bag file
    b = bagpy.bagreader(fileCompleteName)
    # extracts the topic table
    topicTable = b.topic_table
    # from the topic_table creates a new pandas dataframe with the two topics
    interestingTopics = topicTable.loc[                                     \
        (topicTable['Topics'] == '/device_0/sensor_0/Depth_0/image/data') | \
        (topicTable['Topics'] == '/device_0/sensor_1/Color_0/image/data')      ]
    # from the new dataframe, extracts the value
    frequency = np.ceil(interestingTopics.loc[:,"Frequency"].mean())
    numberOfFrames = interestingTopics.loc[:,"Message Count"].max()
    
    return frequency, numberOfFrames

def getDataFromIndex(fileCompleteName, index):
    '''
    Given a bag file and the index, returns:
        - time stamp
        - rgb image
        - depth image
    at the given index
    
    To do so, a playback of the file is executed. Consequently, the highest the
    index, the slowest is the function

    Parameters
    ----------
    fileCompleteName : bag file from realsense recording
        contains the data of rgb and depth images
    index : int
        index of the data that are required

    Returns
    -------
    timestamp_s : int
        timestamp corresponding to the recording of the file
        to print the corresponding date:
        >>> print(datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f'))
            
    color_image_rgb : matrix w*h*3
        Contains the rgb channel values of every pixel
    depth_image : matrix w*h*1
        Contains the depth value of every pixel

    '''
    if not fileCompleteName[-4:] == '.bag':
        fileCompleteName = fileCompleteName + '.bag'
    # =============================================================================
    # START THE STREAM OF THE PIPELINE
    # =============================================================================
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, fileCompleteName, repeat_playback = False)
    
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 1)  # jet
    aligned_stream = rs.align(rs.stream.color)  # alignment depth -> color
    
    # =============================================================================
    # INITIALIZATION
    # =============================================================================
    # so at the first executuion becomes 0
    frameCounter = -1 
    
    try:
        while frameCounter <= index:
            try:
                frame = pipeline.wait_for_frames()
            except: 
                break
            # =============================================================================
            # DEBUGGING
            # =============================================================================
            frameCounter = frameCounter + 1
            
            # =============================================================================
            # GET THE REQUIRED DATA FROM THE BAG FILE
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
            
            depth_image = np.asanyarray(depth_frame.get_data())
    finally:
        # =============================================================================
        # OTHER OPERATIONS
        # =============================================================================
        # stop the pipeline
        pipeline.stop()
        # close all the windows
        cv2.destroyAllWindows()
    
        return timestamp_s, color_image_rgb, depth_image
    
def loadTopic(bagreaderElement, topicName, printLoadingTime):
    
    """
    Uses the functions of the library bagpy to extract topics from the bag file
    For every topic, a csv file is generated and then loaded

    Parameters
    ----------
    bagreaderElement : return of the bagreader function
        example: b = bagreader(bagFileCompletePath)
    topicName : String
        The name of the topic that wants to be loaded
    printLoadingTime : Boolean
        If True, the elapsed time to load the topic is printed

    Returns
    -------
    A pandas dataframe corresponding to the topic

    """
    if printLoadingTime:
        start_time = time.time() 
    
    # creates a csv file and returns its location
    message = bagreaderElement.message_by_topic(topic = topicName)
     
    if printLoadingTime:
        time_elapsed = time.time() - start_time 
        logging.info('Time elapsed: {:.2f} [s]'.format(time_elapsed))
    
    # loads the csv file previously generated
    dataframe = pd.read_csv(message)
     
    if printLoadingTime:
        time_elapsed = time.time() - start_time 
        logging.info('Time elapsed: {:.2f} [s]'.format(time_elapsed))
     
    return dataframe

def createTimesDataFrame(metaDataframe, freq, rgb_depth):
    
    """
    The metadata table contains 24 (21) lines for every acquired frame of the 
    depth (rgb) channel;

    In both tables, among the other values, different times are expressed:
    - index_time
    - system_time
    - Time of Arrival
    - Backend TimeStamp
    
    New dataframe is created, contains the four times already present and the 
    nominal time (the theorical one, if the acquision would work perfectly, 
    taking into account the length of the others)

    Parameters
    ----------
    metaDataframe : pandas dataframe of metadata
        Can come from depth or rgb channel
    freq : int
        Frequency of acquisition of the frames
    rgb_depth : string
        Declares if the metadata dataframe is from depth or rgb


    Returns
    -------
    time_df : pandas dataframe containing 5 columns
        'index time';
        'system time'; 
        'arrival time';
        'backend time';
        'nominal time'.
        
    global_system_time : a pandas dataframe containing 1 column
            

    """
    # renaming for shorter handling
    df = metaDataframe
    # recognition if it's an rgb or a depth dataframe
    if rgb_depth == 'rgb':
        # how many rows for each frame
        skipRows = 21
        # index of the first element related to that magnitude on the table 
        system_time_row = 0
        time_of_arrival_row = 6
        backend_timestamp_row = 7
    elif rgb_depth == 'depth' or rgb_depth == 'stereo' or rgb_depth == '3d':
        # how many rows for each frame
        skipRows = 24
        # index of the first element related to that magnitude on the table 
        system_time_row = 0
        time_of_arrival_row = 8
        backend_timestamp_row = 9
    else:
        logging.error('not recognized dataframe')
        return None
    
    # obtaining the shape of the dataframe
    (rows, columns) = df.shape
    
    # extracting the lines from the data frames
    index_time = df.iloc[np.arange(0, rows, skipRows), 0]
    global_system_time = df.iloc[np.arange(system_time_row, rows, skipRows), 2].astype(float)
    time_of_arrival = df.iloc[np.arange(time_of_arrival_row, rows, skipRows), 2].astype(float)
    backend_timestamp = df.iloc[np.arange(backend_timestamp_row, rows, skipRows), 2].astype(float)
    
    # some arrays are giving absolute time
    system_time = (global_system_time - global_system_time.iloc[0])
    time_of_arrival = (time_of_arrival - time_of_arrival.iloc[0])
    backend_timestamp = (backend_timestamp - backend_timestamp.iloc[0])
    
    # converting to numpy array
    index_time_array = index_time.to_numpy()
    global_system_time_array = global_system_time.to_numpy()
    system_time_array = system_time.to_numpy()
    time_of_arrival_array = time_of_arrival.to_numpy()
    backend_timestamp_array = backend_timestamp.to_numpy()
    # creating also the nominal time array
    nominal_time_array = np.arange(0, len(index_time_array)*1/freq, 1/freq)
    # since different precisions on len()*1/freq and np.arange is different,
    # an element can be added, double check the array
    nominal_time_array = nominal_time_array[0 : len(index_time_array)]
    # explication of different precisions: try the code below
    # print(len(index_time_array) * 1/depth_freq)
    # print(nominal_time_array[-5:])
    
    # conversion of every array from s to ms
    index_time_array = index_time_array * 1000
    #system_time_array # is alreay in ms
    #time_of_arrival_array # is alreay in ms
    #backend_timestamp_array # is alreay in ms 
    nominal_time_array = nominal_time_array * 1000
    
    # creating a dataframe
    d = {'index time': index_time_array, \
         'system time': system_time_array, \
         'arrival time': time_of_arrival_array, \
         'backend time': backend_timestamp_array, \
         'nominal time': nominal_time_array}
    
    time_df = pd.DataFrame(data=d)
    #display(time_df)
    
    # check the types
    #dataTypeSeries = time_df.dtypes
    #print(dataTypeSeries)
    
    d = {'global system time': global_system_time_array}
    global_system_time = pd.DataFrame(data=d)
    
    return time_df, global_system_time

    
def plotTiming(timeDataframe, freq, title, essentialPlots): 
    """
    Creates 4 subplots displaying timing information
    
    Upper left: time elapsed at the acquisition of every frame with respect to
    the start of the acquisition
    Upper right: time elapsed between each couple of frames
    Lower left: drift with respect to the nominal time (the final value is the 
    delay with respect to the theorically perfect recording)
    Lower Right: Histogram of the time elapsed between each couple of frames

    Parameters
    ----------
    timeDataframe : pandas dataframe containing the timing information
        use the one returned from "createTimesDataFrame"
    freq : int
        Frequency of acquisition of the frames
    rgb_depth : string
        Declares if the metadata dataframe is from depth or rgb
    essentialPlot : bool
        If True, only 'system time' is plotted
    Returns
    -------
    None.

    """
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(title, fontsize=16)
    
    # renaming for shorter handling
    if essentialPlots: # only system time is considered
        df = timeDataframe[['system time', 'nominal time']]
    else:
        df = timeDataframe
    
    # obtaining the shape of the dataframe
    (rows, columns) = df.shape
    
    # elapsed time
    this_ax = axes[0,0]
    df.plot(ax = this_ax, style = '.-')
    this_ax.grid()
    this_ax.set_xlabel("frame number")
    this_ax.set_ylabel("[ms]")
    this_ax.set_title("elapsed time to acquire each frame")
    
    # time difference
    this_ax = axes[0,1]
    df.diff().plot(ax = this_ax, style = '.-')
    this_ax.grid()
    this_ax.set_xlabel("frame number")
    this_ax.set_ylabel("[ms]")
    this_ax.set_title("dt between each frame and previous one")
    
    # distribution of time difference (gaussian hopefully)
    this_ax = axes[1,1]
    
    # solution 1: doesn't plot nominal time and resizes automatically
    df.diff().loc[:,df.diff().columns != 'nominal time'].plot.hist(bins = 30, ax = this_ax, alpha = 0.5)
    # solution 2: plots also nominal time but doesn't resize automatically
    # plot = df.diff().plot(kind = 'density', ax = this_ax)
    # this_ax.set_ylim(-0.1, 1.5)
    # to give a reference with the nominal time
    if freq != 0:
        this_ax.axvline(1/freq*1000, label = 'nominal', color = 'C4')
    
    this_ax.grid()
    this_ax.set_xlabel("[ms]")
    this_ax.set_ylabel("frequency")
    # if freq != 0:
    #     this_ax.set_xlim(1/freq*0.7*1000, 1/freq*1.3*1000)
        
    this_ax.set_title("time distribution")
    this_ax.legend()
    
    if freq != 0:
        # new dataframe containing the difference with the nominal time
        # creating an empty data frame
        tmp_df = pd.DataFrame()
        # getting the names of the columns from the previous database
        columnNames = df.columns.values.tolist()
        for column in range(0,columns):
            # computing the difference, storing it in tmp
            tmp = df.iloc[:,column] - df['nominal time'] 
            # adding the tmp column to the dataframe
            tmp_df[columnNames[column]] = tmp
        
    else:
        # new dataframe containing the difference between each couple
        # creating an empty data frame
        tmp_df = pd.DataFrame()
        # getting the names of the columns from the previous database
        columnNames = df.columns.values.tolist()
        for i in range(columns): # for every column
            for j in range(i, columns): # from i to the max number to avoid rep
                if i != j: # to avoid the difference between two same array
                    tmp = df.iloc[:,i] - df.iloc[:,j]
                    tmp_df[str(columnNames[i] + ' - ' + columnNames[j])] = tmp
                        
    df = tmp_df
    
    this_ax = axes[1,0]
    df.plot(ax = this_ax, style = '.-')
    this_ax.grid()
    this_ax.set_xlabel("frame number")
    this_ax.set_ylabel("[ms]")
    this_ax.set_title("drift with respect to nominal time")
    
    # plt.show(block=False)
    # plt.pause(0.1)

def infoTiming(timeDataFrame, columnName, freq):
    """
    Given a time dataframe containing a column called as specified in 
    columnName, for this application, the most reliable is "system time",
    returns a dictionary containing information regarding the timing execution:
    - 'freq th', 
    - 'mean freq real', 
    - 'std dev freq real', 
    - 'time stamp th [ms]', 
    - 'mean time stamp real [ms]', 
    - 'std dev time stamp real [ms]', 
    - 'elapsed time real [ms]', 
    - 'number of samples real', 
    - 'elapsed time th [ms]', (to acquire a number of samples equal to 
       number_of_samples_real, the theorical required time should be)
    - 'number of samples th' {in the elapsed_time_real should have been acquired 
       a number of samples equal to:}

    Parameters
    ----------
    timeDataFrame : pandas dataframe
        Usually system time is the most reliable one
    columnName : string
        Name of the column that wants to be analyzed, usually system time
    freq : int
        Theorical frequency of acquisition

    Returns
    -------
    d : dictionary
        Contains all timing parameters characterizing the test

    """
    
    # renaming the dataframe for a better handling
    df = timeDataFrame
    
    (rows, columns) = df.shape
    
    # comparison of frequencies
    freq_th = float(freq)
    # the number of time stamps is equal to the number of elements - 1
    mean_freq_real = float((rows-1)/df[columnName].iloc[-1]*1000) #freq in Hz
    std_freq_real = float(np.nanstd(1/df[columnName].diff()) * 1000) #freq in Hz
    
    # comparison of time stamps
    time_stamp_theorical = 1/freq * 1000 # from s to ms
    mean_time_stamp_real = float(np.nanmean(df[columnName].diff()))
    std_time_stamp_real = float(np.nanstd(df[columnName].diff()))
    
    # comparison of elapsed time and number of samples
    elapsed_time_real = float(df[columnName].iloc[-1])
    number_of_samples_real = float(rows)
    # to acquire a number of samples equal to number_of_samples_real, 
    # the theorical required time should be:
    elapsed_time_theorical = number_of_samples_real / freq * 1000 # from s to ms
    # in the elapsed_time_real should have been acquired a number of samples equal to:
    number_of_samples_theorical = float(np.floor(elapsed_time_real/1000 * freq))
    
    # creating the dictionary
    d = {'freq th': freq_th, \
         'mean freq real': mean_freq_real, \
         'std dev freq real' : std_freq_real, \
         'time stamp th [ms]': time_stamp_theorical, \
         'mean time stamp real [ms]': mean_time_stamp_real, \
         'std dev time stamp real [ms]' : std_time_stamp_real, \
         'elapsed time real [ms]': elapsed_time_real, \
         'number of samples real': number_of_samples_real, \
         'elapsed time th [ms]': elapsed_time_theorical, \
         'number of samples th' : number_of_samples_theorical}
        
    return d

# def compareTiming(arrayOfTimes,arrayNames, *title):
#     # creating the dataframe with the given arrays
#     df = pd.DataFrame(arrayOfTimes).T
#     # for the tile title
#     if title:
#         pass
#     else:
#         title = "comparison"
#     # for the labels
#     if arrayNames:
#         df.columns = arrayNames
#     # calling the plotTiming function with frequency = 0
#     freq = 0
#     plotTiming(df, freq, title, essentialPlots = False)
    
def logBagFile(bagFileCompletePath, depth_freq, rgb_freq, printLoadingTime, \
               showPlots, essentialPlots, showTimingTable):
    
    """
    Given a bag file, loads the metadata files regarding the rgb and the depth 
    channel and plots figures to show the timing execution

    Parameters
    ----------
    bagFileCompletePath : String
        path to the bag file
    depth_freq : Int
        Frequency of acquisition of the depth channel
    rgb_freq : Int
        Frequency of acquisition of the rgb channel
    printLoadingTime : Bool
        If True, the elapsed time to load the topic is printed
        It's passed to the function loadTopic
    showPlots : Bool
        If True, shows the plots regarding the timing execution.
        It's a flag in this function
    essentialPlots : Bool
        If True, only system time is plotted, 
        It's passed to the function plotTiming
    showTimingTable : Bool
        If True, from the two dictionaries containing the timing information 
        (the one that are also returned), creates a pandas dataframe and prints it
                


    Returns
    -------
    dictDEP : dictionary
        Contains all parameters characterizing the test of the depth channel
    dictRGB : dictionary
        Contains all parameters characterizing the test of the rgb channel
    df_depth_time: 
    df_rgb_time: 
    global_depth_time: 
    global_rgb_time: 

    """
    
    # to get the name of the file
    path, fileName = os.path.split(bagFileCompletePath)
    
    logging.info('Loading information on the file: ' + fileName)
    
    # creates the bagreader element
    b = bagpy.bagreader(bagFileCompletePath)
    
    # loading the metadata topics (the data topics are too heavy)
    df_depth_meta = loadTopic(b, '/device_0/sensor_0/Depth_0/image/metadata', printLoadingTime)
    df_rgb_meta = loadTopic(b, '/device_0/sensor_1/Color_0/image/metadata', printLoadingTime)
    
    df_depth_time, global_depth_time = createTimesDataFrame(df_depth_meta, depth_freq, 'depth')
    df_rgb_time, global_rgb_time = createTimesDataFrame(df_rgb_meta, rgb_freq, 'rgb')
    
    if showPlots:
        plotTiming(df_depth_time, depth_freq, (fileName + ' - DEPTH'), essentialPlots)
        plotTiming(df_rgb_time, rgb_freq, (fileName + ' - RGB'), essentialPlots)
    
    dictDEP = infoTiming(df_depth_time, 'system time', depth_freq)
    dictRGB = infoTiming(df_rgb_time, 'system time', rgb_freq)
    
    if showTimingTable:
        results = pd.DataFrame({'depth':pd.Series(dictDEP),'rgb':pd.Series(dictRGB)})
        print(results)
    
    return dictDEP, dictRGB, df_depth_time, df_rgb_time, global_depth_time, global_rgb_time

def getTimeStampArray(bagFileCompleteName, printInfo = False):
    """
    Executes a playback of the whole test to get the time stamp array
    Parameters
    ----------
    bagFileCompleteName : String
        directory to the bag file        
    printInfo : bool, optional
        Set true if you want to print the timeframe stored at each iteration. 
        The default is False.

    Returns
    -------
    time_stamp_array : float64 array
        array containing the corresponding ms of acquisition of each frame

    """
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bagFileCompleteName, repeat_playback = False)
    
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)
    # initialize the array
    time_stamp_array = []
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except: 
                break
    
            tmp = frames.get_timestamp()
            if printInfo:
                print(datetime.datetime.fromtimestamp(tmp/1000).strftime('%Y-%m-%d %H:%M:%S.%f'))
    
            time_stamp_array = np.append(time_stamp_array, tmp)
    
    finally:
        pipeline.stop()
        if printInfo:
            print('all the frames were analyzed')
            
    return time_stamp_array

def extractAviVideosFromBag(fileCompleteName, outputDir, frequency = 60, numberOfFrames = 20000, color = True, depth_splitted = True, depth_colorized = True, textOnImage = True):
    '''
    Saves in the specified folder a folder with the name of the test. 
    The subfolder contains a csv file with the timestamp of each paired frame and
    two avi videos: COL and DEP channel.
    For the COL video, it's simply the extraction of the rgb channel
    For the DEPcolored video, it's a rendering of the depth info through a colormap
    For the DEP video, a conversion of the 16 bit depth information is done in 
    the 3 channels where the avi video is saved:
        ***
        # CREATE DEPTH IMAGE through conversion 
        dep_image_height, dep_image_width = depth_image.shape
        zerosbit = np.zeros([dep_image_height, dep_image_width], dtype = np.uint8) # 480,848...
        # less significan bits are the rest of the division for 256
        lsb = (depth_image % 256).astype(np.uint8)
        # most significan bits are the division for 256 without rest
        msb = (depth_image / 256).astype(np.uint8)
        
        depth_image_3ch = cv2.merge([zerosbit, msb, lsb])
        ***
        When using this function, keep in mind that the avi video is a compression
        of the information that each frame has
        

    Parameters
    ----------
    fileCompleteName : .bag file
        .bag file containing the rgb/bgr frames, the depth frames and the time array
    outputDir : string
        directory where the files will be saved
    frequency : int, optional
        nominal frequency of recording, frequency for the video saved in .avi
        The default is 60.
    numberOfFrames : int, optional
        attended number of frames in the recording. The extractor will do 
        numberOfFrames iterations, or, if the extraction is complete, 
        will stop earlier. Better put a larger number than the actual one. 
        Useful to print the loading bar.
        The default is 20000.
    textOnImage : bool, optional
        set true if you want to add the timing information on the images. 
        The default is True.

    Returns
    -------
    time_exec_array: array
        contains information about the execution of the extraction

    '''
    if textOnImage:
        # =============================================================================
        #         WRITE ON THE IMAGE PARAMS
        # =============================================================================
        font = cv2.FONT_HERSHEY_SIMPLEX
        origin = (20, 20)
        fontScale = 0.8
        color = (255, 255, 255)
        thickness = 1

    # check extension of the file
    fileCompleteName = utils.checkExtension(fileCompleteName, '.bag')
    # get only the file name excluding ".bag"
    fileName = os.path.split(fileCompleteName)[1][:-4]

    # in order to give a unique name to the execution
    thisExecutionDate = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y%m%d%H%M%S')

    # create folder for the given execution of the given file
    outputFileDir = os.path.join(outputDir, fileName + '-' + thisExecutionDate)

    # create the folder if it doesn't exist
    os.makedirs(outputFileDir, exist_ok=True)

    # create the complete directory to the 3 different outputs
    if color:
        videoRGBCompleteName = os.path.join(outputFileDir, fileName + '-color.avi')
    if depth_splitted:
        videoDEPCompleteName = os.path.join(outputFileDir, fileName + '-depth splitted.avi')
    if depth_colorized:
        videoDEPcolorizedCompleteName = os.path.join(outputFileDir, fileName + '-depth colorized.avi')
    timeCSVCompleteName = os.path.join(outputFileDir, fileName + '-timestamp.csv')
    
    logging.info('working on: ' + fileName)

# =============================================================================
#     # sometimes the function to load the bag file gets stuck, better avoid this
#     # get the number of frames
#     frequency, numberOfFrames = getInfoTopicTable(fileCompleteName)
#     # since the method getInfoTopicTable gives an estimation of the number
#     # of frames, it's better to increase this value. Executing the for loop and
#     # catching the exception won't give any problem
#     numberOfFrames = int(numberOfFrames * 1.2)
# =============================================================================
    
# =============================================================================
#     START THE STREAM OF THE PIPELINE
# =============================================================================
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, fileCompleteName, repeat_playback = False)
    
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
    # to save the starting of the execution
    startTime = time.time()
    # at each iteration add a new row containing landMarkArray and timestamp_s
    timestamp_array = [0] * numberOfFrames

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
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # CREATE COLOR IMAGE
            # cv2 displays images in bgr
            color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_BGR2RGB)
            
            # CREATE DEPTH IMAGE through conversion 
            dep_image_height, dep_image_width = depth_image.shape
            zerosbit = np.zeros([dep_image_height, dep_image_width], dtype = np.uint8) # 480,848...
            # less significan bits are the rest of the division for 256
            lsb = (depth_image % 256).astype(np.uint8)
            # most significan bits are the division for 256 without rest
            msb = (depth_image / 256).astype(np.uint8)
            
            depth_image_3ch = cv2.merge([zerosbit, msb, lsb])

            # CREATE DEPTH IMAGE COLORIZED through colorizer
            depth_image_colorized = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            if textOnImage:
                stringForImage = 'frame: {:05d} - '.format(frameCounter) + \
                    datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f')
                # puts text on the image
                if color:
                    color_image_bgr = cv2.putText(color_image_bgr, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)
                if depth_splitted:
                    depth_image_3ch = cv2.putText(depth_image_3ch, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)
                if depth_colorized:
                    depth_image_colorized = cv2.putText(depth_image_colorized, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)

            if frameCounter == 0:
                # create the folder if it doesn't exist
                os.makedirs(os.path.split(videoRGBCompleteName)[0], exist_ok=True)
                if color:
                    # initialize the video saver for BGR
                    image_height, image_width, _ = color_image_bgr.shape
                    videoOutCol = cv2.VideoWriter(videoRGBCompleteName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                if depth_splitted:
                    # initialize the video saver for DEP
                    image_height, image_width, _ = depth_image_3ch.shape
                    videoOutDep = cv2.VideoWriter(videoDEPCompleteName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
                if depth_colorized:
                    # initialize the video saver for DEP colorized
                    image_height, image_width, _ = depth_image_colorized.shape
                    videoOutDepCol = cv2.VideoWriter(videoDEPcolorizedCompleteName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frequency, (image_width, image_height))
            if color:
                videoOutCol.write(color_image_bgr)
            if depth_splitted:
                videoOutDep.write(depth_image_3ch)
            if depth_colorized:
                videoOutDepCol.write(depth_image_colorized)
            timestamp_array[frameCounter] = timestamp_s
    finally:
        # cut the files preallocated with
        timestamp_array = timestamp_array[:frameCounter]
        time_exec_array = time_exec_array[:frameCounter]

        # create the folder if it doesn't exist
        os.makedirs(os.path.split(timeCSVCompleteName)[0], exist_ok=True)
        # create the pandas dataframe
        df = pd.DataFrame(np.vstack(timestamp_array),  columns=['timestamp'])
        # saves the pandas dataframe in a csv file
        df.to_csv(timeCSVCompleteName, index = False)

# =============================================================================
#         OTHER OPERATIONS
# =============================================================================
        # stop the pipeline
        pipeline.stop()
        # close all the windows
        cv2.destroyAllWindows()
        # gives few information to the user

        elapsedTime = time.time()-startTime
        freqOfExecution = frameCounter/elapsedTime
        
        logging.info("{:d} frames were analyzed in {:.2f} seconds ({:.2f} frames per second)"\
              .format(frameCounter, elapsedTime, freqOfExecution))
            
        return time_exec_array

def extractPngFramesFromBag(fileCompleteName, outputDir, frequency = 60, numberOfFrames = 20000, textOnImage = True):
    '''
    Saves in the specified folder a folder with the name of the test. 
    The subfolder contains a csv file with the timestamp of each paired frame and
    two other subfolders: COL and DEP channel.
    For the COL folder, it's the extraction of the rgb frame, 
    in format w*h*3 of integer 8bit (0->255)
    For the DEP folder, it's the extraction of the dep frame,
    in format w*h*1 of integer 16bit (0->65535)
        

    Parameters
    ----------
    fileCompleteName : .bag file
        .bag file containing the rgb/bgr frames, the depth frames and the time array
    outputDir : string
        directory where the files will be saved
    frequency : int, optional
        nominal frequency of recording, frequency for the video saved in .avi
        The default is 60.
    numberOfFrames : int, optional
        attended number of frames in the recording. The extractor will do 
        numberOfFrames iterations, or, if the extraction is complete, 
        will stop earlier. Better put a larger number than the actual one. 
        Useful to print the loading bar.
        The default is 20000.
    textOnImage : bool, optional
        set true if you want to add the timing information on the images. 
        The default is True.

    Returns
    -------
    time_exec_array: array
        contains information about the execution of the extraction

    '''
    if textOnImage:
        # =============================================================================
        #         WRITE ON THE IMAGE PARAMS
        # =============================================================================
        font = cv2.FONT_HERSHEY_SIMPLEX
        origin = (20, 20)
        fontScale = 0.8
        color = (255, 255, 255)
        thickness = 1

    # check extension of the file
    fileCompleteName = utils.checkExtension(fileCompleteName, '.bag')
    # get only the file name excluding ".bag"
    fileName = os.path.split(fileCompleteName)[1][:-4]

    # in order to give a unique name to the execution
    thisExecutionDate = datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y%m%d%H%M%S')

    # create folder for the given execution of the given file
    outputFileDir = os.path.join(outputDir, fileName + '-' + thisExecutionDate)

    # create directory of folders for saving col and dep
    outputCOLDir = os.path.join(outputFileDir, 'col')
    outputDEPDir = os.path.join(outputFileDir, 'dep')

    # create the folders if they don't exist
    os.makedirs(outputFileDir, exist_ok=True)
    os.makedirs(outputCOLDir, exist_ok = True)
    os.makedirs(outputDEPDir, exist_ok = True)

    # create the complete directory
    timeCSVCompleteName = os.path.join(outputFileDir, 'timestamp.csv')
    
    logging.info('working on: ' + fileName)

# =============================================================================
#     # sometimes the function to load the bag file gets stuck, better avoid this
#     # get the number of frames
#     frequency, numberOfFrames = getInfoTopicTable(fileCompleteName)
#     # since the method getInfoTopicTable gives an estimation of the number
#     # of frames, it's better to increase this value. Executing the for loop and
#     # catching the exception won't give any problem
#     numberOfFrames = int(numberOfFrames * 1.2)
# =============================================================================
    
# =============================================================================
#     START THE STREAM OF THE PIPELINE
# =============================================================================
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, fileCompleteName, repeat_playback = False)
    
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    aligned_stream = rs.align(rs.stream.color)  # alignment depth -> color
    
# =============================================================================
#    INITIALIZATION
# =============================================================================
    # so at the first executuion becomes 0
    frameCounter = -1 
    # to save the timing execution of each loop (debug)
    time_exec_array = [0] * numberOfFrames
    # to save the starting of the execution
    startTime = time.time()
    # at each iteration add a new row containing landMarkArray and timestamp_s
    timestamp_array = [0] * numberOfFrames

    try:
        for i in tqdm.tqdm(range(numberOfFrames)):
            try:
                frame = pipeline.wait_for_frames()
            except: 
                break

            if i == 322:
                debugFlag = True
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
            # the one required from mediapipe as well,
            # the one for cv2 should be in bgr
            color_image_rgb = np.asanyarray(color_frame.get_data())
            color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_BGR2RGB)
            
            depth_image = np.asanyarray(depth_frame.get_data())

            if textOnImage:
                stringForImage = 'frame: {:05d} - '.format(frameCounter) + \
                    datetime.datetime.fromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S.%f')
                # puts text on the image
                color_image_bgr = cv2.putText(color_image_bgr, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)
                # makes no sense write on the image since it's saved in 16 bit format
                # depth_image = cv2.putText(depth_image, stringForImage, origin, font, fontScale, color, thickness, cv2.LINE_AA)
            frameName = '{:05d}'.format(frameCounter)
            cv2.imwrite(os.path.join(outputCOLDir,frameName+'.png'), color_image_bgr)
            cv2.imwrite(os.path.join(outputDEPDir,frameName+'.png'), depth_image)
            timestamp_array[frameCounter] = timestamp_s

    finally:
        # cut the files preallocated with
        timestamp_array = timestamp_array[:frameCounter]
        time_exec_array = time_exec_array[:frameCounter]

        # create the folder if it doesn't exist
        os.makedirs(os.path.split(timeCSVCompleteName)[0], exist_ok=True)
        # create the pandas dataframe
        df = pd.DataFrame(np.vstack(timestamp_array),  columns=['timestamp'])
        # saves the pandas dataframe in a csv file
        df.to_csv(timeCSVCompleteName, index = False)

# =============================================================================
#         OTHER OPERATIONS
# =============================================================================
        # stop the pipeline
        pipeline.stop()
        # close all the windows
        cv2.destroyAllWindows()
        # gives few information to the user

        elapsedTime = time.time()-startTime
        freqOfExecution = frameCounter/elapsedTime
        
        logging.info("{:d} frames were analyzed in {:.2f} seconds ({:.2f} frames per second)"\
              .format(frameCounter, elapsedTime, freqOfExecution))
            
        return time_exec_array
