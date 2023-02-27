# -*- coding: utf-8 -*-
"""
General purpose functions for HPPD project
"""
#%% IMPORTS
import matplotlib.pyplot as plt
import pandas as pd
import winsound
import time
import numpy as np
import scipy
import os
import csv
import vlc
#%% FUNCTIONS            

def testImport():
    '''
    To simply test the import

    Returns
    -------
    None.

    '''

    print('\n' + '='*80 + 'import of hppdWC package successfull!\nReady to roll')

def findFileInDirectory(folder, partialFileName):
    '''
    Given a folder and a part of the name of the file, returns a list with the complete
    path of all the files whose name contains the partialFileName

    Parameters
    ----------
    folder : string
        directory where the file should be searched
    partialFileName : string
        string that should be present in the file name

    Returns
    -------
    filesList : list of string
        list containing all the complete directories to the files whose name 
        contains partialFileName

    '''
    filesList = []
    # loop on all the files contained in the specified folder
    for fileName in os.listdir(folder):
        # if the name of the file contains the partialFileName specified
        if partialFileName in fileName:
            # compose the whole path of the file
            completeFileName = os.path.join(folder, fileName)
            filesList.append(completeFileName)
    return filesList

def playSound(startFreq = 5000, endFreq = 0, stepFreq = -500, duration = 500):
    '''
    Plays a sound, simply call this function at the end of the script to know 
    when it finishes
    
    The played sound is a range of frequency from startFreq to endFreq with step
    stepFreq, each one reproduced for duration ms.
    
    NB: frequency values must be between 37 and 32767

    Parameters
    ----------
    startFreq : int, optional
        frequency of the first freq played. The default is 5000.
    endFreq : int, optional
        frequency of the last freq played. The default is 0.
    stepFreq : int, optional
        step of frequencies in the loop. The default is -500.
    duration : int, optional
        duration of each sound in ms. The default is 500.

    Returns
    -------
    None.

    '''
    # frequency values must be between 37 and 32767
    if startFreq > endFreq:
        startFreq = min(startFreq, 32767)
        endFreq = max(endFreq, 37)
    elif startFreq < endFreq:
        endFreq = min(endFreq, 32767)
        startFreq = max(startFreq, 37)


    for freq in range(startFreq, endFreq, stepFreq):
        winsound.Beep(freq, duration)
        time.sleep(0.01)

def playVLC(source, duration = -1):
    '''
    Plays the media in the given path and pauses the execution till 
        - the end of playing if duration is not specified
        - duration if it is specified

    Parameters
    ----------
    source : string
        path to the media.
    duration : float, optional
        for how many seconds should the media be played before being stopped 
        and continuing the execution? 
        The default is -1, which plays the media for the whole duration. 

    Returns
    -------
    duration : float
        duration in seconds of the media played.

    '''
     
    # creating a vlc instance
    vlc_instance = vlc.Instance()
     
    # creating a media player
    player = vlc_instance.media_player_new()
     
    # creating a media
    media = vlc_instance.media_new(source)
     
    # setting media to the player
    player.set_media(media)

    # play the video
    player.play()

    time.sleep(0.1)

    if duration == -1:
        # getting the duration of the video in s
        duration = player.get_length() / 1000

    # wait time
    time.sleep(duration)

    # stop the player
    player.stop()

    return duration

def addIntToList(thisList, integerToAdd):
    '''
    Adds to each element of the list the given integer

    Parameters
    ----------
    thisList : list
        list of integers.
    integerToAdd : int
        integer to be added to each element of the list.

    Returns
    -------
    thisList : list
        list with the integer added to each element.

    '''
    thisArr= np.array(thisList)
    thisArr = thisArr + integerToAdd
    thisList = thisArr.tolist()
    return thisList

def mulIntToList(thisList, integerToMultiply):
    '''
    Multiplies each element of the list for the given integer

    Parameters
    ----------
    thisList : list
        list of integers.
    integerToMultiply : int
        integer to be multiplied to each element of the list.

    Returns
    -------
    thisList : list
        list with the integer multiplied to each element.

    '''
    thisArr= np.array(thisList)
    thisArr = thisArr * integerToMultiply
    thisList = thisArr.tolist()
    return thisList

def removeNanXYZ(XYZdata, axis = 1):
    '''
    Removes nan from numpy array in the given direction

    Parameters
    ----------
    XYZdata : <class 'numpy.ndarray'>
        numpy array containing nans.
    axis : int, optional
        direction of scanning for removal. 
        if 0, scans each column... 
        if 1, scans each row...
        ... and deletes the ones containing at least one nan
        The default is 1.

    Returns
    -------
    XYZdata : <class 'numpy.ndarray'>
        numpy array without nans.

    '''
    return XYZdata[~np.isnan(XYZdata).any(axis=axis), :]

def addInHead(arr, n = 1):
    '''
    Repeats the first element in arr n times

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.
    n : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    '''
    return np.concatenate([np.array([arr[0]]*n), np.array(arr)])
def addInTail(arr, n=1):
    '''
    repeat the last element in arr n times

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.
    n : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    '''
    return np.concatenate([np.array(arr), np.array([arr[-1]]*n)])

def checkExtension(fileCompleteName, extension):
    '''
    Check if the file has the correct extension.
    If yes, does nothing
    If no, adds the extension

    Parameters
    ----------
    fileCompleteName : string
        name of the file
    extension : string
        required extension

    Returns
    -------
    fileCompleteName : string
        name of the file with the required extension

    '''
    # check if extension starts with '.'
    if not extension[0] == '.':
        extension = '.' + extension 
    # check if extension is in the end of the filename
    if not fileCompleteName[-len(extension):] == extension:
        fileCompleteName = fileCompleteName + extension
    return fileCompleteName

def containsScalars(iterable):
    '''
    Checks if the iterable contains scalar or other iterables
    eg:
    iterable = [1,2,3] -> contains scalar
    iterable = [[1,2,3], [4,5,6]] -> contains list
    iterable = [[1,2,3]] -> contains list

    Parameters
    ----------
    iterable : list or array or other iterable
        you want to know if it contains scalar or iterables inside.

    Returns
    -------
    bool
        DESCRIPTION.

    '''
    # if the first element is a scalar
    if np.isscalar(iterable[0]):
        return True
    else: # could be a list of lists or list of other iterables
        return False

def iterablesInsideList(iterable):
    '''
    If the iterable has one dimension, puts it into a list
    iterable = [1,2,3] -> 1dim -> [[1,2,3]]
    iterable = [[1,2,3], [4,5,6]] -> contains list -> ok
    iterable = [[1,2,3]] -> contains list -> ok

    Parameters
    ----------
    iterable : list or array or other iterable
        you want to put it into a list if it's not already.

    Returns
    -------
    iterable inside the list
    '''
    if containsScalars(iterable):
        return [iterable]
    else: # it a nested list or other nested iterables
        return iterable

def makeList(maybeAList):
    if isinstance(maybeAList, list):
        return maybeAList
    else:
        forSureAList = [maybeAList]
        return forSureAList

def makeNpArray(maybeANpArray):
    if isinstance(maybeANpArray, np.array):
        return maybeANpArray
    else:
        forSureANpArray = np.array(maybeANpArray)
        return forSureANpArray

def toFloatNumpyArray(variable):
    '''
    given a variable, if possible, creates a numpy array of type float

    Parameters
    ----------
    variable : the possible ones are:
        - pd.core.series.Series (column of a pandas.core.frame.DataFrame)
        - pandas.core.frame.DataFrame
        - list
        variable to be converted in numpy array of float

    Returns
    -------
    variable : numpy.ndarray
        variable converted in numpy array
    flag : int
        gives information regarding the conversion:
            -  0 -> no need of conversion, simply astype("float")
            - +1 -> variable was a pandas.series
            - +2 -> varialbe was a pandas.Dataframe
            - +3 -> variable was a list
            - -1 -> variable was none of above and wasn't converted

    '''
    flag = 0
    if isinstance(variable, np.ndarray):
        variable = variable.astype("float")
    elif isinstance(variable, pd.core.series.Series):
        variable = variable.to_numpy()
        variable = variable.astype("float")
        flag = 1
    elif isinstance(variable, pd.core.frame.DataFrame):
        variable = variable.to_numpy() 
        variable = variable.astype("float")
        flag = 2
    elif isinstance(variable, list):
        variable = np.array(variable)
        variable = variable.astype("float")
        flag = 3
    else:
        flag = -1
        
    return variable, flag

def writeCSV(CSVfileName, newRow, mode = 'a'):
    '''
    Simplifies the operation of writing in a csv file

    Parameters
    ----------
    CSVfileName : string
        complete path to the csv file.
    newRow : list
        row to be added.

    Returns
    -------
    None.

    '''
    # eventually create the csv folder
    os.makedirs(os.path.split(CSVfileName)[0], exist_ok= True)
    f = open(CSVfileName, mode, encoding='UTF8', newline='')
    writer = csv.writer(f)
    writer.writerow(newRow)
    f.close()

def cropDFInTime(df, startTime, endTime = np.nan, timeColumnName = 'time', resetZero = False, startFromZero = False, resetIndex = False):
    """
    Returns a copy of the dataframe where time is between startTime and endTime, 
    by default, keeps the original time.
    If resetZero, the time column is decremented of startTime.
    As a result, the time column starts from 0 if startTime corresponds to the 
    first element, otherwise from slightly more than 0.
    If startFromZero, the time column is decremented of the first element.
    As a result, the time column starts from 0.
    If resetIndex, the indexes start from 0, otherwise the original indexing is kept
    

    Parameters
    ----------
    df : pandas dataframe
        The one to be cropped.
    startTime : float
        starting time for cropping.
    endTime : float, optional
        ending time for cropping. 
        The default is np.nan, which means no cutting of the end.
    timeColumnName : string, optional
        name of the column containing the time. The default is 'time'.
    resetZero : bool
        if True, the time starts from startTime.
        if False, the original time is kept.
        The default is False.
    startFromZero : bool
        if True, the time starts from startTime.
        if False, the original time is kept.
        The default is False.
    resetIndex : bool
        if True, the indexes start from 0
        if False, the original indexing is kept
        The default is False.

    Returns
    -------
    df_cropped : pandas dataframe
        The one with time between startTime and endTime.

    """
    df_cropped = df.copy()

    # consider only the part between startTime and endTime
    if not np.isnan(endTime): # if endTime is defined
        df_cropped = df_cropped[df_cropped[timeColumnName] <= endTime]
    df_cropped = df_cropped[df_cropped[timeColumnName] >= startTime]

    # the initial moment is startTime
    if resetZero:
        df_cropped[timeColumnName] -= startTime

    # the initial moment is the first available on the time array
    if startFromZero:
        df_cropped[timeColumnName] -= df_cropped[timeColumnName].iloc[0]

    if resetIndex:
        df_cropped.reset_index(drop = True, inplace = True)

    return df_cropped

def cropDataframeInTimeBackup(df, startTime, endTime, timeColumnName = 'time', resetZero = False, resetIndex = False):
    """
    Returns a copy of the dataframe where time is between startTime and endTime

    Parameters
    ----------
    df : pandas dataframe
        The one to be cropped.
    startTime : float
        starting time for cropping.
    endTime : float
        ending time for cropping.
    timeColumnName : string, optional
        name of the column containing the time. The default is 'time'.
    newZero : bool
        if True, the time starts from 0
        if False, simply crops the dataframe

    Returns
    -------
    df_cropped : pandas dataframe
        The one with time between startTime and endTime.

    """
    df_cropped = df.copy()

    if not resetZero:
        df_cropped = df_cropped[df_cropped[timeColumnName] < endTime]
        df_cropped = df_cropped[df_cropped[timeColumnName] > startTime]

    if resetZero:
        df_cropped[timeColumnName]-=startTime
        df_cropped = df_cropped[df_cropped[timeColumnName] < endTime - startTime]
        df_cropped = df_cropped[df_cropped[timeColumnName] > 0]

    if resetIndex:
        df_cropped.reset_index(drop = True, inplace = True)

    return df_cropped

def correspondingColumns(listOfKeyPoints, letters = ['x','y','z']):
    '''
    From a list of keypoints, creates a list of column name
    
    eg:
        listOfKeyPoints = [5, 12]
        letters = ['x','y','z']
        
        the result would be:
        names = [['x05', 'y05', 'z05'], ['x12', 'y12', 'z12']]
        
    Parameters
    ----------
    listOfKeyPoints : list
        of the interested keypoints (specified as integers).
    letters : list, optional
        list of values to be added in front of the keypoint. The default is ['x','y','z'].

    Returns
    -------
    names : list of list
        contains the name of the corresponding columns.

    '''
    names = [] 
    for number in listOfKeyPoints:
        names.append([str(letter) + '{:02d}'.format(number)  for letter in letters])
        
    return names

def depImgToThreeCol(image):
    '''
    From a dep image, containing only 1 value per pixel: 

        |----------------------------...------> x
        |0.0       1.0       2.0     ...  img_w.0    
        |0.1       1.1       2.1     ...  img_w.1 
        |0.2       1.2       2.2     ...  img_w.2 
        |0.3       1.3       2.3     ...  img_w.3 
        ...        ...       ...     ...  ...
        |0.img_h   1.img_h   2.img_h ...  img_w.img_h
        v y

    returns an 2D array with 3 columns (pointCloud):
        x         y       dep
        0         0       0.0
        1         0       1.0
        2         0       2.0
        ...       ...     ...
        img_w     0       img_w.0
        --------------------------- first row of the image
        0         1       0.1
        1         1       1.1
        2         1       2.1
        ...       ...     ...
        img_w     1       img_w.1
        --------------------------- second row of the image
        ...
        ...
        0         img_h   0.img_h
        1         img_h   1.img_h
        2         img_h   2.img_h
        ...       ...     ...
        img_w     img_h   img_w.img_h
        --------------------------- last row of the image

    Parameters
    ----------
    image : matrix
        contains z values.

    Returns
    -------
    data : array
        contains x y z values.

    '''
    image_h, image_w = image.shape

    hline = np.arange(0, image_w, 1)
    xmask = np.repeat([hline], image_h, axis = 0)

    vline = np.expand_dims(np.arange(0, image_h, 1), axis = 1)
    ymask = np.repeat(vline, image_w, axis = 1)
    
    # create a matrix x y z
    data = np.zeros([image_h*image_w,3])
    data[:,0] = xmask.flatten()
    data[:,1] = ymask.flatten()
    data[:,2] = image.flatten()

    return data

def threeColToDepImg(data, x_col_index = 0, y_col_index = 1, z_col_index = 2):
    '''
    From an 2D array with 3 columns:
        x         y       dep
        0         0       0.0
        1         0       1.0
        2         0       2.0
        ...       ...     ...
        img_w     0       img_w.0
        --------------------------- first row of the image
        0         1       0.1
        1         1       1.1
        2         1       2.1
        ...       ...     ...
        img_w     1       img_w.1
        --------------------------- second row of the image
        ...
        ...
        0         img_h   0.img_h
        1         img_h   1.img_h
        2         img_h   2.img_h
        ...       ...     ...
        img_w     img_h   img_w.img_h
        --------------------------- last row of the image

    returns a dep image, containing only 1 value per pixel: 

        |----------------------------...------> x
        |0.0       1.0       2.0     ...  img_w.0    
        |0.1       1.1       2.1     ...  img_w.1 
        |0.2       1.2       2.2     ...  img_w.2 
        |0.3       1.3       2.3     ...  img_w.3 
        ...        ...       ...     ...  ...
        |0.img_h   1.img_h   2.img_h ...  img_w.img_h
        v y

     Parameters
     ----------
     data : array
         contains x y z values.
    x_col_index : int, optional
        column of the x values. The default is 0.
    y_col_index : int, optional
        column of the y values. The default is 1.
    z_col_index : int, optional
        column of the z values. The default is 2.

    Returns
    -------
    image : matrix
        contains z values.

    '''
    # # removing nan values
    # data = data[~np.isnan(data).any(axis=1), :]

    image = np.reshape(data[:,z_col_index], [int(round(data[-1,y_col_index])+1),int(round(data[-1,x_col_index])+1)],'C')
    return image


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timer:
    """
    Simple timer for timing code(blocks).

    Parameters
    ----------
    name : str
        name of timer, gets saved in Timer.timers optional
    text : str
        custom text, optional
    start : bool
        automatically start the timer when it's initialized, default is True

    Methods
    -------
    start
        start the timer

    stop
        stop the timer, prints and returns the time
        
    lap
        print the time between this lap and the previous one

    """
    timers = dict()

    def __init__(self, name="", text="{:0.4f} seconds", string_lap = 'lap  : ', string_stop = 'stop : ', start=True):
        self._start_time = None
        self._lap_time = 0.
        self.name = name
        self.text = text
        self.string_lap = string_lap
        self.string_stop = string_stop

        if name:
            # Add new named timers to dictionary of timers
            self.timers.setdefault(name, 0)
        if start:
            self.start()

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is already running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def lap(self, lap_name="", printTime = True):
        """Report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        if self._lap_time:
            current_lap = time.perf_counter() - self._lap_time - self._start_time
            self._lap_time += current_lap
        else:
            self._lap_time = time.perf_counter() - self._start_time
            current_lap = self._lap_time
        if printTime:
            if lap_name:
                print(self.string_lap + self.text.format(current_lap) + ' [' + lap_name + ']')
            else:
                print(self.string_lap + self.text.format(current_lap))
        return current_lap

    def stop(self, printTime = True):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        if printTime:
            print(self.string_stop + self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time
        return elapsed_time
