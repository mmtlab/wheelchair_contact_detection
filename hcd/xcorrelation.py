# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:14:59 2023

@author: giamp
"""
import scipy
import logging
import numpy as np
from hppdWC import utils
from hppdWC import plots

def cutInTime(x, y, interval):
    '''
    Given x (time array), y (values), and interval, erases all the x,y pairs whose x is 
    before interval[0] and after interval[1]
    

    Parameters
    ----------
    x : np.array or list
        time array.
    y : np.array or list
        correspoding values.
    interval : list of 2 values [start finish]
        DESCRIPTION.

    Returns
    -------
    x : np.array or list
        only the values after start and before stop.
    y : np.array or list
        only the corresponding values to x.

    '''
    start = interval[0]
    stop = interval[1]
    if start < stop:
        # first cut signal and then time, time gives the condition
        # cut the tails            
        y = y[x<=stop]
        x = x[x<=stop]
        
        # cut the heads
        y = y[x>=start]
        x = x[x>=start]
        
        # reset the time
        x = x - x[0]
    else:
        logging.warning('not cutting the arrays since stop is before start')
    return x, y


def syncXcorr(signal1, signal2, time1, time2, step = 0.01, \
              interval1 = [0, 0], interval2 = [0, 0]):
    '''
    Computes the delay of signal2 with respect to signal1 using cross correlation.
    
    To do so, a similar pattern should be present in both signals.
    
    "time1" and "time2" contain the relative time of the recording and should: 
        - be in the same measurement unit (eg: seconds)
        - start both from 0
    The returned value "delay" will be in the same measurement unit.
    
    "signal1" is the one that gives the t=0, while the advance/delay in the 
    starting of the recording of "signal2" is computed.
    The returned value is "delay", which expresses:
        - the timing delay of signal2 wrt to signal1, 
        - the OPPOSITE (minus sign) of the timing delay in the recording
        
    If the recording of 2 starts before 1, when plotting the two signals,
    you see the event happening in 1 first and then in 2.
    
    To graphically synchronize them, it's necessary to move 2 towards right
    To timewise synchronize them, it's necessary to cut the first frames of 2 
    (the ones when 2 was already recording and 1 wasn't) and to reset the timer of 2
    
    If "delay" is *POSITIVE*, then signal2 started to be recorded AFTER "delay" time. 
    To synchronize the two signals, it's necessary to add values in the head of
    signal2
    NOT SYNC SIGNALS
    -----------****------- signal1
    
    --------****------- signal2
    
    delay = 3 -> signal2 started to be recorded 3 after 
    SYNC SIGNALS
    -----------****------- signal1
    
    add--------****------- signal2
    
    If "delay" is *NEGATIVE*, then signal2 started to be recorded BEFORE "delay" time. 
    To synchronize the two signals, it's necessary to cut values from the head of
    signal2
    NOT SYNC SIGNALS
    -----------****------- signal1
    
    --------------****------- signal2
    
    delay = -3 -> signal2 started to be recorded 3 before
    SYNC SIGNALS
    -----------****------- signal1
    
    -----------****------- signal2

    Parameters
    ----------
    signal1 : array
        Contains the y value of signal 1
    signal2 : array
        Contains the y value of signal 2
    time1 : array
        Contains the x value of signal 1
    time2 : array
        Contains the x value of signal 2
    step : int, optional
        To perform cross correlation, both signals should be at the same 
        frequency, it's necessary to resample them. The step should be in the 
        same measurement units of time1 and time2
        The default is 0.01.
    interval1 : list of 2 values: [startTime endTime], optional
        Part of the signal1 that should be considered when executing the xcorr. 
        The default is [0, 0], which means the whole signal.
    interval2 : list of 2 values: [startTime endTime], optional
        Part of the signal2 that should be considered when executing the xcorr. 
        The default is [0, 0], which means the whole signal.
    showPlot : bool, optional
        If the function should display a plot regarding the execution. 
        The default is False.
    device1 : string, optional
        Name of device 1 in the plot. 
        The default is 'device 1'.
    device2 : string, optional
        Name of device 2 in the plot. 
        The default is 'device 2'.
    userTitle : string, optional
        To be added in the title
        The default is ''.
        

    Returns
    -------
    delay : float
        Delay in the same temporal measurement unit of the two signals
        If POSITIVE, signal2 started to be recorded AFTER signal1
        If NEGATIVE, signal2 started to be recorded BEFORE signal1
    maxError : float
        maxError = step / 2 

    '''
    # keeping sure that the variables are numpy.arrays
    signal1, _ = utils.toFloatNumpyArray(signal1)
    signal2, _ = utils.toFloatNumpyArray(signal2) 
    time1, _ = utils.toFloatNumpyArray(time1) 
    time2, _ = utils.toFloatNumpyArray(time2)
    
    signal1 = fillNanWithInterp(signal1, time1)
    signal2 = fillNanWithInterp(signal2, time2)

    # # eventually cutting the signal1
    # if interval1 != [0, 0]:
    #     time1, signal1 = cutInTime(time1, signal1, interval1)
            
    # # eventually cutting the signal2
    # if interval2 != [0, 0]:
    #     time2, signal2 = cutInTime(time2, signal2, interval2)
            
    # user delay
    # since the xcorrelation works on the y values only, the cutting of the 
    # signals should be taken into account as an additional delay
    userDelay = interval1[0] - interval2[0]
    
    # resampling both signals on the same frequency
    y1, x1, _ = resampleWithInterp(signal1, time1, step, 'time step')
    y2, x2, _ = resampleWithInterp(signal2, time2, step, 'time step')
    
    # eventually cutting the signal1
    if interval1 != [0, 0]:
        x1, y1 = cutInTime(x1, y1, interval1)
            
    # eventually cutting the signal2
    if interval2 != [0, 0]:
        x2, y2 = cutInTime(x2, y2, interval2)
   
    # putting the values around 0
    y1 = y1 - np.mean(y1)
    y2 = y2 - np.mean(y2)
    
    # normalizing from -1 to 1
    y1 = y1 / np.max(np.abs(y1))
    y2 = y2 / np.max(np.abs(y2))
    
    # compute correlation
    corr = scipy.signal.correlate(y1, y2)
    lags = scipy.signal.correlation_lags(len(y1), len(y2))
    # where there is max correlation
    index = np.argmax(corr)
    
    delay =  lags[index]*step
    # adding the userDelay to the one computed on the signals
    delay = delay + userDelay
    maxError = step/2
    results=[x1, y1, interval1, x2, y2, interval2, delay, lags, step, userDelay, maxError, corr, index]
    return results

def plot_syncXcorr(results, device1, device2, userTitle = '', col1 = 'C0', col2 = 'C1'):
    
    [x1,y1,interval1,x2,y2,interval2,delay,lags,step,userDelay,maxError,corr,index]=results
    
    if delay > 0:
        mainTitle = r"{} ({:.2f}-{:.2f}) started {:.3f} $\pm$ {:.3f} after {} ({:.2f}-{:.2f})".format(device2, interval2[0], interval2[1], np.absolute(delay), maxError, device1, interval1[0], interval1[1])
        mainTitle = r"{} ({:.2f}-{:.2f}) started {:.3f} after {} ({:.2f}-{:.2f})".format(device2, interval2[0], interval2[1], np.absolute(delay), device1, interval1[0], interval1[1])
    elif delay < 0:
        mainTitle = r"{} ({:.2f}-{:.2f}) started {:.3f} $\pm$ {:.3f} before {} ({:.2f}-{:.2f})".format(device2, interval2[0], interval2[1], np.absolute(delay), maxError, device1, interval1[0], interval1[1])
        mainTitle = r"{} ({:.2f}-{:.2f}) started {:.3f} before {} ({:.2f}-{:.2f})".format(device2, interval2[0], interval2[1], np.absolute(delay), device1, interval1[0], interval1[1])
    else:
        mainTitle = r"{} started at the same time of {}".format(device2, device1)
    if userTitle != '':
        mainTitle = mainTitle + ' - ' + userTitle

    fig, ax = plots.drawInSubPlots(\
    listXarrays = \
        [[(x1 + interval1[0]).tolist(),(x2 + interval2[0]).tolist()],\
        (lags*step + userDelay).tolist(), \
        [(x1 + interval1[0]).tolist(),(x2 + interval2[0] +delay).tolist()]],\
    listYarrays = \
        [[y1.tolist(), y2.tolist()], \
        corr,\
        [y1.tolist(), y2.tolist()]], \
    listOfTitles = \
        ['not synchronized signals', \
         'correlation according to shift',\
         'synchronized signals'], \
    sharex = False, nrows = 3, mainTitle = mainTitle, listOfkwargs=[[{'color': col1},{'color': col2}],{'marker':''}], listOfLegends = [[device1, device2], ['']])

    for this_ax in [ax[0], ax[2]]:
        this_ax2 = this_ax.twinx()
        this_ax.set_xlabel('time [s]')
        this_ax.set_ylabel(device1, color = col1)
        this_ax2.set_ylabel(device2, color = col2)
        this_ax.set_xlim(np.min([np.min(x1 + interval1[0]), np.min(x2 + interval2[0]), np.min(x2 + interval2[0] + delay)]), np.max([np.max(x1 + interval1[0]), np.max(x2 + interval2[0]), np.max(x2 + interval2[0] + delay)]))

    this_ax = ax[1]
    this_ax.axvline(lags[index]*step + userDelay, color = 'r')
    this_ax.set_xlabel('lag (time [s])')
    this_ax.set_ylabel('correlation')
    this_ax.set_xlim(np.min(lags*step + userDelay), np.max(lags*step + userDelay))

    return fig, ax
#plots.syncXcorr(x1, y1, interval1, device1, x2, y2, interval2, device2, delay, lags, step, userDelay, maxError, corr, index, userTitle, col1 = col1, col2 = col2)
# plots.syncXcorrOld(x1, y1, interval1, device1, x2, y2, interval2, device2, delay, lags, step, userDelay, maxError, corr, index, userTitle = '', col1 = 'C0', col2 = 'C1')
    

def fillNanWithInterp(y, x = 0, mode = 'linear'):
    '''
    Given an  array containing nans, fills it with the method specified in mode.
    If x is given, the y values returned are the one corresponding to the x specified
    If x is not given, y is assumed to be sampled at a fixed frequency

    Parameters
    ----------
    y : np.array
        original array of values containing nans to be corrected
    x : np.array, optional
        time array of acquisition of signal y. 
        The default is 0, which assumes that y is sampled at a fixed frequency
    mode : string, optional
        kind of interpolation to be performed, passed to scipy.interpolate.interp1d(kind = )
        Please refer to documentation 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html 
        The default is 'linear'.

    Returns
    -------
    yinterp : np.array
        contains the data with nan replaced from interpolated value

    '''
    # keeping sure that the variables are numpy.arrays
    x, _ = utils.toFloatNumpyArray(x)
    y, _ = utils.toFloatNumpyArray(y) 
    
    # if x is not given, it's assumed that the y array is equally spaced
    if np.array_equal(0, x):
        x = np.arange(0, len(y), 1)
        
    # find the indexes where y is not nan
    notNanIndexes = ~np.isnan(y)
    
    # if the first or the last value of y are nan, copy the closest value
    if notNanIndexes[0] == False:
        y[0] = y[notNanIndexes][0]
    if notNanIndexes[-1] == False:
        y[-1] = y[notNanIndexes][-1] 
    
    # find again the indexes where y is not nan
    # now the first and the last value are not nan, and they're the extremes of 
    # the interpolation
    notNanIndexes = ~np.isnan(y)
      
    # considering only the not nan value
    yClean = y[notNanIndexes]
    xClean = x[notNanIndexes]
    
    # feeding the interpolator with only the not nan values and obtaining a function
    finterp = scipy.interpolate.interp1d(xClean, yClean, mode)
    
    # computing the values of function on the original x
    yinterp = finterp(x)
    
    return yinterp

def resampleWithInterp(y, x = 0, xparam = 0.01, param = 'time step', mode = 'linear'):
    '''
    Given a signal y and his time array x, resamples it using interpolation
    the three modes to use this function are:
    - specifying the time *step*: 
        the output is resampled with the given step
    - specifying the *frequency*:
        the output is resampled with the given frequency
    - specifying the *time array*:
        the output is resampled on the given time array
    If signal y has contains nan, they are filled with the function fillNanWithInterp()
    
    Parameters
    ----------
    y : np.array
        original array of values
    x : np.array, optional
        time array of acquisition of signal y. 
        The default is 0, which assumes that y is sampled at a fixed frequency
    xparam : float, integer or array, optional
        if param == 'time step'
            specifies the time step
        if param == 'frequency'
            specifies the frequency
        if param == 'time array'
            is equal to the time array where the resampling should be done. 
        The default is 0.01 and goes with 'time step' specified in param
    param : string, optional
        To specify if the resampling should be done on a signal computed on the 
        given time step, frequency or on the given time array. 
        The default is 'time step' and goes with '0.001' specified in xparam
    mode : string, optional
        kind of interpolation to be performed, passed to scipy.interpolate.interp1d(kind = )
        Please refer to documentation 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html 
        The default is 'linear'.

    Returns
    -------
    yinterp : np.array
        Values of the resampled signal
    xinterp : np.array
        Time array of the resampled signal
    finterp : function
        Interpolator function, only works between the extremities of x 

    '''
    # keeping sure that the variables are numpy.arrays
    x, _ = utils.toFloatNumpyArray(x)
    y, _ = utils.toFloatNumpyArray(y) 
    xparam, _ = utils.toFloatNumpyArray(xparam) 
    
    # if x is not given, it's assumed that the y array is equally spaced
    if np.array_equal(0, x):
        if mode != 'time array':
            x = np.arange(0, len(y), 1)
        else:
            logging.error('asking to resample on a given time array but not \
                  specifiying the input time array')
            return None             
        
    # if y contains at least one nan, fill the space
    if np.isnan(y).any():
        logging.warning('nan values detected, filling them with ' + mode + ' method')
        y = fillNanWithInterp(y, x, mode)
        
    # the three modes to use this function are:
    # - specifying the time *step*
    # - specifying the *frequency*
    # - specifying the *time array*
    validParams = ['time step', 'frequency', 'time array']
    
    if param == validParams[0]: # given step
        step = xparam
        xinterp = np.arange(np.min(x), np.max(x), step)  
    elif param == validParams[1]: # given freq
        freq = xparam
        step = 1/freq
        xinterp = np.arange(np.min(x), np.max(x), step)
    elif param == validParams[2]: # given time array
        xinterp = xparam
        # # eventually cutting the time array 
        # xinterp = xinterp[xinterp<=np.max(x)]
        # xinterp = xinterp[xinterp>=np.min(x)]
        # warning the user if the time array specified exceeds the limits
        if (xinterp[0] < np.min(x) or xinterp[-1] > np.max(x)):
            logging.warning('Using extrapolation: ' + \
                  '\nInterpolator has values between {:.2f} and {:.2f}'\
                      .format(np.min(x), np.max(x)) + \
                  ' and computation between {:.2f} and {:.2f} is asked.'\
                      .format(xparam[0], xparam[-1]))
    else:
        logging.error('not valid param. Valid params are: ' + str(validParams))
        return None
    
    # feeding the interpolator with the input values and obtaining a function
    finterp = scipy.interpolate.interp1d(x, y, kind = mode, fill_value = 'extrapolate')
    
    # computing the values of the function on the xinterp
    yinterp = finterp(xinterp)
    
    return yinterp, xinterp, finterp

