"""
date: 2022-12-22 15:03:12
note: extension of pandas operations
"""
import numpy as np

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