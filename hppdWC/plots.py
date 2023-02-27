"""

"""
#%% imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib.patches
import cv2
import scipy
from scipy.spatial.transform import Rotation as R

import seaborn as sns

from . import utils
from . import geom
#%% utilities
def checkColorList(color_list, nofelements = 2):
    if isinstance(color_list, list):
        if len(color_list) == nofelements:
            return color_list
    else:
        color_list_tmp = []
        color_list_tmp.append(color_list)
        color_list = color_list_tmp

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

def createSubPlots(nOfPlots = 0, sharex = False, sharey = False, nrows = 0, ncols = 0, mainTitle = '', listOfTitles = ['']):
    '''
    Given a number of plots to be created in the same figure, prepares the figure
    and returns the figure and the array (of array if ncols > 1 and nrows > 1) 
    of the axis.

    If nrows is specified, the figure presents nrows rows, the number of columns
    is obtained according to nOfPlots
    If ncols is specified, the figure presents ncols columns, the number of rows
    is obtained according to nOfPlots
    If both are specified, the figure satisfies both requirements, disregarding
    nOfPlots
    If none of the two is specified, nrows and ncols is computed in order to have
    ncols = nrows


    Parameters
    ----------
    nOfPlots : int
        number of subplots to be created.
    sharex : bool, optional
        see documentation of matplotlib.pyplot. The default is False.
    sharey : bool, optional
        see documentation of matplotlib.pyplot. The default is False.
    nrows : int, optional
        number of rows. The default is 0.
    ncols : int, optional
        number of columns. The default is 0.
    mainTitle : string, optional
        supTitle of the figure. The default is ''.
    listOfTitles : list of strings, optional
        list of titles applied to the subplots in the given order. 
        The placeholder for empty title is ''.
        The default is [''].

    Returns
    -------
    fig : figure
        DESCRIPTION.
    ax : array (of array) of axis used to plot
        it's possible to modify the plots recalling 
        ax[..][..] if nrows > 1 and ncols > 1;
        ax[..] if nrows == 1 or ncols == 1

    '''

    # detect number of rows and of columns
    if nrows == 0 and ncols == 0:
        nrows = int(np.ceil(np.sqrt(nOfPlots)))
        ncols = int(np.ceil(nOfPlots/nrows))
    elif nrows == 0 and ncols != 0:
        nrows = int(np.ceil(nOfPlots/ncols))
    elif nrows != 0 and ncols == 0:
        ncols = int(np.ceil(nOfPlots/nrows))

    # add empty titles in the end if some are missing
    listOfTitles.extend(['']*(nOfPlots-len(listOfTitles)))

    # create the figure with subplots
    fig, ax = plt.subplots(nrows, ncols,sharex = sharex, sharey = sharey)
    plt.suptitle(mainTitle)

    # add the titles for every subplot
    for counter in range(nOfPlots):
        if nrows > 1 and ncols > 1:
            this_ax = ax[int(np.floor(counter/ncols)), int(counter % ncols)]
        else:
            this_ax = ax[counter]
        this_ax.set_title(listOfTitles[counter])

    return fig, ax

def drawInSubPlots(listXarrays, listYarrays, sharex = False, sharey = False, nrows = 0, ncols = 0, mainTitle = '', listOfTitles = [''], listOfLegends = [''], listOfkwargs = [{}], common_kwargs = {'marker': '.'}):
    '''
    Given a list of x arrays and a list of y arrays, plots them in subplots
    NB: both listXarrays and listYarrays should be lists.
    If it's necessary to plot more x/y values in one subplot, put them into a list 
    and make sure these arrays are either all vertical or horizontal 
    (can't make some x or y vertical and some x or y horizontal but it's possible to
     make all the x vertical and y horizontal or viceversa)
    use np.ndim / np.squeeze / np.expand_dims eventually
    If only one x array for all the subplots, put it into a list with square 
    brackets: [x], it will be repeated for all the suplots.
    If no x array, use an empty placeholder to declare x: [].

    If only one kwargs for all the subplots, two solutions:
    - put it into a list with square bracket: 
        listOfkwargs= [dictionary], it will be repeated for all the suplots
        (not the best idea, tho)
    - give the dictionary to:
        common_kwargs = dictionary, it will be repeated for all the suplots
        (good idea, this is the aim of common_kwargs)
    Use common_kwargs to apply the style to all the subplots, and listOfkwargs to
    modify every single subplot. Duplicate parameters in listOfkwargs will 
    overwrite the ones in common_kwargs.

    eg0.0: easiest application
        drawInSubplots([x0, x1, x2, x3], [y0, y1, y2, y3])
    will draw a figure with 4 subplots:
        x0,y0         | x1,y1
        ---------------------------
        x2,y2        | x3,y3

    eg0.1: more lines in the same subplot with same x
        drawInSubplots([x0, x1, x2, x3],   [y0, y1, y2,[y0, y1, y2]])
        drawInSubplots([x0, x1, x2, [x3]], [y0, y1, y2,[y0, y1, y2]])
    will draw a figure with 4 subplots:
        x0,y0         | x1,y1
        ---------------------------
        x2,y2        | x3,y0 & x3,y1 & x3,y2
    
    eg0.2: more lines in the same subplot with different x
        drawInSubplots([x0, x1, x2, [x0, x1, x2]], [y0, y1, y2, [y0, y1, y2]])
    will draw a figure with 4 subplots:
        x0,y0         | x1,y1
        ---------------------------
        x2,y2        | x0,y0 & x1,y1 & x2,y2

    eg0.3: if some elements are forgotten, they are replaced by repetition
        drawInSubplots([x0, x1, x2, [x0, x1]], [y0, y1, y2, [y0, y1, y2]])
    will draw a figure with 4 subplots:
        x0,y0         | x1,y1
        ---------------------------
        x2,y2        | x0,y0 & x1,y1 & x0,y2
    Note that for subplot bottom right only x0 and x1 are given for y0, y1, y2.
    As a consequence, y2 is plotted using x0

     eg1.0: it's possible not to declare x
         drawInSubplots([], [y0, y1, y2, y3])
     will draw a figure with 4 subplots:
         y0            | y1
         ---------------------------
         y2           | y3

    eg1.1: it's possible not to declare x partially for different subplots
        drawInSubplots([x0, [], x2, []], [y0, y1, y2, y3])
    will draw a figure with 4 subplots:
        x0,y0         | y1
        ---------------------------
        x2,y2         | y3

    eg1.2: it's not possible not to declare x partially for different lines in the same subplot
    eg1.2a: x is declared [] for all the lines of the subplot bottom right
        drawInSubplots([x0, x1, x2, []], [y0, y1, y2, [y0, y1, y2]])
    will draw a figure with 4 subplots:
        x0,y0         | x1,y1
        ---------------------------
        x2,y2        | y0 & y1 & y2

    eg1.2b: x is not declared for any line of the subplot bottom right
        drawInSubplots([x0, x1, x2], [y0, y1, y2, [y0, y1, y2]])
    will draw a figure with 4 subplots:
        x0,y0         | x1,y1
        ---------------------------
        x2,y2        | x0,y0 & x0,y1 & x0,y2
    Note that since x axis for subplot bottom right is not given, x0 by means of repetition

    eg1.2c: x is declared for all the lines of the subplot bottom right by means 
    of rep (same as eg0.1)
        drawInSubplots([x0, x1, x2, x3], [y0, y1, y2, [y0, y1, y2]])
        drawInSubplots([x0, x1, x2, [x3]], [y0, y1, y2, [y0, y1, y2]])
    will draw a figure with 4 subplots:
        x0,y0         | x1,y1
        ---------------------------
        x2,y2        | x3,y0 & x3,y1 & x3,y2

    eg1.2d: x is declared partially for the lines of the same subplot[1,1] -> ERROR
        drawInSubplots([x0, x1, x2, [x0,[], x2]], [y0, y1, y2, [y0, y1, y2]])
    will give error since for subplot[1,1]:
        - x0,y0
        - [],y1
        - x2,y2       

    To use kwargs, create a list of dictionaries:
    eg1: 
        listOfkwargs = [*[{'color': 'red'}]*3, *[{'marker': 'o'}]]
        drawInSubplots([x], [y0, y1, y2,[y0, y1, y2]], listOfkwargs = listOfkwargs)
        will draw the first 3 subplots in red and the fourth one with 
        the 'o' marker for all the three of them
        (in this case listOfkwargs is:
         [{'color': 'red'}, {'color': 'red'}, {'color': 'red'}, {'marker': 'o'}])

    eg2: 
        listOfkwargs = [*[{'color': 'red'}]*3, *[[{'marker': 'o', 'linewidth': 0.5}, {'color': 'blue'}]]]
        drawInSubplots([x], [y0, y1, y2,[y0, y1, y2]], listOfkwargs = listOfkwargs)
        will draw the first 3 subplots in red and, for the fourth one:
            'o' marker and 2 linewidth for the first and the third
            'blue' color for the second
        (in this case listOfkwargs is:
            [{'color': 'red'},
             {'color': 'red'},
             {'color': 'red'},
             [{'marker': 'o', 'linewidth': 0.5}, {'color': 'blue'}]])

    Parameters
    ----------
    listXarrays : TYPE
        DESCRIPTION.
    listYarrays : TYPE
        DESCRIPTION.
    sharex : bool, optional
        see documentation of matplotlib.pyplot. The default is False.
    sharey : bool, optional
        see documentation of matplotlib.pyplot. The default is False.
    nrows : int, optional
        number of rows. The default is 0.
    ncols : int, optional
        number of columns. The default is 0.
    mainTitle : string, optional
        supTitle of the figure. The default is ''.
    listOfTitles : list of strings, optional
        list of titles applied to the subplots in the given order. 
        The placeholder for empty title is ''.
        The default is [''].
    listOfkwargs : list of dict, optional
        list of kwargs applied to the subplots in the given order. 
        The placeholder for no kwargs is {}.
        The default is [{}].
    common_kwargs : dict, optional
        kwargs applied to every line of the plot. If some parameters are in 
        contrast with the one specified in listOfkwargs, listOfkwargs wins.
        The default is {'marker': '.'}.

    Returns
    -------
    fig : figure
        DESCRIPTION.
    ax : array (of array) of axis used to plot
        it's possible to modify the plots recalling 
        ax[..][..] if nrows > 1 and ncols > 1;
        ax[..] if nrows == 1 or ncols == 1

    '''

    nOfPlots = len(listYarrays)

    listXarrays = makeList(listXarrays)
    if not listXarrays:
        listXarrays = [[]]

    # expand lists
    # if only one, it's repeated for all the this_ax, otherwise it simply overflows
    listXarrays = listXarrays * nOfPlots
    listOfLegends = listOfLegends * nOfPlots
    listOfkwargs = listOfkwargs * nOfPlots

    fig, ax = createSubPlots(nOfPlots, sharex, sharey, nrows, ncols , mainTitle, listOfTitles)

    nrows = len(ax)
    if np.ndim(ax) == 2:
        ncols = len(ax[0])
    elif np.ndim(ax) == 1:
        nrows = 1

    # for every plot to draw
    for counter in range(nOfPlots):
        # detect the axis
        if nrows > 1 and ncols > 1:
            this_ax = ax[int(np.floor(counter/ncols)), int(counter % ncols)]
        else:
            this_ax = ax[counter]
        this_ax.grid()

        # squeezing to remove an eventual redundant dimension and conversion to list


        listYarrays[counter] = np.squeeze(listYarrays[counter]).tolist()
        listXarrays[counter] = np.squeeze(listXarrays[counter]).tolist()

        # if more than one y array to plot for this axis
        if not np.isscalar(listYarrays[counter][0]):
            # if no x array is given -> ok: repeated for all the y
            if listXarrays[counter] == []:
                listXarrays[counter] = [listXarrays[counter]] * len(listYarrays[counter])
            # if only one x array is given -> ok: repeated for all the y
            elif np.isscalar(listXarrays[counter][0]):
               listXarrays[counter] = [listXarrays[counter]] * len(listYarrays[counter])
            # if more than one x array is given -> ok: corresponding to the y
            else:
                # in case less x arrays than y arrays are given, repeat the x ones
                listXarrays[counter] = listXarrays[counter] * len(listYarrays[counter])

            # if only one dictionary of kwargs, could also be empty (default)
            if isinstance(listOfkwargs[counter], dict):
                # the same kwargs is applied to all the elements of this subplot
                # creation of the list
                listOfkwargs[counter] = [listOfkwargs[counter]] * len(listYarrays[counter])
            # the list is repeated for all the elements, if it's empty, it's passed an empty dict
            if isinstance(listOfkwargs[counter], list):
                listOfkwargs[counter] = listOfkwargs[counter] * len(listYarrays[counter])
             # for every x, y to be plotted and kwargs to be applied
            for x, y, plt_kwargs in zip(np.squeeze(listXarrays[counter]), listYarrays[counter], listOfkwargs[counter], ):
                # make a copy of the common k_wargs
                this_plt_kwargs = common_kwargs.copy()
                # add the single parameters contained in listOfkwargs
                this_plt_kwargs.update(plt_kwargs)
                if len(x):
                    this_ax.plot(x, y, **this_plt_kwargs)
                else:
                    this_ax.plot(y, **this_plt_kwargs)

        # if only one y array to plot for this axis
        # elif np.ndim(listYarrays[counter]) == 1:
        else:
            # if no x array is given -> ok
            if listXarrays[counter] == []:
                x = listXarrays[counter]
            # # if only one x array is given -> ok
            elif np.isscalar(listXarrays[counter][0]):
               x = listXarrays[counter]
            # if more than one x array is given -> shouldn't be -> extract from list
            else:
                x = listXarrays[counter][0]

            y = listYarrays[counter]

            # if kwargs are packed in a dictionary -> ok
            if isinstance(listOfkwargs[counter], dict):
                plt_kwargs = listOfkwargs[counter]
            # if inside a list (might happen due to iteration of kwargs) -> extract from list
            elif isinstance(listOfkwargs[counter], list):
                plt_kwargs = listOfkwargs[counter][0]

            # if there aren't kwargs for this plot
            if listOfkwargs[counter] == {}:
                if len(x):
                    this_ax.plot(x, y, **common_kwargs)
                else:
                    this_ax.plot(y, **common_kwargs)
            else: # if there are
                # make a copy of the common k_wargs
                this_plt_kwargs = common_kwargs.copy()
                # add the single parameters contained in listOfkwargs
                this_plt_kwargs.update(plt_kwargs)
                if len(x):
                    this_ax.plot(x, y, **this_plt_kwargs)
                else:
                    this_ax.plot(y, **this_plt_kwargs)
        # eventually add the legend
        if listOfLegends[counter] != '':
            this_ax.legend(listOfLegends[counter])

    return fig, ax

def drawInPlot(listXarrays, listYarrays, mainTitle = '', legend = [], listOfkwargs = [{}], common_kwargs = {'marker': '.'}):

    listXarrays = makeList(listXarrays)
    listYarrays = makeList(listYarrays)

    nOfPlots = len(listYarrays)

    if not listXarrays:
        listXarrays = [[]]

    # expand lists
    # if only one, it's repeated for all the this_ax, otherwise it simply overflows
    listXarrays = listXarrays * nOfPlots
    listOfkwargs = listOfkwargs * nOfPlots

    fig = plt.figure()
    plt.grid()

    # if more than one y array to plot for this axis
    if not np.isscalar(listYarrays[0]):
        # if no x array is given -> ok: repeated for all the y
        if listXarrays == []:
            listXarrays = listXarrays * len(listYarrays)
        # if only one x array is given -> ok: repeated for all the y
        elif np.isscalar(listXarrays[0]):
           listXarrays = [listXarrays] * len(listYarrays)
        # if more than one x array is given -> ok: corresponding to the y
        else:
            # in case less x arrays than y arrays are given, repeat the x ones
            listXarrays = listXarrays * len(listYarrays)

        # if only one dictionary of kwargs, could also be empty (default)
        if isinstance(listOfkwargs, dict):
            # the same kwargs is applied to all the elements of this subplot
            # creation of the list
            listOfkwargs = [listOfkwargs] * len(listYarrays)
        # the list is repeated for all the elements, if it's empty, it's passed an empty dict
        if isinstance(listOfkwargs, list):
            listOfkwargs = listOfkwargs * len(listYarrays)
         # for every x, y to be plotted and kwargs to be applied
        #for x, y, plt_kwargs in zip(np.squeeze(listXarrays), listYarrays, listOfkwargs):
        for x, y, plt_kwargs in zip(listXarrays, listYarrays, listOfkwargs):
            # make a copy of the common k_wargs
            this_plt_kwargs = common_kwargs.copy()
            # add the single parameters contained in listOfkwargs
            # this_plt_kwargs.update(plt_kwargs)
            for el in plt_kwargs:
                this_plt_kwargs.update(el)
            if len(x):
                plt.plot(x, y, **this_plt_kwargs)
            else:
                plt.plot(y, **this_plt_kwargs)

    # if only one y array to plot for this axis
    # elif np.ndim(listYarrays) == 1:
    else:
        # if no x array is given -> ok
        if listXarrays == []:
            pass
        # # if only one x array is given -> ok
        elif np.isscalar(listXarrays[0]):
           x = listXarrays
        # if more than one x array is given -> shouldn't be -> extract from list
        else:
            x = listXarrays[0]

        y = listYarrays

        # if kwargs are packed in a dictionary -> ok
        if isinstance(listOfkwargs, dict):
            plt_kwargs = listOfkwargs
        # if inside a list (might happen due to iteration of kwargs) -> extract from list
        elif isinstance(listOfkwargs, list):
            plt_kwargs = listOfkwargs[0]

        # if there aren't kwargs for this plot
        if listOfkwargs== {}:
            if len(x):
                plt.plot(x, y, **common_kwargs)
            else:
                plt.plot(y, **common_kwargs)
        else: # if there are
            # make a copy of the common k_wargs
            this_plt_kwargs = common_kwargs.copy()
            # add the single parameters contained in listOfkwargs
            this_plt_kwargs.update(plt_kwargs)
            if len(x):
                plt.plot(x, y, **this_plt_kwargs)
            else:
                plt.plot(y, **this_plt_kwargs)
    if legend != []:
        plt.legend(legend)

    #plt.suptitle(mainTitle)
    plt.suptitle(mainTitle)

    return fig


#%% images
def highlightPartOfImage(image, maskInterest, coeff = 0.7, colorNotInterest = [255,255,255]):
    '''
    Given an image and a mask where True is "highlight this part", adds a shadow to the other parts

    Parameters
    ----------
    image : matrix M*N*3 or M*N*1 
        where it's necessary to highlight one part.
    maskInterest : matrix M*N*1 of boolean
        True corresponding to the parts where the image is interesting.
    coeff : float between 0 and 1, optional
        intensity of the not interest part modification. 
        The default is 0.7.
    colorNotInterest : list 1*3, optional
        color of the shade assumed. The default is [255,255,255].

    Returns
    -------
    highlightedImg : image

    '''
    mask = image.copy()
    mask[~maskInterest] = colorNotInterest
    highlightedImg = cv2.addWeighted(mask, coeff, image, 1, 0)
    return highlightedImg

def circlesOnImage(image, circles, centreCol = (255,0,0), circleCol = (0,255,0)):
    '''
    Draws all the circles on the image

    Parameters
    ----------
    image : matrix M*N*3
        image, presumably the one where the lines are found.
    circles : array of arrays
        array([[[xc, yc, r]],
               [[xc, yc, r]],
               ...
               [[xc, yc, r]]])
        the coordinates of centre and radius defining a circle
    centreCol : array 1*3, optional
        color of the centre of the circle. The default is (255,0,0).
    circleCol : array 1*3, optional
        color of the circles. The default is (0,255,0).

    Returns
    -------
    circle_image : image
        with circles drawn on it.

    '''
    circlesInt = np.uint16(np.around(circles))
    circle_image = image.copy()
    for i in circlesInt[0,:]:
        # draw the outer circle
        cv2.circle(circle_image,(i[0],i[1]),i[2],circleCol,2)
        # draw the center of the circle
        cv2.circle(circle_image,(i[0],i[1]),1,centreCol,3)
    return circle_image

def linesOnImage(image, lines, lineCol = (255,0,0)):
    '''
    Draws all the lines on the image

    Parameters
    ----------
    image : matrix M*N*3
        image, presumably the one where the lines are found.
    lines : array of arrays
        array([[[x1, y1, x2, y2]],
               [[x1, y1, x2, y2]],
               ...
               [[x1, y1, x2, y2]]])
        contains the coordinates of each pair of points defining a line
    lineCol :  array 1*3, optional
        color of the lines. The default is (255,0,0).

    Returns
    -------
    lines_edges : image
        with lines drawn on it.

    '''
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),1)
    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return lines_edges

def roiWcHrOnImage(image, wheel_centre, handrim, xmin = 0, xmax = 0, ymin = 0, ymax = 0):
    '''
    Draws reference frame of the wheel, handrim and region of interest on image

    Parameters
    ----------
    image : img
        DESCRIPTION.
    wheel_centre : list of 2 floats
        ['wxc', 'wyc']
    handrim : list of 3 floats
        ['hrxc', 'hryc', 'hrr']
    xmin : int, optional
        of hand roi. The default is 0.
    xmax : int, optional
        of hand roi. The default is 0.
    ymin : int, optional
        of hand roi. The default is 0.
    ymax : int, optional
        of hand roi. The default is 0.

    Returns
    -------
    image_drawn : img
        DESCRIPTION.

    '''
    image_drawn = image.copy()
    # draw roi
    cv2.rectangle(image_drawn, (xmin, ymin), (xmax, ymax), (255,255,255), 2)
    # horizontal line for wheel centre
    cv2.line(image_drawn,(int(wheel_centre[0])-20, int(wheel_centre[1])),(int(wheel_centre[0]+20), int(wheel_centre[1])),(0,0,255),2)
    # vertical line for wheel centre
    cv2.line(image_drawn,(int(wheel_centre[0]), int(wheel_centre[1])-20),(int(wheel_centre[0]), int(wheel_centre[1])+20),(0,0,255),2)
    # handrim
    image_drawn = circlesOnImage(image_drawn, [[list(handrim[i] for i in [0,1,len(handrim)-1])]], centreCol = (0,255,255), circleCol = (0,255,0))

    return image_drawn

#%% plots
def showPBPDetection(ergo_data, ergo_data_pbp, mainTitle = ''):
    '''
    Given 

    Parameters
    ----------
    ergo_data : pandas dataframe of 11 columns. 
    The variables used in this function are: time, force, torque and power
       time      force     speed  ...       dist      work     uforce
3001  30.01  37.883817  0.077042  ...   0.000173  0.008107  39.086478
3002  30.02  40.473189  0.093611  ...   0.000410  0.010524  41.758052
     
    ergo_data_pbp : padas dataframe of 24 columns.
        The variables used in this function are: tstart, tstop and tpeak
    start  stop  peak  tstart  ...  ctime    reltime      cwork   negwork
1    3153  3157  3154   31.53  ...   0.47   8.510638  -0.771177 -1.719112
2    3200  3222  3218   32.00  ...   0.73  30.136986   6.275269 -1.902011

    mainTitle : string
        for the title.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    '''
    fig = plt.figure()
    plt.plot(ergo_data['time'], ergo_data['force'], '.-')
    plt.plot(ergo_data['time'], ergo_data['torque'], '.-')
    plt.plot(ergo_data['time'], ergo_data['power'], '.-')
    plt.legend(['force', 'torque', 'power'])
    for starts in ergo_data_pbp['tstart']:
        plt.axvline(starts, linestyle = '--', color = 'black')
    for stops in ergo_data_pbp['tstop']:
        plt.axvline(stops, linestyle = '--', color = 'black')
    for peaks in ergo_data_pbp['tpeak']:
        plt.axvline(peaks, linestyle = '--', color = 'red')
     
    for start, end in zip(ergo_data_pbp['tstart'], ergo_data_pbp['tstop']):
       plt.axvspan(start, end, alpha=0.2, color = 'darkturquoise')
    plt.grid()
    plt.suptitle(mainTitle)
    plt.tight_layout()

    return fig

def startStopInPlot(x, listYarrays, starts, stops, peaks = 0, mainTitle = '', legend = [], listOfkwargs = [{}], common_kwargs = {'marker': '.'}):

    fig = drawInPlot(x, listYarrays, mainTitle, legend, listOfkwargs, common_kwargs)

    for start, stop in zip(starts, stops):
        plt.axvline(start, linestyle = '--', color = 'black')
        plt.axvline(stop, linestyle = '--', color = 'black')
        plt.axvspan(start, stop, alpha = 0.2, color = 'darkturquoise')
    if not(np.isscalar(peaks)):
        for peak in peaks:
            plt.axvline(peak, linestyle = '--', color = 'red')
    return fig

def compareStartStopInPlot(x, listYarrays, starts1, stops1, starts2, stops2, peaks1 = 0, peaks2 = 0, mainTitle = '', legend = [], listOfkwargs = [{}], common_kwargs = {'marker': '.'}):

    fig = drawInPlot(x, listYarrays, mainTitle, legend, listOfkwargs, common_kwargs)
    counter = -1
    for starts, stops, peaks in zip([starts1, starts2], [stops1, stops2], [peaks1, peaks2]):
        counter += 1
        for start, stop in zip(starts, stops):
            if counter == 0:
                pass
                # plt.axvline(start, linestyle = '--', color = 'black')
                # plt.axvline(stop, linestyle = '--', color = 'black')
                plt.axvspan(start, stop, alpha = 0.2, color = 'darkturquoise')
            elif counter == 1:
                plt.axvline(start, linestyle = '--', color = 'C3')
                plt.axvline(stop, linestyle = '--', color = 'C3')
                plt.axvspan(start, stop, alpha = 0.1, color = 'red')
        if not(np.isscalar(peaks)):
            for peak in peaks:
                plt.axvline(peak, linestyle = '--', color = 'red')
    return fig

def startStopInSubPlots(x, listYarrays, starts, stops, peaks = 0, \
    sharex = False, sharey = False, nrows = 0, ncols = 0, \
    mainTitle = '', listOfTitles = [''], listOfLegends = [''], listOfkwargs = [{}], common_kwargs = {'marker': '.'}):

    fig, ax = drawInSubPlots(x, listYarrays, sharex, sharey, nrows, ncols, mainTitle, listOfTitles, listOfLegends, listOfkwargs, common_kwargs)


    counter = -1
    for axes in ax:
        # in case ax is only one rows or one columns
        if np.ndim(axes)==0:
            axes = makeList(axes)
        for this_ax in axes:
            counter+=1
            # to avoid drawing lines in empty subplots
            if counter>=len(listYarrays):
                break
            for start, stop in zip(starts, stops):
                this_ax.axvline(start, linestyle = '--', color = 'black')
                this_ax.axvline(stop, linestyle = '--', color = 'black')
                this_ax.axvspan(start, stop, alpha = 0.2, color = 'darkturquoise')
            if not(np.isscalar(peaks)):
                for peak in peaks:
                    this_ax.axvline(peak, linestyle = '--', color = 'red')
    return fig, ax


def syncXcorrOld(x1, y1, interval1, device1, x2, y2, interval2, device2, delay, lags, step, userDelay, maxError, corr, index, userTitle = '', col1 = 'C0', col2 = 'C1'):

    fig, ax = plt.subplots(3)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
    # plt.tight_layout()
    if delay > 0:
        mainTitle = r"{} ({:.2f}-{:.2f}) started {:.3f} $\pm$ {:.3f} after {} ({:.2f}-{:.2f})".format(device2, interval2[0], interval2[1], np.absolute(delay), maxError, device1, interval1[0], interval1[1])
    elif delay < 0:
        mainTitle = r"{} ({:.2f}-{:.2f}) started {:.3f} $\pm$ {:.3f} before {} ({:.2f}-{:.2f})".format(device2, interval2[0], interval2[1], np.absolute(delay), maxError, device1, interval1[0], interval1[1])
    else:
        mainTitle = r"{} started at the same time of {}".format(device2, device1)
    if userTitle != '':
        mainTitle = mainTitle + ' - ' + userTitle   
        
    fig.suptitle(mainTitle)

    this_ax = ax[0]
    this_ax.plot(x1 + interval1[0], y1, '.', color = col1)
    this_ax2 = this_ax.twinx()
    this_ax2.plot(x2 + interval2[0], y2, '.', color = col2)
    this_ax.grid()
    this_ax.set_title('not synchronized signals')
    this_ax.set_xlabel('time')
    this_ax.set_ylabel(device1, color = col1)
    this_ax2.set_ylabel(device2, color = col2)

    this_ax = ax[1]
    this_ax.plot(lags*step + userDelay, corr)
    this_ax.axvline(lags[index]*step + userDelay, color = 'r')
    this_ax.grid()
    this_ax.set_title('correlation according to shift')
    this_ax.set_xlabel('lag [time]')
    this_ax.set_ylabel('correlation')

    this_ax = ax[2]
    this_ax.plot(x1 + interval1[0], y1, '.', color = col1)
    this_ax2 = this_ax.twinx()
    this_ax2.plot(x2 + interval2[0] + delay, y2, '.', color = col2)
    this_ax.grid()
    this_ax.set_title('synchronized signals')
    this_ax.set_xlabel('time')
    this_ax.set_ylabel(device1, color = col1)
    this_ax2.set_ylabel(device2, color = col2)

    return fig, ax

def syncXcorr(x1, y1, interval1, device1, x2, y2, interval2, device2, delay, lags, step, userDelay, maxError, corr, index, userTitle = '', col1 = 'C0', col2 = 'C1'):

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

    fig, ax = drawInSubPlots(\
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

def checkTiming(d_time_array, step, mainTitle):
    fig, ax = plt.subplots(1,2, sharey = True)
    plt.suptitle(mainTitle)

    this_ax = ax[0]
    this_ax.plot(d_time_array, '.-', alpha = 0.2, lw = 10)
    this_ax.axhline(step, color = 'r', ls = '--', lw = 1)
    this_ax.grid(True)
    this_ax.set_xlabel('sample #')
    this_ax.set_title('during the test')
     
    this_ax = ax[1]
    this_ax.hist(d_time_array, bins = 20, alpha = 0.5, orientation='horizontal')
    this_ax.axhline(step, color = 'r', ls = '--', lw = 1)
    this_ax.grid(True)
    this_ax.set_xlabel('n of samples')
    this_ax.set_title('distribution')

    return fig, ax

def checkTimingMultipleDevices(dt_all, step_all, freq_all, dt_mean_all, dt_std_all, mainTitle = '', devicesName = None):

    nDevices = len(dt_all)

    if mainTitle == '':
        mainTitle = 'timing of acquisition'
    # create the name of the devices, if not specified
    if not devicesName:
        devicesName = ['dev' + str(x) for x in range(nDevices)]
    colors = ['C' + str(x) for x in range(nDevices)]

    fig, ax = createSubPlots(nrows = nDevices, ncols = 2, sharey = 'row', mainTitle = mainTitle)
    for i in range(nDevices):
        this_ax = ax[i, 0]
        this_ax.plot(dt_all[i], '.-', alpha = 0.2, lw = 10, color = colors[i])
        this_ax.axhline(step_all[i], color = 'r', ls = '--', lw = 1)
        this_ax.grid(True)

        # only for the bottom graph
        if i == nDevices-1:
            this_ax.set_xlabel('sample #')

        this_ax = ax[i, 1]
        this_ax.hist(dt_all[i], bins = 20, alpha = 0.5, orientation='horizontal', color = colors[i])
        this_ax.axhline(step_all[i], color = 'r', ls = '--', lw = 1)
        this_ax.grid(True)
        # only for the bottom graph
        if i == nDevices-1:
            this_ax.set_xlabel('n of samples')
        this_ax.set_title(r"{}: mean: {:.4f} - std: {:.4f} s. Nominal value: {:.4f} s [{:.2f} Hz]".\
        format(devicesName[i].upper(), dt_mean_all[i], dt_std_all[i], step_all[i], freq_all[i]))

    return fig, ax

def compareDataFilling(raw, fillNanLin, fillNanCub, mainTitle = ''):

    fig, ax = drawInSubPlots([raw['time']], \
    [raw['angle 0'], fillNanLin, fillNanCub,[fillNanCub, fillNanLin, raw['angle 0']]],\
    mainTitle = mainTitle, listOfTitles = ['only the data available', 'nan filled with linear', 'nan filled with cubic spline', 'comparison'], sharex = True, sharey = True,\
    listOfkwargs = [{'color':'C0'}, {'color':'C1'}, {'color':'C2'}, [{'color':'C2'}, {'color':'C1'}, {'color':'C0'}]])

    this_ax = ax[0,0]
    for axes in ax:
        for this_ax in axes:
            this_ax.set_xlabel('time')
    return fig, ax

def linesAndLinesOnImage(lines, image, mainTitle = '', indexes = []):
    '''
    Given lines, structure in the following shape:
    array([[[x1, y1, x2, y2]],
           [[x1, y1, x2, y2]],
           ...
           [[x1, y1, x2, y2]]])
    containing the coordinates of each pair of points defining a line,
    and an image (theorically the one where the lines are found)
    plots:
        on the left the found lines with plt.plot()
        on the right the found lines in red on the image with plt.imshow()
    

    Parameters
    ----------
    lines : array
        array([[[x1, y1, x2, y2]],
               [[x1, y1, x2, y2]],
               ...
               [[x1, y1, x2, y2]]]).
    image : matrix M*N*1 or M*N*3
        image, presumably the one where the lines are found.
    mainTitle : string, optional
        title to add to the plot. The default is ''.

     Returns
     -------
     fig : figure
         DESCRIPTION.
     ax : array (of array) of axis used to plot
         it's possible to modify the plots recalling 
         ax[..][..] if nrows > 1 and ncols > 1;
         ax[..] if nrows == 1 or ncols == 1

    '''
    fig, ax = plt.subplots(1,2, sharey = True, sharex = True, subplot_kw={'aspect': 1})
    if mainTitle != '':
        fig.suptitle(mainTitle)

    this_ax = ax[0]
    counter = -1
    for line in lines:
        counter += 1
        for x1,y1,x2,y2 in line:
            if indexes == []:
                this_ax.plot([x1, x2],[y1, y2],'-o', color = 'C' + str(counter), label = 'line '+ str(counter))
            else:
                this_ax.plot([x1, x2],[y1, y2],'-o', color = 'C' + str(indexes[counter]), label = 'line '+ str(indexes[counter]))

    this_ax.grid()
    this_ax.invert_yaxis()
    this_ax.legend()

    this_ax = ax[1]
    this_ax.grid()
    this_ax.imshow(linesOnImage(image,lines))
    return fig, ax

def circlesImagesSubplot(image, circles, mainTitle = ''):
    '''
    Given an array of n circles, plots n times the image in 2 columns, for each 
    image one circle

    Parameters
    ----------
    image : matrix M*N*1 or M*N*3
        image, presumably the one where the circles are found.
    circles : array of arrays
        array([[[xc, yc, r]],
               [[xc, yc, r]],
               ...
               [[xc, yc, r]]])
        the coordinates of centre and radius defining a circle
    mainTitle : TYPE, optional
        DESCRIPTION. The default is ''.

    
    Returns
    -------
    fig : figure
        DESCRIPTION.
    ax : array (of array) of axis used to plot
        it's possible to modify the plots recalling 
        ax[..][..] if nrows > 1 and ncols > 1;
        ax[..] if nrows == 1 or ncols == 1

    '''
    ncircles = circles.shape[1]
    fig, ax = plt.subplots(int(np.ceil(ncircles/2)),2, sharey = True, sharex = True, subplot_kw={'aspect': 1})
    fig.suptitle(mainTitle)
    counter = -1
    for circle in circles[0,:]:
        counter += 1
        img = circlesOnImage(image, [[circle]])

        this_ax = ax[int(counter/2),int(counter%2)]
        this_ax.grid()
        this_ax.set_title('xc: {:.2f}, yc: {:.2f}, r: {:.2f}'.format(circle[0], circle[1], circle[2]))
        this_ax.imshow(img)
    return fig, ax

def colorsAlongTheLine(linesColors, color_order = ['red', 'green', 'blue'], mainTitle = ''):
    '''
    Plots the value of each pixel for the RGB (or BGR) channel    

    Parameters
    ----------
    linesColors : array
        of n lines containing for each line 3 columns (corresponding to the RGB or BGR channels) of npoints rows.
    color_order : array of string, optional
        order of the color channels. The default is ['red', 'green', 'blue'].
    mainTitle : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    fig : figure
        DESCRIPTION.
    ax : array (of array) of axis used to plot
        it's possible to modify the plots recalling 
        ax[..][..] if nrows > 1 and ncols > 1;
        ax[..] if nrows == 1 or ncols == 1

    '''
    fig, ax = plt.subplots(len(linesColors), sharey = True, sharex = True)
    fig.suptitle(mainTitle)
    counter = -1
    for line in linesColors:
        counter += 1
        ch0 = line[0][:,0]
        ch1 = line[0][:,1]
        ch2 = line[0][:,2]
    
        this_ax = ax[counter]
    
        this_ax.plot(ch0, color = color_order[0])
        this_ax.plot(ch1, color = color_order[1])
        this_ax.plot(ch2, color = color_order[2])
        this_ax.set_facecolor('C'+str(counter))
        this_ax.patch.set_alpha(0.2)
        this_ax.grid()
        this_ax.set_ylim(0,255)
        this_ax.set_title('line'+str(counter))
    return fig, ax

def gaussColorsAlongTheLine(lines_df, color_order = ['red', 'green', 'blue'], mainTitle = ''):
    '''
    Plots the gaussian distribution of color channel each row of lines_df in
    the specified order

    Parameters
    ----------
    lines_df : pandas dataframe
        contains mean chX and std chX.
    color_order : array of string, optional
        order of the color channels. The default is ['red', 'green', 'blue'].
    mainTitle : TYPE, optional
        DESCRIPTION. The default is ''.

    
    Returns
    -------
    fig : figure
        DESCRIPTION.
    ax : array (of array) of axis used to plot
        it's possible to modify the plots recalling 
        ax[..][..] if nrows > 1 and ncols > 1;
        ax[..] if nrows == 1 or ncols == 1

    '''

    fig, ax = plt.subplots(lines_df.shape[0], sharey = True, sharex = True)
    fig.suptitle(mainTitle)

    counter = -1
    for i in range(lines_df.shape[0]):
        counter += 1
        this_ax = ax[counter]
        line = lines_df.iloc[i,:]
        for ch_index in range(3):
    
            mu = line['mean ch'+str(ch_index)]
            sigma = line['std ch'+str(ch_index)]
    
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            this_ax.plot(x, scipy.stats.norm.pdf(x, mu, sigma), color = color_order[ch_index], label = color_order[ch_index] + ' mean: {:.2f} std {:.2}'.format(mu, sigma))

        mu = line['mean of mean ch']
        sigma = line['mean of std ch']

        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        this_ax.plot(x, scipy.stats.norm.pdf(x, mu, sigma), color ='k', label = 'all channels mean: {:.2f} std {:.2}'.format(mu, sigma))

        this_ax.set_facecolor('C'+str(counter))
        this_ax.patch.set_alpha(0.2)
        this_ax.grid()
        this_ax.legend()
        this_ax.set_title('line'+str(counter))
    return fig, ax

def detectedCentreOfWheel(image, x_centre_abs, y_centre_abs, mainTitle = '', lines = np.array([]), xmin = 0, ymin = 0, indexes = []):
    '''
    Plots the image with the detected centre of the wheel. If lines are given, draws them as wheel according to the shift given from xmin and ymin. If indexes is given, draws the lines of the given colors

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    x_centre_abs : TYPE
        DESCRIPTION.
    y_centre_abs : TYPE
        DESCRIPTION.
    mainTitle : TYPE, optional
        DESCRIPTION. The default is ''.
    lines : TYPE, optional
        DESCRIPTION. The default is np.array([]).
    xmin : TYPE, optional
        DESCRIPTION. The default is 0.
    ymin : TYPE, optional
        DESCRIPTION. The default is 0.
    indexes : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    fig : figure object
        to manipulate the whole figure.

    '''
    fig = plt.figure()
    plt.imshow(image)
    if not lines.size == 0:
        counter = -1
        for line in lines:
            counter += 1
            for x1,y1,x2,y2 in line:
                plt.plot([x1+xmin, x2+xmin],[y1+ymin, y2+ymin],'-o', color = 'C' + str(indexes[counter]), label = 'line '+ str(indexes[counter]))
        plt.legend()
    plt.suptitle('detected centre of the wheel')
    plt.grid()
    plt.plot(x_centre_abs, y_centre_abs, 'r*', markersize = 10)
    return fig

#%% 3D plots
def drawRefFrame(axis, xx, yy, zz):
    axis.quiver(np.nanmin(xx),np.nanmin(yy),np.nanmin(zz),np.nanmax(xx)-np.nanmin(xx),0,0, color = 'red')
    axis.quiver(np.nanmin(xx),np.nanmin(yy),np.nanmin(zz),0,np.nanmax(yy)-min(yy),0, color = 'green')
    axis.quiver(np.nanmin(xx),np.nanmin(yy),np.nanmin(zz),0,0,np.nanmax(zz)-min(zz), color = 'blue')

def orthogonalProjectionRCamView(imageOrXYZdata, flag = 'image', color_list = '', colormap = 'jet', mainTitle = '', alpha = 0.2, showRefFrame = True):
    '''
    Given multiple images in one channel, or multiple arrays of 3 columns 
    containing XYZ coordinates, draws 3D plot from different point of view: 
        front view | rear view
        -----------------------
        top view   | 

    NB:
        - this function slows down the execution, it's very demanding
        - the axis configuration is thought for an image, so:
            ----------->X
            |
            |
            |
            v Y
    
    Parameters
    ----------
    imageOrXYZdata : matrix M*N*1 or matrix M*3
        image containing the depth info or 3 columns array containing XYZ coordinates
    flag : string, optional
        to specify if data are already in the format XYZ or it's an image.
        The default is "image".
    mainTitle : string, optional
        title of the plot. The default is ''.
    color_list : array of strings, optional
        color to be used for every imageOrXYZdata. If specified, alpha is applied,
        if not specified, the colormap is applied.
        The default is '', which means that the color map is used for all the objects
    colormap : string, optional
        colormap for the 3D points. The default is 'jet'.
    alpha :  double between 0 and 1
        transparency applied when the color is specified
    showRefFrame : bool, optional
        if showing the ref frame (x red, y green, z blue) in the plots.
        The default is True.
    
    Raises
    ------
    NameError
        if specified flag is not "image" or "XYZdata".

     Returns
     -------
     fig : figure
         DESCRIPTION.
     ax : array (of array) of axis used to plot
         it's possible to modify the plots recalling 
         ax[..][..] if nrows > 1 and ncols > 1;
         ax[..] if nrows == 1 or ncols == 1

    '''
    # if image(s) passed
    if flag.lower() == 'image'.lower():
        # if a list of images
        if isinstance(imageOrXYZdata, list):
            data_list = []
            # for every image, apply the three column conversion
            for image in imageOrXYZdata:
                data = utils.depImgToThreeCol(image)
                data_list.append(data)
        # if only one image, apply the three column conversion
        else:
            data_list = utils.depImgToThreeCol(imageOrXYZdata)
    # if XYZ array(s) passed
    elif flag.lower() == 'XYZdata'.lower():
        data_list = imageOrXYZdata
    else:
        raise NameError('possible flags are "image" or "XYZdata"')

    # if only one element to be plot, put into a list to make the for loop work
    if isinstance(data_list, list):
       pass
    else:
        data_list_tmp = []
        data_list_tmp.append(data_list)
        data_list = data_list_tmp
    # now there is a list of three columns XYZ arrays

    if isinstance(color_list, list):
       pass
    else:
        color_list_tmp = []
        color_list_tmp.append(color_list)
        color_list = color_list_tmp

    if not len(color_list) == len(data_list):
        raise TypeError('you should specifiy as many colors as elements to plot with color_list.                        if you want to use a colormap, write: " " ')

    fig, ax = plt.subplots(2,2, subplot_kw=dict(projection='3d'))
    plt.suptitle(mainTitle)

    xx = []; yy = []; zz = [] # to save min max for axis drawing
    this_ax = ax[0,0]
    for (data, color) in zip(data_list, color_list):
        # removing nan values
        data = data[~np.isnan(data).any(axis=1), :]
        X = data[:,0]
        Y = data[:,1]
        Z = data[:,2]
        xx.append(X); yy.append(Y); zz.append(Z) # for axis drawing
        if len(X) == 1:
            this_ax.scatter(X, Y, Z, marker='*', c = color, s = 50)
        else:
            if not color == '':
                this_ax.scatter(X, Y, Z, marker='.', c = color, alpha = alpha)
            else:
                this_ax.scatter(X, Y, Z, marker='.', c = Z, cmap = colormap)
    if showRefFrame:
        X = np.concatenate(xx, axis=None)
        Y = np.concatenate(yy, axis=None)
        Z = np.concatenate(zz, axis=None)
        drawRefFrame(this_ax, X, Y, Z)
    this_ax.set_xlabel('X axis')
    this_ax.set_ylabel('Y axis')
    this_ax.set_zlabel('Z axis')
    this_ax.view_init(elev=-90, azim=-90)
    this_ax.set_title('front view')

    xx = []; yy = []; zz = [] # to save min max for axis drawing
    this_ax = ax[0,1]
    rotBack = R.from_euler('xyz', [0, -90, 0], degrees = True)
    for (data, color) in zip(data_list, color_list):
        data = data[~np.isnan(data).any(axis=1), :]
        dataBack = rotBack.apply(data)
        X = dataBack[:,0]
        Y = dataBack[:,1]
        Z = dataBack[:,2]
        xx.append(X); yy.append(Y); zz.append(Z) # for axis drawing
        if len(X) == 1:
            this_ax.scatter(X, Y, Z, marker='*', c = color, s = 50)
        else:
            if not color == '':
                this_ax.scatter(X, Y, Z, marker='.', c = color, alpha = alpha)
            else:
                this_ax.scatter(X, Y, Z, marker='.', c = -X, cmap = colormap)
    if showRefFrame:
        X = np.concatenate(xx, axis=None)
        Y = np.concatenate(yy, axis=None)
        Z = np.concatenate(zz, axis=None)
        # since there was a rotation of axis, instead of calling the function,
        # the drawing of the axis is done manually
        this_ax.quiver(max(X),min(Y),min(Z), 0,0,max(Z)-min(Z), color = 'red')
        this_ax.quiver(max(X),min(Y),min(Z), 0,max(Y)-min(Y),0, color = 'green')
        this_ax.quiver(max(X),min(Y),min(Z), min(X)-max(X),0,0, color = 'blue')
    this_ax.set_xlabel('(-)Z axis')
    this_ax.set_ylabel('Y axis')
    this_ax.set_zlabel('X axis')
    this_ax.view_init(elev=-90, azim=-90)
    # this_ax.view_init(elev=-50, azim=-90)
    this_ax.set_title('side view')

    xx = []; yy = []; zz = [] # to save min max for axis drawing
    this_ax = ax[1,0]
    for (data, color) in zip(data_list, color_list):
        data = data[~np.isnan(data).any(axis=1), :]
        dataTop = data
        X = dataTop[:,0]
        Y = dataTop[:,1]
        Z = dataTop[:,2]
        xx.append(X); yy.append(Y); zz.append(Z) # for axis drawing
        if len(X) == 1:
            this_ax.scatter(X, Y, Z, marker='*', c = color, s = 50)
        else:
            if not color == '':
                this_ax.scatter(X, Y, Z, marker='.', c = color, alpha = alpha)
            else:
                this_ax.scatter(X, Y, Z, marker='.', c = Z, cmap = colormap)
    if showRefFrame:
        X = np.concatenate(xx, axis=None)
        Y = np.concatenate(yy, axis=None)
        Z = np.concatenate(zz, axis=None)
        drawRefFrame(this_ax, X, Y, Z)
    this_ax.set_xlabel('X axis')
    this_ax.set_ylabel('Y axis')
    this_ax.set_zlabel('Z axis')
    this_ax.view_init(elev=0, azim=-90)
    this_ax.set_title('top view')

    xx = []; yy = []; zz = [] # to save min max for axis drawing
    this_ax = ax[1,1]
    rotTop = R.from_euler('xyz', [90, 0, 0], degrees = True)
    for (data, color) in zip(data_list, color_list):
        data = data[~np.isnan(data).any(axis=1), :]
        dataTop = rotTop.apply(data)
        X = dataTop[:,0]
        Y = dataTop[:,1]
        Z = dataTop[:,2]
        xx.append(X); yy.append(Y); zz.append(Z) # for axis drawing
        if len(X) == 1:
            this_ax.scatter(X, Y, Z, marker='*', c = color, s = 50)
        else:
            if not color == '':
                this_ax.scatter(X, Y, Z, marker='.', c = color, alpha = alpha)
            else:
                this_ax.scatter(X, Y, Z, marker='.', c = -Y, cmap = colormap)
    if showRefFrame:
        X = np.concatenate(xx, axis=None)
        Y = np.concatenate(yy, axis=None)
        Z = np.concatenate(zz, axis=None)
        # since there was a rotation of axis, instead of calling the function,
        # the drawing of the axis is done manually
        this_ax.quiver(min(X),max(Y),min(Z), max(X)-min(X),0,0, color = 'red')
        this_ax.quiver(min(X),max(Y),min(Z), 0,min(Y)-max(Y),0, color = 'blue')
        this_ax.quiver(min(X),max(Y),min(Z), 0,0,max(Z)-min(Z), color = 'green')
    this_ax.set_xlabel('X axis')
    this_ax.set_ylabel('(-Z) axis')
    this_ax.set_zlabel('Y axis')
    this_ax.view_init(elev=-165, azim=-120)

    return fig, ax

def orthogonalProjection(imageOrXYZdata, flag = 'image', color_list = '', colormap = 'jet', mainTitle = '', alpha = 0.2, showRefFrame = True):
    '''
    Given multiple images in one channel, or multiple arrays of 3 columns 
    containing XYZ coordinates, draws 3D plot from different point of view: 
        front view | rear view
        -----------------------
        top view   | 

    NB:
        - this function slows down the execution, it's very demanding
    
    Parameters
    ----------
    imageOrXYZdata : matrix M*N*1 or matrix M*3
        image containing the depth info or 3 columns array containing XYZ coordinates
    flag : string, optional
        to specify if data are already in the format XYZ or it's an image.
        The default is "image".
    mainTitle : string, optional
        title of the plot. The default is ''.
    color_list : array of strings, optional
        color to be used for every imageOrXYZdata. If specified, alpha is applied,
        if not specified, the colormap is applied.
        The default is '', which means that the color map is used for all the objects
    colormap : string, optional
        colormap for the 3D points. The default is 'jet'.
    alpha :  double between 0 and 1
        transparency applied when the color is specified
    showRefFrame : bool, optional
        if showing the ref frame (x red, y green, z blue) in the plots.
        The default is True.
    
    Raises
    ------
    NameError
        if specified flag is not "image" or "XYZdata".

    Returns
    -------
    fig : figure
        DESCRIPTION.
    ax : array (of array) of axis used to plot
        it's possible to modify the plots recalling 
        ax[..][..] if nrows > 1 and ncols > 1;
        ax[..] if nrows == 1 or ncols == 1

    '''
    # if image(s) passed
    if flag.lower() == 'image'.lower():
        # if a list of images
        if isinstance(imageOrXYZdata, list):
            data_list = []
            # for every image, apply the three column conversion
            for image in imageOrXYZdata:
                data = utils.depImgToThreeCol(image)
                data_list.append(data)
        # if only one image, apply the three column conversion
        else:
            data_list = utils.depImgToThreeCol(imageOrXYZdata)
    # if XYZ array(s) passed
    elif flag.lower() == 'XYZdata'.lower():
        data_list = imageOrXYZdata
    else:
        raise NameError('possible flags are "image" or "XYZdata"')

    # if only one element to be plot, put into a list to make the for loop work
    if isinstance(data_list, list):
       pass
    else:
        data_list_tmp = []
        data_list_tmp.append(data_list)
        data_list = data_list_tmp
    # now there is a list of three columns XYZ arrays

    if isinstance(color_list, list):
       pass
    else:
        color_list_tmp = []
        color_list_tmp.append(color_list)
        color_list = color_list_tmp

    if not len(color_list) == len(data_list):
        raise TypeError('you should specifiy as many colors as elements to plot with color_list.                        if you want to use a colormap, write: " " ')

    fig, ax = plt.subplots(2,2, subplot_kw=dict(projection='3d'))
    plt.suptitle(mainTitle)

    xx = []; yy = []; zz = [] # to save min max for axis drawing
    this_ax = ax[0,0]
    for (data, color) in zip(data_list, color_list):
        # removing nan values
        data = data[~np.isnan(data).any(axis=1), :]
        X = data[:,0]
        Y = data[:,1]
        Z = data[:,2]
        xx.append(X); yy.append(Y); zz.append(Z) # for axis drawing
        if len(X) == 1:
            this_ax.scatter(X, Y, Z, marker='*', c = color, s = 50)
        else:
            if not color == '':
                this_ax.scatter(X, Y, Z, marker='.', c = color, alpha = alpha)
            else:
                this_ax.scatter(X, Y, Z, marker='.', c = Z, cmap = colormap)
    if showRefFrame:
        X = np.concatenate(xx, axis=None)
        Y = np.concatenate(yy, axis=None)
        Z = np.concatenate(zz, axis=None)
        drawRefFrame(this_ax,[0,np.abs(np.nanmax(X))], [0,np.abs(np.nanmax(Y))],[0,np.abs(np.nanmax(Z))])
        # drawRefFrame(this_ax, X, Y, Z)
    this_ax.set_xlabel('X axis')
    this_ax.set_ylabel('Y axis')
    this_ax.set_zlabel('Z axis')
    this_ax.view_init(elev=90, azim=-90)
    this_ax.set_title('front view')

    xx = []; yy = []; zz = [] # to save min max for axis drawing
    this_ax = ax[0,1]
    rotBack = R.from_euler('xyz', [0, 90, 0], degrees = True)
    for (data, color) in zip(data_list, color_list):
        data = data[~np.isnan(data).any(axis=1), :]
        dataBack = rotBack.apply(data)
        X = dataBack[:,0]
        Y = dataBack[:,1]
        Z = dataBack[:,2]
        xx.append(X); yy.append(Y); zz.append(Z) # for axis drawing
        if len(X) == 1:
            this_ax.scatter(X, Y, Z, marker='*', c = color, s = 50)
        else:
            if not color == '':
                this_ax.scatter(X, Y, Z, marker='.', c = color, alpha = alpha)
            else:
                this_ax.scatter(X, Y, Z, marker='.', c = X, cmap = colormap)
    if showRefFrame:
        X = np.concatenate(xx, axis=None)
        Y = np.concatenate(yy, axis=None)
        Z = np.concatenate(zz, axis=None)
        # since there was a rotation of axis, instead of calling the function,
        # the drawing of the axis is done manually
        this_ax.quiver(0,0,0, np.nanmax(X),0,0, color = 'blue')
        this_ax.quiver(0,0,0, 0,np.nanmax(Y),0, color = 'green')
        this_ax.quiver(0,0,0, 0,0,np.abs(np.nanmax(Z)), color = 'red')
    this_ax.set_xlabel('Z axis')
    this_ax.set_ylabel('Y axis')
    this_ax.set_zlabel('X axis')
    this_ax.view_init(elev=90, azim=-90)
    # this_ax.view_init(elev=-50, azim=-90)
    this_ax.set_title('side view')

    xx = []; yy = []; zz = [] # to save min max for axis drawing
    this_ax = ax[1,0]
    for (data, color) in zip(data_list, color_list):
        # removing nan values
        data = data[~np.isnan(data).any(axis=1), :]
        X = data[:,0]
        Y = data[:,1]
        Z = data[:,2]
        xx.append(X); yy.append(Y); zz.append(Z) # for axis drawing
        if len(X) == 1:
            this_ax.scatter(X, Y, Z, marker='*', c = color, s = 50)
        else:
            if not color == '':
                this_ax.scatter(X, Y, Z, marker='.', c = color, alpha = alpha)
            else:
                this_ax.scatter(X, Y, Z, marker='.', c = Z, cmap = colormap)
    if showRefFrame:
        X = np.concatenate(xx, axis=None)
        Y = np.concatenate(yy, axis=None)
        Z = np.concatenate(zz, axis=None)
        drawRefFrame(this_ax,[0,np.abs(np.nanmax(X))], [0,np.abs(np.nanmax(Y))],[0,np.abs(np.nanmax(Z))])
        # drawRefFrame(this_ax, X, Y, Z)
    this_ax.set_xlabel('X axis')
    this_ax.set_ylabel('Y axis')
    this_ax.set_zlabel('Z axis')
    this_ax.view_init(elev=180, azim=-90)
    this_ax.set_title('top view')

    xx = []; yy = []; zz = [] # to save min max for axis drawing
    this_ax = ax[1,1]
    for (data, color) in zip(data_list, color_list):
        data = data[~np.isnan(data).any(axis=1), :]
        X = data[:,0]
        Y = data[:,1]
        Z = data[:,2]
        xx.append(X); yy.append(Y); zz.append(Z) # for axis drawing
        if len(X) == 1:
            this_ax.scatter(X, Y, Z, marker='*', c = color, s = 50)
        else:
            if not color == '':
                this_ax.scatter(X, Y, Z, marker='.', c = color, alpha = alpha)
            else:
                this_ax.scatter(X, Y, Z, marker='.', c = Z, cmap = colormap)
    if showRefFrame:
        X = np.concatenate(xx, axis=None)
        Y = np.concatenate(yy, axis=None)
        Z = np.concatenate(zz, axis=None)
        # since there was a rotation of axis, instead of calling the function,
        # the drawing of the axis is done manually
        drawRefFrame(this_ax,[0,np.abs(np.nanmax(X))], [0,np.abs(np.nanmax(Y))],[0,np.abs(np.nanmax(Z))])

    this_ax.set_xlabel('X axis')
    this_ax.set_ylabel('Y axis')
    this_ax.set_zlabel('Z axis')

    return fig, ax



def zzzTestElevAndAzim():
    '''
    to simply see the different ways of 3D plotting according to elevation and azimuth angle

   Returns
   -------
   fig : figure
       DESCRIPTION.
   ax : array (of array) of axis used to plot
       it's possible to modify the plots recalling 
       ax[..][..] if nrows > 1 and ncols > 1;
       ax[..] if nrows == 1 or ncols == 1

    '''
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    # for r in range(0,360,90):
    for r in [0]: #roll not yet implemented, wait for matplotlib version 3.6
        fig, ax = plt.subplots(4,4, sharey = True, sharex = True, subplot_kw=dict(projection='3d'))
    
        plt.suptitle('test view. roll = ' +str(r))
    
        for e in range(0,360,90):
            for a in range(0,360,90):
    
                this_ax = ax[int(e/90),int(a/90)]
                this_ax.scatter(x, y , z, marker='o', c=z, cmap = 'jet')
                drawRefFrame(this_ax, x, y, z)
                this_ax.set_xlabel('X Label')
                this_ax.set_ylabel('Y Label')
                this_ax.set_zlabel('Z Label')
    
                # this_ax.view_init(elev=e, azim=a, roll=r)
                this_ax.view_init(elev=e, azim=a)
                this_ax.set_title('elev = ' + str(e) + ' azim = ' + str(a))

#%% orthogonal projection
def plotOrtProj(x,y,z, circleRadius):
    fig = plt.figure()
    ax0 = fig.add_subplot(221)
    ax0.scatter(x, y, marker = '.')
    ax0.grid()
    ax0.set_title('front view')
    ax1 = fig.add_subplot(222, sharey = ax0)
    ax1.scatter(z, y, marker = '.')
    ax1.grid()
    ax1.set_title('side view')
    ax2 = fig.add_subplot(223, sharex = ax0)
    ax2.scatter(x, z, marker = '.')
    ax2.invert_yaxis()
    ax2.grid()
    ax2.set_title('top view')
    addPatchesToOrtProj(ax0, ax1, ax2, circleRadius)
    return fig, [ax0, ax1, ax2]

def addPatchesToOrtProj(ax0, ax1, ax2, circleRadius):
    amp = circleRadius/100
    frontView = matplotlib.patches.Circle((0,0), radius=circleRadius, fill = False, color = 'red')
    ax0.add_patch(frontView)

    sideView = matplotlib.patches.Rectangle((-amp/2,-circleRadius),amp,2*circleRadius, color = 'red', alpha = 0.5)
    ax1.add_patch(sideView)

    topView = matplotlib.patches.Rectangle((-circleRadius,-amp/2),2*circleRadius,amp, color = 'red', alpha = 0.5)
    ax2.add_patch(topView)

def addHandLandmarks(axs, df, listOfLandmarks):
    # [1,5,9,13,17] proximal
    # [4,8,12,16,20] distal
    for landmark in listOfLandmarks:
        x = df['x{:02d}'.format(landmark)]
        y = df['y{:02d}'.format(landmark)]
        z = df['z{:02d}'.format(landmark)]
        axs[0].scatter(x,y,marker = '.')
        axs[1].scatter(z,y,marker = '.')
        axs[2].scatter(x,z,marker = '.')

#%% statistical plots
def histCompareSamePlot(df_original, groups, variables, bins = 20, alpha = 0.4):
    variables = makeList(variables)
    groups = makeList(groups)
    df = df_original.copy()
    for variable in variables:
        for group in groups:
            # how many classes in this group
            classes = sorted(df[group].unique())
            nclasses = len(classes)
            # create figure
            fig = plt.figure()
            plt.suptitle(variable + ' according to ' + group)
            
            # to manage the color
            i = -1
            # for the left side: one row per class
            for element in classes:
                # to manage color and axis creation
                i+=1
                col = 'C'+str(i)
                # axis creation, sharing x and y with the previous ax
                if i==0:
                    this_ax = df[df[group] == element][variable].hist(bins = bins, alpha = alpha, color = col)
                else:
                    df[df[group] == element][variable].hist(bins = bins, alpha = alpha, color = col, ax = this_ax)
            plt.legend(classes)

def histCompareSubPlots(df_original, groups, variables, bins = 20, alpha = 0.4,
    sharex = False, sharey = False, nrows = 0, ncols = 0, mainTitle = ''):
    variables = makeList(variables)
    groups = makeList(groups)
    df = df_original.copy()
    for variable in variables:
        for group in groups:
            # how many classes in this group
            classes = sorted(df[group].unique())
            nclasses = len(classes)
            nOfPlots = nclasses

            # _ in the tail of nrows and ncols to avoid overwriting
            if nrows == 0 and ncols == 0:
                nrows_ = int(np.ceil(np.sqrt(nOfPlots)))
                ncols_ = int(np.ceil(nOfPlots/nrows_))
            elif nrows == 0 and ncols != 0:
                ncols_ = ncols
                nrows_ = int(np.ceil(nOfPlots/ncols_))
            elif nrows != 0 and ncols == 0:
                nrows_ = nrows
                ncols_ = int(np.ceil(nOfPlots/nrows_))

            fig, ax = createSubPlots(nclasses, sharex, sharey, nrows_, ncols_, variable + mainTitle, classes)

            
            # to manage the color
            i = -1
            # for the left side: one row per class
            for element in classes:
                # to manage color and axis creation
                i+=1
                col = 'C'+str(i)
                if np.ndim(ax) == 2:
                    this_ax = ax[int(np.floor(i/ncols_)), int(i % ncols_)]
                elif np.ndim(ax) == 1:
                    this_ax = ax[i]
                df[df[group] == element][variable].hist(bins = bins, alpha = alpha, color = col, ax = this_ax)
                this_ax.set_title(element)


def histLR(df_original, groups, variables, bins = 20, alpha = 0.4):
    '''
    Given a dataframe:
        for every variable in variables (one column of the df)
            for every group in groups (one column of the df)
    on the left: draws the histograms of variable according to group
    on the top right: draws the density distribution
    on the bottom right: draws the standard gaussian according to mean and std dev

    LR because: 
        on the left the pile of histograms 
        on the right comparison between density distribution and gaussian


    Parameters
    ----------
    df : pandas dataframe.
        Should contain as columns the strings specified in groups and in variables
    groups : array of strings
        How to group the element of the dataframe?
    variables : array of strings
        Of which columns do you want the histograms?
    bins : int, optional
        Number of bins. 
        The default is 20.
    alpha : double between 0 and 1
        Transparency of histograms.
        The default is 0.4.

    example:
        variables = ['nan perc mean', 'nan perc 02', 'nan perc 05']
        groups = ['block', 'condition']
        plotHistograms(df, groups, variables)

        produces 6 plots: 
            nan perc mean according to block, 
            nan perc mean according to condition, 
            nan perc 02 according to block, 
            nan perc 02 according to condition, 
            nan perc 05 according to block, 
            nan perc 05 according to condition. 
        

    Returns
    -------
    None.

    '''
    variables = makeList(variables)
    groups = makeList(groups)
    df = df_original.copy()

    for variable in variables:
        for group in groups:
            # how many classes in this group
            classes = sorted(df[group].unique())
            nclasses = len(classes)
            # nclasses rows on the left, 2 rows on the right
            nrows = np.lcm(nclasses, 2) 
            # create figure
            fig = plt.figure(constrained_layout=True)
            gs = grd.GridSpec(nrows, 2, figure=fig)
            fig.suptitle(variable + ' according to ' + group)
            
            # to manage the color
            i = -1
            # for the left side: one row per class
            for element in classes:
                # to manage color and axis creation
                i+=1
                col = 'C'+str(i)
                i_axis = i * int(nrows/nclasses)
                # axis creation, sharing x and y with the previous ax
                if i==0:
                    this_ax = fig.add_subplot(gs[i_axis : (i_axis + int(nrows/nclasses)), 0])
                else:
                    this_ax = fig.add_subplot(gs[i_axis : (i_axis + int(nrows/nclasses)), 0], sharex = this_ax, sharey = this_ax)
                # bar plot 
                df[df[group] == element][variable].hist(bins = bins, alpha = alpha, color = col, ax = this_ax)
                this_ax.set_title(element)
            
            # upper right: density distribution
            i = -1
            this_ax = fig.add_subplot(gs[0:(int(nrows/2)),1])
            for element in classes:
                # to manage color and axis creation
                i+=1
                col = 'C'+str(i)
                # plot density function
                df[df[group] == element][variable].plot.density(c = col, ax = this_ax, label = element)
                this_ax.axvline(df[df[group] == element][variable].mean(), ls = '--', color = col, label='_Hidden Label')
            this_ax.set_title("density distribution")
            this_ax.grid()
            this_ax.legend()

            # lower right: gaussian distribution
            i = -1
            this_ax = fig.add_subplot(gs[(int(nrows/2)):,1], sharex = this_ax, sharey = this_ax)
            for element in classes:
                # to manage color and axis creation
                i+=1
                col = 'C'+str(i)
                # plot gaussian
                mu = df[df[group] == element][variable].mean()
                sigma = df[df[group] == element][variable].std()
                x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                this_ax.plot(x, scipy.stats.norm.pdf(x, mu, sigma), color = col, label = element + ' mean: {:.2f} std {:.2}'.format(mu, sigma))
                this_ax.axvline(df[df[group] == element][variable].mean(), ls = '--', color = col, label='_Hidden Label')
            this_ax.set_title("assumption of statistical analysis")
            this_ax.grid()
            this_ax.legend()

def histV(df_original, groups, variables, bins = 20, alpha = 0.4):
    '''
    Given a dataframe:
        for every variable in variables (one column of the df)
            for every group in groups (one column of the df)
    on the top: draws the histograms of variable according to group
    on the center: draws the density distribution
    on the bottom: draws the standard gaussian according to mean and std dev

    V because the three plots are vertically stacked


    Parameters
    ----------
    df : pandas dataframe.
        Should contain as columns the strings specified in groups and in variables
    groups : array of strings
        How to group the element of the dataframe?
    variables : array of strings
        Of which columns do you want the histograms?
    bins : int, optional
        Number of bins. 
        The default is 20.
    alpha : double between 0 and 1
        Transparency of histograms.
        The default is 0.4.

    example:
        variables = ['nan perc mean', 'nan perc 02', 'nan perc 05']
        groups = ['block', 'condition']
        plotHistograms(df, groups, variables)

        produces 6 plots: 
            nan perc mean according to block, 
            nan perc mean according to condition, 
            nan perc 02 according to block, 
            nan perc 02 according to condition, 
            nan perc 05 according to block, 
            nan perc 05 according to condition. 
        

    Returns
    -------
    None.

    '''
    variables = makeList(variables)
    groups = makeList(groups)
    df = df_original.copy()
    for variable in variables:
        for group in groups:
            # create figure
            fig, axes = plt.subplots(3,1, sharex = True)
            fig.suptitle(variable + ' according to ' + group)
            # to handle the color
            i = -1
            classes = sorted(df[group].unique())
            for element in classes:
                i+=1
                col = 'C'+str(i)
                # bar plot 
                this_ax = axes[0]
                df[df[group] == element][variable].hist(bins = bins, alpha = alpha, color = col, ax = this_ax, label = element)
                this_ax.set_title("bar plot")
                # plot density
                this_ax = axes[1]
                df[df[group] == element][variable].plot.density(c = col, ax = this_ax, label = element)
                this_ax.axvline(df[df[group] == element][variable].mean(), ls = '--', color = col, label='_Hidden Label')
                this_ax.set_title("density distribution")
                this_ax.grid()
                
                # plot gaussian
                this_ax = axes[2]
                mu = df[df[group] == element][variable].mean()
                sigma = df[df[group] == element][variable].std()
                x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                this_ax.plot(x, scipy.stats.norm.pdf(x, mu, sigma), color = col, label = element + ' mean: {:.2f} std {:.2}'.format(mu, sigma))
                this_ax.axvline(df[df[group] == element][variable].mean(), ls = '--', color = col, label='_Hidden Label')
                this_ax.set_title("assumption of stat analysis")
                this_ax.grid()
                
            # legend for all plots
            axes[0].legend()
            axes[1].legend()
            axes[2].legend()
            plt.grid(True)



def barEachTest(df_original, variables, xcolumn, mainTitle = '', flagLabelEachBar = True, strformat = '{:.1f}'):

    variables = makeList(variables)
    df = df_original.copy()

    fig, axis = plt.subplots(len(variables)+1,1, sharex = True, sharey = True)
    fig.suptitle(mainTitle)
    fig.subplots_adjust(bottom=0.15)
    this_ax = axis[0]
    df.plot.bar(x = xcolumn, y = variables, ax = this_ax, grid = True)

    for i in range(len(variables)):
        this_ax = axis[i+1]
        df.plot.bar(x = xcolumn, y = variables[i], ax = this_ax, grid = True, color = 'C'+str(i))
        if flagLabelEachBar:
            this_ax.bar_label(this_ax.containers[0], labels=[strformat.format(p) for p in df[variables[i]]], color = 'C'+str(i))


def barWithStd(df_original, groups, variables, lolims = True, mainTitle = '', flagLabelEachBar = True, strformat = '{:.1f}'):

    variables = makeList(variables)
    groups = makeList(groups)
    df = df_original.copy()
    for variable in variables:
        # only extract the relevant part
        df_var = df[[*groups, *[variable]]]
        for group in groups:
            grouped_df = df_var.groupby(group)
            mean_df = grouped_df.mean()
            std_df = grouped_df.std()
            plot = mean_df.plot(kind='bar', grid = True, \
            title = mainTitle + variable +' according to ' + group, \
            legend = False, yerr = std_df,  \
            error_kw = dict(lw=3, capsize=10, capthick=3, lolims = lolims),\
            label = [strformat.format(p) for p in df[variable]])

            if flagLabelEachBar:
                for c in plot.containers[1::2]:
                    plot.bar_label(c, label_type='center')

            # to erase the error bar below
            for ch in plot.get_children():
                if str(ch).startswith('Line2D'):
                    ch.set_marker('_')
                    ch.set_markersize(20)

def barWithOutStd(df_original, groups, variables, mainTitle = '', flagLabelEachBar = True, strformat = '{:.1f}'):

    variables = makeList(variables)
    groups = makeList(groups)
    df = df_original.copy()
    for variable in variables:
        # only extract the relevant part
        df_var = df[[*groups, *[variable]]]
        for group in groups:
            grouped_df = df_var.groupby(group)
            mean_df = grouped_df.mean()
            plot = mean_df.plot(kind='bar', grid = True, \
            title = mainTitle + variable +' according to ' + group, \
            legend = False, \
            label = [strformat.format(p) for p in df[variable]])

            # plot.yaxis.set_major_formatter('{x:1.0f}%')

            if flagLabelEachBar:
                for c in plot.containers[1::2]:
                    plot.bar_label(c, label_type='center')


def barWithStdCompareVar(df_original, groups, variables, lolims = True, mainTitle = '', flagLabelEachBar = True, strformat = '{:.1f}'):
    variables = makeList(variables)
    groups = makeList(groups)
    df = df_original.copy()
    df_var = df[[*groups, *variables]]
    for group in groups:
        grouped_df = df_var.groupby(group)
        mean_df = grouped_df.mean()
        std_df = grouped_df.std()
        plot = mean_df.plot(kind='bar', grid = True,\
        title = group, legend = True, yerr = std_df,  \
        error_kw = dict(lw=3, capsize=10, capthick=3, lolims = lolims))

        if flagLabelEachBar:
            for c in plot.containers[1::2]:
                plot.bar_label(c, label_type='center')

        # to erase the error bar below
        for ch in plot.get_children():
            if str(ch).startswith('Line2D'):
                ch.set_marker('_')
                ch.set_markersize(20)

def barWithStdGrouped(df_original, groups, variables, lolims = True, mainTitle = '', flagLabelEachBar = True, strformat = '{:.1f}'):
    variables = makeList(variables)
    groups = makeList(groups)
    df = df_original.copy()
    for variable in variables:
        fig, axes = plt.subplots(1, len(groups), sharey = True)
        fig.suptitle(mainTitle + variable)
        # only extract the relevant part
        df_var = df[[*groups, *[variable]]]
        i = -1
        for group in sorted(groups):
            i+=1
            this_ax = axes[i]
            grouped_df = df_var.groupby(group)
            mean_df = grouped_df.mean()
            std_df = grouped_df.std()
            plot = mean_df.plot(kind='bar', grid = True, color = 'C'+str(i), \
            ax = this_ax, title = group, legend = False, yerr = std_df,  \
            error_kw = dict(lw=3, capsize=10, capthick=3, lolims = lolims))

            if flagLabelEachBar:
                for c in plot.containers[1::2]:
                    plot.bar_label(c, label_type='center')

            # to erase the error bar below
            for ch in plot.get_children():
                if str(ch).startswith('Line2D'):
                    ch.set_marker('_')
                    ch.set_markersize(20)


def barWithStdGroupedCrossed(df_original, groups, variables, lolims = True, mainTitle = '', flagLabelEachBar = True, strformat = '%.2f', ylim = None, xlab = '', ylab = ''):

    #todo only works for 2 groups
    variables = makeList(variables)
    groups = makeList(groups)
    df = df_original.copy()

    for variable in variables:
        fig, axes = plt.subplots(1, len(groups)+1, sharey = True)
        fig.suptitle(mainTitle + variable)
        df_var = df[[*groups, *[variable]]]
        i = -1
        for group in sorted(groups):
            i+=1
            this_ax = axes[i]
            this_ax.set_xlabel(xlab)
            this_ax.set_ylabel(ylab)
            # this_ax.yaxis.set_major_formatter('{x:1.0f}%')
            grouped_df = df_var.groupby(group)
            mean_df = grouped_df.mean()
            std_df = grouped_df.std()
            plot = mean_df.plot(kind='bar', grid = True, color = 'C'+str(i), \
            ax = this_ax, title = group, legend = False, yerr = std_df,  \
            error_kw = dict(lw=3, capsize=10, capthick=3, lolims = lolims))

            if flagLabelEachBar:
                for c in plot.containers[1::2]:
                    plot.bar_label(c, label_type='center', fmt=strformat)

            # to erase the error bar below
            for ch in plot.get_children():
                if str(ch).startswith('Line2D'):
                    ch.set_marker('_')
                    ch.set_markersize(20)


        i+=1
        this_ax = axes[i]
        grouped_df = df_var.groupby(groups)
        mean_df = grouped_df.mean()
        std_df = grouped_df.std()
        plot = mean_df.plot(kind='bar', grid = True, color = 'C'+str(i), \
        ax = this_ax, title = groups[0]+' x '+groups[1], legend = False, yerr = std_df,  \
        error_kw = dict(lw=3, capsize=10, capthick=3, lolims = lolims))

        if flagLabelEachBar:
            for c in plot.containers[1::2]:
                plot.bar_label(c, label_type='center', fmt=strformat)

        # to erase the error bar below
        for ch in plot.get_children():
            if str(ch).startswith('Line2D'):
                ch.set_marker('_')
                ch.set_markersize(20)

        if ylim != None:
            plt.ylim(ylim)

        plt.tight_layout()




#%% histograms
def histograms(listOfArrays, sharex = False, sharey = False, nrows = 0, ncols = 0, listOfTitles = '', mainTitle = ''):

    fig, ax = createSubPlots(len(listOfArrays), sharex = False, sharey = False, nrows = 0, ncols = 0, listOfTitles = '', mainTitle = '')

    counter = -1
    for array, title in zip(listOfArrays, listOfTitles):
        counter += 1
        if nrows > 1 and ncols > 1:
            this_ax = ax[int(np.floor(counter/ncols)), int(counter % ncols)]
        else:
            this_ax = ax[counter]
        this_ax.hist(array)
        this_ax.grid()
        this_ax.set_title(title)
    return fig, ax

def compareHistograms(listYarrays, sharex = False, sharey = False, nrows = 0, ncols = 0, mainTitle = '', listOfTitles = [''], listOfLegends = [''], listOfkwargs = [{}], common_kwargs = {'alpha': 0.5}, bins = 20, superimpose = True):
    '''
    see documentation of drawInSubPlots

    '''

    nOfPlots = len(listYarrays)

    # expand lists
    # if only one, it's repeated for all the this_ax, otherwise it simply overflows
    listOfLegends = listOfLegends * nOfPlots
    listOfkwargs = listOfkwargs * nOfPlots

    fig, ax = createSubPlots(nOfPlots, sharex, sharey, nrows, ncols , mainTitle, listOfTitles)

    nrows = len(ax)
    if np.ndim(ax) == 2:
        ncols = len(ax[0])
    elif np.ndim(ax) == 1:
        nrows = 1

    # for every plot to draw
    for counter in range(nOfPlots):
        # detect the axis
        if nrows > 1 and ncols > 1:
            this_ax = ax[int(np.floor(counter/ncols)), int(counter % ncols)]
        else:
            this_ax = ax[counter]
        this_ax.grid()

        # squeezing to remove an eventual redundant dimension and conversion to list
        listYarrays[counter] = np.squeeze(listYarrays[counter]).tolist()

        # if more than one y array to plot for this axis
        if not np.isscalar(listYarrays[counter][0]):
            # if only one dictionary of kwargs, could also be empty (default)
            if isinstance(listOfkwargs[counter], dict):
                # the same kwargs is applied to all the elements of this subplot
                # creation of the list
                listOfkwargs[counter] = [listOfkwargs[counter]] * len(listYarrays[counter])
            # the list is repeated for all the elements, if it's empty, it's passed an empty dic
            if isinstance(listOfkwargs[counter], list):
                listOfkwargs[counter] = listOfkwargs[counter] * len(listYarrays[counter])
            # for every y to be plotted and kwargs to be applied

            if superimpose: # the histograms are superimposed
                for y, plt_kwargs in zip(np.squeeze(listYarrays[counter]), listOfkwargs[counter]):
                    # make a copy of the common k_wargs
                    this_plt_kwargs = common_kwargs.copy()
                    # add the single parameters contained in listOfkwargs
                    this_plt_kwargs.update(plt_kwargs)
                    this_ax.hist(y, **this_plt_kwargs, bins = bins)

            else: # each bar is next to the other, it's necessary to rewrite the kwargs
                # first: merge common and singular kwargs
                for i in range(len(listOfkwargs[counter])):
                    # make a copy of the common k_wargs
                    this_plt_kwargs = common_kwargs.copy()
                    # add what is contained in list of kwargs
                    this_plt_kwargs.update(listOfkwargs[counter][i])
                    listOfkwargs[counter][i] = this_plt_kwargs
                # extract colors and alpha
                colors = []; alphas = [];
                for dictionary in listOfkwargs[counter]:
                    try:
                        colors.append(dictionary['color'])
                    except:
                        pass
                    try:
                        alphas.append(dictionary['alpha'])
                    except:
                        alphas.append(np.nan)
                tmp = len(np.squeeze(listYarrays[counter]))
                alpha = np.nanmean(alphas)
                if np.isnan(alpha):
                    alpha = 1
                if len(colors)<tmp: # if not enough colors were given
                    colors = ['C'+str(i) for i in range(tmp)]
                this_ax.hist(np.squeeze(listYarrays[counter]), color = colors[0:tmp], alpha = alpha, bins = bins)

        # if only one y array to plot for this axis
        else:
            y = listYarrays[counter]
            # if kwargs are packed in a dictionary -> ok
            if isinstance(listOfkwargs[counter], dict):
                plt_kwargs = listOfkwargs[counter]
            # if inside a list (might happen due to iteration of kwargs) -> extract from list
            elif isinstance(listOfkwargs[counter], list):
                plt_kwargs = listOfkwargs[counter][0]
            # if there aren't kwargs for this plot
            if listOfkwargs[counter] == {}:
                this_ax.hist(y, **common_kwargs, bins = bins)
            else: # if there are
                # make a copy of the common k_wargs
                this_plt_kwargs = common_kwargs.copy()
                # add the single parameters contained in listOfkwargs
                this_plt_kwargs.update(plt_kwargs)
                this_ax.hist(y, **this_plt_kwargs, bins = bins)

        # eventually add the legend
        if listOfLegends[counter] != '':
            this_ax.legend(listOfLegends[counter])

    return fig, ax

#%% scatter
def scatterFeatures(X, y, featNames = [''], labelColors = [''], mainTitle = '', alpha_scatter = 0.5, marker_scatter = '.', s_scatter = 1, alpha_hist = 0.5, bins = 20, superimpose = True, labelNames = None, ticklabels = 'bottom'):
    '''
    Given X, nOfSamples rows * nOfFeatures columns, containing the features
    and y, nOfSamples rows * 1 column, containing the labels

    plots in a matrix one feature with respect to the other.
    On the diagonal, draws an histogram with the distribution of the given 
    feature for the different labels

    Parameters
    ----------
    X : array
        nOfSamples rows * nOfFeatures columns, containing the features
    y : array
        nOfSamples rows * 1 column, containing the labels
    featNames : list of string, optional
        name of the features. The default is [''].
    labelColors : list of string, optional
        colors of each feature. The default is [''].
    mainTitle : string, optional
        title of the figure. The default is ''.
    alpha_scatter : float [0,1], optional
        alpha of the scatters. Useful when there are many point, so the cloud 
        is more dense where there are more points. The default is 0.5.
    marker_scatter : string, optional
        marker shape for scatter plot. The default is '.'.
    s_scatter : TYPE, optional
        marker size for scatter plot. The default is 1.
    alpha_hist : float[0,1], optional
        alpha of the histograms. The default is 0.5.
    bins : int, optional
        number of bars for the histograms. The default is 20.
    superimpose : bool, optional
        if True, the histrograms are superimposed
        if False, each bar or each label is close to the other

    Returns
    -------
    fig : figure
        DESCRIPTION.
    ax : array (of array) of axis used to plot
        it's possible to modify the plots recalling 
        ax[..][..] if nrows > 1 and ncols > 1;
        ax[..] if nrows == 1 or ncols == 1

    '''
    assert ticklabels == 'bottom' or ticklabels == 'none',\
        f"ticklabels can be \'bottom\' or \'none\', got: \'{ticklabels}\'"



    nOfFeatures = len(X[0])
    labels = np.unique(y)

    # chose colors
    if labelColors == [''] or len(labelColors) != len(labels):
        labelColors = ['C'+str(x) for x in range(len(labels))]

    fig, ax = plt.subplots(nOfFeatures, nOfFeatures, sharex = 'col', sharey = 'row')
    plt.suptitle(mainTitle)

    for counter in range(nOfFeatures**2):
        # detect the axis
        row = int(np.floor(counter/nOfFeatures))
        col = int(counter % nOfFeatures)

        this_ax = ax[row, col]
        this_ax.grid()

        # Instead of plotting a dimension against itself, display histograms
        if row == col:
            this_ax.get_shared_y_axes().remove(this_ax)

            if superimpose: # the histograms are superimposed
                for (label, labelColor) in zip(labels, labelColors):
                    this_ax.hist([X[:,row][y==label]], \
                    color = labelColor, alpha = alpha_hist, bins = bins)
            else: # each bar is next to the other
                this_ax.hist([X[:,row][y==label] for label in labels], \
                color = labelColors, alpha = alpha_hist, bins = bins)
            # this_ax.set_title(featNames[row])
            if row == 0 and labelNames: # eventually adding the legend
                this_ax.legend(labelNames)

        # Plot row against col
        else:
            for (label, labelColor) in zip(labels, labelColors):
                this_ax.scatter([X[:,col][y==label]], [X[:,row][y==label]], \
                color = labelColor, s = s_scatter, \
                marker = marker_scatter, alpha = alpha_scatter)

            #this_ax.set_title(featNames[row] + ' / ' + featNames[col])

        # to create more space
        # only last row has xlabel
        if row == nOfFeatures-1:
            this_ax.set_xlabel(featNames[col])
        else:
            if ticklabels == 'bottom':
                plt.setp(this_ax.get_xticklabels(), visible=False)

        # only first column has ylabel
        if col == 0:
            this_ax.set_ylabel(featNames[row])
            if ticklabels == 'bottom':
                plt.setp(this_ax.get_yticklabels(), visible=False)

        # in anycase, if ticklabels is none, erase them
        if ticklabels == 'none':
            plt.setp(this_ax.get_xticklabels(), visible=False)
            plt.setp(this_ax.get_yticklabels(), visible=False)

        this_ax.grid(True)

    return fig, ax


def scatterFeaturesBackup(X, y, featNames = [''], labelColors = [''], mainTitle = '', alpha_scatter = 0.5, marker_scatter = '.', s_scatter = 1, alpha_hist = 0.5, bins = 20, superimpose = True, labelNames = None):
    '''
    Given X, nOfSamples rows * nOfFeatures columns, containing the features
    and y, nOfSamples rows * 1 column, containing the labels

    plots in a matrix one feature with respect to the other.
    On the diagonal, draws an histogram with the distribution of the given 
    feature for the different labels

    Parameters
    ----------
    X : array
        nOfSamples rows * nOfFeatures columns, containing the features
    y : array
        nOfSamples rows * 1 column, containing the labels
    featNames : list of string, optional
        name of the features. The default is [''].
    labelColors : list of string, optional
        colors of each feature. The default is [''].
    mainTitle : string, optional
        title of the figure. The default is ''.
    alpha_scatter : float [0,1], optional
        alpha of the scatters. Useful when there are many point, so the cloud 
        is more dense where there are more points. The default is 0.5.
    marker_scatter : string, optional
        marker shape for scatter plot. The default is '.'.
    s_scatter : TYPE, optional
        marker size for scatter plot. The default is 1.
    alpha_hist : float[0,1], optional
        alpha of the histograms. The default is 0.5.
    bins : int, optional
        number of bars for the histograms. The default is 20.
    superimpose : bool, optional
        if True, the histrograms are superimposed
        if False, each bar or each label is close to the other

    Returns
    -------
    fig : figure
        DESCRIPTION.
    ax : array (of array) of axis used to plot
        it's possible to modify the plots recalling 
        ax[..][..] if nrows > 1 and ncols > 1;
        ax[..] if nrows == 1 or ncols == 1

    '''

    nOfFeatures = len(X[0])
    labels = np.unique(y)

    # chose colors
    if labelColors == [''] or len(labelColors) != len(labels):
        labelColors = ['C'+str(x) for x in range(len(labels))]

    fig, ax = plt.subplots(nOfFeatures, nOfFeatures)
    plt.suptitle(mainTitle)

    for counter in range(nOfFeatures**2):
        # detect the axis
        row = int(np.floor(counter/nOfFeatures))
        col = int(counter % nOfFeatures)

        this_ax = ax[row, col]
        this_ax.grid()

        # Instead of plotting a dimension against itself, display histograms
        if row == col:
            if superimpose: # the histograms are superimposed
                for (label, labelColor) in zip(labels, labelColors):
                    this_ax.hist([X[:,row][y==label]], \
                    color = labelColor, alpha = alpha_hist, bins = bins)
            else: # each bar is next to the other
                this_ax.hist([X[:,row][y==label] for label in labels], \
                color = labelColors, alpha = alpha_hist, bins = bins)
            this_ax.set_title(featNames[row])
            if row == 0 and labelNames: # eventually adding the legend
                this_ax.legend(labelNames)

        # Plot row against col
        else:
            for (label, labelColor) in zip(labels, labelColors):
                this_ax.scatter([X[:,row][y==label]], [X[:,col][y==label]], \
                color = labelColor, s = s_scatter, \
                marker = marker_scatter, alpha = alpha_scatter)

            #this_ax.set_title(featNames[row] + ' / ' + featNames[col])
            this_ax.set_xlabel(featNames[row])
            this_ax.set_ylabel(featNames[col])

        plt.tight_layout()

    return fig, ax

#%% matrices

def heatmap(matrix, showValues = True, fmt = '.2f', xticklabels = 'auto', yticklabels = 'auto', cmap = "viridis", cbar = False, mainTitle = ''):

    assert len(np.shape(matrix)) == 2, \
        f"matrix dimension is {np.shape(matrix)} but should be (row,col)"

    plt.figure()
    if showValues:
        sns.heatmap(matrix, annot = matrix, fmt = fmt, cmap = cmap, xticklabels = xticklabels, yticklabels = yticklabels, cbar = cbar)
    else:
        sns.heatmap(matrix, cmap = cmap, xticklabels = xticklabels, yticklabels = yticklabels, cbar = cbar)

    plt.suptitle(mainTitle)


def heatmapRaw(values, xticklabels, yticklabels, addMeanRowRight = True, addMeanColBottom = True, showValues = True, fmt = '.2f', cmap = "viridis", cbar = False, mainTitle = ''):
    '''
    Given a 1d array, reshapes it to the size specified from the length of
    xticklabels and yticklabels and passes it to heatmap
    The reshaping is done 'rowwise':
        values = [0, 1, 2, 3, 4, 5]
        xticklabels = ['a', 'b', 'c']
        yticklabels = ['A', 'B']
          a  b  c
        A 0, 1, 2,
        B 3, 4, 5

    Parameters
    ----------
    values : TYPE
        DESCRIPTION.
    xticklabels : TYPE
        DESCRIPTION.
    yticklabels : TYPE
        DESCRIPTION.
    addMeanRowRight : TYPE, optional
        DESCRIPTION. The default is True.
    addMeanColBottom : TYPE, optional
        DESCRIPTION. The default is True.
    showValues : TYPE, optional
        DESCRIPTION. The default is True.
    fmt : TYPE, optional
        DESCRIPTION. The default is '.2f'.
    cmap : TYPE, optional
        DESCRIPTION. The default is "viridis".
    cbar : TYPE, optional
        DESCRIPTION. The default is False.
    mainTitle : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    None.

    '''

    assert len((values)) == len(xticklabels)*len(yticklabels), \
        f"values has {len(values)} items. Not corresponding with {len(xticklabels)}*{len(yticklabels)} = {len(xticklabels)*len(yticklabels)} "

    matrix = np.reshape(values,(len(yticklabels),len(xticklabels)))
    if addMeanRowRight:
        # add row wise mean value on the right
        matrix = np.concatenate((matrix,np.expand_dims(np.mean(matrix, axis = 1), axis=1)), axis = 1)
        xticklabels = np.append(xticklabels, 'mean')

    if addMeanColBottom:
        # add column wise mean value in the bottom
        matrix = np.concatenate((matrix,np.expand_dims(np.mean(matrix, axis = 0), axis=0)), axis = 0)
        yticklabels = np.append(yticklabels, 'mean')

    heatmap(matrix, showValues, fmt = fmt, xticklabels = xticklabels, yticklabels = yticklabels, cmap = cmap, cbar = cbar, mainTitle = mainTitle)
