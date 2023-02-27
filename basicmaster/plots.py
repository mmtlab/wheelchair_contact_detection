# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:38:34 2022

@author: eferlius
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
from . import utils
# import utils

def createSubPlots(nOfPlots = 0, sharex = False, sharey = False,
                   nrows = 0, ncols = 0, mainTitle = '', listTitles = [''],
                   listXlabels = [''], listYlabels = ['']):
    '''
    Creates a grid of subplots with nOfPlots or nrows and ncols specified

    Parameters
    ----------
    nOfPlots : int, optional
        number of plots to be created, by default 0
    sharex : bool, optional
        how x is shared between axes (True, 'row', 'col', False), 
        by default False
    sharey : bool, optional
        how y is shared between axes (True, 'row', 'col', False), 
        by default False
    nrows : int, optional
        number of rows for subplot, by default 0
    ncols : int, optional
        number of cols for subplot, by default 0
    mainTitle : str, optional
        main title of the plot, by default ''
    listTitles : list, optional
        list of titles of each axis, ordered in a horizontal list as the axes appear. 
        If no title is associated with an axis, use '' as a placeholder, by default ['']
    listXlabels : list, optional
        list of x label of each axis, ordered in a horizontal list as the axes appear. 
        If no x label is associated with an axis, use '' as a placeholder, by default ['']
    listYlabels : list, optional
        list of y label of each axis, ordered in a horizontal list as the axes appear. 
        If no y label is associated with an axis, use '' as a placeholder, by default ['']

    Returns
    -------
    fig : matplotlib figure
        DESCRIPTION.
    ax : 2d array of axes
        DESCRIPTION.
    '''

    # detect number of rows and of columns
    if nrows*ncols < nOfPlots and nrows != 0 and ncols != 0:
        nrows = 0
        ncols = 0
    if nrows == 0 and ncols == 0:
        nrows = int(np.ceil(np.sqrt(nOfPlots)))
        ncols = int(np.ceil(nOfPlots/nrows))
    elif nrows == 0 and ncols != 0:
        nrows = int(np.ceil(nOfPlots/ncols))
    elif nrows != 0 and ncols == 0:
        ncols = int(np.ceil(nOfPlots/nrows))
    else:
        pass
        # both ncols and nrows are specified

    # add empty titles in the end if some are missing
    listTitles = utils.make_list(listTitles)
    listXlabels = utils.make_list(listXlabels)
    listYlabels = utils.make_list(listYlabels)
    listTitles.extend(['']*(nOfPlots-len(listTitles)))
    listXlabels.extend(['']*(nOfPlots-len(listXlabels)))
    listYlabels.extend(['']*(nOfPlots-len(listYlabels)))

    # create the figure with subplots
    fig, ax = plt.subplots(nrows, ncols, sharex = sharex, sharey = sharey, squeeze = False)
    plt.suptitle(mainTitle)
            
    ac = -1 #ax counter
    for row in range(nrows):
        for col in range(ncols):
            ac += 1
            if ac >= nOfPlots:
                break
            this_ax = ax[row][col]
            this_ax.set_title(listTitles[ac])
            this_ax.set_xlabel(listXlabels[ac])
            this_ax.set_ylabel(listYlabels[ac])
            this_ax.grid()
    return fig, ax
       
def plts(X = [], Y = [], sharex = False, sharey = False, nrows = 0, ncols = 0, 
mainTitle = '', listTitles = [''], listXlabels = [''], listYlabels = [''], 
listLegLabels = [''], listOfkwargs = [{}], common_kwargs = {'marker': '.'}):
    '''
    Given (X,Y), plots them
    X and Y can be 
    - 1D-list
    - 1D-np.array, 
    - list of 1D-lists
    - list of 1D-np.arrays

    *warning*: if instead of 1D the lists or np.arrays are 2D, it's up to the 
    user resize and transpose them in the correct way

    Assuming xn and yn are either lists or np.arrays, it's possible to obtain:
    - plts(x0,y0) -> one axis: x0y0
    - plts([x0,x1],[y0,y1]) -> two axes: x0y0 and x1y1
    - plts([x0,x1,x2],[y0,y1,y2]) -> three axes: x0y0, x1y1 and x2y2
    - plts([x0,[x1,x2]],[y0,[y1,y2]]) -> two axes: x0y0 and x1y1x2y2
    - plts([[x0,x1]],[[y0,y1]]) -> one axis x0y0x1y1 (mind the double square bracket)

    sharex and sharey allow to define the axis sharing (in case of more than 
    one axis, otherwise it's ignored).

    nrows and ncols allow to decide the layout of the subplot (in case of more 
    than one axis, otherwise it's ignored).

    mainTilte is the title of the plot.

    listTitles is a 1D-list containing the title of each axis.

    listLegLabels and listOfkwargs are 1D-lists containing respectively the label 
    and the kwargs to be applied to each plot. 
    If X and Y are 2D (ex: plts([x0,[x1,x2]],[y0,[y1,y2]]), it's not necessary 
    to follow the same structure to specify labels and kwargs, just use the 1D-list 
    (in this case listLegLabels = [l0, l1, l2] and listOfkwargs = [kw0, kw1, kw2]). 
    If no label for a plot, use '' as a placeholder.
    If no kwargs for a plot, use {} as a placeholder.

    common_kwargs are applied to all the plots, the same parameter can be overwritten 
    by means of the corresponding value in listOfkwargs.  
    
    Parameters
    ----------
    X : list, optional
        list of x arrays, can be list, np.array, list of list or list of 
        np.array, by default []
    Y : list, optional
        list of y arrays, can be list, np.array, list of list or list of 
        np.array, by default []
    sharex : bool, optional
        how x is shared between axes (True, 'row', 'col', False), 
        by default False
    sharey : bool, optional
        how y is shared between axes (True, 'row', 'col', False), 
        by default False
    nrows : int, optional
        number of rows for subplot, by default 0
    ncols : int, optional
        number of cols for subplot, by default 0
    mainTitle : str, optional
        main title of the plot, by default ''
    listTitles : list, optional
        list of titles of each axis, ordered in a horizontal list as the axes appear. 
        If no title is associated with an axis, use '' as a placeholder, by default ['']
    listXlabels : list, optional
        list of x label of each axis, ordered in a horizontal list as the axes appear. 
        If no x label is associated with an axis, use '' as a placeholder, by default ['']
    listYlabels : list, optional
        list of y label of each axis, ordered in a horizontal list as the axes appear. 
        If no y label is associated with an axis, use '' as a placeholder, by default ['']
    listLegLabels : list, optional
        list of labels of each plot, ordered in a horizontal list as the plots appear. 
        If no label is associated with a plot, use '' as a placeholder, by default ['']
    listOfkwargs : list, optional
        list of kwargs of each plot, ordered in a horizontal list as the plots appear. 
        If no kwarg is associated with a plot, use {} as a placeholder,
        by default [{}]
    common_kwargs : dict, optional
        kwarg applied to all the plots, by default {'marker': '.'}      
    
    Returns
    -------
    fig : matplotlib figure
        DESCRIPTION.
    ax : 2d array of axes
        DESCRIPTION.
    '''

    Y = utils.make_listOfList_or_listOfNpArray(Y)
    nOfPlots = len(Y)

    if utils.is_emptyList_or_emptyNpArray(X):
        X = [[]] * nOfPlots
    else:
        X = utils.make_listOfList_or_listOfNpArray(X) 

    listLegLabels = utils.make_list(listLegLabels)
    listOfkwargs = utils.make_list(listOfkwargs)

    fig, ax = createSubPlots(nOfPlots, sharex, sharey, nrows, ncols, mainTitle, 
    listTitles, listXlabels, listYlabels)

    nrows = len(ax)
    ncols = len(ax[0])

    lkc = -1 # listLegLabels and listOfKwargs counter
    ac = -1 #ax counter
    for row in range(nrows):
        for col in range(ncols):
            ac += 1
            if ac >= nOfPlots:
                continue

            this_ax = ax[row, col]
            this_X = utils.make_listOfList_or_listOfNpArray(X[ac])
            this_Y = utils.make_listOfList_or_listOfNpArray(Y[ac])

            tac = -1 #this ax counter
            for x, y in zip (this_X, this_Y):
                lkc += 1
                tac += 1

                this_plt_kwargs = common_kwargs.copy()
                try:
                    this_plt_kwargs.update(listOfkwargs[lkc])
                except:
                    pass
                finally:
                    try:
                        # if label for this plot
                        this_plt_label = listLegLabels[lkc]
                        if this_plt_label == '':
                            raise Exception()
                        if not utils.is_emptyList_or_emptyNpArray(x):
                            this_ax.plot(x, y, **this_plt_kwargs, label = this_plt_label)
                        else:
                            this_ax.plot(y, **this_plt_kwargs, label = this_plt_label)
                        this_ax.legend()
                    except:
                        # if no label for this plot
                        if not utils.is_emptyList_or_emptyNpArray(x):
                            this_ax.plot(x, y, **this_plt_kwargs)
                        else:
                            this_ax.plot(y, **this_plt_kwargs)
    plt.tight_layout()
    return fig, ax
     
def pltsImg(imgs, sharex = False, sharey = False, nrows = 0, ncols = 0, 
mainTitle = '', listTitles = [''], listXlabels = [''], listYlabels = ['']):
    '''
    Given (X,Y), plots them
    X and Y can be 
    - 1D-list
    - 1D-np.array, 
    - list of 1D-lists
    - list of 1D-np.arrays

    *warning*: if instead of 1D the lists or np.arrays are 2D, it's up to the 
    user resize and transpose them in the correct way

    Assuming xn and yn are either lists or np.arrays, it's possible to obtain:
    - plts(x0,y0) -> one axis: x0y0
    - plts([x0,x1],[y0,y1]) -> two axes: x0y0 and x1y1
    - plts([x0,x1,x2],[y0,y1,y2]) -> three axes: x0y0, x1y1 and x2y2
    - plts([x0,[x1,x2]],[y0,[y1,y2]]) -> two axes: x0y0 and x1y1x2y2
    - plts([[x0,x1]],[[y0,y1]]) -> one axis x0y0x1y1 (mind the double square bracket)

    sharex and sharey allow to define the axis sharing (in case of more than one 
    axis, otherwise it's ignored).

    nrows and ncols allow to decide the layout of the subplot (in case of more 
    than one axis, otherwise it's ignored).

    mainTilte is the title of the plot.

    listTitles is a 1D-list containing the title of each axis.

    listLegLabels and listOfkwargs are 1D-lists containing respectively the label 
    and the kwargs to be applied to each plot. 
    If X and Y are 2D (ex: plts([x0,[x1,x2]],[y0,[y1,y2]]), it's not necessary 
    to follow the same structure to specify labels and kwargs, just use the 1D-list 
    (in this case listLegLabels = [l0, l1, l2] and listOfkwargs = [kw0, kw1, kw2]). 
    If no label for a plot, use '' as a placeholder.
    If no kwargs for a plot, use {} as a placeholder.

    common_kwargs are applied to all the plots, the same parameter can be 
    overwritten by means of the corresponding value in listOfkwargs.  
    
    Parameters
    ----------
    X : list, optional
        list of x arrays, can be list, np.array, list of list or list of np.array, by default []
    Y : list, optional
        list of y arrays, can be list, np.array, list of list or list of np.array, by default []
    sharex : bool, optional
        how x is shared between axes (True, 'row', 'col', False), 
        by default False
    sharey : bool, optional
        how y is shared between axes (True, 'row', 'col', False), 
        by default False
    nrows : int, optional
        number of rows for subplot, by default 0
    ncols : int, optional
        number of cols for subplot, by default 0
    mainTitle : str, optional
        main title of the plot, by default ''
    listTitles : list, optional
        list of titles of each axis, ordered in a horizontal list as the axes appear. 
        If no title is associated with an axis, use '' as a placeholder, by default ['']
    listXlabels : list, optional
        list of x label of each axis, ordered in a horizontal list as the axes appear. 
        If no x label is associated with an axis, use '' as a placeholder, by default ['']
    listYlabels : list, optional
        list of y label of each axis, ordered in a horizontal list as the axes appear. 
        If no y label is associated with an axis, use '' as a placeholder, by default ['']
    listLegLabels : list, optional
        list of labels of each plot, ordered in a horizontal list as the plots appear. 
        If no label is associated with a plot, use '' as a placeholder, by default ['']
    listOfkwargs : list, optional
        list of kwargs of each plot, ordered in a horizontal list as the plots appear. 
        If no kwarg is associated with a plot, use {} as a placeholder,
        by default [{}]
    common_kwargs : dict, optional
        kwarg applied to all the plots, by default {'marker': '.'}      
    
    Returns
    -------
    fig : matplotlib figure
        DESCRIPTION.
    ax : 2d array of axes
        DESCRIPTION.
    '''

    imgs = utils.make_listOfList_or_listOfNpArray(imgs)
    nOfPlots = len(imgs)

    fig, ax = createSubPlots(nOfPlots, sharex, sharey, nrows, ncols, mainTitle, listTitles, listXlabels, listYlabels)

    nrows = len(ax)
    ncols = len(ax[0])

    ac = -1 #ax counter
    for row in range(nrows):
        for col in range(ncols):
            ac += 1
            if ac >= nOfPlots:
                continue

            this_ax = ax[row, col]
            this_ax.imshow(imgs[ac], interpolation = None)
    plt.tight_layout()
    return fig, ax

def pltsImgColorPalette(num = 9):
    imgs = []
    for r in np.linspace(0,num-1,num):
        r = int(r)
        img = np.ones((num,num,3))
        img[:,:,0] = r/(num-1)*255
        for g in np.linspace(0,num-1,num):
            g = int(g)             
            img[g,:,1] = g/(num-1)*255
            for b in np.linspace(0,num-1,num):
                b = int(b)          
                img[g,b,2] = b/(num-1)*255
        imgs.append(img.astype(np.uint8))

    pltsImg(imgs, mainTitle = 'RGB Palette', 
    listTitles = ['R:{}'.format(i/(num-1)*255) for i in range(num)], 
    listYlabels = ['G (0->255)']*(num), listXlabels= ['B (0->255)']*(num))

#%% just to figure out how does it work
if __name__ == '__main__':
    start = 0
    stop = 100
    step = 1
    x = np.arange(start, stop, step)
    y = np.arange(start, stop, step)+np.random.rand(len(x))*10


    fig, ax = plts(x,y, mainTitle = 'plts with one x and y', listLegLabels = 'x+ 0')

    fig, ax = plts([x,x+5],[y,y], mainTitle = 'plts with two x and y detatched', 
    listLegLabels = ['x + 0', 'x + 5'], listOfkwargs = [{'color': 'C4'}, {}], 
    sharex = True, sharey = True, ncols = 2, listTitles = ['here x + 0', 'here x + 5'])

    fig, ax = plts([[x,x+5]],[[y,y]], mainTitle = 'plts with two x and y on the same', 
    listLegLabels = ['x + 0', 'x + 5'], listOfkwargs = [{'color': 'C4'}, {'color' : 'C2'}], 
    sharex = True, sharey = True, listTitles = ['here x + 0 and x + 5'])

    fig, ax = plts([[x,x+5], x+10],[[y,y], y], mainTitle = 'plts with two x and y on the same', 
    listLegLabels = ['x + 0', 'x + 5', 'x + 10'], 
    listOfkwargs = [{'color': 'C4'}, {}, {'color' : 'C2', 'linewidth' : '0', 'markersize' : '10'}], 
    sharex = True, sharey = True, ncols = 1, listTitles = ['here x + 0 and x + 5', 'here x + 10'])
  
    fig, ax = plts([[x,x+5], x+10],[[y,y], y], mainTitle = 'plts with two x and y on the same - np.array()', 
    listLegLabels = ['', 'x + 5', 'x + 10'], sharex = True, sharey = True, ncols = 1, 
    listTitles = ['here x + 0 and x + 5', 'here x + 10'])

    fig, ax = plts([[x,x+5], x+10],[[y,y], y], mainTitle = 'plts with two x and y on the same', 
    sharex = True, sharey = True, ncols = 1, listTitles = ['here x + 0 and x + 5', 'here x + 10'])

    fig, ax = plts([[x,x+5], x+10],[[y,y], y], 
    mainTitle = 'plts with two x and y on the same - with x and y labels', 
    sharex = True, sharey = True, ncols = 1, listTitles = ['here x + 0 and x + 5', 'here x + 10'], 
    listXlabels=['','x'], listYlabels=['couple of y', 'y alone'])

    xm = np.array([[1],[2],[3]])
    ym = np.array([[4],[6],[5]])

    fig, ax = plts(np.squeeze(np.transpose(xm)),np.squeeze(np.transpose(ym)), 
    mainTitle = 'use of vertical numpy arrays -> need to transpose them')



    img00 = [[[255,0,0],[255,0,0],[0,255,0]],[[0,0,255],[0,0,255],[0,255,0]]]
    img01 = [[[255,0,0],[255,0,0],[0,0,0]],[[0,0,255],[0,0,255],[0,0,0]]]
    imggray = [[0, 128, 255],[255,128,0]]

    pltsImg([img00, img01, imggray], listTitles=['img00', 'img01','img gray'], 
    mainTitle = 'list of lists for RGB image')
    pltsImg([np.array(img00), np.array(img01), np.array(imggray)], 
    listTitles=['img00', 'img01','img gray'], mainTitle = 'list of np arrays for RGB image')

    pltsImg([img01], mainTitle = 'list for RGB image with square brackets -> ok')
    pltsImg(img01, mainTitle= 'list for RGB image without square brakets -> not ok')
    pltsImg(np.array(img01), mainTitle= 'np array for RGB image without square brackets -> ok')

    pltsImg([imggray], mainTitle = 'list for gray image with square brackets -> ok')   
    # pltsImg(imggray, mainTitle = 'list for gray image without square brackets -> error')
    pltsImg(np.array(imggray), mainTitle = 'np array for gray image without square brackets -> ok')

    pltsImgColorPalette(4)

    
    plt.draw()
    plt.pause(0.001)
    _ = input('press any key to continue')
    plt.close('all')

# %%
