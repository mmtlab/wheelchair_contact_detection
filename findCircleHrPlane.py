# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:16:09 2023

@author: giamp
"""
import opencv as cv2
import numpy as np
import pandas as pd
import hppdWC


def fromCirclesToCirclesDF(circles):
    '''
    Given circles, structure in the following shape:
    array([[[xc, yc, r]],
           [[xc, yc, r]],
           ...
           [[xc, yc, r]]])
    containing the coordinates of centre and radius defining a circle,
    Creates a dataframe with the following columns:
       xc  yc  r


    Parameters
    ----------
    circles : array of arrays
        array([[[xc, yc, r]],
               [[xc, yc, r]],
               ...
               [[xc, yc, r]]])
        the coordinates of centre and radius defining a circle

    Returns
    -------
    circles_df : pandas dataframe
        with the following columns:
    xc  yc  r

    '''
    circles = np.squeeze(circles)
    xc = circles[:,0]
    yc = circles[:,1]
    r  = circles[:,2]

    # creation of the dataframe
    d = {'xc': xc.flatten(), \
         'yc': yc.flatten(), \
         'r' : r.flatten()}

    circles_df = pd.DataFrame(data=d)

    return circles_df
def fromLinesToLinesDF(lines):
    '''
    Given lines, structure in the following shape:
    array([[[x1, y1, x2, y2]],
           [[x1, y1, x2, y2]],
           ...
           [[x1, y1, x2, y2]]])
    containing the coordinates of each pair of points defining a line,
    Creates a dataframe with the following columns:
        x1  y1  x2  y2   distance      slope


    Parameters
    ----------
    lines : array of arrays
        array([[[x1, y1, x2, y2]],
               [[x1, y1, x2, y2]],
               ...
               [[x1, y1, x2, y2]]])
        contains the coordinates of each pair of points defining a line

    Returns
    -------
    lines_df : pandas dataframe
        with the following columns:
    x1  y1  x2  y2   distance      slope

    '''
    lines = np.squeeze(lines)
    x1 = lines[:,0]
    y1 = lines[:,1]
    x2 = lines[:,2]
    y2 = lines[:,3]

    # creation of the dataframe
    d = {'x1': x1.flatten(), \
         'y1': y1.flatten(), \
         'x2': x2.flatten(), \
         'y2': y2.flatten()}

    lines_df = pd.DataFrame(data=d)

    lines_df['distance'] = np.sqrt((lines_df['x1']-lines_df['x2'])**2+(lines_df['y1']-lines_df['y2'])**2)

    lines_df['slope'] = (lines_df['y1']-lines_df['y2'])/(lines_df['x1']-lines_df['x2'])

    return lines_df
def splitDFProperty(df, column, threshold):
    '''
    Splits a dataframe according to the threshold applied on a column
    Doesn't reset the indexes

    Parameters
    ----------
    df : pandas dataframe
        dataframe that has to be splitted.
    column : string
        name of the column whose property defines the splitting.
    threshold : value
        threshold value for splitting.

    Returns
    -------
    df_above_threshold : pandas dataframe
        contains all the rows whose column is EQUAL or GREATER than the treshold.
    df_below_threshold : pandas dataframe
        contains all the rows whose column is LOWER than the treshold.

    '''
    df_above_threshold = df[df[column] >= threshold]
    df_below_threshold = df[df[column] < threshold]

    return df_above_threshold, df_below_threshold

def pickDFProperty(df, column, nofelements = 1, ascending = False):
    '''
    Given a dataframe, returns only rows whose specified columns expresses the GREATEST (if ascending == False) or LOWEST (if ascending == True) values

    Parameters
    ----------
   df : pandas dataframe
       dataframe that has to be filtered.
   column : string
       name of the column whose property defines the splitting.
    nofelements : int, optional
        number of elements to be picked.
        The default is 1.
    ascending : bool, optional
        if False, the GREATEST values are returned 
        if True, the LOWEST. 
        The default is False.

    Returns
    -------
    df_filtered : pandas dataframe
        dataframe containing only the chosen row(s).
    indexes : array of one or more int
        indexes of the chosen rows of the original dataframe.

    '''
    # sort according to column
    df_sorted = df.sort_values(by = column, ascending = ascending)
    # pick only the first nofelements lines
    df_filtered = df_sorted.iloc[0:nofelements,:]

    # df_filtered is a dataframe -> look for index
    if nofelements>1:
        indexes =  df_filtered.index
    # df_filtered is a series -> look for name corresponding to the index of the dataframe where it was picked
    if nofelements==1:
        indexes = [df_filtered.index[0]]

    return df_filtered, indexes
def findCirclesOnImage(img, minDist=1, param1=50, param2=60, minRadius=180, maxRadius=200):
    '''
    find circles on the img, doesn't matter if RGB or BGR since it's converted 
    in gray scale. All the other parameters are the one of cv2.HoughCircles:
        https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Parameters
    ----------
    img : image
        where the circles should be detected.
    minDist : int
        between centers.
    param1 : int
        refer to documentation.
    param2 : int
        refer to documentation. In this case, param2 is recursively decreased 
        until when at least one circle is detected.
    minRadius : int
        minimum radius of the detected circles.
    maxRadius : int
        maximum radius of the detected circles.

    Returns
    -------
    circles : structure in the following shape:
        array([[[xc, yc, r]],
               [[xc, yc, r]],
               ...
               [[xc, yc, r]]])
        containing the coordinates of centre and radius defining a circle

    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = None
    coeff = 1
    # while loop changing param2: make the detection less picky till at least one circle is detected
    while circles is None:
        if coeff < 0:
            return None
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist = minDist,\
        param1 = param1, param2 = param2 * coeff, minRadius = minRadius, maxRadius = maxRadius)
        coeff = coeff - 0.1/coeff
    
    return circles
def findLinesOnImage(img, edge_low_thresh, edge_high_thresh, rho, theta, threshold, min_line_length, max_line_gap):
    '''
    find lines on the img, doesn't matter if RGB or BGR since it's converted 
    in gray scale. All the other parameters are the one of cv2.HoughLinesP:
        https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga8618180a5948286384e3b7ca02f6feeb
    

    Parameters
    ----------
    img : image
        where the circles should be detected.
    edge_low_thresh : int
        for canny edge detection
    edge_high_thresh : int
        for canny edge detection
    rho : double
        distance resolution in pixels of the Hough grid.
    theta : double
       angular resolution in radians of the Hough grid.
    threshold : int
        minimum number of votes (intersections in Hough grid cell).
        In this case, threshold is recursively decreased until when at least 
        one line is detected.
    min_line_length : double
        minimum number of pixels making up a line.
    max_line_gap : double
        maximum gap in pixels between connectable line segments.

    Returns
    -------
    lines :  array of arrays
        array([[[x1, y1, x2, y2]],
               [[x1, y1, x2, y2]],
               ...
               [[x1, y1, x2, y2]]])
        contains the coordinates of each pair of points defining a line.

    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, edge_low_thresh, edge_low_thresh)
    # plt.figure()
    # plt.title('detected edges')
    # plt.imshow(edges)
    lines = None
    coeff = 1
    # while loop changing threshold: make the detection less picky till at least one line is detected
    while lines is None:
        if coeff < 0:
            return None
        lines = cv2.HoughLinesP(edges, rho, theta, threshold * coeff, None, min_line_length, max_line_gap)
        coeff = coeff - 0.1/coeff
    return lines
def findWheelCentreAndHandrim(rgb_img, dep_img, ppx, ppy, fx, fy,
                              fitPlaneMethod = 'RANSAC',
                              minDist = 1, param1 = 50, param2 =60,
                              minRadius = 180, maxRadius = 200,
                              div = 4, edge_low_thresh = 50, edge_high_thresh = 150,
                              rho = 1, theta = np.pi/180, threshold = 30,
                              min_line_length_coeff = 0 , max_line_gap = 20,
                              w = 5, tolerance = 5000,
                              maxMinDepthHandrimPlaneDetection = np.nan,
                              showPlot = False, mainTitle = ''):
    '''
    Given an rgb image and corrispondent depth image with instrinsic camera coordinates, 
    performs circles and lines detection to find the handrim and the centre of the wheel. 
    Returns the centre of the wheel coordinates both in image and metric coordinates, 
    thecentre of the handrim coordinates both in image (+ the radius of the handrim in the image) and metric coordinates, 
    the plane where the handrim lays and the coordinates of the points of the handrim
    
    NB: slows down the execution

    Parameters
    ----------
    rgb_img : matrix M*N*3
        contains RGB or BGR information for every pixel.
    dep_img : matrix M*N*1
        contains DEP information for every pixel.
    ppx : float
        x centre of the metric camera on image
    ppy : float
        y centre of the metric camera on image
    fx : float
        focal distance on x
    fy : float
        focal distance on y
    minDist : int
        between centers. The default is 1.
    param1 : int
        refer to documentation. The default is 50.
    param2 : int
        refer to documentation. The default is 60.
    minRadius : int
        minimum radius of the detected circles. The default is 180.
    maxRadius : int
        maximum radius of the detected circles. The default is 200.
    div : float, optional
        once found the possible handrims, the image considered for centre 
        detection is cropped in a square from the centre till radius / div. 
        The default is 4.
    edge_low_thresh : int
        for canny edge detection.
        The default is 50.
    edge_high_thresh : int
        for canny edge detection.
        The default is 150.
    rho : double
        distance resolution in pixels of the Hough grid.
        The default is 1.
    theta : double
       angular resolution in radians of the Hough grid.
       The default is np.pi/180.
    threshold : int
        minimum number of votes (intersections in Hough grid cell).
        The default is 30.
    min_line_length : double
        minimum number of pixels making up a line.
        The default is 0.
    max_line_gap : double
        maximum gap in pixels between connectable line segments.
        The default is 20.
    w : int, optional
        amplitude of the padding when computing mean and std of the colors 
        crossed by the line. 
        The default is 5.
    tolerance : double, optional
        amplitude of the circular crown. The default is 5000.
    maxDepthHandrimPlaneDetection : double, optional
        during LSTSQ computation to detect the handrim, values more far away 
        than maxDepthHandrimPlaneDetection are ignored. 
        The default is -1, which doesn't delete any value
    minDepth : double, optional
        during LSTSQ computation to detect the handrim, values closer 
        than minDepth are ignored. 
        The default is 0.
    maxX : double, optional
        during LSTSQ computation to detect the handrim, values of x bigger than 
        maxX are ignored
        The default is -1, which doesn't delete any value
    minX : double, optional
        during LSTSQ computation to detect the handrim, values of x smaller than 
        maxY are ignored are ignored. 
        The default is 0.
    maxY : double, optional
        during LSTSQ computation to detect the handrim, values of y bigger than 
        maxY are ignored
        The default is -1, which doesn't delete any value
    minY : double, optional
        during LSTSQ computation to detect the handrim, values of y smaller than 
        maxY are ignored are ignored. 
        The default is 0.
    showPlot : boolean, optional
        if plots are shown. The default is False
    mainTitle : string, optional
        title to be added to the plots
    
    Returns
    -------
    wc_img : array 1*2
        x, y coordinates of the wheel on image 
    hrc_img : array 1*3
        x, y coordinates of the handrim on image + radius
    centre_metric : array 1*3
        x, y, z coordinates of the wheel on metric 
    handrimPlane : hppdWC.geom.Plane3d object
        plane where the handrim is laying.
    dataHandrim : array n*3
        contains x y z coordinates of the handrim, modeled as a circular crown

    '''
    assert fitPlaneMethod == 'LSTSQ' or fitPlaneMethod == 'RANSAC', \
        f"fitPlaneMethod not valid, possible are LSTSQ or RANSAC, got: {fitPlaneMethod}"
    dep_img = dep_img.astype('float')

# =============================================================================
#     #%%0.0 all possible handrims detection on rgb image
# =============================================================================
    circles = findCirclesOnImage(rgb_img, minDist = minDist,\
    param1 = param1, param2 = param2, minRadius = minRadius, maxRadius = maxRadius)

    # if only one circle is found, add another one so the structure is mantained
    if len(circles)==1:
        circles = np.append(circles, circles, axis=1)

    circles_df = fromCirclesToCirclesDF(circles)
    xc_hr, yc_hr, rhr_img = circles_df.mean(axis=0)

# =============================================================================
#     #%%1 wheel centre detection on rgb image
# =============================================================================
    image = rgb_img.copy()
    image_h, image_w, _ = image.shape
# =============================================================================
#     #%%1.0 find lines on cropped image
# =============================================================================
    # initial guess of the area of interest according to the found handrims
    # get the boundaries for crop
    xmin = int(np.maximum(0, xc_hr - rhr_img/div))
    xmax = int(np.minimum(xc_hr + rhr_img/div, image_w-1))
    ymin = int(np.maximum(0, yc_hr - rhr_img/div))
    ymax = int(np.minimum(yc_hr + rhr_img/div, image_h-1))

    # crop the image
    img = image[ymin : ymax + 1, xmin : xmax + 1]
    img_h, img_w, _ = img.shape

    # find lines on the image
    lines = findLinesOnImage(img, \
    edge_low_thresh = edge_low_thresh, edge_high_thresh = edge_high_thresh, \
    rho = rho, theta = theta, threshold = threshold,\
    min_line_length = np.minimum(img_h,img_w) * min_line_length_coeff ,\
    max_line_gap = max_line_gap)

# =============================================================================
#     # %%1.1 three longest lines with m>0 and m<0
# =============================================================================
    # convert lines into dataframe
    lines_df = fromLinesToLinesDF(lines)

    # cancel the lines with slope == 0 or inf
    lines_df_000_slope = lines_df[lines_df['slope'] == 0]
    lines_df_inf_slope = lines_df[lines_df['slope'] == np.inf]

    # if there are both vertical lines and horizontal lines:
    if not lines_df_000_slope.empty and not lines_df_inf_slope.empty:
        lines_df_pos_slope = lines_df_inf_slope.copy()
        lines_df_neg_slope = lines_df_000_slope.copy()
    else:
        # erase lines that are horizontal or vertical, they're on the edge between m>0 and m<0
        lines_df = lines_df[lines_df['slope'] != 0]
        lines_df = lines_df[lines_df['slope'] != np.inf]
        # SPLIT THE DF into lines with pos and neg slope
        lines_df_pos_slope, lines_df_neg_slope = splitDFProperty(lines_df, 'slope', 0.000001)

    # pick the 3 longest lines of each df
    lines_df_pos_slope, index_pos_slope = pickDFProperty(lines_df_pos_slope, 'distance', 3)
    lines_df_neg_slope, index_neg_slope = pickDFProperty(lines_df_neg_slope, 'distance', 3)
    #0.000001 instead of 0 so horizontal lines are considered as neg slope and vertical lines, whoSe slope is inf, are considered as pos slope

    # considering only the chosen lines
    indexes = np.append(index_pos_slope, index_neg_slope)
    validLines = lines[:][indexes]
    # updating the dataframe as well
    validLines_df = lines_df.loc[indexes.tolist(), :].reset_index(drop=True)

    # renaming
    lines = validLines
    lines_df = validLines_df.copy()


# =============================================================================
#     #%%1.2 only the line with highest red mean value with m>0 and m<0
# =============================================================================
    # find which lines are covering more red areas
    linesColors = hppdWC.analysis.colorsOnTheLineImage(img, lines, w = w)
    linesColors_df = hppdWC.analysis.fromLinesColorsToDF(linesColors)

    lines_df = pd.concat([lines_df, linesColors_df], axis = 1)

    # line with smallest std dev of channels with m>0 and m<0
    # split into lines with pos and neg slope
    lines_df_pos_slope, lines_df_neg_slope = splitDFProperty(lines_df, 'slope', 0.00001)
    #0.000001 instead of 0 so horizontal lines are considered as neg slope and vertical lines, whose slope is inf, are considered as pos slope

    # pick the one with highest red value of each df
    lines_df_pos_slope, index_pos_slope = pickDFProperty(lines_df_pos_slope, 'mean ch0', 1, ascending = False)
    lines_df_neg_slope, index_neg_slope = pickDFProperty(lines_df_neg_slope, 'mean ch0', 1, ascending = False)

    # pick the one with smallest std dev of each df
    # lines_df_pos_slope, index_pos_slope = pickDFProperty(lines_df_pos_slope, 'mean of std ch', 1, ascending = True)
    # lines_df_neg_slope, index_neg_slope = pickDFProperty(lines_df_neg_slope, 'mean of std ch', 1, ascending = True)

    # considering only the chosen lines
    indexes = np.append(index_pos_slope, index_neg_slope)
    validLines = lines[:][indexes]
    # updating the dataframe as well
    validLines_df = lines_df.loc[indexes.tolist(), :].reset_index(drop=True)

    # renaming
    lines = validLines
    lines_df = validLines_df.copy()

# =============================================================================
#     #%%1.3 use the class geom to compute the intersection of each pair of lines
# =============================================================================
    # creation of the lines
    validLines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            p1 = hppdWC.geom.Point2d(x1, y1)
            p2 = hppdWC.geom.Point2d(x2, y2)
            myLine = hppdwc.geom.Line2d(p1 = p1, p2 = p2)
            validLines.append(myLine)

    # in case of multiple lines
    # intersection_x = []
    # intersection_y = []        
    # for line1 in validLines:
    #     for line2 in validLines:
    #         # not computing the intersection for lines going in the same direction
    #         if line1.m * line2.m < 0:
    #             x, y = line1.intersection(line2)
    #             intersection_x.append(x)
    #             intersection_y.append(y)

    # only two lines
    intersection_x, intersection_y = validLines[0].intersection(validLines[1])
# =============================================================================
#      #%%1.4 add to intersection the crop of the original image
# =============================================================================
    x_centre = np.mean(intersection_x)
    y_centre = np.mean(intersection_y)

    # value on the complete image
    xwc_img = xmin + x_centre
    ywc_img = ymin + y_centre

# =============================================================================
#     #%%2.0 handrim detection on rgb image
# =============================================================================
    # pick the handrims whose x distance is minimum with respect to the centre of the wheel
    circles_df_min_dist = circles_df[abs(circles_df['xc'] - xwc_img) == min(abs(circles_df['xc'] - xwc_img))]

    # among, them pick the handrim with the greatest radius
    circles_df_min_dist_biggest_radius = circles_df_min_dist[circles_df_min_dist['r'] == max(circles_df_min_dist['r'])]

    # get parameters of the chosen one
    xhrc_img = circles_df_min_dist_biggest_radius['xc'].iloc[0]
    yhrc_img = circles_df_min_dist_biggest_radius['yc'].iloc[0]
    rhrc_img = circles_df_min_dist_biggest_radius['r'].iloc[0]

# =============================================================================
#     #%%3 handrim plane detection on dep image
# =============================================================================
    image_h, image_w = dep_img.shape
    xmask, ymask = np.meshgrid(np.arange(0, image_w, 1), np.arange(0, image_h, 1))
# =============================================================================
#     #%%3.0 extract 3D points for LSTSQ
# =============================================================================
    dep_image = dep_img.copy()
    # remove invalid values
    dep_image[dep_image <= 0] = np.nan
    # creating a mask of the area of interest (circular crown)
    maskValidValues = abs(((xmask-xhrc_img)**2+(ymask-yhrc_img)**2-(rhrc_img)**2))<tolerance
    # give nan values to all the pixel outside of the area of interest
    dep_image[~maskValidValues] = np.nan

    # find points of the mask in the real 3D world
    pc = geom.convert_depth_frame_to_pointcloud_pp_ff(dep_image, ppx, ppy, fx, fy)
    x,y,z = pc
    data = np.transpose([x,y,z])
    # # remove points outside the range
    if not np.isnan(maxMinDepthHandrimPlaneDetection).all():
        data[:,2][data[:,2]>maxMinDepthHandrimPlaneDetection[0]] = np.nan
        data[:,2][data[:,2]<maxMinDepthHandrimPlaneDetection[1]] = np.nan
    # if maxX > 0:
    #     data[:,2][data[:,0]>maxX-ppx] = np.nan
    # data[:,2][data[:,0]<minX-ppx] = np.nan
    # if maxY > 0:
    #     data[:,2][data[:,1]>maxY-ppy] = np.nan
    # data[:,2][data[:,1]<minY-ppy] = np.nan
    #remove nan rows
    data = data[~np.isnan(data).any(axis=1)]

    if showPlot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:,0],data[:,1],data[:,2],c=data[:,2], marker = '.')
        ax.view_init(elev=-90, azim=-90)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    if showPlot:
        highlightedImage = plots.highlightPartOfImage(image, maskValidValues, coeff = 0.7, colorNotInterest = [255, 255, 255])
        handrim = circles[:,circles_df_min_dist_biggest_radius.index]
        plt.figure()
        plt.grid()
        plt.title(mainTitle + ' - detected handrim')
        plt.imshow(plots.circlesOnImage(highlightedImage, handrim))
        plt.axvline(xwc_img, color = 'r')
        plt.axhline(ywc_img, color = 'r')
        plt.axhline(handrim[:,0,1], color = (0,1,0))

        plots.orthogonalProjectionRCamView(data, flag = 'xyzdata', mainTitle = mainTitle + ' - available data')

# =============================================================================
#     #%%3.1 fit plane with least squares
# =============================================================================
    # data are expressed in the depth camera ref, not on top left
    if fitPlaneMethod == 'LSTSQ':
        coeffX, coeffY, constant, normal = geom.fitPlaneLSTSQ(data)
    if fitPlaneMethod == 'RANSAC':
        coeffX, coeffY, constant, normal = geom.fitPlaneRANSAC(data)
    handrimPlane = geom.Plane3d(coeffX = coeffX, coeffY = coeffY, constant = constant)

# =============================================================================
#     #%%3.2 3D coordinates of the handrim, modeled as a 2D circular crown
# =============================================================================
    dep_image = dep_img.copy()
    # creating a mask of the area of interest (circular crown)
    maskValidValues = abs(((xmask-xhrc_img)**2+(ymask-yhrc_img)**2-(rhrc_img)**2))<tolerance
    # give nan values to all the pixel outside of the area of interest
    dep_image[~maskValidValues] = np.nan

    # find dataHandrim in the real 3D world
    pc = geom.convert_depth_frame_to_pointcloud_pp_ff(dep_image, ppx, ppy, fx, fy)
    x,y,z = pc
    dataHandrim = np.transpose([x,y,z])
    #remove nans
    dataHandrim = dataHandrim[~np.isnan(dataHandrim).any(axis=1)]
    X = dataHandrim[:,0]
    Y = dataHandrim[:,1]
    Z = handrimPlane.findZ(X,Y)
    dataHandrim = np.transpose(np.row_stack((X,Y,Z)))

# =============================================================================
#     #%%4.0 3D coordinates of the centres of the wheel and of the handrim
# =============================================================================
    # laying on the handrimPlane
    x_centre_metric, y_centre_metric  = geom.convert_pixel_coord_to_metric_coordinate_pp_ff(xwc_img, ywc_img, ppx, ppy, fx, fy)
    z_centre_metric = handrimPlane.findZ(x_centre_metric, y_centre_metric)

# =============================================================================
#     #%%5.0 pack outputs
# =============================================================================
    # 2D coordinates of wheel centre
    wc_img = [xwc_img, ywc_img]
    # 2D coordinates of handrim on the image + radius
    hrc_img = [xhrc_img, yhrc_img, rhr_img]
    # 3D coordinates of wheel centre in the real world
    centre_metric = [x_centre_metric, y_centre_metric, z_centre_metric]


    # the values of these two should be close to the real handrim in m
    # np.nanmean(np.sqrt((dataHandrim[:,0]-centre_metric[0])**2+(dataHandrim[:,1]-centre_metric[1])**2))
    # np.nanmean(np.sqrt((dataHandrim[:,0]-centre_metric[0])**2+(dataHandrim[:,1]-centre_metric[1])**2+(dataHandrim[:,2]-centre_metric[2])**2))

    # to plot available data, the detected plane and the handrim circular crown
    xmin = np.min(X)
    xmax = np.max(X)
    ymin = np.min(Y)
    ymax = np.max(Y)

    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
    #xx, yy = geom.convert_pixel_coord_to_metric_coordinate_pp_ff(xx, yy, ppx, ppy, fx, fy)
    Zplane = handrimPlane.findZ(xx, yy)
    dataPlane = np.transpose(np.row_stack((xx.flatten(),yy.flatten(), Zplane.flatten())))
    if showPlot:
        fig, ax = plots.orthogonalProjectionRCamView([data, dataPlane, dataHandrim, np.array([centre_metric])], \
        flag = 'xyzdata', mainTitle = mainTitle + ' - final solution', alpha = 0.1, \
        color_list = ['', 'k', '', 'r'])


    return wc_img, hrc_img, centre_metric, handrimPlane, dataPlane
