# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:43:16 2023

@author: giamp
"""

import math
def fromCartToCylindricalCoordinates(cart_coord):
    '''
    Transform x,y,z coordinates of a point to r,theta,z (cylindrical) coordinates.

    Parameters
    ----------
    cart_coord : array n*3
        XYZ coordinates of the point.

    Returns
    -------
    cyl_coord : array n*3
        Cylindrical coordinates of the point.
        Following structure [r,theta,z].

    '''
    r=math.sqrt((cart_coord[0])**2 + (cart_coord[1])**2)
    if cart_coord[0]>=0 and cart_coord[1]>=0:
        theta=math.atan(cart_coord[1]/cart_coord[0])
    elif cart_coord[0]<0:
        theta=math.atan(cart_coord[1]/cart_coord[0])+math.pi
    else:
        theta=math.atan(cart_coord[1]/cart_coord[0])+2*math.pi
    cyl_coord=[r,theta,cart_coord[2]]
    return cyl_coord