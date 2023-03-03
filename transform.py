# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:31:27 2023

@author: giamp
"""

def changeRefFrameTR(data, XYZcentre, r):
    '''
    Applies a rototranslation matrix to the given data.
    First a translation and then a rotation, both expressed in the original ref frame

    Parameters
    ----------
    data : array n*2 or n*3
        XY(Z) coordinates to be rototranslated.
    XYZcentre : array 1*2 or 1*3
        Coordinates of the centre of the new reference frame seen from the original one.
    r : <scipy.spatial.transform._rotation.Rotation>
        Rotation applied to the data.

    Returns
    -------
    data_new_ref : array n*2 or n*3
        Expressed in the new reference frame.

    '''
    # translation
    data_translated = []
    for i in range(len(data)):
        data_translated.append(data[i] - XYZcentre[i])
    # rotation
    data_new_ref = r.apply(data_translated)

    return data_new_ref

