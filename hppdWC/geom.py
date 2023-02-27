# -*- coding: utf-8 -*-
"""
Created on Tue May 24 18:53:52 2022

@author: eferlius
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn import linear_model

from . import utils

class Point2d:
    def __init__(self, x, y):        
        self.x = x
        self.y = y
    def __str__(self):
        return "({:.2f}, {:.2f})".format(self.x, self.y) 

    def dist(self, otherPoint2d):
        dist = np.sqrt((self.x-otherPoint2d.x)**2+(self.y-otherPoint2d.y)**2)
        return dist

class Point3d:
    def __init__(self, x, y, z):        
        self.x = x
        self.y = y
        self.z = z
    def __str__(self):
        return "({:.2f}, {:.2f}, {:.2f})".format(self.x, self.y, self.z) 

    def dist(self, otherPoint3d):
        dist = np.sqrt((self.x-otherPoint3d.x)**2+(self.y-otherPoint3d.y)**2+(self.z-otherPoint3d.z)**2)
        return dist

class Line2d:
    def __init__(self, p1 = None, p2 = None, m = None, q = None, a = None, b = None, c = None):
        '''
        it's possible to create the class in different ways:
            - given 2 Point2d
            - with m and q: EXPLICIT FORM y = m*x + q
            - with a, b and c: IMPLICIT FORM a*x + b*y + c = 0
        

        Parameters
        ----------
        p1 : TYPE, optional
            DESCRIPTION. The default is None.
        p2 : TYPE, optional
            DESCRIPTION. The default is None.
        m : TYPE, optional
            DESCRIPTION. The default is None.
        q : TYPE, optional
            DESCRIPTION. The default is None.
        a : TYPE, optional
            DESCRIPTION. The default is None.
        b : TYPE, optional
            DESCRIPTION. The default is None.
        c : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        NotImplementedError
            if parameters are not 2 points or m and q or a, b and c.

        Returns
        -------
        None.

        '''
        # given 2 Point2d
        if isinstance(p1, Point2d) and isinstance(p2, Point2d) and not m and not q and not a and not b and not c:
            # if same Point2d is given, it's not possible to compute it
            if p1.x == p2.x and p1.y == p2.y:
                return None
            # if same x, the line is vertical. explicit form doesn't exist
            elif p1.x == p2.x:
                self.m = float('inf')
                self.q = float('nan')
                
                self.a = 1
                self.b = 0
                self.c = - p1.x
            # if same y, the line is horizontal.
            elif p1.y == p2.y:
                self.m = 0
                self.q = p1.y
                
                self.a = 0
                self.b = 1
                self.c = - p1.y

            else:
                self.m = (p1.y - p2.y) / (p1.x - p2.x)
                self.q = - self.m * p2.x + p2.y

                if self.q != 0:
                    self.a = self.m / self.q
                    self.b = - 1 / self.q
                    self.c = 1
                else:
                    self.a = self.m
                    self.b = -1
                    self.c = 0
        # with m and q: EXPLICIT FORM y = m*x + q
        elif not p1 and not p2 and m!=None and q!=None and not a and not b and not c:
            self.m = m
            self.q = q
            
            self.a = self.m / self.q
            self.b = - 1 / self.q
            self.c = 1
        # with a, b and c: IMPLICIT FORM a*x + b*y + c = 0
        elif not p1 and not p2 and not m and not q and a!=None  and b!=None  and c!=None :
            if b != 0:
                self.a = a / c
                self.b = b / c
                self.c = 1
                
                self.m = - self.a / self.b
                self.q = - self.c / self.b
            else:
                self.a = a / c
                self.b = b / c
                self.c = 1
                
                self.m = float('inf')
                self.q = float('nan')       
        else:
            raise NotImplementedError('specify either 2 2d points, or m and q or a, b and c')
            
    def __str__(self):
        return "y = {:.2f}*x + {:.2f} or {:.2f}*x + {:.2f}*y + {:.2f} = 0".\
            format(self.m, self.q, self.a, self.b, self.c)
            
    def getExplicitParam(self):
        return self.m, self.q
    
    def getImplicitParam(self):
        return self.a, self.b, self.c

    def intersection(self, otherLine2d):
        if self.m == 'inf' or otherLine2d.m == 'inf':
            if self.m == 'inf':
                x =  - self.c / self.a
                y = otherLine2d.findY(x)
            elif otherLine2d.m == 'inf':
                x =  - otherLine2d.c / otherLine2d.a
                y = self.findY(x)
            p = Point2d(x,y)

        elif self.m == otherLine2d.m and self.q != otherLine2d.q:
            return None # the lines are parallel
        elif self.m == otherLine2d.m and self.q == otherLine2d.q:
            return float('inf') # the lines are the same
        else:
            # Cramer's rule
            # A1 * x + B1 * y = C1
            # A2 * x + B2 * y = C2            
            # x = Dx/D
            # y = Dy/D
            # where D is main determinant of the system:
            # A1 B1
            # A2 B2
            # and Dx:
            # C1 B1
            # C2 B2
            # and Dy
            # A1 C1
            # A2 C2
            
            D  = + self.a * otherLine2d.b - self.b * otherLine2d.a
            Dx = - self.c * otherLine2d.b + self.b * otherLine2d.c
            Dy = - self.a * otherLine2d.c + self.c * otherLine2d.a
            
            p = Point2d(Dx/D, Dy/D)
            
            return p.x, p.y
        
    def findY(self, x):
        '''
        Computes the corresponding y given x
        If the line is vertical (m == inf), None is returned

        Parameters
        ----------
        x : number or array of numbers
            x where you want to know y.

        Returns
        -------
        y : number or array of numbers
            y corresponding to the x values in input.

        '''
        x, _ = utils.toFloatNumpyArray(x)
        if self.m == float('inf'):
            return None
        else:
            y = self.m * x + self.q
        return y
            
    def plot(self, x, ax = None, **kwargs):
        '''
        Plots the line according to x values given on the given ax
        

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        ax : axes, optional
            axes to plot on. If not given, creates a new figure
            The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        ax : axes
            axes where the line is plotted

        '''
        if not ax:
            f, ax = plt.subplots()
        else:
            pass
        ax.plot(x, self.findY(x), **kwargs)
        return ax
        
class Line3d:
    def __init__(self, p1, p2):
        """
        From 2 points in 3d space, computes the direction connecting the two

        Parameters
        ----------
        p1 : Point3d
            DESCRIPTION.
        p2 : Point3d
            DESCRIPTION.

        Returns
        -------
        None.

        """

        # considering the only way we need for hppdWC project: 3d line passing for 2 point
        if isinstance(p1, Point3d) and isinstance(p2, Point3d):
            # if same Point2d is given, it's not possible to compute it
            if p1.x == p2.x and p1.y == p2.y and p1.z == p2.z:
                return None          
            else:                  
                self.l = float(p2.x - p1.x)
                self.m = float(p2.y - p1.y)
                self.n = float(p2.z - p1.z)
                
    def __str__(self):
        return "v = ({:.2f}, {:.2f}, {:.2f})".format(self.l, self.m, self.n)
    
    def cosAngle(self, otherLine3d):
        '''
        Computes the cosine of the angle between the two lines in 3d space.
        The used formula is:
                                l*l' + m*m' + n*n'
            cos(rs) = ---------------------------------------------
                          ________________     ___________________
                        V(l^2 + m^2 + n^2) * V(l'^2 + m'^2 + n'^2)
                                               
        Only the direction of the two lines is taken into account                                       

        Parameters
        ----------
        otherLine3d : Line3d object
            DESCRIPTION.

        Returns
        -------
        result : float
            cosine of the angle between the two lines

        '''
        num = self.l * otherLine3d.l + self.m * otherLine3d.m + self.n * otherLine3d.n
        den1 = np.sqrt(self.l**2 + self.m**2 + self.n**2)
        den2 = np.sqrt(otherLine3d.l**2 + otherLine3d.m**2 + otherLine3d.n**2)
        result = num / (den1*den2)
        return result

class Plane3d:
    def __init__(self, a = None, b = None, c = None, d = None, coeffX = None, coeffY = None, constant = None):
        '''
        

        Parameters
        ----------
        a : TYPE, optional
            DESCRIPTION. The default is None.
        b : TYPE, optional
            DESCRIPTION. The default is None.
        c : TYPE, optional
            DESCRIPTION. The default is None.
        d : TYPE, optional
            DESCRIPTION. The default is None.
        coeffX : TYPE, optional
            DESCRIPTION. The default is None.
        coeffY : TYPE, optional
            DESCRIPTION. The default is None.
        constant : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        if a!=None and b!=None and c!=None and d!=None and not coeffX and not coeffY and not constant:
            self.a = a
            self.b = b
            self.c = c
            self.d = d

            self.coeffX = -a/c
            self.coeffY = -b/c
            self.constant = -d/c
        elif not a and not b and not c and not d and coeffX!=None and coeffY!=None and constant!=None:
            self.coeffX = coeffX
            self.coeffY = coeffY
            self.constant = constant

            self.a = coeffX
            self.b = coeffY
            self.c = -1
            self.d = constant

        # computing the normal direction
        normal = (self.a, self.b, self.c)
        nn = np.linalg.norm(normal)
        self.normal = normal / nn

    def __str__(self):
        return "{:.2f}*x + {:.2f}*y + {:.2f}*z + {:.2f} = 0".\
        format(self.a, self.b, self.c, self.d)

    def describe(self, variable):
        if variable == '':
            return "{:.2f}*x + {:.2f}*y + {:.2f}*z + {:.2f} = 0".\
            format(self.a, self.b, self.c, self.d)
        elif variable == 'x':
            return "x = {:.2f}*y + {:.2f}*z + {:.2f}".\
            format(-self.b/self.a, -self.c/self.a, -self.d/self.a)
        elif variable == 'y':
            return "y = {:.2f}*x + {:.2f}*z + {:.2f}".\
            format(-self.a/self.b, -self.c/self.b, -self.d/self.b)
        elif variable == 'z':
            return "z = {:.2f}*x + {:.2f}*y + {:.2f}".\
            format(-self.a/self.c, -self.b/self.c, -self.d/self.c)
        else:
            raise NameError('possible variables are "x", "y" or "z"')

    def findX(self, y, z):
        y, _ = utils.toFloatNumpyArray(y)
        z, _ = utils.toFloatNumpyArray(z)
        if self.a == 0: # TODO implement properly
            return None
        else:
            x = - (self.b * y + self.c * z - self.d) / self.a
        return x

    def findY(self, x, z):
        x, _ = utils.toFloatNumpyArray(x)
        z, _ = utils.toFloatNumpyArray(z)
        if self.b == 0: # TODO implement properly
            return None
        else:
            y = - (self.a * x + self.c * z + self.d) / self.b
        return y

    def findZ(self, x, y):
        # three different ways
        # solution 1
        # point = np.array([0.0, 0.0, self.constant])
        # d = -point.dot(self.normal)
        # z = (-self.normal[0]*xx - self.normal[1]*yy - d)*1. / self.normal[2]

        # solution 2
        # z = (-self.coeffX*xx - self.coeffY*yy - self.constant) * -1

        # solution 3
        # z = - (self.a*xx + self.b*yy + self.d) / self.c

        x, _ = utils.toFloatNumpyArray(x)
        y, _ = utils.toFloatNumpyArray(y)
        if self.c == 0: # TODO implement properly
            return None
        else:
            z = - (self.a * x + self.b * y + self.d) / self.c
        return z

    def dist(self, point = None, x = None, y = None, z = None, XYZ = None):
        if point!=None and not x and not y and not z and not XYZ.all():
            pass
        elif not point and x!=None and y!=None and z!=None and not XYZ.all():
            point = Point3d(x,y,z)
        elif not point and not x and not y and not z and XYZ.all()!=None:
            point = Point3d(XYZ[0], XYZ[1], XYZ[2])
        num = abs(self.a*point.x + self.b*point.y + self.c*point.z +self.d)
        den = np.sqrt(self.a**2 + self.b**2 + self.c**2)
        dist = num/den
        return dist

    def plot(self, xmin = None, xmax = None, ymin = None, ymax = None, \
             zmin = None, zmax = None, ax = None, **kwargs):

        # compute needed points for plane plotting
        if xmin!=None and xmax!=None and ymin!=None and ymax!=None and not zmin and not zmax:
            xx, yy = np.meshgrid([xmin, xmax], [ymin, ymax])
            zz = self.findZ(xx,yy)

        elif xmin!=None and xmax!=None and zmin!=None and zmax!=None and not ymin and not ymax:
            xx, zz = np.meshgrid([xmin, xmax], [zmin, zmax])
            yy = self.findY(xx,zz)

        elif ymin!=None and ymax!=None and zmin!=None and zmax!=None and not xmin and not xmax:
            yy, zz = np.meshgrid([ymin, ymax], [zmin, zmax])
            xx = self.findX(yy,zz)

        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        else:
            pass
        ax.plot_surface(xx, yy, zz, **kwargs)
        return ax

def fitPlaneLSTSQ(XYZ):
    '''
    Given the 3D coordinates of the points, returns the equation of the plane
    that best fits them in the given form:

    z = coeffX * x + coeffY * y + constant

    Parameters
    ----------
    XYZ : array of 3 columns
        containing x, y and z coordinates.

    Returns
    -------
    coeffX : float
        coefficient of x = coeffX * x + coeffY * y + constant.
    coeffY : float
        coefficient of y = coeffX * x + coeffY * y + constant.
    constant : float
        constant of z = coeffX * x + coeffY * y + constant.
    normal : array 1*3
        direction of the normal to the plane.

    '''
    (rows, cols) = XYZ.shape
    G = np.ones((rows, 3))
    G[:, 0] = XYZ[:, 0]  #X
    G[:, 1] = XYZ[:, 1]  #Y
    Z = XYZ[:, 2]
    (coeffX, coeffY, constant),resid,rank,s = np.linalg.lstsq(G, Z, rcond = -1)
    normal = (coeffX, coeffY, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return coeffX, coeffY, constant, normal

def fitPlaneRANSAC(XYZ):
    X_train = XYZ[:, 0:2]
    y_train = XYZ[:, 2]

    ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    ransac.fit(X_train, y_train)

    coeffX = ransac.estimator_.coef_[0]
    coeffY = ransac.estimator_.coef_[1]
    constant = ransac.estimator_.intercept_

    normal = (coeffX, coeffY, -1)
    nn = np.linalg.norm(normal)
    normal = normal / nn
    return coeffX, coeffY, constant, normal

def rotMatrixToFitPlane(handrimPlane, wheel_centre):
    '''
    Given a plane and the centre of the required ref frame, computes the 
    rotation to obtain the new reference frame by means of aligning two vectors:
    called XYZ the axis of the original frame and xyz of the final one,
        - z is imposed to be coincident with the normal to the plane expressed in XYZ
        - y is imposed to be opposite to original y (since in the image ref 
         frame y is going down and we want y going up)

    Parameters
    ----------
    handrimPlane : hppdWC.geom.Plane3d
        plane where the handrim lies. From this plane it's obtained the normal.
    wheel_centre : array 1*3
        XYZ coordinates of the centre of the new fre frame expressed in the original one.

    Returns
    -------
    rot : <scipy.spatial.transform._rotation.Rotation>
        rotation from original frame to final one.
    rmsd : double
        If the vectors are normals and of unitary norm, is 0.
        check documentation at https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html
    sens : matrix
        Sensitivity matrix.
        check documentation at https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html

    '''
    a = [[0,0,1], [0,-1,0]]
    a_norm = []
    for i in range(len(a)):
        a_norm.append(a[i] / np.linalg.norm(a[i], axis = 0))
    
    vertical =  [0,1,handrimPlane.findZ(wheel_centre[0], wheel_centre[1]+1)-wheel_centre[2]]

    b = [handrimPlane.normal, vertical]
    b_norm = []
    for i in range(len(b)):
        b_norm.append(b[i] / np.linalg.norm(b[i], axis = 0))
    
    rot, rmsd, sens = R.align_vectors(a = a_norm, b = b_norm, return_sensitivity = True)
    return rot, rmsd, sens
def rotMatrixToFitPlaneWRTimg(handrimPlane, wheel_centre):
    '''
    Given a plane and the centre of the required ref frame, computes the 
    rotation to obtain the new reference frame by means of aligning two vectors:
    called XYZ the axis of the original frame and xyz of the final one,
        - z is imposed to be coincident with the normal to the plane expressed in XYZ
        - y is imposed to be opposite to original y (since in the image ref 
         frame y is going down and we want y going up)

    Parameters
    ----------
    handrimPlane : hppdWC.geom.Plane3d
        plane where the handrim lies. From this plane it's obtained the normal.
    wheel_centre : array 1*3
        XYZ coordinates of the centre of the new fre frame expressed in the original one.

    Returns
    -------
    rot : <scipy.spatial.transform._rotation.Rotation>
        rotation from original frame to final one.
    rmsd : double
        If the vectors are normals and of unitary norm, is 0.
        check documentation at https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html
    sens : matrix
        Sensitivity matrix.
        check documentation at https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html

    '''
    a = [[0,0,1], [0,-1,0]]
    a_norm = []
    for i in range(len(a)):
        a_norm.append(a[i] / np.linalg.norm(a[i], axis = 0))
    
    vertical =  [0,1,handrimPlane.findZ(wheel_centre[0], wheel_centre[1]+1)-wheel_centre[2]]

    b = [handrimPlane.normal, vertical]
    b_norm = []
    for i in range(len(b)):
        b_norm.append(b[i] / np.linalg.norm(b[i], axis = 0))
    
    rot, rmsd, sens = R.align_vectors(a = a_norm, b = b_norm, return_sensitivity = True)
    return rot, rmsd, sens

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
    data_translated = data - XYZcentre
    # rotation
    data_new_ref = r.apply(data_translated)

    return data_new_ref

def changeRefFrameTRS(data, XYZcentre, r, coeffX = 1, coeffY = 1, coeffZ = 1):
    '''
    Applies first a rototranslation with the function changeRefFrameTR and
    in the end scales the values on the final x, y, z of the given coeff

    Parameters
    ----------
    data : array n*2 or n*3
        XY(Z) coordinates to be rototranslated.
    XYZcentre : array 1*2 or 1*3
        Coordinates of the centre of the new reference frame seen from the original one.
    r : <scipy.spatial.transform._rotation.Rotation>
        Rotation applied to the data.
    coeffX : double, optional.
        Coefficient for scaling x axis.
        The default is 1.
    coeffY : double, optional.
        Coefficient for scaling y axis.
        The default is 1.
    coeffZ : double, optional.
        Coefficient for scaling z axis.
        The default is 1.

    Returns
    -------
    data_new_ref_scaled : array n*2 or n*3
        Expressed in the new reference frame with scaled values on the axis.

    '''
    # translation and rotation
    data_new_ref = changeRefFrameTR(data, XYZcentre, r)
    # scaling the values
    data_new_ref_scaled = data_new_ref*np.array([coeffX,coeffY,coeffZ])
    return data_new_ref_scaled

def convert_pixel_coord_to_metric_coordinate(pixel_x, pixel_y, camera_intrinsic):
    ppx, ppy, fx, fy = camera_intrinsic.ppx, camera_intrinsic.ppy, camera_intrinsic.fx, camera_intrinsic.fy
    pixel_x = np.array(pixel_x)
    pixel_y = np.array(pixel_y)

    X = (pixel_x - ppx)/fx
    Y = (pixel_y - ppy)/fy
    return X, Y

def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y, camera_intrinsics):
    """
	Convert the depth and image point information to metric coordinates
	Parameters:
	-----------
	depth : double
        The depth value of the image point
	pixel_x : double
		The x value of the image coordinate
	pixel_y : double
		The y value of the image coordinate
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	X : double
		The x value in meters
	Y : double
		The y value in meters
	Z : double
		The z value in meters
	"""
    depth /= 1000
    X = (pixel_x - camera_intrinsics.ppx)/camera_intrinsics.fx *depth
    Y = (pixel_y - camera_intrinsics.ppy)/camera_intrinsics.fy *depth
    return X, Y, depth

def convert_depth_frame_to_pointcloud(depth_image, camera_intrinsics):
	"""
	Convert the depthmap to a 3D point cloud
	Parameters:
	-----------
	depth_frame 	 	 : rs.frame()
						   The depth_frame containing the depth map
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	x : array
		The x values of the pointcloud in meters
	y : array
		The y values of the pointcloud in meters
	z : array
		The z values of the pointcloud in meters
	"""
	
	[height, width] = depth_image.shape

	nx = np.linspace(0, width-1, width)
	ny = np.linspace(0, height-1, height)
	u, v = np.meshgrid(nx, ny)
	x = (u.flatten() - camera_intrinsics.ppx)/camera_intrinsics.fx
	y = (v.flatten() - camera_intrinsics.ppy)/camera_intrinsics.fy

	z = depth_image.flatten() / 1000;
	x = np.multiply(x,z)
	y = np.multiply(y,z)

	x = x[np.nonzero(z)]
	y = y[np.nonzero(z)]
	z = z[np.nonzero(z)]

	return x, y, z


def convert_pointcloud_to_depth(pointcloud, camera_intrinsics):
	"""
	Convert the world coordinate to a 2D image coordinate
	Parameters:
	-----------
	pointcloud : numpy array with shape 3xN
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	x : array
		The x coordinate in image
	y : array
		The y coordiante in image
	"""

	assert (pointcloud.shape[0] == 3)
	x_ = pointcloud[0,:]
	y_ = pointcloud[1,:]
	z_ = pointcloud[2,:]

	m = x_[np.nonzero(z_)]/z_[np.nonzero(z_)]
	n = y_[np.nonzero(z_)]/z_[np.nonzero(z_)]

	x = m*camera_intrinsics.fx + camera_intrinsics.ppx
	y = n*camera_intrinsics.fy + camera_intrinsics.ppy

	return x, y

def convert_pixel_coord_to_metric_coordinate_pp_ff(pixel_x, pixel_y, ppx, ppy, fx, fy):
    pixel_x = np.array(pixel_x)
    pixel_y = np.array(pixel_y)

    X = (pixel_x - ppx)/fx
    Y = (pixel_y - ppy)/fy
    return X, Y

def convert_depth_pixel_to_metric_coordinate_pp_ff(depth, pixel_x, pixel_y, ppx, ppy, fx, fy):
	"""
	Convert the depth and image point information to metric coordinates
	Parameters:
	-----------
	depth : double
        The depth value of the image point
	pixel_x : double
		The x value of the image coordinate
	pixel_y : double
		The y value of the image coordinate
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	X : double
		The x value in meters
	Y : double
		The y value in meters
	Z : double
		The z value in meters
	"""
	X = (pixel_x - ppx)/fx *depth
	Y = (pixel_y - ppy)/fy *depth
	return X, Y, depth

def convert_depth_frame_to_pointcloud_pp_ff(depth_image, ppx, ppy, fx, fy):
	"""
	Convert the depthmap to a 3D point cloud
	Parameters:
	-----------
	depth_frame 	 	 : rs.frame()
						   The depth_frame containing the depth map
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	x : array
		The x values of the pointcloud in meters
	y : array
		The y values of the pointcloud in meters
	z : array
		The z values of the pointcloud in meters
	"""
	
	[height, width] = depth_image.shape

	nx = np.linspace(0, width-1, width)
	ny = np.linspace(0, height-1, height)
	u, v = np.meshgrid(nx, ny)
	x = (u.flatten() - ppx)/fx
	y = (v.flatten() - ppy)/fy

	z = depth_image.flatten() / 1000;
	x = np.multiply(x,z)
	y = np.multiply(y,z)

	x = x[np.nonzero(z)]
	y = y[np.nonzero(z)]
	z = z[np.nonzero(z)]

	return x, y, z


def convert_pointcloud_to_depth_pp_ff(pointcloud, ppx, ppy, fx, fy):
	"""
	Convert the world coordinate to a 2D image coordinate
	Parameters:
	-----------
	pointcloud : numpy array with shape 3xN
	camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
	Return:
	----------
	x : array
		The x coordinate in image
	y : array
		The y coordiante in image
	"""

	assert (pointcloud.shape[0] == 3)
	x_ = pointcloud[0,:]
	y_ = pointcloud[1,:]
	z_ = pointcloud[2,:]

	m = x_[np.nonzero(z_)]/z_[np.nonzero(z_)]
	n = y_[np.nonzero(z_)]/z_[np.nonzero(z_)]

	x = m*fx + ppx
	y = n*fy + ppy

	return x, y


                
    
