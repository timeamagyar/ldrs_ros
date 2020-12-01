import numpy as np



class Projection(object):
    """
    Class for projecting 3D points into a 2D image frame,
    specific to the RealSense D435 camera model.

    Project points using the Projection.project() method.
    
    Attributes
    ----------
    focal_lengths: tuple
    principal_points: tuple
       Intrinsic camera parameters for raw images
       Projects 3D points in the camera coordinate frame to 2D pixel
       coordinates using the focal lengths (fx, fy) and principal points (ppx, ppy)
    """

    def __init__(self, focal_lengths, principal_points):
        self.focal_lengths = focal_lengths
        self.principal_points = principal_points

    def project(self, points):
        """
        Project points to pixel coordinates.
        
        Parameters
        ----------
        points: numpy.ndarray
            n by 3 numpy array.
            Each row represents a 3D point, as [x, y, z]

        Returns
        -------
        numpy.ndarray
            n by 2 array.
            Each row represents a point projected to 2D camera coordinates
            as [row, col]

        """
        # x / z
        x = points[:, 0] / points[:, 2]
        # y / z
        y = points[:, 1] / points[:, 2]
        # focal lengths
        fx = self.focal_lengths[0]
        fy = self.focal_lengths[1]
        # principal point
        ppx = self.principal_points[0]
        ppy = self.principal_points[1]
        """
        Modified perspective projection equations
        The ppx and ppy offsets place the origin of the image plane in the center of the top left pixel
        The original RealSense SDK uses an Inverse Brown-Conrady distortion model for the left and right infrared imagers
        with zero coefficients, meaning no distortion
        """
        pixel_x = (x * fx) + ppx
        pixel_y = (y * fy) + ppy
        projected = np.column_stack([pixel_x, pixel_y])

        return projected



