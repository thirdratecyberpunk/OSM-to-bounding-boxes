"""
Maps a given image to another image
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import transform
from skimage import data

import cv2

class FootageTransformation:
    def __init__(self):
        # import the two images
        # cctv_footage = imread('example_data/frame4538.jpg')
        # to_transform = imread('example_data/junction_timestep_001.png')
        self.dst = np.genfromtxt('example_data/dst.csv', delimiter=',', encoding='utf-16')
        self.src = np.genfromtxt('example_data/src.csv', delimiter=',', encoding='utf-16')
        self.camera_tform = transform.ProjectiveTransform()
        self.camera_tform.estimate(self.src, self.dst)
        self.mtx = cv2.getPerspectiveTransform(self.src, self.dst)

    def camera_translation(self):
        """
        Returns the transformation object that warps the SUMO images to match
        the CCTV camera
        """
        return self.camera_tform
    
    def coordinate_translation(self):
        """
        Returns the transformation object that warps the coordinates to match
        the homography
        """
        return self.mtx


# warped = transform.warp(to_transform, tform)

# need to figure out how to apply these transformations to
# the normalised coordinates - calculate changes made to image 
# then calculate new normalised values?

# fig, ax = plt.subplots(nrows=3, figsize=(8, 3))

# ax[0].imshow(to_transform, cmap=plt.cm.gray)
# ax[0].plot(dst[:, 0], dst[:, 1], '.r')
# ax[1].imshow(cctv_footage)
# ax[1].plot(src[:, 0], src[:, 1], '.r')
# ax[2].imshow(warped, cmap=plt.cm.gray)

# plt.show()

