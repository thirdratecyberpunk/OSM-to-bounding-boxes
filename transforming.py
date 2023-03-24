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
        # import the reference points from .csv files as float32s
        self.dst = np.float32(np.genfromtxt('example_data/dst.csv', delimiter=',', encoding='utf-16'))
        self.src = np.float32(np.genfromtxt('example_data/src.csv', delimiter=',', encoding='utf-16'))
        
        # image transformation from skimage
        self.camera_tform = transform.ProjectiveTransform()
        self.camera_tform.estimate(self.src, self.dst)

        # calculating transformation object for coordinates
        # using cv2 
        # TODO: check if this can be used for image coords as well
        self.mtx = cv2.getPerspectiveTransform(self.src, self.dst)

    def get_camera_tform(self):
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
    
    def translate_image(self, to_transform):
        """
        Takes the image to warp as a numpy array,
        returns the perspective warped image
        """
        return transform.warp(to_transform, self.camera_tform)
    
    def translate_coordinates(self, to_transform):
        """
        Takes a set of x/y coordinates and returns the homographed set
        """
        return cv2.perspectiveTransform(to_transform, self.mtx)
    
    def show_homography_plot(self):
        """
        Helper function to display the points homography
        uses as reference and translation result
        """
        cctv_footage = imread('example_data/frame4538.jpg')
        to_transform = imread('example_data/junction_timestep_001.png')
        
        warped = self.translate_image(to_transform)

        fig, ax = plt.subplots(nrows=3, figsize=(8, 3))

        ax[0].imshow(to_transform, cmap=plt.cm.gray)
        ax[0].plot(self.dst[:, 0], self.dst[:, 1], '.r')
        ax[1].imshow(cctv_footage)
        ax[1].plot(self.src[:, 0], self.src[:, 1], '.r')
        ax[2].imshow(warped, cmap=plt.cm.gray)

        plt.show()

