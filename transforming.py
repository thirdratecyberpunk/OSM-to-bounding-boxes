"""
Maps a given image to another image
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import transform
from skimage import data

# import the two images
cctv_footage = imread('example_data/frame4538.jpg')
to_transform = imread('example_data/junction_timestep_001.png')

# estimate transformation based on source/destination coordinates

# SOURCE is the image we're mapping 
# DST is the image we're attempting to mimic with the homography

dst = np.genfromtxt('example_data/dst.csv', delimiter=',', encoding='utf-16')
src = np.genfromtxt('example_data/src.csv', delimiter=',', encoding='utf-16')

print(dst,src)

tform = transform.ProjectiveTransform()
tform.estimate(src, dst)
warped = transform.warp(to_transform, tform)

# need to figure out how to apply these transformations to
# the normalised coordinates - calculate changes made to image 
# then calculate new normalised values?

fig, ax = plt.subplots(nrows=3, figsize=(8, 3))

ax[0].imshow(to_transform, cmap=plt.cm.gray)
ax[0].plot(dst[:, 0], dst[:, 1], '.r')
ax[1].imshow(cctv_footage)
ax[1].plot(src[:, 0], src[:, 1], '.r')
ax[2].imshow(warped, cmap=plt.cm.gray)

plt.show()

