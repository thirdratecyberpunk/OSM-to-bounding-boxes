"""
Maps a given image to another image
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import transform
from skimage import data

# text = data.text()

# src = np.array([[0, 0], [0, 50], [300, 50], [300, 0]])
# dst  = np.array([[155, 15], [65, 40], [260, 130], [360, 95]])

# tform3 = transform.ProjectiveTransform()
# tform3.estimate(src, dst)
# warped = transform.warp(text, tform3, output_shape=(50, 300))

# fig, ax = plt.subplots(nrows=2, figsize=(8, 3))

# ax[0].imshow(text, cmap=plt.cm.gray)
# ax[0].plot(dst[:, 0], dst[:, 1], '.r')
# ax[1].imshow(warped, cmap=plt.cm.gray)

# for a in ax:
#     a.axis('off')

# plt.tight_layout()
# plt.show()

# import the two images
cctv_footage = imread('example_data/frame4538.jpg')
to_transform = imread('example_data/junction_timestep_001.png')

# estimate transformation based on source/destination coordinates

# SOURCE is the image we're mapping 

# dst = np.array([[259, 290], 
# [311,123], 
# [161,138]])

# src = np.array([[305, 207], 
# [302,94], 
# [192,164]])

dst = np.genfromtxt('example_data/dst.csv', delimiter=',', encoding='utf-16')
src = np.genfromtxt('example_data/src.csv', delimiter=',', encoding='utf-16')

tform = transform.ProjectiveTransform()
tform.estimate(src, dst)
warped = transform.warp(to_transform, tform)

fig, ax = plt.subplots(nrows=3, figsize=(8, 3))

ax[0].imshow(to_transform, cmap=plt.cm.gray)
ax[0].plot(dst[:, 0], dst[:, 1], '.r')
ax[1].imshow(cctv_footage)
ax[1].plot(src[:, 0], src[:, 1], '.r')
ax[2].imshow(warped, cmap=plt.cm.gray)

plt.show()

