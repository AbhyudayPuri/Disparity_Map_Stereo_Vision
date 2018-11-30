import numpy as np 
import cv2 as cv 
import math
import Disparity as dis

# Reading the images
L = cv.imread('/Users/abhyudaypuri/Downloads/sawtooth/im0.ppm')
R = cv.imread('/Users/abhyudaypuri/Downloads/sawtooth/im6.ppm')

# L = cv.imread('/Users/abhyudaypuri/Downloads/tsukuba/scene1.row3.col1.ppm')
# R = cv.imread('/Users/abhyudaypuri/Downloads/tsukuba/scene1.row3.col3.ppm')

# Pre-processing and converting them into grayscale
L_gray = cv.cvtColor(L, cv.COLOR_BGR2GRAY)
R_gray = cv.cvtColor(R, cv.COLOR_BGR2GRAY)

# Using histogram equalization on both the images
L_gray = cv.equalizeHist(L_gray)
R_gray = cv.equalizeHist(R_gray)
# Res = np.hstack((L_gray, L_eq, R_gray, R_eq))
# cv.imshow('image stacked together', Res)
# cv.waitKey(0)

# Computing the disparity map using OpenCV
stereo = cv.StereoSGBM_create(numDisparities = 32, blockSize = 16)
disparity = stereo.compute(L_gray, R_gray) / 16
disparity = (disparity - np.min(disparity)) / 32
# Plotting the disparity map 
# cv.imshow('image', disparity)
# cv.waitKey(0)


# Finding disparity using my own function
print("Starting now")

block_size = [7, 7]

D_map = dis.compute_disparity_map(L_gray, R_gray, block_size)


print("Done")
# D_map = np.uint8(((D_map - np.min(D_map))/np.max(D_map))*255)
# D_map = np.uint8(D_map * 8)
cv.imshow('image', D_map)
cv.waitKey(0)
# cv.imwrite('Unfiltered_MI.png', D_map)

D_map_filtered = cv.medianBlur(D_map, 11)
cv.imshow('filtered image', D_map_filtered)
cv.waitKey(0)
# cv.imwrite('Filtered_MI.png', D_map_filtered)
