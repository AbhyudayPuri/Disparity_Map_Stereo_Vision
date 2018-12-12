import numpy as np 
import cv2 as cv 
import math
import Disparity as dis
import host_code

# Reading the images
# L = cv.imread('/Users/abhyudaypuri/Downloads/sawtooth/im0.ppm')
# R = cv.imread('/Users/abhyudaypuri/Downloads/sawtooth/im6.ppm')

L = cv.imread('Data/sawtooth/im0.ppm')
R = cv.imread('Data/sawtooth/im6.ppm')

# L = cv.imread('Data/tsukuba/scene1.row3.col1.ppm')
# R = cv.imread('Data/tsukuba/scene1.row3.col3.ppm')

# L = cv.imread('/Users/abhyudaypuri/Downloads/tsukuba/scene1.row3.col1.ppm')
# R = cv.imread('/Users/abhyudaypuri/Downloads/tsukuba/scene1.row3.col3.ppm')

# Pre-processing and converting them into grayscale
L_gray = cv.cvtColor(L, cv.COLOR_BGR2GRAY)
R_gray = cv.cvtColor(R, cv.COLOR_BGR2GRAY)

# Using histogram equalization on both the images
# L_gray = cv.equalizeHist(L_gray)
# R_gray = cv.equalizeHist(R_gray)
# Res = np.hstack((L_gray, L_eq, R_gray, R_eq))
# cv.imshow('image stacked together', Res)
# cv.waitKey(0)

L_gray = L_gray - np.mean(L_gray)
R_gray = R_gray - np.mean(R_gray)


# Computing the disparity map using OpenCV
# stereo = cv.StereoSGBM_create(numDisparities = 32, blockSize = 16)
# disparity = stereo.compute(L_gray, R_gray) / 16
# disparity = (disparity - np.min(disparity)) / 32
# cv.imwrite('CV.png', disparity)
# Plotting the disparity map 
# cv.imshow('image', disparity)
# cv.waitKey(0)


# Finding disparity using my own function
print("Starting now")

block_size = [9, 9]

# D_map = np.zeros(L_gray.shape)
# for i in range(2):
# 	D_map = D_map + dis.compute_disparity_map(L_gray, R_gray, [5 + 5*i, 5 + 5*i])

# D_map = dis.compute_disparity_map(L_gray, R_gray, block_size)
# D_map_m = dis.compute_disparity_map(R_gray, L_gray, block_size)
D_map = host_code.compute_disparity_gpu(L_gray, R_gray, block_size)
# D_map_b[np.abs(D_map_b - D_map_m) > 1] = 0
# D_map = np.uint8((D_map_b + D_map_m)/2)

print("Done")
# D_map = np.uint8(((D_map - np.min(D_map))/np.max(D_map))*255)
# cv.imshow('image', D_map)
# cv.waitKey(0)
cv.imwrite('Output/Unfiltered_MI.png', D_map)

D_map_filtered = cv.medianBlur(D_map, 13)
# cv.imshow('filtered image', D_map_filtered)
# cv.waitKey(0)
cv.imwrite('Output/Filtered_MI.png', D_map_filtered)
