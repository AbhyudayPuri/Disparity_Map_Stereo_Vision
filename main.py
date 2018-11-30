import numpy as np 
import cv2 as cv 
import math
import Mutual_Information as mi 
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

block_size = [9, 9]

# i_range = L_gray.shape[0] // block_size[0]
# j_range = L_gray.shape[1] // block_size[1]
# k_range = L_gray.shape[1] - block_size[1] - 1

# D_map = np.zeros(L_gray.shape)

D_map = dis.compute_disparity_map(L_gray, R_gray, block_size)


# for i in range(i_range):
# 	print("i = ", i)
# 	for j in range(j_range):
# 		cost = math.inf
# 		L_sub = L_gray[block_size[0]*i : block_size[0]*(i+1), block_size[1]*j : block_size[1]*(j+1)]

# 		#for k in range(k_range):
# 		l = block_size[1]*j - 50
# 		for k in range(100):
# 			if (l >= 0 and l < L_gray.shape[1] - block_size[1]):
# 				R_sub = R_gray[block_size[0]*i : block_size[0]*(i+1), l : l + block_size[1]]
# 				# R_sub = R_gray[block_size[0]*i : block_size[0]*(i+1), k : k + block_size[1]]
# 				curr_cost = -mi.mutual_information(L_sub, R_sub)
# 				# curr_cost = np.sqrt(np.sum((L_sub - R_sub)**2))
# 				# curr_cost = np.sum(np.abs(L_sub-R_sub))
# 				if curr_cost < cost:
# 					cost = curr_cost
					
# 					# Sub-pixel estimation
# 					C1 = 0
# 					C3 = 0

# 					if (l-1 >= 0  and l+1 < L_gray.shape[1] - block_size[1]):
# 						R_sub = R_gray[block_size[0]*i : block_size[0]*(i+1), l-1 : l + block_size[1]-1]
# 						C1 = -mi.mutual_information(L_sub, R_sub)
# 						# C1 = np.sqrt(np.sum((L_sub - R_sub)**2))
# 						R_sub = R_gray[block_size[0]*i : block_size[0]*(i+1), l+1 : l + block_size[1]+1]
# 						C3 = -mi.mutual_information(L_sub, R_sub)
# 						# C3 = np.sqrt(np.sum((L_sub - R_sub)**2))

# 					d = np.abs((j*block_size[1] - l))

# 					d_est = d - (1/2) * (C3-C1) / (C1 - 2*curr_cost + C3)
# 					D_map[block_size[0]*i : block_size[0]*(i+1), block_size[1]*j : block_size[1]*(j+1)]	= d_est
# 			l += 1


print("Done")
# D_map = np.uint8(((D_map - np.min(D_map))/np.max(D_map))*255)
# D_map = np.uint8(D_map * 8)
cv.imshow('image', D_map)
cv.waitKey(0)
# cv.imwrite('Unfiltered_MI.png', D_map)

D_map_filtered = cv.medianBlur(D_map, 15)
cv.imshow('filtered image', D_map_filtered)
cv.waitKey(0)
# cv.imwrite('Filtered_MI.png', D_map_filtered)
