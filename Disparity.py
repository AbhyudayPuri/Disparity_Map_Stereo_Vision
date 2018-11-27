import numpy as np 
import cv2 as cv 
import math

# Reading the images
L = cv.imread('/Users/abhyudaypuri/Downloads/sawtooth/im0.ppm')
R = cv.imread('/Users/abhyudaypuri/Downloads/sawtooth/im8.ppm')

# Pre-processing and converting them into grayscale
L_gray = cv.cvtColor(L, cv.COLOR_BGR2GRAY)
R_gray = cv.cvtColor(R, cv.COLOR_BGR2GRAY)
# D_map = np.zeros(L_gray.shape)
# print(D_map.shape)

# Computing the disparity map 
stereo = cv.StereoSGBM_create(numDisparities = 32, blockSize = 16)
disparity = stereo.compute(L_gray, R_gray) / 16
disparity = (disparity - np.min(disparity)) / 32
# Plotting the disparity map 
# cv.imshow('image', disparity)
# cv.waitKey(0)


# Finding disparity using my own function
print("Starting now")

block_size = [5, 5]

i_range = L_gray.shape[0] // block_size[0]
j_range = L_gray.shape[1] // block_size[1]
k_range = L_gray.shape[1] - block_size[1] - 1

D_map = np.zeros(L_gray.shape)


for i in range(i_range):
	for j in range(j_range):
		cost = math.inf
		L_sub = L_gray[block_size[0]*i : block_size[0]*(i+1), block_size[1]*j : block_size[1]*(j+1)]

		#for k in range(k_range):
		l = block_size[1]*j - 50
		for k in range(100):
			if (l >= 0 and l < L_gray.shape[1] - block_size[1]):
				R_sub = R_gray[block_size[0]*i : block_size[0]*(i+1), l : l + block_size[1]]
				# R_sub = R_gray[block_size[0]*i : block_size[0]*(i+1), k : k + block_size[1]]
				# R_sub = R_gray[5*i : 5*(i+1) , 7*k : 7*(k+1)]
				curr_cost = np.sum((L_sub - R_sub)**2)
				if curr_cost < cost:
					cost = curr_cost
					D_map[block_size[0]*i : block_size[0]*(i+1), block_size[1]*j : block_size[1]*(j+1)]	= np.abs((j*block_size[1] - l))
					# D_map[block_size[0]*i : block_size[0]*(i+1), block_size[1]*j : block_size[1]*(j+1)] = np.sum((L_sub - R_sub)**2)
					# print(D_map[5*i : 5*(i+1), 7*j : 7*(j+1)])
			l += 1


print("Done")
print("minimum", np.min(D_map))
print("maximum", np.max(D_map))
D_map = np.uint8(((D_map - np.min(D_map))/np.max(D_map))*255)
cv.imshow('image', D_map)
cv.waitKey(0)
