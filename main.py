import numpy as np 
import cv2 as cv 
import math
import Disparity as dis
import host_code

################################################################
# Here is where you give the path to read the 2 input images   #
################################################################
# Reading the images
L = cv.imread('./data/im0.ppm')
R = cv.imread('./data/im6.ppm')

# Converting images into grayscale
L_gray = cv.cvtColor(L, cv.COLOR_BGR2GRAY)
R_gray = cv.cvtColor(R, cv.COLOR_BGR2GRAY)

# Pre-processing by mean adjusting the images
L_gray = L_gray - np.mean(L_gray)
R_gray = R_gray - np.mean(R_gray)

print("Starting now")

# Select block size over here 
block_size = [9, 9]

##############################################################
# Calling the disparity map function                         #
# Both the CPU and GPU function calls need to be made here   #                                                         
##############################################################

# Call to CPU function 
# Un-comment this next line to run the code on the CPU
# D_map = dis.compute_disparity_map(L_gray, R_gray, block_size)

# Call to GPU function
D_map = host_code.compute_disparity_gpu(L_gray, R_gray, block_size)

# Smoothening the result by passing it through a median filter
D_map_filtered = cv.medianBlur(D_map, 13)

# Saving the raw and filtered disparity map
cv.imwrite('./data/raw_disparity.png', D_map)
cv.imwrite('./data/filtered_disparity.png', D_map_filtered)
