import cv2 as cv
import numpy as np 

###############################################################################
# This file can be used to compute the error between your generated disparity #
# and the ground truth disparity                                              #
###############################################################################

def mse_err(gt, res):
	err = np.mean((gt - res)**2)
	return err

# Reading the ground truth image
img_gt = cv.imread('./data/disp6.pgm')
# Reading the generated disparity
img_comp = cv.imread('./data/filtered_disparity.png')

# Converting the read images into grayscale
img_gt = cv.cvtColor(img_gt, cv.COLOR_BGR2GRAY)
img_comp = cv.cvtColor(img_comp, cv.COLOR_BGR2GRAY)

# Calculating the error between the 2 images
err = mse_err(img_gt, img_comp)

# Print the output on the STDOUT
print("Error:", err_SSE)
