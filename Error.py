import cv2 as cv
import numpy as np 

# Checking deviation from ground truth

def mse_err(gt, res):
	err = np.mean((gt - res)**2)
	return err

img_gt = cv.imread('/Users/abhyudaypuri/Downloads/sawtooth/disp6.pgm')
img_MI = cv.imread('/Users/abhyudaypuri/Downloads/Filtered_MI.png')
img_SSE = cv.imread('/Users/abhyudaypuri/Downloads/Sub-Pixel/Filtered_MI.png')

print(img_gt.shape)
img_gt = cv.cvtColor(img_gt, cv.COLOR_BGR2GRAY)
img_MI = cv.cvtColor(img_MI, cv.COLOR_BGR2GRAY)
img_SSE = cv.cvtColor(img_SSE, cv.COLOR_BGR2GRAY)

err_MI = mse_err(img_gt, img_MI)
err_SSE = mse_err(img_gt, img_SSE)

print("normal: ", err_MI, "subpixel: ", err_SSE)
