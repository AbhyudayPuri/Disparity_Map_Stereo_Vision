import numpy as np 

def mutual_information(X, Y):
	X = X.flatten()
	Y = Y.flatten()

	c_XY = np.histogram2d(X, Y)[0]
	c_X = np.histogram(X)[0]
	c_Y = np.histogram(Y)[0]

	H_X = channon_entropy(c_X)
	H_Y = channon_entropy(c_Y)
	H_XY = channon_entropy(c_XY)

	MI = H_X + H_Y - H_XY
	return MI


def channon_entropy(c):
	c_normalized = c / float(np.sum(c))
	c_normalized = c_normalized[np.nonzero(c_normalized)]
	
	H = -np.sum(c_normalized * np.log2(c_normalized))  
	return H