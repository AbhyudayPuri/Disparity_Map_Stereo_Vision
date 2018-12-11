import numpy as np 
import cv2 as cv 
# import math
import Mutual_Information as mi 

def compute_disparity_map(L, R, block_size = [7, 7]):
	i_range = L.shape[0] // block_size[0]
	j_range = L.shape[1] // block_size[1]
	k_range = L.shape[1] - block_size[1] - 1

	D_map = np.zeros(L.shape)

	for i in range(i_range):
		print("i = ", i)
		for j in range(j_range):
			cost = np.inf
			L_sub = L[block_size[0]*i : block_size[0]*(i+1), block_size[1]*j : block_size[1]*(j+1)]

			#for k in range(k_range):
			l = block_size[1]*j - 50
			for k in range(100):
				if (l >= 0 and l < L.shape[1] - block_size[1]):
					R_sub = R[block_size[0]*i : block_size[0]*(i+1), l : l + block_size[1]]
					# R_sub = R[block_size[0]*i : block_size[0]*(i+1), k : k + block_size[1]]
					curr_cost = -mi.mutual_information(L_sub, R_sub)
					# curr_cost = np.sqrt(np.sum((L_sub - R_sub)**2))
					# curr_cost = np.sum(np.abs(L_sub-R_sub))
					if curr_cost < cost:
						cost = curr_cost
						
						# Sub-pixel estimation
						C1 = 0
						C3 = 0

						if (l-1 >= 0  and l+1 < L.shape[1] - block_size[1]):
							R_sub = R[block_size[0]*i : block_size[0]*(i+1), l-1 : l + block_size[1]-1]
							C1 = -mi.mutual_information(L_sub, R_sub)
							# C1 = np.sqrt(np.sum((L_sub - R_sub)**2))
							R_sub = R[block_size[0]*i : block_size[0]*(i+1), l+1 : l + block_size[1]+1]
							C3 = -mi.mutual_information(L_sub, R_sub)
							# C3 = np.sqrt(np.sum((L_sub - R_sub)**2))

						d = np.abs((j*block_size[1] - l))

						d_est = d - (1/2) * (C3-C1) / (C1 - 2*curr_cost + C3)
						D_map[block_size[0]*i : block_size[0]*(i+1), block_size[1]*j : block_size[1]*(j+1)]	= d_est
				l += 1

	D_map = np.uint8(D_map * 8)
	return D_map
