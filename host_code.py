#!/usr/bin/env python

from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import numpy as np 
import time

def compute_disparity_gpu(L_cpu, R_cpu):
	L_cpu = np.int32(np.array(L_cpu))
	R_cpu = np.int32(np.array(R_cpu))
	D_map_cpu = np.int32(np.zeros(L_cpu.shape))

	# Copying the images to device memory
	L_gpu = gpuarray.to_gpu(L_cpu)
	R_gpu = gpuarray.to_gpu(R_cpu)
	D_map_gpu = gpuarray.empty(L_cpu.shape, L_cpu.dtype)

	kernel= """
	#include<stdio.h>
	__global__ void compute_disparity(int *L, int *R, int *D, unsigned L_width, unsigned L_height)



	"""
	kernel = kernel % {
		}

	mod = compiler.SourceModule(kernel)
	compute_disparity = mod.get_function("compute_disparity")
	compute_disparity()
	D_map_gpu = D_map_gpu.get()