import ctypes
import pathlib
import numpy as np
import cv2 as cv


# Load the shared library into ctypes
libname = pathlib.Path().absolute() / "prim" / "lib" / "libprim.so"
lib = ctypes.CDLL(libname)
rp_fun = lib.rp

alpha = np.genfromtxt('configs/alpha.dat', delimiter=',')

if not alpha.flags['C_CONTIGUOUS']:
	alpha = np.ascontiguousarray(alpha)

def RP(img, n_proposals, segment_mask=None):
	# Preporcessing image
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	if not img.flags['C_CONTIGUOUS']: img = np.ascontiguousarray(img)
	# Preprocessing segment mask
	if segment_mask is not None:
		segment_mask = segment_mask.astype(np.float64).T
		if not segment_mask.flags['C_CONTIGUOUS']: segment_mask = np.ascontiguousarray(segment_mask)
	out = np.full((n_proposals, img.shape[0], img.shape[1]), False)
	# Execute Random Prim algorithm
	rp_fun(
		# Image 
		img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 
		img.ctypes.shape_as(ctypes.c_uint),
		# Segment mask 
		segment_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if segment_mask is not None else None, 
		# Number of proposals
		ctypes.c_int(n_proposals),
		# Alphas parameters
		alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
		ctypes.c_int(len(alpha)),
		# Output matrix with results
		out.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
	)
	return out


