import ctypes
import pathlib
import numpy as np
import cv2 as cv

# Load the shared library into ctypes
libname = pathlib.Path().absolute() / "lib" / "libprim.so"
lib = ctypes.CDLL(libname)
rp_fun = lib.rp

alpha = np.genfromtxt('../configs/alpha.dat', delimiter=',')

if not alpha.flags['C_CONTIGUOUS']:
	alpha = np.ascontiguousarray(alpha)


def RP(img, n_proposals):
	if not img.flags['C_CONTIGUOUS']: img = np.ascontiguousarray(img)
	out = np.full((n_proposals, img.shape[0], img.shape[1]), False)
	rp_fun(img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 
			img.ctypes.shape_as(ctypes.c_uint),
			ctypes.c_int(n_proposals),
			alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), 
			ctypes.c_int(len(alpha)),
			out.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
		)
	return out


img = cv.imread('frog.png')
res = RP(img, 1000)

for i in range(res.shape[0]):
	masked = img.copy()
	masked[~res[i]] = [0, 0, 0]
	cv.imshow('image', masked)
	cv.waitKey(0)
	cv.destroyAllWindows()