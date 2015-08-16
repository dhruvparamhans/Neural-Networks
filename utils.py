##Class for defining on the auxiliary utilities we will need 
import numpy as np 
def im2col(input_matrix, filter_size, stride = 1):
	"""
	Performs the im2Col implementation
	input is in general a np.ndarray 
	filter_size is a list of 2 numbers of the kernel size
	
	Input: Matrix of size widthXheight (with no padding)
    (Eventually will need to write a function which performs padding as well.)
	Output: Matrix of size nb_filters X ( (Vx-fx)/stride +1 * (Vy-fy)/stride + 1)
	"""

	width, height = input_matrix.shape
	fx = filter_size[0]
	fy = filter_size[1] 
	col_extent = (height - fy)/stride  + 1
	row_extent = (width - fx)/stride + 1

	start_idx = np.arange(fx)[:,None]*height + np.arange(fy)

	# didx = width*height*np.arange(depth)

	# start_idx = (didx[:,None]+start_idx.ravel()).reshape((-1,fx,fy))

	offset_idx = np.arange(row_extent)[:,None]*height + np.arange(col_extent)

	out = np.take(input_matrix, start_idx.ravel()[:,None] + offset_idx.ravel())

	return out


def col2im(R, size, width):
	dy, dx = size
	xsz = width-dx +1
	ysz = R.shape[0]//xsz
	A = np.empty((ysz + (dy -1), width), dtype = R.dtype)
	for i in xrange(ysz):
		for j in xrange(xsz):
			A[i:i+dy, j:j+dx] = R[i*xsz+j, :].reshape(dy,dx)
	return A 
