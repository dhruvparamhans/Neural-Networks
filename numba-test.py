# ##Testing whether I can initialize
# ## the class instance variables outside
# ## the init call 

# class Test:

# 	def __init__(self, name = None):
# 		name = name 

# 		var2 = None

# 	def test(self, V):
# 		in_act = V

# 	def func(self):
# 		print var2 * 2
		

# test = Test()

# ##Testing an im2Col implementation in Python
# ##First with stride and then without it 

import numpy as np 
# A = np.arange(12).reshape(3,4)
# # A = np.random.randint(0,9, (8,6))
# B = [2,2] #Blocksize for scanning D

# M, N  = A.shape
# stride = 1
# col_extent = (N - B[1])/stride  + 1
# row_extent = (M - B[0])/stride + 1

# start_idx = np.arange(B[0])[:,None]*N + np.arange(B[1])

# # didx = M*N*np.arange(D)

# # start_idx = (didx[:,None]+start_idx.ravel()).reshape((-1,B[0],B[1]))

# offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

# out = np.take(A, start_idx.ravel()[:,None] + offset_idx.ravel())


def col2im(R, size, width):
	dy, dx = size
	xsz = width-dx +1
	ysz = R.shape[0]//xsz
	A = np.empty((ysz + (dy -1), width), dtype = R.dtype)
	for i in xrange(ysz):
		for j in xrange(xsz):
			A[i:i+dy, j:j+dx] = R[i*xsz+j, :].reshape(dy,dx)
	return A 
def get_im2col_indices(x_shape, field_height, field_width, padding=0, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = (H + 2 * padding - field_height) / stride + 1
  out_width = (W + 2 * padding - field_width) / stride + 1

  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  # p = padding
  # x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols


##Dummy Implementation of the forward pass in the Conv Network 
##We use the im2col technique used in the CS 231 course


#Test the implementation for the Convolutional step in the Conv Layer Class 

# from Vol import Vol 
# import numpy as np 
# from utils import im2col

# #Lets calculate each step of the calculation 
# nb_filters = 96 



# ##These define the input image parameters 
# in_sx = 227 
# in_sy = 227 
# in_depth = 3 

# filter_sx = 11
# filter_sy = 11

# stride = 4 
# pad = 0
# out_depth = nb_filters
# out_sx = int(np.floor((in_sx + pad * 2 - filter_sx)/ stride + 1))
# out_sy = int(np.floor((in_sy + pad * 2 - filter_sy)/ stride + 1))

# A = Vol(out_sx, out_sy, out_depth, 0.0)

# filters = [Vol(filter_sx, filter_sy, in_depth) for i in range(out_depth)]
# V = Vol(in_sx, in_sy, in_depth)
# input_vol = V.w 
# 		##W

# input_vol = input_vol.reshape(V.depth, V.sx, V.sy)

# inputim2col = im2col(input_vol, [filter_sx, filter_sy], 4)

# ##Create the filter matrix 
# filtercol = np.zeros((out_depth,filter_sx*filter_sy*in_depth))
# for i in xrange(out_depth):
# 	filtercol[i] = filters[i].w.flatten()
# 		##Perform the convolution step 

# convolve = np.dot(filtercol, inputim2col)
# A.w = convolve.flatten()





# class User():
# 	def __init__(self):
# 		self._out_sx = None 

# 	def _setinsx(self, in_sx = None):
# 		self._in_sx = in_sx
# 	def _getinsx(self):
# 		return self._in_sx
# 	in_sx = property(_getinsx, _setinsx)
# 	def _setoutsx(self, out_sx = None):
# 		self._out_sx = 2 + in_sx
# 	def _getoutsx(self):
# 		return self._out_sx

	
# 	out_sx = property(_getoutsx, _setoutsx)


# test = User()
# test.in_sx = 3 
# print test.in_sx 
# print test.out_sx