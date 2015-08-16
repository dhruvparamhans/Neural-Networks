import numpy as np 
from numba import double 
from numba.decorators import jit 
from DropoutLayer import Dropout
from FullyConnectedLayer import FullyConnected
from SoftmaxLayer import Softmax 
from Activations import Relu, Sigmoid
from InputLayer import Input 
from Layer import Layer 
# def pairwise_python (X, D):
# 	M, N = X.shape
# 	for i in range(M):
# 		for j in range(M):
# 			d = 0.0
# 			for k in range(N):
# 				tmp = X[i,k] - X[j,k]
# 				d += tmp * tmp
# 			D[i,j] = np.sqrt(d)

# @jit(arg_types = [double[:, :], double[:,:]])
# def pairwise_numba (X, D):
# 	M, N = X.shape
# 	for i in range(M):
# 		for j in range(M):
# 			d = 0.0
# 			for k in range(N):
# 				tmp = X[i,k] - X[j,k]
# 				d += tmp * tmp
# 			D[i,j] = np.sqrt(d)


class switch(object):
	def __init__(self, value):
		self.value = value
		self.fall = False

	def __iter__(self):
		yield self.match
		raise StopIteration 

	def match(self, *args):
		if self.fall or not args:
			return True
		elif self.value in args:
			self.fall = True
			return True
		else:
			return False

l1 = Layer(layer_type = 'input', in_sx = 10, in_sy = 10, in_depth = 1)
l2 = Layer(layer_type = 'fullyconnect', num_neurons = 20)
l3 = Layer(layer_type = 'fullyconnect', num_neurons = 40)
layer_list = [l1, l2, l3]
layers = [Layer()]*len(layer_list)

layers[0] = Input(layer_list[0].in_sx, layer_list[0].in_sy, layer_list[0].in_depth) 

for i in range(1, len(layer_list)):
	layer = layer_list[i]
	previous_layer = layers[i-1] 
	layer.in_sx = previous_layer.out_sx
	layer.in_sy = previous_layer.out_sy 
	layer.in_depth = previous_layer.out_depth
	v = layer.layer_type
	in_sx = layer.in_sx 
	in_sy = layer.in_sy 
	in_depth = layer.in_depth 
	num_neurons = (layer.num_neurons, None)[layer.num_neurons is None]
	droput_prob = (layer.dropout_prob, 0.5)[layer.dropout_prob is None]
	for case in switch(v):
		if case('dropout'):
			layers[i] = Dropout(in_sx, in_sy, in_depth)
			break 
		if case('fullyconnect'):
			layers[i] = FullyConnected(num_neurons,in_sx, in_sy, in_depth)
			break 
		if case('softmax'):
			layers[i] = Softmax(in_sx, in_sy, in_depth)
			break
		if case('relu'):
			layers[i] = Relu(in_sx, in_sy, in_depth)
			break
		if case('sigmoid'):
			layers[i] = Sigmoid(in_sx, in_sy, in_depth)
			break
		if case():
			print "Error Unrecognized error type"