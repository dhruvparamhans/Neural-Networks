from Vol import Vol 
import copy 
import numpy as np 


class FullyConnected:

	"""
	This class implements the fully connected layer 
	"""

	def  __init__(self, num_neurons, in_sx, in_sy, in_depth,
	 l1_decay_mul = 0.0, l2_decay_mul = 1.0, bias = 0.0):

		self.in_sx = in_sx 
		self.in_sy = in_sy 
		self.in_depth = in_depth 

		self.bias = bias

		self.out_depth = num_neurons 

		self.out_sx = 1
		self.out_sy = 1
		self.layer_type = 'fullyconnect'

		self.num_inputs = self.in_sx * self.in_sy * self.in_depth
		self.filters = []
		for i in xrange(self.out_depth):
			self.filters.append(Vol(1,1,self.num_inputs))
		# self.filters = [Vol(1,1,self.num_inputs) for i in range(self.out_depth)]

		self.biases = Vol(1,1,self.out_depth, bias) 

	def forward(self, V):
		"""
		Performs the forward pass. 
		Straightforward implementation since every neuron connected 
		to every neuron in the previous layer 
		"""

		self.in_act = V

		input_weights = V.w 

		output = Vol(1,1,self.out_depth, 0.0)

		for i in xrange(self.out_depth):
			a = 0.0
			filter_weights = self.filters[i].w
			# a = np.dot(input_weights, filter_weights) + self.biases.w[i]
			for d in xrange(self.num_inputs):
				a  += input_weights[d] * filter_weights[d]
			a += self.biases.w[i]
			output.w[i] = a
		self.out_act = output
		return self.out_act 

	def backward(self):

		"""
		Performs the backward pass. 
		Simpler than the convolutional layer since
		we just chain the gradients together 
		"""

		input_weights = self.in_act

		input_weights.dw = np.zeros(input_weights.w.shape)

		for i in xrange(self.out_depth):
			filt = self.filters[i]
			grad_prec = self.out_act.dw[i]
			self.biases.dw[i] += self.out_act.dw[i]
			for d in xrange(self.num_inputs):
				input_weights.dw[d] += filt.w[d] * grad_prec
				filt.dw[d] += input_weights.w[d] * grad_prec

	def getParamsAndGrads(self):
		param_list = []
		for i in xrange(self.out_depth):
			param_list.append({'params': self.filters[i].w, 'grads': self.filters[i].dw, 
			'l1_decay_mul': self.l1_decay_mul, 'l2_decay_mul': self.l2_decay_mul})
		param_list.append({'params': self.biases.w, 'grads': self.biases.dw, 
							'l1_decay_mul' : 0.0, 'l2_decay_mul' : 0.0})
		return param_list

	def getDetails(self):
		print "Details of the layer"

		print ({'input_sx': self.in_sx, 'input_sy': self.in_sy, 'in_depth': self.in_depth, \
    			'output_sx': self.out_sx, 'output_sy': self.out_sy, 'output_depth': self.out_depth, \
    			'num_inputs': self.num_inputs, 'num_neurons': self.out_depth}) 



## Test Code 

input_Vol = Vol(2,2,1)
test_fc = FullyConnected(2, input_Vol.sx, input_Vol.sy, input_Vol.depth)

fp = test_fc.forward(input_Vol)

print fp.w
print fp.dw

test_fc.backward()

print test_fc.getDetails() 
