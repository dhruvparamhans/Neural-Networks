import numpy as np 
from DropoutLayer import Dropout
from FullyConnectedLayer import FullyConnected
from SoftmaxLayer import Softmax 
from Activations import Relu, Sigmoid
from InputLayer import Input 
from Layer import Layer 
##I wanted to use a switch-case statement in Python. 
##Just for fun. I could have done it using a dictionary as well. 
##Its just more fun to know how things can be differently in python 

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


class Network:
	"""
	This class defines the complete network with all the layers 
	The available layers will be 
	1. Convolutional Layer 
	2. Max Pooling Layer 
	3. Fully Connected Layer 
	4. Softmax Layer. 

	The Softmax Layer will be the output layer. Finally, 
	the first layer will always be an input layer
	"""
	def __init__(self):
		self.layers = []

	def makelayers(self, layer_list):
		"""
		Takes a list of layers and creates the network 
		"""
		## The list layer contains an instance of each of the layers
		## present in the network 
		## Thus all the necessary details like the input height and width 
		## should be retrievable as instance variables of each class 
		## This changes from my earlier approach of having dictionary
		## definitions for the layers 

		##I guess we will have to pass a dictionary of the layers that we want 
		## where the user has specified the layer type and the parameters she 
		## wants for the layer 

		## I can just create a Layer class which contains all the necessary variables 
		## for each layer. The class is just a container for the variables and nothing else 
		## prevents me from passing in unnecessarily long lists 
		if (len(layer_list) <2):
			print "You must have input and softmax layers"
		if layer_list[0].layer_type != 'input' :
			print "The first layer should be an input layer"

		
		self.layers = [Layer()]*len(layer_list)
		self.layers[0] = Input(layer_list[0].in_sx, layer_list[0].in_sy, layer_list[0].in_depth)

		for i in range(1, len(layer_list)):
			layer = layer_list[i]
							
			previous_layer = self.layers[i-1] 
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
				if case('input'):
					self.layers[i] = Input(in_sx, in_sy, in_depth)
					break 
				if case('dropout'):
					self.layers[i] = Dropout(in_sx, in_sy, in_depth)
					break 
				if case('fullyconnect'):
					self.layers[i] = FullyConnected(num_neurons,in_sx, in_sy, in_depth)
					break 
				if case('softmax'):
					self.layers[i] = Softmax(in_sx, in_sy, in_depth)
				if case('relu'):
					self.layers[i] = Relu(in_sx, in_sy, in_depth)
				if case('sigmoid'):
					self.layers[i] = Sigmoid(in_sx, in_sy, in_depth)
				if case():
					print "Error Unrecognized error type"



		##Probably not the most pythonic way to do things 
		##but aiming for correctness and less for optimization
		##in the first iteration of the code 
	def getParamsAndGrads(self):
		param_list = []
		for i in xrange(len(self.layers)):
			layer_param_list = self.layers[i].getParamsAndGrads()
			for j in xrange(len(layer_param_list)):
				param_list.append(layer_param_list[j])
		return param_list


	def forward(self, V, is_training=False):
		"""
		Perform the forward pass through each of the 
		layers of the network. The pass is performed in 
		a recursive manner 
		"""

		forward_pass = self.layers[0].forward(V, is_training)
		for i in range(1, len(self.layers)):
			forward_pass = self.layers[i].forward(forward_pass, is_training)
		return forward_pass


	def backward(self, y):
		"""
		Performs the backward pass in a recursive fashion 
		"""
		N = len(self.layers)
		##The last layer is supposed to be a softmax layer 
		loss = self.layers[N-1].backward(y) 
		for i in range(N-2, -1, -1):
			self.layers[i].backward()
		return loss

	def getCostLoss(self, V,y):
		"""
		Returns the loss from the last layer (Softmax)
		"""
		self.forward(V, is_training = False)
		return self.layers[len(self.layers)-1].backward(y)


	def prediction(self):

		return np.max(self.layers[len(self.layers)-1].out_act.w)





##For testing purposes only 

l1 = Layer(layer_type = 'input', in_sx = 10, in_sy = 10, in_depth = 1)
l2 = Layer(layer_type = 'fullyconnect', num_neurons = 20)
l3 = Layer(layer_type = 'fullyconnect', num_neurons = 40)
layer_list = [l1, l2, l3]

net = Network()

net.makelayers(layer_list)



			
