import random 

import numpy as np 

class Network():

	def __init__(self, sizes):
		"""Sizes is a list which defines the architecture
		of the network. The number of layers in the network
		is the length of the list. For example, a 3 layer 
		network could be [2,4,2], where the first layer is
		the input layer with 2 neurons, a hidden layer with
		4 neurons and a final output layer with 2 neurons"""


		self.num_layers = len(sizes)
		self.sizes = sizes

		# We randomly initiate the biases and the weights
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.weights = [np.random.randn(y,x)
						for x,y in zipe(sizes[:-1], sizes[1:])]


	def feedforward(self, a):
		"""Here a is the input"""
		for b,w in zip(self.biases, self.weights):
			a = sigmoid_vec(np.dot(w,a)+b)
		return a 

	def SGD(self, training_data, epochs, mini_batch_size,
		eta, test_data=None):
	""" This is the implementation for the Stochastic Gradient 
	Descent algorithm with a given mini batch size with learning
	rate eta. The training data is given in the form of a tuple 
	(x,y) where x is the input and y is the desired output """


	if test_data: n_test = len(test_data)
	n = len(training_data)
	for j in xrange(epochs):
		random.shuffle(training_data)
		mini_batches = [
					training_data[k:k+mini_batch_size]
					for k in xrange(0,n, mini_batch_size)]
		for mini_batch in mini_batches:
			self.update_mini_batch(mini_batch,eta)
		if test_data:
			print "Epoch {0}: {1} / {2}".format(
				j, self.evaluate(test_data), n_test)
		else:
			print "Epoch {0} complete".format(j)

	def update_mini_batch(self, mini_batch, eta):
		"""This is where we apply the gradient descent
		using backprop on a single mini batch"""

		## The partial derivatives for the bias update 
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		## The partial derivatives for the weight update
		nabla_w = [np.zeros(w.shape) for w in self.weights]


		##feedforward step
		activation = x
		activations = [x]
		zs = []

		for b,w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid_vec(z)
			activations.append(activation)

		## Backward propagation 

		delta = self.cost_derivative(activations[-1],y) * \
				sigmoid_prime_vec(zs[-1])

		nabla_b[-1] = delta 
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		for l in xrange(2, self.num_layers):
			z = zs[-l]
			spv = sigmoid_prime_vec(z)
			delta = np.dot(self.weights[-l+1].transpose(),delta) * spv
			nabla_b[-l] = delta 
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

		return (nabla_b, nabla_w)


	def evaluate(self, test_data):
	""" This function evaluates the performance of 
	the network on some test data"""

		test_results = [(np.argmax(self.feedforward(x)),y)
					for (x,y) in test_data]
		return sum(int(x==y) for (x,y) in test_results)
	
	def cost_derivative(self, output_activations, y):
		return (output_activations-y)

def sigmoid(z):
	return 1.0/(1.0+np,exp(-z))
sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)

