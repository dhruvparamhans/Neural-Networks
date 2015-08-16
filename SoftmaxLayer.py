from Vol import Vol 
import copy 

class Softmax:
	"""
	This class implements the SoftMax Layer. 
	This in general would be the last layer in our network
	since this is the layer we will use to predict the classes
	"""

	def __init__(self, in_sx, in_sy, in_depth):

		self.out_sx = 1
		self.out_sy = 1 
		self.nb_inputs = in_sx * in_sy * in_depth
		self.out_depth = self.nb_inputs
		self.layer_type = 'softmax'
		self.in_act = None 
		self.out_act = None 

		self.exponentials = None

	def forward(self, V):
		self.in_act = V

		V2 = Vol(1,1,self.out_depth, 0.0)

		##To prevent the exponentials from exploding, 
		##we follow the advice from the CS231n 
		##by subtracting the maximum weight 
		##thereby making the calculation stable 
		
		V2_weights = V.w

		max_weight = np.amax(V2_weights)

		exponentials = np.exp(V2_weights - max_weight)
		denom = np.sum(exponentials)

		exponentials /= denom 

		V2.w = exponentials
		##We will use the exponentials for backprop
		self.exponentials = exponentials 
		self.out_act = V2 
		return self.out_act 

	def backward(self, y):

		##First we get the weights and biases for the
		## input to this layer 

		input_matrix = self.in_act

		##We first set the weight matrix of the imput matrix to zero 
		input_matrix.dw = np.zeros(input_matrix.w.shape[0])

		for i in xrange(self.out_depth):
			indicator = 1 if i == y else 0
			mul = -(indicator- self.exponentials[i])
			x.dw[i] = mul

		return -np.log(self.exponential[y])

	def getParamsAndGrads(self):
		return []

	def save_to_json(self, filename):
		json_dict = {}
		json_dict['out_depth'] = self.out_depth
		json_dict['out_sy'] = self.out_sy
		json_dict['out_sx'] = self.out_sx 
		json_dict['layer_type'] = self.layer_type
		json_dict['nb_inputs'] = self.nb_inputs

		json_dict_dump = json.dumps(json_dict, 
						cls = NumpyEncoder)
		with open(filename, 'w') as f:
			f.write(json_dict_dump)
		print "Wrote data to " +  filename
	##Function to read the data from a json file 

	def load_from_json(self, filename):
		with open(filename, 'r') as f:
			data = json.loads(f.read(), object_hook = json_numpy_obj_hook)
		self.out_sx = data['width']
		self.out_sy = data['height']
		self.out_depth = data['depth']

		self.layer_type = data['layer_type']
		self.nb_inputs = data['nb_inputs']

		print "Data loaded"






