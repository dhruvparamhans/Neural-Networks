from Vol import Vol 
import copy 
from Layer import Layer 
import numpy as np 
class Dropout:
	"""
	This class implements the Dropout Layer 
	"""

	def __init__(self, in_sx, in_sy, in_depth, drop_prob = 0.5):

		self.out_sx = in_sx
		self.out_sy = in_sy 
		self.out_depth = in_depth

		self.layer_type = 'dropout'

		self.drop_prob = drop_prob

		n = self.out_sx * self.out_sy * self.out_depth
		self.dropped = np.zeros(n)

		self.in_act = None 
		self.out_act = None 

	def forward(self, V, is_training = False):
		"""
		Perform the forward pass
		The default setting is prediction and not 
		training. 
		The dropout probability by default is 0.5 
		So either we drop the neurons from the network
		or we scale them by the random number generated
		"""

		self.in_act = V

		V2 = copy.deepcopy(V)
		N = V.w.shape[0]

		if (is_training):
			for i in xrange(N):
				if (np.random.random() < self.drop_prob):
					V2.w[i]=0
					self.dropped[i] = True 
				else:
					self.dropped[i] = False
		else:
			for i in xrange(N):
				V2.w[i] *= self.drop_prob
		self.out_act = V2
		return self.out_act

	def backward(self):
		"""
		Performs the backpropagation step 
		"""

		V = self.in_act
		chain_grad = self.out_act
		N = V.w.shape[0]
		V.dw = np.zeros(N)
		for i in xrange(N):
			if (self.dropped[i] == False ):
				V.dw[i] = chain_grad.dw[i]

	def getParamsAndGrads(self):
		return []

	def save_to_json(self, filename):
		json_dict = {}
		json_dict['out_depth'] = self.out_depth
		json_dict['out_width'] = self.out_sx
		json_dict['out_height'] = self.out_sy
		json_dict['layer_type'] = self.layer_type
		json_dict['drop_prob'] = self.drop_prob
		json_dict_dump = json.dumps(json_dict, 
						cls = NumpyEncoder)
		with open(filename, 'w') as f:
			f.write(json_dict_dump)
		print "Wrote data to " +  filename
	##Function to read the data from a json file 

	def load_from_json(self, filename):
		with open(filename, 'r') as f:
			data = json.loads(f.read(), object_hook = json_numpy_obj_hook)
		self.out_sx = data['out_width']
		self.out_sy = data['out_height']
		self.out_depth = data['out_depth']
		self.drop_prob = data['drop_prob']
		self.layer_type = data['layer_type']

		print "Data loaded"

## Just for testing purposes 

# test_layer = DropoutLayer(3,3,3)
# #Store in test,json 

# test_layer.save_to_json('test.json')

# test_layer_2 =DropoutLayer(1,1,1)

# test_layer_2.load_from_json('test.json')

# print test_layer_2.out_depth 












