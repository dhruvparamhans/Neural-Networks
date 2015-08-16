import Vol
import copy 

import numpy as np 

class Relu:
	"""
	This implements the Rectified Linear Unit Activation
	"""

	def __init__(self,in_sx, in_sy, in_depth):

		self.out_sx = in_sx
		self.out_sy = in_sy 
		self.out_depth = in_depth 

		self.layer_type = 'relu'

		self.in_act = None 
		self.out_act = None

	def forward(self, V):
		self.in_act = V 
		V2 = copy.deepcopy(V)

		N = V.w.shape[0]
		V2_weights = V2.w
		for i in xrange(N):
			V2_weights[i] = (V2.w[i], 0)[V2.w[i] < 0]
		self.out_act = V2
		return self.out_act 

	def backward(self):
		V1 = self.in_act
		V2 = self.out_act
		N = V1.w.shape[0]
		V1.dw = np.zeros(N)
		for i in xrange(N):
			V1.dw[i] = (V2.dw[i], 0)[V2.dw[i] <=0]
	def getParamsAndGrads(self):
		return [] 

	def save_to_json(self, filename):
		json_dict = {}
		json_dict['out_depth'] = self.out_depth
		json_dict['out_sy'] = self.out_sy
		json_dict['out_sx'] = self.out_sx 
		json_dict['layer_type'] = self.layer_type

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

		print "Data loaded"

class Sigmoid:
	"""
	This implements the Sigmoid activation 
	"""

	def __init__(self, in_sx, in_sy, in_depth):

		self.out_sx = in_sx
		self.out_sy = in_sy 
		self.out_depth = in_depth 

		self.layer_type = 'sigmoid'

		self.in_act = None 
		self.out_act = None 

	def forward(self, V):
		self.in_act = V 
		V2 = Vol.Vol(V.sx, V.sy, V.depth, 0.0)
		num = np.ones(V.w.shape[0])
		denom = np.exp((-1)*V.w)
		denom = 1+ denom
		V2.w = num/denom

		self.out_act = V2
		return self.out_act

	def backward(self):
		V1 = self.in_act
		V2 = self.out_act 
		##The magic of numpy arrays :) 
		V1.dw = V2.w * (1-V2.w) * V2.dw


	def getParamsAndGrads(self):
		return [] 

	def save_to_json(self, filename):
		json_dict = {}
		json_dict['out_depth'] = self.out_depth
		json_dict['out_sy'] = self.out_sy
		json_dict['out_sx'] = self.out_sx 
		json_dict['layer_type'] = self.layer_type

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

		print "Data loaded"



