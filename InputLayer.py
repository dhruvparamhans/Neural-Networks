import Vol 

class Input:

	def __init__(self, in_sx, in_sy, in_depth, out_sx=None, out_sy = None, out_depth = None):

		##TO DO: Use kwargs to make the user any number of arguments as he wants 
		# out_sx = kwargs.get('out_sx', None)
		# out_sy = kwargs.get('out_sy', None)
		# out_depth = kwargs.get('out_depth', None)
		# in_sx = kwargs.get('in_sx', None')
		# in_sy = kwargs.get('in_sy', None)
		# in_depth = kwargs.get('in_depth', None) 

		self.in_sx = in_sx
		self.in_sy = in_sy 
		self.in_depth = in_depth
		##Allows user to input either the input parameters or the output parameters. 
		##Probably not required. Will remove it in the next iteration 
		self.out_sx = out_sx if out_sx is not None else self.in_sx
		self.out_sy = out_sy if out_sy is not None else self.in_sy
		self.out_depth = out_depth if out_depth is not None else self.in_depth 
		self.layer_type = 'input'
		self.in_act = None 
		self.out_act = None 

	def forward(self, V, is_training = None):
		"""
		Performs the forward pass 
		Here will just be an identity function 
		"""

		self.in_act = V
		self.out_act = V
		return self.out_act

	def backward(self):
		return 

	def getParamsAndGrads(self):
		return [] 

	def save_to_json(self, filename):
		json_dict = {}
		json_dict['out_depth'] = self.out_depth
		json_dict['out_width'] = self.out_sx
		json_dict['out_height'] = self.out_sy
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
		self.out_sx = data['out_width']
		self.out_sy = data['out_height']
		self.out_depth = data['out_depth']

		self.layer_type = data['layer_type']

		print "Data loaded"


## Just for testing purposes 

# test_layer = InputLayer(3,3,3)
# #Store in test,json 
# test_Vol = Vol.Vol(3,3,3)
# test_layer.save_to_json('test.json')

# test_layer.forward(test_Vol)

# test_layer_2 =InputLayer()

# test_layer_2.load_from_json('test.json')

# print test_layer_2.out_depth 



