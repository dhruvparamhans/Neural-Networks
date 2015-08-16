import base64  
import numpy as np 
import json 

## Helper class written to write numpy array
##to JSON 

class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):

		if isinstance(obj, np.ndarray):
			data_b64 = base64.b64encode(obj.data)
			return dict(__ndarray__=data_b64, 
						dtype = str(obj.dtype),
						shape = obj.shape)

		return json.JSONEncoder(self, obj)

## Helper function to decode the data
	## from json 

def json_numpy_obj_hook(dct):
	""" This decodes the numpy array 
	written in a json file """

	if isinstance(dct, dict) and '__ndarray__' in dct:
		data = base64.b64decode(dct['__ndarray__'])
		return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
	return dct


class Vol:
	""" 
	Class which stores the data for the network. Holds data for the all the filters, 
the weights and also the gradients. 3D volume with height (sy), width (sx) and depth (depth)
	"""

	def __init__(self, sx, sy, depth, init_weights = None):
		self.sx = sx
		self.sy = sy 
		self.depth = depth 

		n = sx*sy*depth 

		##We have flattened out the weight
		## and the gradient matrix 
		self.w = np.zeros(n)
		self.dw = np.zeros(n)

		self.init_weights = init_weights

		##We random;y intialize the weights
		##However, we perform scaling on the 
		##weights to equalize the output variance 
		## for each neuron. 
		if self.init_weights is None:

			scale = np.sqrt(1./(self.sx*self.sy*self.depth))
			for i in xrange(n):
				self.w[i] = scale*np.random.randn()
		else:
			for i in xrange(n):
				self.w[i] = init_weights

	##Helper functions to get, set and add
	## weights or gradients 

	def get_weight(self, x ,y ,d):
		ix = ((self.sx*y) + x)*self.depth + d
		return self.w[ix]

	def set_weight(self ,x ,y ,d ,v):
		ix = ((self.sx*y) + x)*self.depth + d
		self.w[ix] = v 

	def add_weight(self, x, y, d, v):
		ix = ((self.sx*y) + x)*self.depth + d
		self.w[ix] += v 

	def get_grad(self, x, y, d):
		ix = ((self.sx*y) + x)*self.depth + d
		return self.dw[ix]

	def set_grad(self, x, y, d, v):
		ix = ((self.sx*y) + x)*self.depth + d
		self.dw[ix] = v

	def add_grad(self, x, y, d, v):
		ix = ((self.sx*y) + x)*self.depth + d
		self.dw[ix] += v 

	##Function to store the volume in a dictionary 
	##which can then be written to a json file 
	def save_to_json(self, filename):
		json_dict = {}
		json_dict['width'] = self.sx
		json_dict['height'] = self.sy
		json_dict['depth'] = self.depth 
		json_dict['weights'] = self.w
		##Saving gradients might not be necessary 
		##Remove if need space 
		json_dict['gradients'] = self.dw

		json_dict_dump = json.dumps(json_dict, 
						cls = NumpyEncoder)
		with open(filename, 'w') as f:
			f.write(json_dict_dump)
		print "Wrote data to " +  filename
	##Function to read the data from a json file 

	def load_from_json(self, filename):
		with open(filename, 'r') as f:
			data = json.loads(f.read(), object_hook = json_numpy_obj_hook)
		self.sx = data['width']
		self.sy = data['height']
		self.depth = data['depth']

		n = self.sx * self.sy * self.depth 
		self.w = np.zeros(n)
		self.dw = np.zeros(n)

		for i in range(n):
			self.w[i] = data['weights'][i] 

		print "Data loaded"


