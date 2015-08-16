from Vol import Vol
import copy 
import numpy as np 
from utils import im2col 

class ConvLayer:

	def __init__(self, nb_filters, filter_depth, filter_sx, in_sx, in_sy, filter_sy = None,
		stride = 1, pad = 0, l1_decay = 0.0, l2_decay = 1.0, bias = 0.0):

		##Note that for the functioning of the class we really only need are the following:

		# 1. filter_sx
		# 2. filter_sy ( = filter_sy if not specified) 
		# 3. nb_filters

		# We shall have default values for the rest of the variables. 
		# The variables in_sx, in_sy will be instantiated when we define the neural net class 

		##The input dimensions will be calculated during the construction of the net 
		##Thus we define them here so that we can use them in the functions later 
		##If for any reason the variables are not defined, python will throw all 
		##sorts of errors (usually type errors or attribute errors)

		self.in_depth = filter_depth
		self.in_sx = in_sx
		self.in_sy = in_sy

		self.out_sx = None 
		self.out_sy = None 

		self.stride = stride
		self.l1_decay = l1_decay
		self.l2_decay = l2_decay
		self.pad = pad 
		self.layer_type = 'conv'

		##Filter variables are initialized 

		self.filter_sx = filter_sx 
		if filter_sy is not None:
			self.filter_sy = filter_sy
		else:
			self.filter_sy = self.filter_sx 

		##Define the dimensions of the output 
		##We follow the presentation given in the stanford course
		##on convolutional neural networks. There we derive the relation 
		##between the out put height and width in terms of the 
		##stride, pad and input height 
		self.out_depth = nb_filters

		self.out_sx = int(np.floor((self.in_sx + self.pad * 2 - self.filter_sx)/ self.stride + 1))
		self.out_sy = int(np.floor((self.in_sy + self.pad * 2 - self.filter_sx)/ self.stride + 1))
		
		##Initialize the filters and the biases 
		self.filters = [Vol(self.filter_sx, self.filter_sy, self.in_depth) for i in range(self.out_depth)]
		self.biases = Vol(1,1,self.out_depth, bias)	
	
	def forward(self, V):
		self.in_act = V 

		## We shall use the technique of im2Col as detailed in the notes for the course 
		## CS231n. However, there are still speed-ups to be had if we could link 
		## numpy with the BLAS Library 

		A = Vol(self.out_sx, self.out_sy, self.out_depth, 0.0)
		input_vol = V.w 
		input_vol = input_vol.reshape(V.depth, V.sx, V.sy)

		inputim2col = im2col(input_vol, [self.filter_sx, self.filter_sy], self.stride)

		##Create the filter matrix 
		filtercol = np.zeros((self.out_depth,self.filter_sx*self.filter_sy*self.in_depth))
		for i in xrange(self.out_depth):
			filtercol[i] = self.filters[i].w.flatten()
		##Perform the convolution step 

		convolve = np.dot(filtercol, inputim2col)
		##Adding biases 
		##Could be done in a neater fashion
		for i in xrange(self.out_depth):
			convolve[i] += self.biases.w[i]
		A.w = convolve.flatten()
		self.out_act = A 
		return self.out_act

	def backward(self):

	def getParamsAndGrads(self):
		param_list = []
		for i in xrange(self.out_depth):
			param_list.append({'params': self.filters[i].w, 'grads': self.filters[i].dw, 'l1_decay_mul': self.l1_decay_mul,
			 			'l2_decay_mul': self.l2_decay_mul})
		param_list.append({'params': self.biases.w, 'grads': self.biases.dw, 
			'l1_decay_mul': 0.0, 'l2_decay_mul': 0.0});

		return param_list

	def save_to_json(self, filename):
		json_dict = {}
		json_dict['out_depth'] = self.out_depth
		json_dict['out_width'] = self.out_sx
		json_dict['out_height'] = self.out_sy
		json_dict['layer_type'] = self.layer_type
		json_dict['filter_sx'] = self.filter_sx
		json_dict['filter_sy'] = self.filter_sy 
		json_dict['nb_filters'] = self.out_depth 
		json_dict['stride'] = self.stride 
		json_dict['pad'] = self.pad 
		json_dict['l1_decay'] = self.l1_decay
		json_dict['l2_decay'] = self.l2_decay
		json_dict['in_sx'] = self.in_sx
		json_dict['in_sy'] = self.in_sy 
		json_dict['in_depth'] = self.in_depth
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
		self.in_sx = data['in_sx']
		self.in_sy = data['in_sy']
		self.in_depth = data['in_depth']
		self.filter_sx = data['filter_sx']
		self.filter_sy = data['filter_sy']
		self.stride = data['stride']
		self.pad = data['pad']
		self.l1_decay = data['l1_decay']
		self.l2_decay = data['l2_decay']
		self.layer_type = data['layer_type']

		print "Data loaded"

##Test Code 

conv = ConvLayer(nb_filters = 96, filter_sx = 11, filter_depth = 3, stride = 4, in_sx = 227, in_sy = 227)

inputVol = Vol(227,227,3)

conv.forward(inputVol)







