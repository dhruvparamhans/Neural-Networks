"""
Base class for all types of layers 
"""


class Layer():

	def __init__(self, *args, **kwargs):

		##Basic Parameters for all networks 
		self.in_sx = kwargs.get('in_sx', None)
		self.in_sy = kwargs.get('in_sy', None)
		self.in_depth = kwargs.get('in_depth', None)
		self.out_sx = kwargs.get('out_sx', None)
		self.out_sy = kwargs.get('out_sy', None)
		self.out_depth = kwargs.get('out_depth', None)
		self.layer_type = kwargs.get('layer_type', None)

		##Specific Parameters for certain layers 

		#dropout
		self.dropout_prob = kwargs.get('dropout_prob', None)

		#Fully Connected Layer 
		self.num_neurons  = kwargs.get('num_neurons', None)

		#Convolutional Layer 
		##To add 

	
