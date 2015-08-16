from Vol import Vol 
import numpy as np 
import copy 

class Training:
	"""
	This class performs all the training using SGD and ada-delta
	"""

	def __init__(self, net, learning_rate = 0.01,  l1_decay = 0.0, l2_decay = 0.0,  batch_size = 1, method = 'sgd', 
				momentum = 0.9, rho = 0.95, eps = 1e-6):
		"""
		Takes in as argument a net to train over. 
		This is done to make the class independent of the particular
		net that we are training over.
		"""
		self.net = net 

		self.l1_decay = l1_decay 
		self.l2_decay = l2_decay

		self.batch_size = batch_size
		self.method = method 

		## These are the variables for Ada-delta 

		self.rho = rho 
		self.eps = eps
		self.momentum = momentum 

		self.iter_count = 0 ##This counts the number of iterations to be performed
		self.grad_sum = []  ## To store the gradients 
		self.xsum = [] ## For the adadelta calculation

		def train(x,y):
			"""
			The function applies the training algorithm on the network
			This is where the magic happens.
			x is the training data and y are the training labels 
			"""

			##We first perform the forward pass 
			self.net.forward(x,is_training = True)

			##Perform the backward pass 
			cost_loss = self.net.backward(y)
			l1_decay_loss = 0.0
			l2_decay_loss = 0.0 

			self.iter_count +=1

			##For adadelta and adagrad, we need to accumulate the 
			##gradients, so we initialize the corresponding lists 

			if (self.iter_count % self.batch_size == 0):

				##We collect all the parameters and Gradients of the whole network
				grad_list = self.net.getParamsAndGrad()

				if (len(self.grad_sum) == 0 & (self.method != 'sgd' or self.momentum > 0.0 )):

					for i in range(len(grad_list)):
						self.grad_sum.append(np.zeros(len(grad_list[i]['params'])))
						if (self.method == 'adadelta') :
							self.xsum.append(np.zeros(len(grad_list[i]['params'])))
						else:
							self.xsum.append([])

				for i in xrange(len(grad_list)):
					grad = grad_list[i]
					params = grad['params']
					grads = grad['grads']

					##Check whether the parameters l1_decay_mul exist 
					##in the params or not 
					if 'l1_decay_mul' in grad.keys():
						l1_decay_mul = grad['l1_decay_mul']
					else:
						l1_decay_mul = 1.0 
					##Similar update for l2_decay_mul 

					if 'l2_decay_mul' in grad.keys():
						l2_decay_mul = grad['l2_decay_mul']
					else:
						l2_decay_mul = 1.0 

					l1_decay = self.l1_decay * l1_decay_mul
					l2_decay = self.l2_decay * l2_decay_mul

					nb_params = len(params)

					for j in xrange(nb_params):
						l2_decay_loss += l2_decay * params[j] * params[j] / 2.0 
						l1_decay_loss += l1_decay * abs(params[j])
						p = (1,-1)[params[j] > 0]
						l1_grad = l1_decay * p
						l2_grad = l2_decay * params[j] 

						gradij = (l2grad + l1grad + grads[j]) / self.batch_size

						gradsumi = self.grad_sum[i]
						xsumi = self.xsum[i]  

						if (self.method == 'adagrad'):
							gradsumi[j] = gradsumi[j] + grads[j] + grads[j] 
							dx = self.learning_rate/ np.sqrt(gradsumi[j] + self.eps) * gradij
							params[j] += dx 

						elif (self.method == 'adadelta'):

							gradsumi[j] = self.rho * gradsumi[j] + (1-self.rho) * gradij * gradij
              				dx = - np.sqrt((xsumi[j] + self.eps)/(gradsumi[j] + self.eps)) * gradij
              				xsumi[j] = self.rho * xsumi[j] + (1-self.rho) * dx * dx
              				params[j] += dx;

              			else:
              				##Default is SGD 
              				if (self.momentum > 0.0):
              					dx = self.momentum * gradsumi[j] - self.learning_rate * gradij; 
                				gradsumi[j] = dx; 
                				params[j] += dx; 
                			else:
                				params[j] +=  -self.learning_rate * gradij 
                		grads[j] = 0.0 	

        	return {'l2_decay_loss': l2_decay_loss, 'l1_decay_loss': l1_decay_loss,
              'cost_loss': cost_loss, 'softmax_loss': cost_loss, 
              'loss': cost_loss + l1_decay_loss + l2_decay_loss}
