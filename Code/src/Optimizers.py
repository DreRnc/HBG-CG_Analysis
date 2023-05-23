import numpy as np

class HeavyBallGradient():
	'''
    Implementation of Heavy Ball gradient descent optimization algorithm.
	Variants: rprop for backpropagation, Nesterov for momentum.

	Attributes
	----------
	self.step (Float) : learning step
	self.momentum (Float) : coefficient for momentum, multiplying last step updates for wieghts and biases
	self.Nesterov (Bool) : whether optimizer must use Nesterov momentum or not
	self.rprop (Bool) : whether to apply rprop variant or standard backprop
	self.alpha (Float) : coefficient for rprop
	self.beta (Float) : coefficient for rprop
	self.eta_min (Float) : minimum value for rprop
	self.eta_max (Float) : maximum value for rprop
	self.ada_grad (AdaGrad) : instance of AdaGrad class for adaptive gradient descent

    Methods
	-------
	__init__(self, step, momentum):
		Input:
			step (Float) : learning step
			momentum (Float) : coefficient for momentum, multiplying last step updates for wieghts and biases
			Nesterov (Bool) : whether optimizer must use Nesterov momentum or not

	__call__(self, grad_weights, grad_biases, last_weights_update, last_biases_update, rprop):
		Input: 
			grad_weights (np.array) : gradient of the loss with respect to the weights
			grad_biases (np.array) : gradient of the loss with respect to the biases
			last_weights_update (np.array) : update on weights of previous step
			last_biases_update (np.array) : update on biases of previous step
			rprop (Bool) : whether to apply rprop variant or standard backprop
		Output:
			updates on weights (np.array) : update to apply to weights for current optimization step
			updates on biases (np.array) : update to apply to biases for current optimization step
    '''
	
	def __init__(self, step, momentum, Nesterov, rprop, adaptive_grad, n_inputs, n_units):
		self.step = step
		self.momentum = momentum
		self.Nesterov = Nesterov
		self.rprop = rprop
		self.alpha = 1.2
		self.beta = 0.5
		self.eta_min = 1e-4
		self.eta_max = 0.1
		self.weights_steps = 0.01*np.ones((n_inputs, n_units))
		self.biases_steps = 0.01*np.ones((1, n_units))

		if adaptive_grad is True:
			self.ada_grad = AdaGrad(n_inputs, n_units)
		else:
			self.ada_grad = None

	def __call__(self, grad_weights, grad_biases, last_weights_update, last_biases_update, last_grad_weights, last_grad_biases):
		if self.rprop:
			# rprop variant
			self.weights_steps = np.where(np.sign(grad_weights) == np.sign(last_grad_weights), np.where(np.abs(self.weights_steps) < self.eta_max, self.weights_steps * self.alpha, self.weights_steps), \
				np.where(np.abs(self.weights_steps) > self.eta_min, self.weights_steps * self.beta, self.weights_steps))
			self.biases_steps = np.where(np.sign(grad_biases) == np.sign(last_grad_biases), np.where(np.abs(self.biases_steps) < self.eta_max, self.biases_steps * self.alpha, self.biases_steps), \
				np.where(np.abs(self.biases_steps) > self.eta_min, self.biases_steps * self.beta, self.biases_steps))
			weights_updates, biases_updates = -self.weights_steps * np.sign(grad_weights), -self.biases_steps * np.sign(grad_biases)
		elif self.ada_grad is None:
			weights_updates, biases_updates = -self.step * grad_weights, -self.step * grad_biases
		else:
			weights_updates, biases_updates = self.ada_grad(grad_weights, grad_biases, self.step)
		
		return weights_updates + self.momentum * last_weights_update, biases_updates + self.momentum * last_biases_update

class AdaGrad():
	def __init__(self, n_inputs, n_units):
		
		self.epsilon = 1e-07
		# create a matrix of zeros with columns as the number of inputs and rows as the number of units
		self.Grad_weights = np.zeros((n_inputs, n_units))
		self.Grad_biases = np.zeros(( 1, n_units))
		self.waiting = 0

	def __call__(self, grad_weights, grad_biases, step):
		self.waiting = self.waiting + 1

		if self.waiting < 0:
			return -step * grad_weights, -step * grad_biases
		else: 

			self.Grad_weights = np.square(grad_weights) + self.Grad_weights
			self.Grad_biases = np.square(grad_biases)	+ self.Grad_biases
		
			return np.multiply(np.divide(- step , np.sqrt(self.Grad_weights + self.epsilon)) , grad_weights) , -step * grad_biases
