import numpy as np
from src.ActivationFunctions import get_activation_instance
from src.RegularizationFunctions import get_regularization_instance
from src.Optimizers import HeavyBallGradient

class Layer:
    """
    A Layer is a collection of neurons.

    Override Methods
    ----------------
    forwardprop
    backprop
    """

    def __init__(self):
        pass

    def forwardprop(self):
        raise NotImplementedError

    def backprop(self):
        raise NotImplementedError
    


class FullyConnectedLayer(Layer):

    """
    A Fully Connected layer is a collection of neurons that are fully connected to the previous layer.
    In the following calculations, W_ij is the weight for input i of unit j.

    Attributes
    ----------
    self.n_units (int) : number of units of the layer
    self.n_inputs_per_unit (int) : number of inputs per unit, i.e. number of units of previous layer
    self._weights (np.array) : dimensions (n_inputs_per_unit x n_units)
    self._biases (np.array) : dimension (1, self.n_units)
    self._last_weights_update (np.array) : update of weights on last optimization step
    self._last_biases_update (np.array) : update of biases on last optimization step
    self.optimizer (HeavyBallGradient): optimizer for weights and biases updates
    self.rprop (Bool) : whether to apply rprop variant or standard backprop
    self.regularization_function (RegularizationFunction) : regularization function instance for layer
    self._input (np.array) : inputs saved at each step, to use for backprop
    
    Methods
    -------
    __init__ : initialize only properties of the layer that are intrinsic to the structure of the MLP
    initialize : initialize properties of the fully connected layer which are specific for each fit
    get_params : gets the parameters from the layer
    set_params : set the parameters of the layer
    forwardprop : performs linear transformation on input
    backprop : performs backpropagation, updating weights and biases, and passing gradient for previous layer

    """
    
    def __init__(self, n_units, n_inputs_per_unit):
        
        """
        Initialize only properties of the layer that are intrinsic to the structure of the MLP.
        
        Parameters
        ----------
        n_units (int): number of units of the layer
        n_inputs_per_unit (int): number of inputs per unit, i.e. number of units of previous layer

        """

        self.n_units = n_units
        self.n_inputs_per_unit = n_inputs_per_unit

    def initialize(self, weights_initialization, weights_scale, weights_mean, regularization, alpha_l1, alpha_l2, step, momentum, Nesterov, rprop, adaptive_gradient):

        """
        Initializes properties of the fully connected layer which are specific for each fit;
        function is infact called whenever starting a new fit.
        
        Parameters
        ----------
        weights_initialization (str): type of initialization for weights
        weights_scale (float): std of the normal distribution for initialization of weights
        weights_mean (float): mean of the normal distribution for initialization of weights
        regularization (RegularizationFunction): regularization function for the layer
        alpha_l1 (Float) : parameter for L1 component of regularization
        alpha_l2 (Float) : parameter for L2 component of regularization
        step (Float) : learning step
        momentum (Float) : coefficient for momentum, multiplying last step updates for wieghts and biases
		Nesterov (Bool) : whether optimizer must use Nesterov momentum or not
        rprop (Bool) : whether to apply rprop variant or standard backprop
        adaptive_gradient (bool) : whether to apply adaptive gradient or not

        """
        
        # Weight initialization
        if weights_initialization == "scaled":
            scale = weights_scale
        elif weights_initialization == "xavier":
            scale = 1 / self.n_inputs_per_unit
        elif weights_initialization == "he":
            scale = 2 / self.n_inputs_per_unit
        else:
            print("invalid weigths initialization: choose one between 'scaled', 'xavier', 'he' ")

        self._weights = np.random.normal(loc = weights_mean, scale = scale, size = (self.n_inputs_per_unit, self.n_units))
        
        self._biases = np.zeros((1, self.n_units))

        # Set all updates and gradients to zero at initialization
        self._last_weights_update = 0
        self._last_biases_update = 0

        # for Rprop
        self._last_grad_weights = 0
        self._last_grad_biases = 0

        # Optimizer initialization
        self.optimizer = HeavyBallGradient(step, momentum, Nesterov, rprop, adaptive_gradient, self.n_inputs_per_unit, self.n_units)

        # Rprop: True or False
        self.rprop = rprop

        # Regularization function
        self.regularization_function = get_regularization_instance(regularization, alpha_l1, alpha_l2)


    def get_params(self):

        """
        Gets the parameters from the layer.
        Function used for early stopping.

        Returns
        -------
        Dictionary of parameters from the layer. 
            "weights" (np.array) : dimensions (n_inputs_per_unit x n_units)
            "bias" (np.array) : dimension (1, self.n_units)

        """

        return {"weights": self._weights.copy(), "biases": self._biases.copy()}

    def set_params(self, params):      
          
        """
        Sets the parameters of the layer.
        Function used for setting best parameters when using early stopping.

        Parameters
        ----------
        params (dict): the values to set for the parameters of the layer.
            "weights" (np.array) : dimensions (n_inputs_per_unit x n_units)
            "bias" (np.array) : dimension (1, self.n_units)

        """

        self._weights = params["weights"]
        self._biases = params["biases"]

    def forwardprop(self, input):
        
        """
        Perform linear transformation to input.

        Parameters
        ----------
        input (np.array) : inputs of whole batch (batch_size x n_inputs_per_unit)

        Returns
        -------
        (np.array) : outputs of whole batch (batch_size x n_units)

        """

        if np.shape(self._biases)[1] != self.n_units:
            raise Exception("Dimension Error!")
        
        # Saves values for backprop
        self._input = input        

        return np.matmul(input, self._weights) + self._biases



    def backprop(self, grad_output):

        """
        Performs backpropagation, updating weights and biases, and passing gradient for next step.
        Starts by calculating various gradients with respect to input, weights and biases.
        Then calls optimizer to update weights and biases.
        Finally, returns gradient with respect to input.

        Parameters
        ----------
        grad_output (np.array) : gradient of loss function with respect to output of this layer

        Returns
        -------
        grad_input (np.array) : gradient of loss function with respect to input of this layer (i.e. output of previous layer)

        """
        
        weights = self._weights
        biases = self._biases

        if self.optimizer.Nesterov:
            weights = weights + self.optimizer.momentum * self._last_weights_update
            biases = biases + self.optimizer.momentum * self._last_biases_update

        grad_input = np.matmul(grad_output, weights.T)
        grad_weights = np.matmul(self._input.T, grad_output) + self.regularization_function.derivative(weights)
        grad_biases = grad_output.sum(axis = 0, keepdims = True)

        weights_update, biases_update = self.optimizer(grad_weights, grad_biases, \
            self._last_weights_update, self._last_biases_update, self._last_grad_weights, self._last_grad_biases)

        self._biases += biases_update
        self._weights += weights_update

        # Saves values for next step
        self._last_grad_weights = grad_weights
        self._last_grad_biases = grad_biases
        self._last_weights_update = weights_update
        self._last_biases_update = biases_update

        return grad_input



class ActivationLayer(Layer):

    """
    An Activation Layer applies the activation function element-wise to the input.

    Attributes
    ----------
    self.activation (ActivationFunction) : activation function instance for layer
    self._input (np.array) : inputs saved at each step, to use for backprop
    
    Methods
    -------
    __init__ : initialize activation layer with its activation function
    forwardprop : performs linear transformation on input
    backprop : performs backpropagation, updating weights and biases, and passing gradient for previous layer

    """

    def __init__(self, activation = "ReLU"):

        """
        Initialize activation layer with its activation function and number of units.
        
        Parameters
        ----------
        activation (str) : Name/alias of the activation function

        """

        self.activation = get_activation_instance(activation)
    
    def forwardprop(self, input):
        
        """
        Applies activation function to input element-wise.

        Parameters
        ----------
        input (np.array) : inputs of whole batch (batch_size x n_inputs_per_unit)

        Returns
        -------
        (np.array) : outputs of whole batch (batch_size x n_units)

        """

        # Saves values for backprop
        self._input = input

        # print(self.activation(input))

        return self.activation(input)

    def backprop(self, grad_output):
        
        """
        Performs backpropagation, computing derivative with respect to inputs.

        Parameters
        ----------
        grad_output (np.array) : gradient of loss function with respect to output of this layer

        Returns
        -------
        grad_input (np.array) : gradient of loss function with respect to input of this layer (i.e. output of previous layer)

        """

        return grad_output * self.activation.derivative(self._input)



class Dense(Layer):

    """
    A Dense layer is a fully connected layer with an activation layer afterwards. 

    Attributes
    ----------
    self._fully_connected_layer (FullyConnectedLayer) : linear combination of the dense layer
    self._activation_layer (ActivationLayer) : activation function of the dense layer
    
    Methods
    -------
    __init__ : initialize only properties of the layer that are intrinsic to the structure of the MLP
    initialize : initialize properties of the fully connected layer which are specific for each fit
    get_params : gets the parameters from the fully connected part of the layer
    set_params : set the parameters of the fully connected part of the layer
    forwardprop : performs linear transformation and activation on input
    backprop : performs backpropagation, updating weights and biases, and passing gradient to previous layer
    """

    def __init__(self, n_units, n_inputs_per_unit, activation):

        """
        Initialize only properties of the layer that are intrinsic to the structure of the MLP.
        
        Parameters
        ----------
        n_units (int): number of units in the layer
        n_inputs_per_unit (int): number of inputs per unit (units in layer before)
        activation (str) : Name/alias of the activation function

        """

        self._fully_connected_layer = FullyConnectedLayer(n_units, n_inputs_per_unit)
        self._activation_layer = ActivationLayer(activation)

    def initialize(self, weights_initialization, weights_scale, weights_mean, regularization, alpha_l1, alpha_l2, step, momentum, Nesterov, rprop, adaptive_gradient):

        """
        Initialize properties of the FCL which are specific for each fit.
        Function is infact called whenever starting a new fit.
        
        Parameters
        ----------
        weights_initialization (str): type of initialization for weights
        weights_scale (float): std of the normal distribution for initialization of weights
        weights_mean (float): mean of the normal distribution for initialization of weights
        regularization (RegularizationFunction): regularization function for the layer
        alpha_l1 (Float) : parameter for L1 component of regularization
        alpha_l2 (Float) : parameter for L2 component of regularization
        step (Float) : learning step
        momentum (Float) : coefficient for momentum, multiplying last step updates for wieghts and biases
		Nesterov (Bool) : whether optimizer must use Nesterov momentum or not
        rprop (Bool) : whether to apply rprop variant or standard backprop
        adaptive_gradient (bool) : whether to apply adaptive gradient or not

        """

        self._fully_connected_layer.initialize(weights_initialization, weights_scale, weights_mean, regularization, alpha_l1, alpha_l2, step, momentum, Nesterov, rprop, adaptive_gradient)

    def get_params(self):

        """
        Gets the parameters from the FC layer.

        (See documentation in fully connected layer class)

        """

        return self._fully_connected_layer.get_params()

    def set_params(self, params):

        """
        Sets the parameters for the FC layer.
        
        (See documentation in fully connected layer class)

        """

        self._fully_connected_layer.set_params(params)

    def forwardprop(self, input):
        
        """
        Computes forward propagation, first on FCL, then on AL.

        (See documentation in FCL / AL  classes)

        """

        output_FCL = self._fully_connected_layer.forwardprop(input)
        return self._activation_layer.forwardprop(output_FCL)


    def backprop (self, grad_output):

        """
        Performs backpropagation, first on AL and then on FCL.

        First calculates gradient with respect to output of FCL.
        Then updates weights and biases.
        Finally calculates gradient with respect to input and returns it.

        (See documentation in AL / FCL classes)

        """

        grad_output_FCL = self._activation_layer.backprop(grad_output)

        return self._fully_connected_layer.backprop(grad_output_FCL)

