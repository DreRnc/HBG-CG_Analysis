import numpy as np
from src.ActivationFunctions import get_activation_instance


class Layer:
    """
    A Layer is a collection of neurons.

    Override Methods
    ----------------
    forward
    backward
    """

    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class FullyConnectedLayer(Layer):

    """
    A Fully Connected layer is a collection of neurons that are fully connected to the previous layer.
    In the following calculations, W_ij is the weight for input i of unit j.

    Attributes
    ----------
    n_units (int): number of units of the layer
    n_inputs_per_unit (int): number of inputs per unit, i.e. number of units of previous layer
    weights (np.array): weights of the layer, shape (n_units, n_inputs_per_unit)
    biases (np.array): biases of the layer, shape (n_units, 1)
    activation_function (str): name/alias of activation function for all activation layers
    activation (np.array): activation of the layer, shape (n_units, 1)

    Methods
    -------
    __init__ : initializes the layer
    initialize : initializes the layer for a specific fit
    forward : computes the activation of the layer
    backward : computes the gradient of the loss with respect to the weights and biases of the layer
    get_params : returns the weights and biases of the layer
    set_params : sets the weights and biases of the layer
    update_params : updates the weights and biases of the layer

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

    def initialize(self, weights_initialization, weights_scale, weights_mean):
        """
        Initializes the weights and biases of the layer.

        Parameters
        ----------
        weights_initialization (str) : name/alias of the initialization method
        weights_scale (float) : scale of the weights initialization
        weights_mean (float) : mean of the weights initialization

        """
        if weights_initialization == "scaled":
            scale = weights_scale
        elif weights_initialization == "xavier":
            scale = 1 / self.n_inputs_per_unit
        elif weights_initialization == "he":
            scale = 2 / self.n_inputs_per_unit
        else:
            raise ValueError(
                "Invalid weights initialization method: must be 'scaled', 'xavier' or 'he'."
            )

        self.weights = np.random.normal(
            loc=weights_mean, scale=scale, size=(self.n_inputs_per_unit, self.n_units)
        )

        self.biases = np.zeros((1, self.n_units))

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

        return {"weights": self.weights.copy(), "biases": self.biases.copy()}

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

        self.weights = params["weights"]
        self.biases = params["biases"]

    def update_params(self, updates):
        """
        Updates the parameters of the layer.
        Function used for updating parameters after each optimization step.

        Parameters
        ----------
        updates (dict): the values to add to the parameters of the layer.
            "weights" (np.array) : dimensions (n_inputs_per_unit x n_units)
            "bias" (np.array) : dimension (1, self.n_units)

        """

        self.weights += updates["weights"]
        self.biases += updates["biases"]

    def forward(self, input):
        """
        Perform linear transformation to input.

        Parameters
        ----------
        input (np.array) : inputs of whole batch (batch_size x n_inputs_per_unit)

        Returns
        -------
        (np.array) : outputs of whole batch (batch_size x n_units)

        """

        if np.shape(self.biases)[1] != self.n_units:
            raise Exception("Dimension Error!")

        # Saves values for backpropagation
        self._input = input

        return np.matmul(input, self.weights) + self.biases

    def backward(self, grad_output, regularization_function):
        """
        Performs backpropagation, calculating gradientd with respect to weights and biases
        and passing gradient for next step.

        Parameters
        ----------
        grad_output (np.array) : gradient of objective function with respect to output of this layer

        Returns
        -------
        grad_layer (np.array) : ((n_inputs_per_unit + 1) x n_units) matrix of gradients with respect to weights and biases
        grad_input (np.array) : gradient of objective function with respect to input of this layer (i.e. output of previous layer)

        """
        grad_input = np.matmul(grad_output, self.weights.T)
        grad_weights = np.matmul(
            self._input.T, grad_output
        ) + regularization_function.derivative(self.weights)
        grad_biases = grad_output.sum(axis=0, keepdims=True)

        grad_layer = {"weights": grad_weights, "biases": grad_biases}

        return grad_layer, grad_input


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
    forward : performs linear transformation on input
    backward : performs backpropagation, updating weights and biases, and passing gradient for previous layer

    """

    def __init__(self, activation="ReLU"):
        """
        Initialize activation layer with its activation function and number of units.

        Parameters
        ----------
        activation (str) : Name/alias of the activation function

        """

        self.activation = get_activation_instance(activation)

    def forward(self, input):
        """
        Applies activation function to input element-wise.

        Parameters
        ----------
        input (np.array) : inputs of whole batch (batch_size x n_inputs_per_unit)

        Returns
        -------
        (np.array) : outputs of whole batch (batch_size x n_units)

        """

        # Saves values for backpropagation
        self._input = input

        return self.activation(input)

    def backward(self, grad_output):
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
    __init__ : initialize dense layer with its activation function
    forward : performs linear transformation on input
    backprop : performs backpropagation, updating weights and biases, and passing gradient for previous layer
    get_params : returns parameters of the layer
    set_params : sets parameters of the layer
    update_params : updates parameters of the layer

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

    def initialize(self, weights_initialization, weights_scale, weights_mean):
        """
        Initialize properties of the FCL which are specific for each fit.
        Function is infact called whenever starting a new fit.

        Parameters
        ----------
        weights_initialization (str): type of initialization for weights
        weights_scale (float): std of the normal distribution for initialization of weights
        weights_mean (float): mean of the normal distribution for initialization of weights

        """
        self._fully_connected_layer.initialize(
            weights_initialization, weights_scale, weights_mean
        )

    def update_params(self, update):
        """
        Updates the parameters of the FC layer.

        (See documentation in fully connected layer class)

        """
        self._fully_connected_layer.update_params(update)

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

    def forward(self, input):
        """
        Computes forward propagation, first on FCL, then on AL.

        (See documentation in FCL / AL  classes)

        """
        output_FCL = self._fully_connected_layer.forward(input)
        return self._activation_layer.forward(output_FCL)

    def backward(self, grad_output, regularization_function):
        """
        Performs backpropagation, first on AL and then on FCL.

        First calculates gradient with respect to output of FCL.
        Then updates weights and biases.
        Finally calculates gradient with respect to input and returns it.

        (See documentation in AL / FCL classes)

        """
        grad_output_FCL = self._activation_layer.backward(grad_output)
        return self._fully_connected_layer.backward(
            grad_output_FCL, regularization_function
        )
