import numpy as np
from src.Layers import Layer, FullyConnectedLayer, Dense


class MLP():
    """
    Implements a multilayer perceptron
    
    Attributes 
    ----------

    self.layers (list) : layers of the MLP
    self.input_size (int) : size of the input of the network
    self.output_size (int) : size of the output of the network
    self.task (str) : ("classification" or "regression") task the network is performing
    self.hidden_layer_units (list) : list of int indicating number of units for each hidden layer
    self.activation_function (str) : name/alias of activation function for all activation layers
    
    """

    def __init__(self, hidden_layer_units, input_size, output_size, activation_function = 'sigm', task = 'regression', random_seed = 0):

        """
        Builds the architecture of the MLP.

        Parameters
        -----------
        hidden_layer_units (list) : list of int indicating number of units for each hidden layer
        input_size (int) : size of the input of the network
        output_size (int) : size of the output of the network
        task (str) : ("classification" or "regression") task the network is performing
        activation_function (str) : name/alias of activation function for all activation layers
        random_seed (int) : seed for random functions of numpy, for random weight initialization

        """
        self.layers = []
        self.input_size = input_size
        self.output_size = output_size
        self.task = task
        
        self.hidden_layer_units = hidden_layer_units
        self.activation_function = activation_function

        layer_units = [input_size] + hidden_layer_units + [output_size]
        
        n_layers = len(layer_units) - 1 

        np.random.seed(random_seed)

        for l in range(1, n_layers + 1):

            if l < n_layers:
                new_layer = Dense(layer_units[l], layer_units[l-1], activation_function)
            elif self.task == 'classification': 
                new_layer = Dense(layer_units[l], layer_units[l-1], "tanh")
            else:
                new_layer = FullyConnectedLayer(layer_units[l], layer_units[l-1])
                
            self.layers.append(new_layer)

    def __call__(self, X):

        """
        Computes the model's output for the input X
        
        Parameters
        ----------
        X (np.array) : (n_samples x n_inputs) input values for the network
         
        Returns
        ------- 
        Y (np.array) : (n_samples x n_output) model's output for the inputs supplied

        """
        input_size = X.shape[1]

        if input_size != self.input_size:
            raise Exception("Dimension Error: input has not the same size as MLP input.")
        
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, grad_output):

        """
        Performs backward pass, computing gradients with respect to all parameters.
        
        Parameters
        ----------
        grad_output (np.array) : (batch_size x n_outputs) the gradient of objective function J with repsect to outputs

        Returns
        -------
        grad_params (np.array) : (params x ) 

        """
        grad_params = []

        for layer in self.layers.reverse():
            grad_layer, grad_output = layer.backward(grad_output)
            grad_params.append(grad_layer)

        return grad_params
    
    def get_params(self):

        """
        Gets parameters of the model.
        
        Returns
        -------
        params (list of dict) : list of dictionaries that are parameters of each layer
    
        """
        params = []

        for layer in self.layers:
            params.append(layer.get_params())

        return params
    
    def set_params(self, params):

        """
        Sets parameters of the model.
        
        Parameters
        ----------
        params (list of dict) : list of dictionaries that are parameters of each layer
    
        """

        for i, layer in enumerate(self.layers):
            layer.set_params(params[i])

    def update_params(self, updates):
        
        """
        Updates parameters of the model.
        
        Parameters
        ----------
        params (list of dict) : list of dictionaries that are parameters of each layer
        updates (list of dict) : list of dictionaries that are updates for each layer
    
        """

        for i, layer in enumerate(self.layers):
            layer.update_params(updates[i])
