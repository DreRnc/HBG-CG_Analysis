import numpy as np
import math
from sklearn.model_selection import train_test_split

from src.Layers import Layer, FullyConnectedLayer, Dense
from src.MetricFunctions import get_metric_instance, MetricFunction
from src.EarlyStopping import EarlyStopping

class MLP:

    """
    Implements multilayer perceptron.

    Attributes
    -----------
    self.layers (list) : layers of the MLP
    self.input_size (int) : size of the input of the network
    self.output_size (int) : size of the output of the network
    self.task (str) : ("classification" or "regression") task the network is performing
    self.hidden_layer_units (list) : list of int indicating number of units for each hidden layer
    self.activation_function (str) : name/alias of activation function for all activation layers
    self._eval_metric (MetriFunction) : instance of metric function for evaluating model performance
    self.learning_curve (np.array) : training error at every epoch; for plotting learning curve
    self.validation_curve (np.array) : validation error at every epoch; for plotting validation curve
    self.learning_accuracy_curve (np.array) : training accuracy at every epoch; for plotting learning accuracy curve
    self.test_accuracy_curve (np.array) : test accuracy at every epoch for classification curves
    self.early_stopping (EarlyStopping) : instance of EarlyStopping class

    Methods
    -------
    __init__ : builds the architecture of the MLP
    evaluate_model : evaluates performance of the model on a set, given a certain metric
    fit : fits the MLP on a set, given an error function and the necessary hyperparameters
    predict : predicts the outputs for the given inputs

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
        self.early_stopping = None
        
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

    def evaluate_model(self, X, y_true, metric = 'generic'):

        """
        Evaluates performance of the model on a set, given a certain metric.

        Parameters
        -----------
        X (np.array) : (n_samples x n_inputs) input values for the network
        y_true (np.array) : (n_samples x n_output) ground truth values of target variables for the inputs supplied
        metric (str) : name/alias of metric used for evaluation

        """

        if metric != 'generic':
            eval_metric = metric
            eval_metric = get_metric_instance(eval_metric)
        else:
            if self.task == "classification":
                eval_metric = get_metric_instance('acc')
            else:
                eval_metric = get_metric_instance('mse')

        y_pred = self.predict(X)

        return eval_metric(y_true, y_pred)

    def fit(self, X, y_true, n_epochs, batch_size = -1, X_test = None, y_test = None, error = "MSE", eval_metric = "default", \
        regularization = "elastic", alpha_l1 = 0, alpha_l2 = 0, weights_initialization = "scaled", weights_scale = 0.01, \
        weights_mean = 0, step = 0.1, momentum = 0, Nesterov = False, rprop = False, early_stopping = True, \
        patience = 10, tolerance = 0.01, validation_split_ratio = 0.1, random_seed = 0, verbose = False, adaptive_gradient = False):

        """
        Fits the MLP on a set, given an error function and the necessary hyperparameters.

        Parameters
        -----------
        X (np.array) : (n_samples x n_inputs) input values for the network
        y_true (np.array) : (n_samples x n_output) ground truth values of target variables for the inputs supplied
        n_epochs (int) : maximum number of training epochs
        batch_size (int) : size of minibatch (default: full batch)
        X_test (np.array) : in classification, inputs of test set
                            in regression, None
        y_test (np.array) : in classification, ground truth values of test set
                            in regression, None
        eval_metric (str) : name/alias of metric used for evaluation
        error (str) : name/alias of error function
        regularization (str) : name/alias of regularization
        alpha_l1 (Float) : parameter for L1 component of regularization
        alpha_l2 (Float) : parameter for L2 component of regularization
        weights_initialization (str) : ("scaled", "xavier", "he") method of initialization of weights
        weights_scale (float) : std deviation of normal distribution of weights random initialization
        weights_mean (float) : mean (default = 0) deviation of normal distribution of weights random initialization
        step (float) : learning step
        momentum (float) : coefficient for momentum, multiplying last step updates for wieghts and biases
        Nesterov (bool) : whether optimizer must use Nesterov momentum or not
        rprop (Bool) : whether to apply rprop variant or standard backprop
        early_stopping (bool) : whether to apply early stopping or not
        patience (int) : patience of the early stopping
        tolerance (float) : tolerance of the early stopping
        validation_split_ratio (float) : split ratio of validation and training set
        random_seed (int) : seed for random functions, for random split of set in training/validation
        verbose (bool) : whether to activate verbose mode or not

        """

        n_epochs = int(n_epochs)

        self.learning_curve = np.zeros(n_epochs)
        self.validation_curve = np.zeros(n_epochs)
        self.learning_accuracy_curve = np.zeros(n_epochs)
        self.test_accuracy_curve = np.zeros(n_epochs)

        if eval_metric == "default":
            self._eval_metric = get_metric_instance(error)
        else:
            self._eval_metric = get_metric_instance(eval_metric)

        error_function = get_metric_instance(error)
        
        # # Initializes EarlyStopping
        # if early_stopping:
        #     self.early_stopping = EarlyStopping(patience = patience, tolerance = tolerance, metric = self._eval_metric)
        #     X, X_test, y_true, y_test = train_test_split(X, y_true, test_size = validation_split_ratio, shuffle = True, random_state = random_seed)

        # Checks on sizes of MLP and sets
        n_samples, input_size = X.shape
        try:
            output_size = y_true.shape[1]
        except:
            output_size = 1
        if input_size != self.input_size or output_size != self.output_size:
            raise Exception("Input/Output sizes do not match MLP architecture!")

        # Batches
        if batch_size == -1:
            batch_size = n_samples
        else:
            batch_size = int(batch_size)
        n_batches = math.ceil(n_samples/batch_size)

        # Initialize layers
        for layer in self.layers:
            layer.initialize(weights_initialization, weights_scale, weights_mean, regularization, alpha_l1, alpha_l2, step, momentum, Nesterov, rprop, adaptive_gradient)

        # Training
        for epoch in range(n_epochs):

            X_batches = np.split(X, range(batch_size, X.shape[0], batch_size), axis = 0)
            y_true_batches = np.split(y_true, range(batch_size, y_true.shape[0], batch_size), axis= 0)

            # Learning
            for batch in range(n_batches):

                X_batch = X_batches[batch]
                y_true_batch = y_true_batches[batch]

                y_pred_batch = self.predict(X_batch)
                
                grad_outputs = error_function.derivative(y_true_batch, y_pred_batch)
                
                for layer in reversed(self.layers):

                    grad_inputs = layer.backprop(grad_outputs)
                    grad_outputs = grad_inputs
            
            # Validation/Test Set: saving statistics and learning curve
            if X_test is not None:
                    y_pred_test = self.predict(X_test)
                    self.validation_curve[epoch] = self._eval_metric(y_test, y_pred_test)
                    if self.task == 'classification':
                        self.test_accuracy_curve[epoch] = get_metric_instance('acc')(y_test, y_pred_test)

            # Statistics recording
            y_pred = self.predict(X)
            self.learning_curve[epoch] = (self._eval_metric(y_true, y_pred))
            if self.task == 'classification':
                self.learning_accuracy_curve[epoch] = get_metric_instance('acc')(y_true, y_pred)

            if verbose:
                print("Epoch " + str(epoch) + ": " + "metric" + " = " + str(self._eval_metric(y_true, y_pred)))

            # # Early stopping
            # if early_stopping:

            #     params = [layer.get_params() for layer in self.layers]
            #     stop = self.early_stopping.on_epoch_end(y_test, y_pred_test, params)

            #     if stop:
            #         if verbose:
            #             print(f"Early stopped training on epoch {epoch}")
            #             print(f'Best epoch was {self.early_stopping._best_epoch}')
            #         best_params = self.early_stopping._best_params
            #         for layer, layer_best_params in zip(self.layers, best_params):
            #             layer.set_params(layer_best_params)

            #         # Chop curves
            #         self.learning_curve = self.learning_curve[:epoch]
            #         self.validation_curve = self.validation_curve[:epoch]
            #         self.learning_accuracy_curve = self.learning_accuracy_curve[:epoch] 
            #         self.test_accuracy_curve = self.test_accuracy_curve[:epoch]
            #         break
            #     elif epoch == n_epochs - 1:
            #         best_params = self.early_stopping._best_params
            #         for layer, layer_best_params in zip(self.layers, best_params):
            #             layer.set_params(layer_best_params)



    def predict(self, X):

        """
        Predicts the outputs for the given inputs.

        Parameters
        -----------
        X (np.array) : (n_samples x n_inputs) input values for the network

        Returns
        -------
        y_pred (np.array) : (n_samples x n_output) predicted values of target variables for the inputs supplied

        """

        n_samples, input_size = X.shape

        if input_size != self.input_size:
            raise Exception("Dimension Error!")
        
        tmp = X
        for layer in self.layers:
            layer.input = tmp
            layer.output = layer.forwardprop(layer.input)
            tmp = layer.output

        y_pred = layer.output

        return y_pred
    

class RandomizedMLP(MLP):

    """
    Implements multilayer perceptron which only updates weights on last layer.

    (See documentation of MLP class)

    """

    def fit(self, X, y_true, n_epochs, batch_size = -1, X_test = None, y_test = None, error = "MSE", eval_metric = "default", \
        regularization = "elastic", alpha_l1 = 0, alpha_l2 = 0, weights_initialization = "scaled", weights_scale = 0.01, \
        weights_mean = 0, step = 0.1, momentum = 0, Nesterov = False, rprop = False, early_stopping = True, \
        patience = 10, tolerance = 0.01, validation_split_ratio = 0.1, random_seed = 0, verbose = False, adaptive_gradient = False):

        """
        Fits the weigths and biases of only the last layer of MLP.
        Function is exactly the same as parent class, except for backpropagation on layers.

        (See documentation of MLP class)

        """

        n_epochs = int(n_epochs)

        self.learning_curve = np.zeros(n_epochs)
        self.validation_curve = np.zeros(n_epochs)
        self.learning_accuracy_curve = np.zeros(n_epochs)
        self.test_accuracy_curve = np.zeros(n_epochs)

        if eval_metric == "default":
            self._eval_metric = get_metric_instance(error)
        else:
            self._eval_metric = get_metric_instance(eval_metric)

        error_function = get_metric_instance(error)
        
        # # Initializes EarlyStopping
        # if early_stopping:
        #     self.early_stopping = EarlyStopping(patience = patience, tolerance = tolerance, metric = self._eval_metric)
        #     X, X_test, y_true, y_test = train_test_split(X, y_true, test_size = validation_split_ratio, shuffle = True, random_state = random_seed)

        # Checks on sizes of MLP and sets
        n_samples, input_size = X.shape
        try:
            output_size = y_true.shape[1]
        except:
            output_size = 1
        if input_size != self.input_size or output_size != self.output_size:
            raise Exception("Input/Output sizes do not match MLP architecture!")

        # Batches
        if batch_size == -1:
            batch_size = n_samples
        else:
            batch_size = int(batch_size)
        n_batches = math.ceil(n_samples/batch_size)

        # Initialize layers
        for layer in self.layers:
            layer.initialize(weights_initialization, weights_scale, weights_mean, regularization, alpha_l1, alpha_l2, step, momentum, Nesterov, rprop, adaptive_gradient)

        # Training
        for epoch in range(n_epochs):

            X_batches = np.split(X, range(batch_size, X.shape[0], batch_size), axis = 0)
            y_true_batches = np.split(y_true, range(batch_size, y_true.shape[0], batch_size), axis= 0)

            # Learning
            for batch in range(n_batches):

                X_batch = X_batches[batch]
                y_true_batch = y_true_batches[batch]

                y_pred_batch = self.predict(X_batch)
                
                grad_outputs = error_function.derivative(y_true_batch, y_pred_batch)
                
                # Only do backprop for last layer
                self.layers[-1].backprop(grad_outputs)
            
            # Validation/Test Set: saving statistics and learning curve
            if X_test is not None:
                y_pred_test = self.predict(X_test)
                self.validation_curve[epoch] = self._eval_metric(y_test, y_pred_test)
                if self.task == 'classification':
                    self.test_accuracy_curve[epoch] = get_metric_instance('acc')(y_test, y_pred_test)

            # Statistics recording
            y_pred = self.predict(X)
            self.learning_curve[epoch] = (self._eval_metric(y_true, y_pred))
            if self.task == 'classification':
                self.learning_accuracy_curve[epoch] = get_metric_instance('acc')(y_true, y_pred)

            if verbose:
                print("Epoch " + str(epoch) + ": " + "metric" + " = " + str(self._eval_metric(y_true, y_pred)))

            # # Early stopping
            # if early_stopping:

            #     params = [layer.get_params() for layer in self.layers]
            #     stop = self.early_stopping.on_epoch_end(y_test, y_pred_test, params)

            #     if stop:
            #         if verbose:
            #             print(f"Early stopped training on epoch {epoch}")
            #             print(f'Best epoch was {self.early_stopping._best_epoch}')
            #         best_params = self.early_stopping._best_params
            #         for layer, layer_best_params in zip(self.layers, best_params):
            #             layer.set_params(layer_best_params)

            #         # Chop curves
            #         self.learning_curve = self.learning_curve[:epoch]
            #         self.validation_curve = self.validation_curve[:epoch]
            #         self.learning_accuracy_curve = self.learning_accuracy_curve[:epoch] 
            #         self.test_accuracy_curve = self.test_accuracy_curve[:epoch]
            #         break
            #     elif epoch == n_epochs - 1:
            #         best_params = self.early_stopping._best_params
            #         for layer, layer_best_params in zip(self.layers, best_params):
            #             layer.set_params(layer_best_params)