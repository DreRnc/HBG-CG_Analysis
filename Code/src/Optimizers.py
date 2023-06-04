import numpy as np
from src.RegularizationFunctions import get_regularization_instance
from src.MetricFunctions import get_metric_instance

class Optimizer:

    """
    Attributes: 
    self.model
    self.loss
    self.regularization
    self.batch_size
    self.stopping_conditions
    self.obj_history



    Methods: 

    self.__init__
    self.initialize
    self._objective_function
    self._forward_backward
    self._update_params
    self._step
    self.fit_model
    self.verify_stop_conditions


    """

    def __init__(self, loss, regularization_function, stopping_criterion):

        """
        Construct an Optimizer object.

        Parameters
        ----------
        loss (str) : loss function to optimize 
        regularization (str) : regularization function to optimize
        stopping_criterion (str) : stopping condition for the optimization (e.g. max number of iterations
        """

        self.loss = get_metric_instance(loss)
        self.regularization_function = get_regularization_instance(regularization_function)

        if stopping_criterion not in ["max_epochs", "obj_tol", "grad_tol"]:
            raise ValueError("Stopping criterion must be one of 'max_epochs', 'obj_tol', 'grad_tol'")
        else:
            self.stopping_criterion = stopping_criterion

    def initialize(self, model, stopping_value, batch_size =- 1, alpha_l1 = 0, alpha_l2 = 0, verbose = False):

        """
        Initialize the optimizer with all the parameters needed for the optimization.
        This is where you can initialize the momentum, the learning rate, etc. for grid search.

        Parameters
        ----------
        batch_size (int) : size of the batch for the gradient descent. If -1, the whole dataset is used.
        alpha_l1 (float) : regularization parameter for l1 regularization
        alpha_l2 (float) : regularization parameter for l2 regularization

        """
        self.model = model
        self.verbose = verbose
        self.batch_size = batch_size
        self.regularization_function.set_coefficients(alpha_l1, alpha_l2)

        match self.stopping_criterion:
            case "max_epochs":
                if type(stopping_value) != int:
                    raise ValueError("Stopping criterion must be an integer")
                self.max_epochs = stopping_value
            case "obj_tol":
                if type(stopping_value) != float:
                    raise ValueError("Stopping criterion must be a float")
                self.obj_tol = stopping_value
            case "grad_tol":
                if type(stopping_value) != float:
                    raise ValueError("Stopping criterion must be a float")
                self.grad_tol = stopping_value
            case _:
                raise ValueError("Stopping criterion must be one of 'max_epochs', 'obj_tol', 'grad_tol'")

    def _objective_function(self, y, y_pred):

        """
        Compute the objective function of the optimization problem.
        
        Parameters
        ----------
        y_pred (np.array) : predicted values
        y (np.array) : ground truth values
        
        Returns
        -------
        J (float) : objective function of the optimization problem
        
        """
        J = self.loss(y, y_pred)
        self.loss_history.append(J)

        params = self.model.get_params()

        for layer_params in params:
            J += self.regularization_function(layer_params["weights"])

        return J

    def _forward_backward(self, X, y):

        """
        Compute the objective function and the gradients of the objective function with respect to the parameters.
        Gradient norm is computed only if needed as stopping condition, as it is a costly operation.
        
        Parameters
        ----------
        X (np.array) : input data
        y (np.array) : ground truth values
        
        Returns
        -------
        J (float) : objective function of the optimization problem
        grad_params (list) : list of dictionaries containing the gradients of the objective function with respect to the parameters
        
        """
        y_pred = self.model(X)
        J = self._objective_function(y, y_pred)
        self.obj_history.append(J)

        grad_output = self.loss.derivative(y, y_pred)
        grad_params = self.model.backward(grad_output, self.regularization_function)

        if self.stopping_criterion == "grad_tol" or True: #### DONE FOR TESTING PURPOSES
            grad_norm = 0
            for grad_layer in grad_params:
                for grad in grad_layer.values():
                    grad_norm += np.sum(grad**2)
            self.grad_norm = np.sqrt(grad_norm)

        return J, grad_params
    
    def _update_params(self):
        raise NotImplementedError

    def _step(self):
        raise NotImplementedError
    
    def get_batches(self, X, y):

        """
        Generator that returns batches of data.
        
        Parameters
        ----------
        X (np.array) : input data
        y (np.array) : ground truth values
        
        Returns
        -------
        X_batch (np.array) : batch of input data
        y_batch (np.array) : batch of ground truth values
        
        """
        if self.batch_size == -1:
            yield X, y
        else:
            for i in range(0, X.shape[0], self.batch_size):
                yield X[i:i+self.batch_size], y[i:i+self.batch_size]

    def fit_model(self, X, y):

        """
        Fit the model to the data.

        Parameters
        ----------
        X (np.array) : input data
        y (np.array) : ground truth values

        """
        
        # Initialize the values for stopping conditions at the beginning of the optimization
        self.n_epochs = 0
        self.obj_history = []
        self.loss_history = []
        self.grad_norm = np.inf
        self.last_update = []

        while not self.verify_stopping_conditions():
            for X_batch, y_batch in self.get_batches(X, y):
                self._step(X_batch, y_batch)
            self.n_epochs += 1
            if self.verbose: 
                print(f"Epoch {self.n_epochs} - Objective function: {self.obj_history[-1]} - Loss: {self.loss_history[-1]} - Gradient norm: {self.grad_norm}")

    def verify_stopping_conditions(self):
        match self.stopping_criterion:
            case "max_epochs":
                return self.max_epochs == self.n_epochs
            case "obj_tolerance":
                return self.obj_tol > self.obj_history[-1] - self.obj_history[-2]
            case "grad_norm":
                return self.grad_norm < self.grad_norm_tol


class HBG(Optimizer):

    """
    Attributes
    ----------
    self.model
    self.loss
    self.regularization_function
    self.alpha : learning rate
    self.beta : momentum
    self.batch_size


    Methods: 

    self.__init__
    self.initialize
    self._objective_function
    self._forward_backward
    self._step
    self.fit_model
    self._update_params

    """

    def initialize(self, model, stopping_value, alpha, beta, batch_size =- 1, alpha_l1 = 0, alpha_l2 = 0, verbose = False): 

        """
        Initialize the optimizer.

        Parameters
        ----------
        model (MLP) : model to be fitted
        stopping_value (int or float) : stopping criterion value
        alpha (float) : learning rate
        beta (float) : momentum
        batch_size (int) : batch size
        alpha_l1 (float) : regularization parameter for l1 regularization
        alpha_l2 (float) : regularization parameter for l2 regularization
        

        """
        super().initialize(model, stopping_value, batch_size, alpha_l1, alpha_l2, verbose)

        self.alpha = alpha
        self.beta = beta    

    def _step(self, X_batch, y_batch):

        """
        Perform one step of the gradient descent algorithm.
        
        Parameters
        ----------
        X_batch (np.array) : batch of input data
        y_batch (np.array) : batch of ground truth values
        
        """
        _ , grad_params = self._forward_backward(X_batch, y_batch)
        
        if self.last_update:
            for i in range(len(self.model.layers)):
                self.last_update[i]["weights"] = self.beta * self.last_update[i]["weights"] - self.alpha * grad_params[i]["weights"]
                self.last_update[i]["biases"] = self.beta * self.last_update[i]["biases"] - self.alpha * grad_params[i]["biases"]
        else: 
            for i in range(len(self.model.layers)):
                self.last_update.append({"weights": - self.alpha * grad_params[i]["weights"], "biases": - self.alpha * grad_params[i]["biases"]})
        
        """
        print(self.model.layers)
        for layer in range(len(self.model.layers)):
            print(layer)
            for value in self.last_update[layer].values():
                print(value.shape)
        """
        
        self.model.update_params(self.last_update)


class CG(Optimizer):

    """
    Attributes : 
    self.model
    self.loss
    self.regularization
    self.regularization_function
    self.batch_size

    Methods: 

    self.__init__
    self.initialize
    self._objective_function
    self._forward_backward
    self._phi
    self._AWLS
    self._step
    self.fit_model

    """

    def initialize(self, beta_variant, m1, m2, MaxFeval, tau, delta, eps, sfgrd):
        
        
        self.beta_variant = beta_variant
        self.m1 = m1
        self.m2 = m2
        self.MaxFeval = MaxFeval
        self.tau = tau
        self.delta = delta
        self.eps = eps
        self.sfgrd = sfgrd

        self.last_grad_params = []
        self.last_d = []
        for layer in self.model.layers:
            self.last_grad_params.append(np.zeros(layer.weights.shape()))
            self.last_grad_params.append(np.zeros(layer.biases.shape()))
            self.last_d.append(np.zeros(layer.weights.shape()))
            self.last_d.append(np.zeros(layer.biases.shape()))

    def _phi(self, alpha, d):

        # compute tomography and its derivative
        self._update_params(alpha, d)
        phi, phip = self.forward_backward()

        # reset model to current params
        self.model.set_params(self._current_params)

        return phi, phip

    def _AWLS(self, d, phi0 , phip0):

        feval = 1
        
        while feval <= self.MaxFeval:
            
            [ phi_as , phip_as ] = self._phi(alpha_s, d)
            
            if ( phi_as <= phi0 + self.m1 * alpha_s * phip0 ) and ( np.abs( phip_as ) <= - self.m2 * phip0 ):
                alpha = alpha_s
                return alpha # Armijo + strong Wolfe satisfied, we are done
            
            if phi_as >= 0:  # derivative is positive
                break
            
            alpha_s = alpha_s / self.tau
            
        alpha_m = 0;
        alpha = alpha_s;
        phipm = phip0;
        
        while ( feval <= self.MaxFeval ) and ( ( alpha_s - alpha_m ) ) > self.delta and ( phips > self.eps ):
            # compute the new value by safeguarded quadratic interpolation
             
            alpha = ( alpha_m * phips - alpha_s * phipm ) / ( phips - phipm );
            alpha = np.max( np.array([alpha_m + ( alpha_s - alpha_m ) * self.sfgrd, \
                                      np.min( np.array([alpha_s - ( alpha_s - alpha_m ) * self.sfgrd, alpha]) ) ]) )
            # compute tomography
            [ phi_a , phip_a ] = self._phi(alpha, d)
            
            if ( phi_a <= phi0 + self.m1 * alpha * phip0 ) and ( np.abs( phip_a ) <= - self.m2 * phip0 ):
                break #Armijo + strong Wolfe satisfied, we are done
            
            # restrict the interval based on sign of the derivative in a
            if phip_a < 0:
                alpha_m = alpha
                phipm = phip_a
            else:
                alpha_s = alpha
                phips = phip_a
                
            return alpha

    def _step(self):
        
        self._current_params = self.model.get_params
        
        J, grad_params = self.forward_backward()

        grad_params_flat = grad_params[0].flatten()
        last_grad_params_flat = self.last_grad_params[0].flatten()
        last_d_flat = self.last_d[0].flatten()

        for l in range(1, 2*len(self.model.layers)):

            grad_params_flat.concatenate(grad_params[l].flatten())
            last_grad_params_flat.concatenate(self.last_grad_params[l].flatten())
            last_d_flat.concatenate(self.last_d[l].flatten())

        if self.beta_variant == "FR":
            beta = np.norm(grad_params_flat)**2/np.norm(self.last_grad_params_flat)**2
    
        d = - grad_params + beta * self.last_d

        alpha = self._AWLS(d, J, grad_params)

        self._update_params(d, alpha)