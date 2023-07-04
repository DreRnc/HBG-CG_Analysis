import numpy as np
from src.RegularizationFunctions import get_regularization_instance, RegularizationFunction
from src.MetricFunctions import get_metric_instance, MetricFunction
from src.EarlyStopping import EarlyStopping

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

    def __init__(self, loss, early_stopping = None, regularization_function = 'L2', stopping_criterion = 'max_epochs'):

        """
        Construct an Optimizer object.

        Parameters
        ----------
        model (MLP) : model to optimize
        loss (str) : loss function to optimize 
        regularization_function  (str) : regularization function to optimize
        early_stopping (EarlyStopping) : early stopping criterion
        stopping_criterion (str) : stopping condition for the optimization (e.g. max number of iterations
        """
        if type(loss) == str:       
            self.loss = get_metric_instance(loss)
        elif isinstance(loss, MetricFunction):
            self.loss = loss
        else:
            raise ValueError("Loss must be a string or a MetricFunction object")
        
        if type(regularization_function) == str:
            self.regularization_function = get_regularization_instance(regularization_function)
        elif isinstance(regularization_function, RegularizationFunction):
            self.regularization_function = regularization_function
        else:
            raise ValueError("Regularization function must be a string or a RegularizationFunction object")

        if early_stopping is not None:
            if not isinstance(early_stopping, EarlyStopping):
                raise ValueError("Early stopping must be an EarlyStopping object")
            else:
                self.early_stopping = early_stopping
        else:
            self.early_stopping = None
        
        if stopping_criterion not in ["max_epochs", "obj_tol", "grad_norm", "n_evaluations"]:
            raise ValueError("Stopping criterion must be one of 'max_epochs', 'obj_tol', 'grad_norm', 'n_evaluations'")
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
                    raise ValueError("Stopping value for max_epochs must be an integer")
                self.max_epochs = stopping_value
            case "obj_tol", "grad_norm":
                if self.early_stopping is None:
                    raise ValueError("EarlyStopping object must be provided at initialization for stopping criterion 'obj_tol' or 'grad_norm'")
            case "n_evaluations":
                if type(stopping_value) != int:
                    raise ValueError("Stopping value for n_evaluations must be an integer")
                self.n_evaluations = stopping_value
            case _:
                raise ValueError("Stopping criterion must be one of 'max_epochs', 'obj_tol', 'grad_norm'")

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

        grad_output = self.loss.derivative(y, y_pred)
        grad_params = self.model.backward(grad_output, self.regularization_function)

        grad_norm = 0
        for grad_layer in grad_params:
            for grad in grad_layer.values():
                grad_norm += np.sum(grad**2)
        grad_norm = np.sqrt(grad_norm)

        self.n_forward_backward += 1

        return J, grad_params, grad_norm
    
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
        self.n_epochs = 0
        self.n_forward_backward = 0
        self.obj_history = []
        self.grad_norm_history = []
        self.last_update = []
        self.early_stopping.initialize()

        while not self.verify_stopping_conditions():
            for X_batch, y_batch in self.get_batches(X, y):
                self._step(X_batch, y_batch)
            if self.verbose: 
                print(f"Epoch {self.n_epochs} - Objective function: {self.obj_history[-1]} - Gradient norm: {self.grad_norm_history[-1]}")
            self.n_epochs += 1
        
        if self.verbose: 
            y_pred = self.model(X)
            _ = self._objective_function(y, y_pred)
            print(f"Stopping condition reached after {self.n_epochs} epochs")
            print(f"Objective function: {self.obj_history[-1]} - Gradient norm: {self.grad_norm_history[-1]}")    


    def verify_stopping_conditions(self):
        """
        Verify if the stopping conditions are met.
        
        Returns
        -------
        bool : True if the stopping conditions are met, False otherwise

        """
        match self.stopping_criterion:
            case "max_epochs":
                return self.max_epochs == self.n_epochs
            case "obj_tolerance":
                return self.early_stopping(self.obj_history[-1])
            case "grad_norm":
                return self.early_stopping(self.grad_norm_history[-1])
            case "n_evaluations":
                return self.n_forward_backward >= self.max_evaluations
            
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

    def _step(self, X, y):

        """
        Perform one step of the gradient descent algorithm.
        
        Parameters
        ----------
        X (np.array) : input data
        y (np.array) : ground truth values
        
        """
        J, grad_params, grad_norm = self._forward_backward(X, y)
        self.obj_history.append(J)
        self.grad_norm_history.append(grad_norm)
        
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
    self.model: model to optimize
    self.loss: loss function
    self.regularization_function: regularization function
    self.batch_size: batch size
    self.beta_type: type of beta update
    self.m1: parameter for FR beta update
    self.m2: parameter for PR beta update
    self.MaxFeval: maximum number of function evaluations
    self.tau: parameter for AWLS
    self.delta: parameter for AWLS
    self.eps: parameter for AWLS
    self.sfgrd: parameter for AWLS
    self.alpha_l1: regularization parameter for l1 regularization
    self.alpha_l2: regularization parameter for l2 regularization
    self.verbose: print information about the optimization process

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

    def initialize(self, model, stopping_value = 1000, batch_size = -1, alpha_l1 = 0, alpha_l2 = 0.001, verbose = False,
                   beta_type = "FR", m1 = 0.25, m2 = 0.4, MaxFeval = 20, tau = 0.9, delta = 1e-4, eps = 1e-6, sfgrd = 0.2):

        """
        Initialize the optimizer.
        
        Parameters
        ----------
        stopping_value (int or float) : stopping criterion value
        batch_size (int) : batch size
        alpha_l1 (float) : regularization parameter for l1 regularization
        alpha_l2 (float) : regularization parameter for l2 regularization
        verbose (bool) : print information about the optimization process
        beta_type (str) : type of beta update
        m1 (float) : parameter for FR beta update
        m2 (float) : parameter for PR beta update
        MaxFeval (int) : maximum number of function evaluations
        tau (float) : parameter for AWLS
        delta (float) : parameter for AWLS
        eps (float) : parameter for AWLS
        sfgrd (float) : parameter for AWLS
        
        """
        super().initialize(model, stopping_value, batch_size, alpha_l1, alpha_l2, verbose)
        
        if beta_type in ['FR', 'HS+', 'PR+']:
            self.beta_type = beta_type
        else:
            raise ValueError('Insert valid beta type (FR, HS+, PR+)')

        self.m1 = m1
        self.m2 = m2
        self.MaxFeval = MaxFeval
        self.tau = tau
        self.delta = delta
        self.eps = eps
        self.sfgrd = sfgrd

        self.last_grad_params = []
        self.last_d = []

        params = self.model.get_params()
        for l in range(len(self.model.layers)):
            self.last_grad_params.append({"weights" : np.zeros(np.shape(params[l]["weights"])),\
                                          "biases" : np.zeros(np.shape(params[l]["biases"]))})
            self.last_d.append({"weights" : np.zeros(np.shape(params[l]["weights"])),\
                                "biases" : np.zeros(np.shape(params[l]["biases"]))})
            
    def _flatten(self,d):
        
        """
        Flatten the gradient d.

        Parameters
        ----------
        d (list) : direction of descent

        Returns
        -------
        d_flat (np.array) : flattened direction of descent

        """
        d_flat = d[0]["weights"].flatten()
        d_flat = np.concatenate((d_flat,d[0]["biases"].flatten()))
        for l in range(1, len(d)):
            d_flat = np.concatenate((d_flat, d[l]["weights"].flatten()))
            d_flat= np.concatenate((d_flat, d[l]["biases"].flatten()))
        return d_flat

    def _update_params(self, alpha, d):

        """
        Update the parameters of the model.
        
        Parameters
        ----------
        alpha (float) : step size
        d (list) : direction of descent
        
        """
        new_params = self._current_params
        for l in range(len(new_params)):
            new_params[l]["weights"] = new_params[l]["weights"] + alpha*d[l]["weights"]
            new_params[l]["biases"] = new_params[l]["biases"] + alpha*d[l]["biases"]
        self.model.set_params(new_params)

    def _phi(self, alpha, d, X, y):
        
        """
        Compute the objective function and its derivative.
        
        Parameters
        ----------
        alpha (float) : step size
        d (list) : direction of descent
        X (np.array) : input data
        y (np.array) : output data

        Returns
        -------
        phi (float) : objective function
        phip (float) : derivative of the objective function

        """

        # compute tomography and its derivative
        self._update_params(alpha, d)
        phi, phip = self._forward_backward(X,y)
        phip = np.matmul(self._flatten(phip), self._flatten(d))

        # reset model to current params
        self.model.set_params(self._current_params)

        return phi, phip

    def _AWLS(self, d, X, y):

        """
        Armijo-Wolfe line search.
        
        Parameters
        ----------
        d (list) : direction of descent
        X (np.array) : input data
        y (np.array) : output data
        
        Returns
        -------
        alpha (float) : step size
        
        """

        feval = 1
        alpha_s = 0.01

        [phi0 , phip0] = self._phi(0, d, X, y)

        while feval <= self.MaxFeval:
            
            [ phi_as , phip_as ] = self._phi(alpha_s, d, X, y)
            feval = feval + 1
            if ( phi_as <= phi0 + self.m1 * alpha_s * phip0) and ( np.abs( phip_as ) <= - self.m2 * phip0 ):
                alpha = alpha_s
                return alpha # Armijo + strong Wolfe satisfied, we are done
            
            if phip_as >= 0:  # derivative is positive
                break
            
            alpha_s = alpha_s / self.tau
            
        alpha_m = 0;
        alpha = alpha_s;
        phip_am = phip0;
        
        while ( feval <= self.MaxFeval ) and ( ( alpha_s - alpha_m ) ) > self.delta and ( phip_as > self.eps ):
            # compute the new value by safeguarded quadratic interpolation
            
            alpha = ( alpha_m * phip_as - alpha_s * phip_am ) / ( phip_as - phip_am );
            alpha = np.max( np.array([alpha_m + ( alpha_s - alpha_m ) * self.sfgrd, \
                                      np.min( np.array([alpha_s - ( alpha_s - alpha_m ) * self.sfgrd, alpha]) ) ]) )
            # compute tomography
            [ phi_a , phip_a ] = self._phi(alpha, d, X, y)
            feval = feval + 1

            if ( phi_a <= phi0 + self.m1 * alpha * phip0 ) and ( np.abs( phip_a ) <= - self.m2 * phip0 ):
                break #Armijo + strong Wolfe satisfied, we are done
            
            # restrict the interval based on sign of the derivative in a
            if phip_a < 0:
                alpha_m = alpha
                phip_am = phip_a
            else:
                alpha_s = alpha
                phip_as = phip_a
                
        return alpha
        

    def _step(self, X, y):

        """
        Compute the step size and update the parameters of the model.
        
        Parameters
        ----------
        X (np.array) : input data
        y (np.array) : output data
        
        """
        
        self._current_params = self.model.get_params()
        
        J, grad_params, grad_norm = self._forward_backward(X,y)
        self.obj_history.append(J)
        self.grad_norm_history.append(grad_norm)

        grad_params_flat = self._flatten(grad_params)
        last_grad_params_flat = self._flatten(self.last_grad_params)
        last_d_flat = self._flatten(self.last_d)

        
        if self.beta_type == "FR":
            num = np.linalg.norm(grad_params_flat)**2
            den = np.linalg.norm(last_grad_params_flat)**2
        elif self.beta_type == "HS+":
            num = np.dot(grad_params_flat, (grad_params_flat-last_grad_params_flat))
            den = np.dot(last_d_flat, (grad_params_flat-last_grad_params_flat))
        else: #PR+
            num = np.dot(grad_params_flat, (grad_params_flat-last_grad_params_flat))
            den = np.linalg.norm(last_grad_params_flat)**2
        
        if den != 0:
            beta = max(0, num/den)
        else:
            beta = 0

        d = []
        for l in range(len(grad_params)):
            d.append({"weights" : - grad_params[l]["weights"] + beta * self.last_d[l]["weights"],\
                     "biases" : - grad_params[l]["biases"] + beta * self.last_d[l]["biases"]})
            
        alpha = self._AWLS(d, X, y)
        self._update_params(alpha, d)

        self.last_d = d
        self.last_grad_params = grad_params