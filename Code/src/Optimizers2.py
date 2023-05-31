import numpy as np
from src.RegularizationFunctions import get_regularization_instance
from src.MetricFunctions import get_metric_instance

class Optimizer:

    """
    Attributes : 
    self.model
    self.loss
    self.regularization
    self.batch_size
    self.stopping_conditions


    Methods: 

    self.__init__
    self.initialize
    self._objective_function
    self._forward_backward
    self._update_params
    self._step
    self.fit_model

    """

    def __init__(self, model, loss, regularization):

        """
        Construct an Optimizer object.

        Parameters
        ----------
        model (Model) : model to optimize
        loss (Loss) : loss function to optimize
        regularization (Regularization) : regularization to optimize

        """

        self.model = model
        self.loss = get_metric_instance(loss)
        self.regularization = get_regularization_instance(regularization)

    def initialize(self, batch_size =- 1, alpha_l1 = 0, alpha_l2 = 0):

        """
        Initialize the optimizer with all the parameters needed for the optimization.
        This is where you can initialize the momentum, the learning rate, etc. for grid search.

        Parameters
        ----------
        batch_size (int) : size of the batch for the gradient descent. If -1, the whole dataset is used.
        alpha_l1 (float) : regularization parameter for l1 regularization
        alpha_l2 (float) : regularization parameter for l2 regularization

        """
        self.batch_size = batch_size
        self.regularization.set_coefficients(alpha_l1, alpha_l2)


    def _objective_function(self, y_pred, y):

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

        J = self.loss(y_pred,y)
        for layer in self.model.layers:
            J = J + self.regularization(layer.weights)

        return J

    def _forward_backward(self, X, y):

        y_pred = self.model(X)
        J = self.objective_function(y_pred, y)
        grad_output = self.loss.derivative(y_pred, y)
        grad_params = self.model.backward(grad_output)
        
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
        while not self.stopping_conditions():
            for X_batch, y_batch in self.get_batches(X, y):
                self._step(X_batch, y_batch)


class HBG(Optimizer):

    """
    Attributes
    ----------
    self.model
    self.loss
    self.regularization
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

    def __init__(self, model, loss, regularization):
        super.__init__(model, loss, regularization)

    def initialize(self, alpha, beta):

        """
        """
        self.alpha = alpha
        self.beta = beta
        self.last_update = []
        for layer in self.model.layers:
            self.last_update.append{"weights": np.zeros(layer.weights.shape), "biases": np.zeros(layer.biases.shape)}

    def _step(self, X_batch, y_batch):

        """
        Perform one step of the gradient descent algorithm.
        
        Parameters
        ----------
        X_batch (np.array) : batch of input data
        y_batch (np.array) : batch of ground truth values
        
        """
        _ , grad_params = self._forward_backward(X_batch, y_batch)
        for i, layer in enumerate(self.model.layers):
            self.last_update[i]["weights"] = self.beta * self.last_update[i]["weights"] - self.alpha * grad_params[i]["weights"]
            self.last_update[i]["biases"] = self.beta * self.last_update[i]["biases"] - self.alpha * grad_params[i]["biases"]

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

    def __init__(self):
        super.__init__()

    def initialize(self, beta, m1, m2, MaxFeval, tau, delta, eps, sfgrd):

        self.beta = beta
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

        if self.beta == "FR":
            beta = np.norm(grad_params_flat)**2/np.norm(self.last_grad_params_flat)**2
    
        d = - grad_params + beta * self.last_d

        alpha = self._AWLS(d, J, grad_params)

        self._update_params(d, alpha)
            

