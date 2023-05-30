import numpy as np

class Optimizer:

    """
    Attributes : 

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

        self.model = model
        self.loss = loss #alias
        self.regularization = regularization #alias

    def initialize():

        pass

    def _objective_function(self, y_pred, y):

        J = self.loss(y_pred,y)
        for layer in self.model.layers:
            J = J + self.regularization(layer.weights)

        return J

    def _forward_backward(self, X, y):

        y_pred = self.model(X)
        J = self.objective_function(y_pred, y)
        grad_output = self.loss.derivative(y_pred, y)
        grad_theta, grad_output = self.model.backward(grad_output)
        
        return J, grad_theta
    
    def _update_params(self):

        pass

    def _step(self):

        pass

    def fit_model(self, X, y):

        while not self.stopping_conditions:
            for batch in batches:
                self._step(batch)

        return self.model

class HBG(Optimizer):

    """
    Attributes : 

    Methods: 

    self.__init__
    self.initialize
    self._objective_function
    self._forward_backward
    self._step
    self.fit_model

    """

    def __init__(self):
        super.__init__()

    def initialize(self):
        pass

    def _step(self):
        pass


class CG(Optimizer):

    """
    Attributes : 

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

        self.last_grad_theta = 0
        self.last_d = 0

    def _phi(self, alpha, d):

        self._update_params(alpha, d)
        phi, phip = self.forward_backward()

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
                if alpha_s <= self.min_alpha:
                    break
                phips = phip_a
                
            return alpha

    def _step(self):
        
        self._current_params = self.model.get_params
        
        J, grad_theta = self.forward_backward()
        
        if self.beta == "FR":
            beta = np.norm(grad_theta)**2/np.norm(self.last_grad_theta)**2
            
        d = - grad_theta + beta * self.last_d

        alpha = self._AWLS(d, J, grad_theta)

        self._update_params(d, alpha)
            

