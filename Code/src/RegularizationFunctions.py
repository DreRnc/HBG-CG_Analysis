import numpy as np

class RegularizationFunction():
    '''
    Base class for regularization functions.

    Methods to override:
        __init__(self, alpha_l1, alpha_l2): Initialises the class with the alpha parameters; not implemented
            Input:
                alpha_l1 (Float) : parameter for L1 component
                alpha_l2 (Float) : parameter for L2 component
        __call__(self,w): Output of function; not implemented
            Input: 
                w (np.array) : weights
            Output: Error
        derivative(self,w): Derivative of function; not implemented
            Input: 
                w (np.array) : weights
            Output: Error
    '''
    def __init__(self, alpha_l1, alpha_l2):
        raise NotImplementedError
    def __call__(self, w):
        raise NotImplementedError
    def derivative(self, w):
        raise NotImplementedError

class ElasticReg(RegularizationFunction):
    '''
    Base class for regularization functions.

    Methods:
        __init__(self, alpha_l1, alpha_l2): Initialises the class with the alpha parameters; not implemented
            Input:
                alpha_l1 (Float) : parameter for L1 component
                alpha_l2 (Float) : parameter for L2 component
        __call__(self,w): Output of function; not implemented
            Input: 
                w (np.array) : weights
            Output: 
                (Float) : value of regularization function
        derivative(self,w): Derivative of function; not implemented
            Input: 
                w (np.array) : weights
            Output: 
                (Float) : derivative of regularization function with respect to weights
    '''

    def __init__(self, alpha_l1, alpha_l2):
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2

    def __call__(self, w):
        return self.alpha_l1 * np.sum(np.abs(w)) + self.alpha_l2 * np.sum(np.square(w))

    def derivative(self, w):
        return self.alpha_l1 * np.sign(w) + 2 * self.alpha_l2 * w

class L1Reg(ElasticReg):
    '''
    Computes the L1 (Lasso) regularization function, which is the sum of the absolute value of the weights in the model.
    '''

    def __init__(self, alpha_l1, alpha_l2):
        super().__init__(alpha_l1, 0)

class L2Reg(ElasticReg):
    '''
    Computes the L2 (Ridge) regularization effect, which is the sum of the squared weights in the model.
    '''

    def __init__(self, alpha_l1, alpha_l2):
        super().__init__(0, alpha_l2)


class NoReg(ElasticReg):
    '''
    Computes no regularization, i.e. call and derivative return zero.
    '''
    def __init__(self, alpha_l1, alpha_l2):
        super().__init__(0, 0)



def get_regularization_instance(reg_type, alpha_l1, alpha_l2):
    '''
    Returns an instance of the regularization function class indicated in the input, if present, else returns ValueError.

    Parameters
    ----------
    reg_type (str) : Name/alias of the regularization function
    alpha_l1 (Float) : value of alpha_l1 parameter
    alpha_l2 (Float) : value of alpha_l2 parameter

    Returns
    -------
    (RegularizationFunction) : Instance of the requested regularization function
    '''
    if reg_type in ['L1', 'Lasso', 'lasso', 'l1']:
        return L1Reg(alpha_l1, alpha_l2)
    elif reg_type in ['L2', 'Ridge', 'ridge', 'l2']:
        return L2Reg(alpha_l1, alpha_l2)
    elif reg_type in ['Elastic', 'ElasticNet', 'elastic', 'elasticnet']:
        return ElasticReg(alpha_l1, alpha_l2)
    elif reg_type in ['None', 'No', 'no', 'none']:
        return NoReg(alpha_l1, alpha_l2)
    else:
        raise ValueError('Regularization function not recognized')
