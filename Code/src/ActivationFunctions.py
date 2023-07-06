import numpy as np


class ActivationFunction:
    """
    Base class for activation functions.

    Methods to override:
        __call__(self,x): Computes function; not implemented
            Input: np.array
            Output: Error
        derivative(self,x): Computes derivative of function not implemented
            Input: np.array
            Output: Error
    """

    def __call__(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Identity(ActivationFunction):
    """
    Identity activation function implementation.

    Methods:
        __call__(self,x): Computes function
            Input: np.array
            Output: np.array
        derivative(self,x): Computes derivative of function
            Input: np.array
            Output: np.array
    """

    def __call__(self, x):
        return x

    def derivative(self, x):
        return np.ones(x.shape)


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function implementation.

    Methods:
        __call__(self,x): Computes function
            Input: np.array
            Output: np.array
        derivative(self,x): Computes derivative of function
            Input: np.array
            Output: np.array
    """

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self(x) * (1 - self(x))


class Tanh(ActivationFunction):
    """
    Hyperbolic tangent activation function implementation.

    Methods:
        __call__(self,x): Computes function
            Input: np.array
            Output: np.array
        derivative(self,x): Computes derivative of function
            Input: np.array
            Output: np.array
    """

    def __call__(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.square(self(x))


class ReLU(ActivationFunction):
    """
    Rectified linear unit activation function implementation.

    Methods:
        __call__(self,x): Computes function
            Input: np.array
            Output: np.array
        derivative(self,x): Computes derivative of function
            Input: np.array
            Output: np.array
    """

    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(int)


def get_activation_instance(activation):
    """
    Returns an instance of the activation function class indicated in the input, if present, else, returns ValueError.

    Parameters
    ----------
    activation (str) : Name/alias of the activation function

    Returns
    -------
    (ActivationFunction) : Instance of the requested activation function
    """
    if activation in ["sigmoid", "Sigmoid", "Sigmoid()", "sig", "Sigm", "sigm"]:
        return Sigmoid()
    elif activation in ["tanh", "Tanh", "Tanh()", "tanh", "ta"]:
        return Tanh()
    elif activation in ["identity", "Identity", "Identity()", "id"]:
        return Identity()
    elif activation in ["relu", "ReLU", "ReLU()", "r", "RELU", "Relu", "re", "reLU"]:
        return ReLU()
    else:
        raise ValueError("Activation function not found")
