import numpy as np


class MetricFunction:
    """
    Base class for metric functions.

    Methods to override:
        __call__(self,y_true, y_pred): Returns the metric; not implemented
            Input: np.array
            Output: Error
    """

    def __call__(self, y_true, y_pred):
        raise NotImplementedError


class Accuracy(MetricFunction):
    """
    Computes the accuracy between two np.arrays of any size.

    Methods
    -------
    __call__(self,y_true, y_pred): Computes accuracy of predictions.
        Input: np.array
            y_true (np.array) : ground truth values
            y_pred (np.array) : predicted values
        Output: Accuracy
    """

    def __call__(self, y_true, y_pred):
        """
        Computes accuracy of predictions.
        Label threshold is 0, as tanh is used in output layer in classification tasks.

        Parameters
        ----------
        y_true (np.array) : ground truth values
        y_pred (np.array) : predicted values

        Returns
        -------
        Accuracy (float), i.e. number of correct labels in y_pred divided by total number of labels.
        """

        if y_true.shape != y_pred.shape:
            raise ValueError("Inputs must have the same shape")

        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = -1

        return np.mean(y_true == y_pred)


class ErrorFunction(MetricFunction):
    """
    Base class for error functions.

    Methods to override:
        __call__(self,y_true, y_pred): Computes error; not implemented
            Input: np.array
                y_true (np.array) : ground truth values
                y_pred (np.array) : predicted values
            Output: Error
        derivative(self,y_true, y_pred): Computes derivative of the error; not implemented
            Input: 2 np.array of the same shape
            Output: Error
    """

    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class MSE(ErrorFunction):
    """
    Computes the mean squared error between two np.arrays of any size.

    Methods:
        __call__(self,y_true, y_pred): Returns the mean squared error
            Input: 2 np.arrays of the same shape
                y_true (np.array) : ground truth values
                y_pred (np.array) : predicted values
            Output: Float
        derivative(self,y_true, y_pred): Returns the derivative of the mean squared error
            Input: 2 np.arrays (n_observations, n_outputs) of the same shape
            Output: np.array (n_observations, n_outputs)

    """

    def __call__(self, y_true, y_pred):
        """
        Computes mean squared error between predictions and ground truth values.

        Parameters
        ----------
        y_true (np.array) : ground truth values
        y_pred (np.array) : predicted values

        Returns
        -------
        MSE (float), i.e. mean squared error between predictions and ground truth values.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("Inputs must have the same shape")

        return np.mean(np.square(y_pred - y_true))

    def derivative(self, y_true, y_pred):
        """
        Computes derivative of mean squared error with respect to output of the network.

        Parameters
        ----------
        y_true (np.array) : ground truth values
        y_pred (np.array) : predicted values

        Returns
        -------
        Derivative (np.array), i.e. derivative of mean squared error with respect to output of the network.
        """

        if y_true.shape != y_pred.shape:
            raise ValueError("Inputs must have the same shape")

        return 2 * (y_pred - y_true) / (y_true.shape[0] * y_true.shape[1])


class MEE(ErrorFunction):
    """
    Computes the mean eculidean distance between two np.arrays of any size.

    Methods:
        __call__(self,y_true, y_pred): Returns the mean euclidean distance
            Input: 2 np.arrays of the same shape
                y_true (np.array) : ground truth values
                y_pred (np.array) : predicted values
            Output: Float

    """

    def __call__(self, y_true, y_pred):
        """
        Computes mean euclidean error between predictions and ground truth values.

        Parameters
        ----------
        y_true (np.array) : ground truth values
        y_pred (np.array) : predicted values

        Returns
        -------
        MEE (float), i.e. mean euclidean error between predictions and ground truth values
        """
        return np.mean(np.linalg.norm(y_pred - y_true, axis=1))


class MAE(ErrorFunction):
    """
    Computes the mean absolute error between two np.arrays of any size.

    Methods:
        __call__(self,y_true, y_pred): Returns the mean absolute error
            Input: 2 np.arrays of the same shape
                y_true (np.array) : ground truth values
                y_pred (np.array) : predicted values
            Output: Float
        derivative(self,y_true, y_pred): Returns the derivative of the mean absolute error
            Input: 2 np.arrays (n_observations, n_outputs) of the same shape
            Output: np.array (n_observations, n_outputs)

    """

    def __call__(self, y_true, y_pred):
        """
        Computes mean absolute error between predictions and ground truth values.

        Parameters
        ----------
        y_true (np.array) : ground truth values
        y_pred (np.array) : predicted values

        Returns
        -------
        MAE (float), i.e. mean absolute error between predictions and ground truth values
        """

        if y_true.shape != y_pred.shape:
            raise ValueError("Inputs must have the same shape")

        return np.mean(np.abs(y_pred - y_true))

    def derivative(self, y_true, y_pred):
        """
        Computes derivative of mean absolute error with respect to output of the network.

        Parameters
        ----------
        y_true (np.array) : ground truth values
        y_pred (np.array) : predicted values

        Returns
        -------
        Derivative (np.array), i.e. derivative of mean absolute error with respect to output of the network.
        """

        if y_true.shape != y_pred.shape:
            raise ValueError("Inputs must have the same shape")

        return np.sign(y_pred - y_true) / (y_true.shape[0] * y_true.shape[1])


def get_metric_instance(metric):
    """
    Returns an instance of the metric function class indicated in the input, if present, else returns ValueError.

    Parameters
    ----------
    metric (str) : Name/alias of the metric function

    Returns
    -------
    (MetricFunction) : Instance of the requested metric function
    """
    if metric in ["MSE", "mean_squared_error", "mse", "mean squared error"]:
        return MSE()
    elif metric in ["MAE", "mean_absolute_error", "mae", "mean absolute error"]:
        return MAE()
    elif metric in ["Accuracy", "accuracy", "acc", "ACC", "ACCURACY", "a"]:
        return Accuracy()
    elif metric in [
        "MEE",
        "mean_expected_error",
        "mee",
        "mean expected error",
        "Mean Expected Error",
    ]:
        return MEE()
    else:
        raise ValueError("Metric function not found")
