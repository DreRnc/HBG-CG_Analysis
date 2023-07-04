import numpy as np

class EarlyStopping():

    def __init__ (self, patience, tolerance):
        """
        Initializes the early stopping criterion.

        Parameters
        ----------
        patience (int): number of epochs to wait before stopping
        tolerance (float): tolerance for the early stopping criterion

        """
        self.patience = patience
        self.tolerance = tolerance

    def initialize (self):
        """
        Initializes the early stopping criterion.
        
        """
        self._n_worsening_epochs = 0
        self._best_objective = np.infty



    def __call__ (self, objective):
        """
        Returns True if the training should stop, False otherwise.
        
        Parameters
        ----------
        objective (float): objective function value
        
        Returns
        -------
        bool: True if the training should stop, False otherwise
        
        """
        if objective < self._best_objective - self.tolerance:
            self._best_objective = objective
            self._n_worsening_epochs = 0
        else:
            self._n_worsening_epochs += 1

        if self._n_worsening_epochs > self.patience:
            return True
        else:
            return False
