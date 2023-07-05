import numpy as np

class EarlyStopping():

    def __init__ (self, mode):
        """
        Initializes the early stopping criterion.

        Parameters
        ----------
        patience (int): number of steps to wait before stopping
        tolerance (float): tolerance for the early stopping criterion

        """
        if mode not in ['grad_norm', 'obj_tol']:
            raise ValueError('mode must be either grad_norm or obj_tol')
        self.mode = mode

    def initialize (self, patience, tolerance = 1e-4):
        """
        Initializes the early stopping criterion.
        
        """
        self.patience = patience
        self.tolerance = tolerance
        self._n_worsening_epochs = 0
        self.last_obj = np.inf

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
        if self.mode == 'grad_norm':
            return self._check_grad_norm(objective)
        elif self.mode == 'obj_tol':
            return self._check_obj_tol(objective)
        
    def _check_obj_tol (self, objective):
        """
        Returns True if the training should stop, False otherwise.
        
        Parameters
        ----------
        objective (float): objective function value
        
        Returns
        -------
        bool: True if the training should stop, False otherwise
        
        """
        if abs(objective - self.last_obj) < self.tolerance:
            self._n_worsening_epochs += 1
        else:
            self._n_worsening_epochs = 0

        self.last_obj = objective

        if self._n_worsening_epochs >= self.patience:
            return True
        else:
            return False
    
    def _check_grad_norm (self, grad_norm):
        """
        Returns True if the training should stop, False otherwise.
        
        Parameters
        ----------
        objective (float): objective function value
        
        Returns
        -------
        bool: True if the training should stop, False otherwise
        
        """
        if grad_norm < self.tolerance:
            self._n_worsening_epochs += 1
        else:
            self._n_worsening_epochs = 0

        if self._n_worsening_epochs >= self.patience:
            return True
        else:
            return False
        
        
        
