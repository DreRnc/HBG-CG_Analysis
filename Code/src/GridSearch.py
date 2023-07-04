import multiprocessing
import random
import numpy as np
from tqdm import tqdm

from itertools import product

from joblib import Parallel, delayed

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

class GridSearch():

    def __init__(self, optimizer, model, objective = 'validation_error'):
        '''
        Initializes the GridSearch object.

        Parameters
        ----------
        optimizer (Optimizer): The optimizer to be used
        model (Model): The model to be used
        parameters_grid (Dictionary): The values of parameters to be tested
        objective (String): The objective of the grid search. It can be either "validation_error" or "training_objective".
            "validation_error" means that the score is computed on the validation set;
            "training_objective" means that the score is computed on the training set, including the regularization term;
        '''
        self.optimizer = optimizer
        self.model = model
        if objective not in ['validation_error', 'training_objective']:
            raise ValueError('objective must be either "validation_error" or "training_objective"')
        self.objective = objective
    
    def create_folds(self, n_folds, stratified):
        """
        Creates the folds for the cross validation.

        Parameters
        ----------
        n_folds (Int > 1): The number of folds to be used in the cross validation
        stratified (Bool): If True the folds are stratified

        """
        if type(n_folds) != int:
            raise TypeError('n_folds must be an integer')
        if n_folds < 1:
            raise ValueError('n_folds must be greater than 1')
        if type(stratified) != bool:
            raise TypeError('stratified must be a boolean')
        if stratified and self.objective == 'training_objective':
            raise Warning('stratified is non influent when the objective is "training_objective"')
        if self.objective == 'training_objective' and n_folds != 1:
            raise Warning('n_folds is non influent when the objective is "training_objective"')
        
        if self.objective == "validation_error":
            if n_folds < 2:
                if stratified:
                    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size = self.test_size, stratify = self.y, random_state = 42)
                else:
                    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size = self.test_size, random_state = 42)
            elif stratified: 
                cv = StratifiedKFold(n_splits = n_folds)
                self.folds = list(cv.split(self.X, self.y))
            else:
                cv = KFold(n_splits = n_folds)
                self.folds = list(cv.split(self.X, self.y))

    def fit(self, X, y, parameters_grid, n_folds = 1, stratified = False, test_size = 0.2, verbose = True, parallel = False, random_search = False, n_random = 10, get_eta = False):
        '''
        Performs the grid search.

        Parameters
        ----------
        X (np.array): the input data
        y (np.array): the output data
        parameters_grid (Dictionary): The values of parameters to be tested
        n_folds (Int > 1): The number of folds to be used in the cross validation
        stratified (Bool): If True the folds are stratified
        test_size (Float): The size of the test set if n_folds < 2
        verbose (Bool): If True prints the results of each combination of parameters
        parallel (Bool): If True uses all the cores of the CPU to compute the results
        random_search (Bool): If True takes n_random random combinations of parameters
        n_random (Int): The number of random combinations to be tested
        get_eta (Bool): If True returns the time it took to compute the results

        '''
        self.verbose = verbose
        self.n_folds = n_folds
        self.scores = []
        self.params_list = []
        self.scores_list = []
        self.X = X
        self.y = y
        self.parameters_grid = parameters_grid
        self.test_size = test_size
        self.parallel = parallel
        self.iter = 0
        

        # Creates the folds
        self.create_folds(n_folds, stratified)
        
        # Creates a list with all the combinations of parameters
        par_combinations = list(product(*list(self.parameters_grid.values())))
        self.num_combinations = len(par_combinations)
        
        # If random search is True, it takes n_random random combinations
        if random_search:
            par_combinations = random.sample(par_combinations, n_random)
            if self.verbose:
                print(f'Random search of: {n_random} combinations')
        else: 
            if self.verbose:
                print(f'Grid search of {len(par_combinations)} combinations.')

        self.grid_search(par_combinations)

        self.clean_output()

    def grid_search(self, par_combinations):
        '''
        Performs the grid search.

        Parameters
        ----------
        par_combinations (List): The list of combinations of parameters to be tested

        '''
        # If parallel is True, it uses all the cores of the CPU
        if self.parallel:
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs = num_cores)(delayed(self.fit_model)(par) for par in tqdm(par_combinations))
        else:
            results = [self.fit_model(params) for params in tqdm(par_combinations)]

        # Gets the results
        self.scores = [result[0] for result in results]
        self.params_list = [result[1] for result in results]
        self.scores_list = [result[2] for result in results]
        
    def fit_model(self, params):
         
        """
        Fits the model with the given parameters.

        Parameters
        ----------
        params (Tuple): The tuple of parameters to be tested

        Returns
        ----------
        score (Float): The score of the model
        params (Dictionary): The parameters of the model
        scores (List): The scores of the model in each fold

        """

        # Creates a dictionary with the parameters
        parameters = dict(zip(self.parameters_grid.keys(), params))

        if self.objective == 'validation':
            if self.n_folds < 2:
                self.fit_model_fold(self.X_train, self.y_train, **parameters)
                score = self.model.evaluate_model(self.X_val, self.y_val)
                scores = [score]
            else:
                scores = []
                for train_index, val_index in self.folds:
                    self.fit_model_fold(self.X[train_index], self.y[train_index], **parameters)
                    score = self.model.evaluate_model(self.X[val_index], self.y[val_index])
                    scores.append(score)
                score = np.mean(scores)
        elif self.objective == 'training_objective':
            self.fit_model_fold(self.X, self.y, **parameters)
            score = self.optimizer.obj_history[-1]
            scores = [score]
        
        self.iter += 1

        return score, parameters, scores

    def fit_model_fold(self, X, y, **params):
        """
        Fits the model with the given parameters.
        
        Parameters
        ----------
        X (np.array): the input data
        y (np.array): the output data
        **params: the parameters of the model
        
        """
        self.model.initialize()
        self.optimizer.initialize(self.model, **params)
        self.optimizer.fit_model(X, y)

    def clean_output(self):
        '''
        Clean the output of the grid search
        '''

        self.results = list(zip(self.scores, self.params_list, self.scores_list))

        if self.model.task == 'regression':
            self.results.sort(key = lambda x: x[0])
        elif self.model.task == 'classification':
            self.results.sort(key = lambda x: x[0], reverse = True)
        
        print('\n')
        print(f'Parameters of best model, evaluated on {self.objective}: {self.results[0][1]}')
        print(f'Validation error on {self.n_folds} folds for best model: {self.results[0][2]}')
        print(f'Mean validation error: {self.results[0][0]}')


        self.best_parameters = self.results[0][1]
        self.best_score = self.results[0][0]

        self.model.initialize()
        self.optimizer.initialize(self.model, **self.best_parameters)
        self.optimizer.fit_model(self.X, self.y)
        
        self.best_model = self.model
    
    def get_best_parameters(self, n_parameters = 1, all = False):
        '''
        Returns the best n parameters

        Parameters
        ----------
        n_results (Int): The number of results to be returned
        all (Bool): If True returns all the results

        Returns
        -------
        List of dictionaries: The best n parameters
        '''
        
        if all:
            return self.results
        else:
            return self.results[:n_parameters]